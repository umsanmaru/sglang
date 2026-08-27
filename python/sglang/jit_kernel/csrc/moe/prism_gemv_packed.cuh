#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// Prism worklist GEMV — **kt packed(AMX) 레이아웃** 판 (2026-08-27).
//
// warm을 kt 포맷 slab 한 벌(pinned, cudaHostRegister)로 두기 위한 decode 커널이다.
// 수학은 `prism_gemv.cuh`의 indexed worklist와 같다: pair p=(m,j)가 e=topk[p]의
// W 슬랩(K 인덱스 kidx로 gather한 x와 곱)을 읽어 out[m, j, off + n]에 bf16으로 쓴다.
// 다른 것은 W 주소식 하나 — `BufferBBF16Impl`의 6D packed 순서(n_block → k_block →
// n_step → k_step → 32×32 타일, 타일 안은 16×16 dword 두 개가 transpose된 VNNI 배치)를
// 그대로 읽는다. 타일 안에서 dword = (k 짝수, k 홀수) 한 쌍 × n 하나이므로,
//   16 B 청크 = 같은 k-페어의 연속 n 4개 = 스레드 하나의 W 로드 단위
// 이고, 페어 sparsity(k2wl2)의 마스크 비트가 정확히 "타일의 16-dword 행 하나(64 B)"에
// 대응한다 — 죽은 페어는 그 행을 읽지 않는다.
//
// 스레드 배치 (block 256 = 64 열 = 타일 2개):
//   ns(0/1) × half(0/1) × kp(0..15) × ng(0..3):  n = n0 + ns·32 + half·16 + ng·4 + {0..3},
//   k-페어 슬롯 kp — 각 K 타일(32행)의 페어 kp를 맡아 ks 루프를 돈다.
//   같은 (ns, half, kp)의 ng 4개가 연속 64 B를 읽고, kp가 연속이면 다음 64 B다.
// 리덕션: kp 16개의 부분합을 smem으로 모아 kp==0 스레드가 쓴다 (순서 고정 → 결정적).
constexpr int kPackedStep = 32;   // kt N_STEP == K_STEP
constexpr int kPackedThreads = 256;
constexpr int kPackedCols = 64;   // 블록 열 = 타일 2개
constexpr int kPackedKTile = 2048;  // x 스테이징 폭 (bf16, 4 KB)

struct PackedSlot {
  const __nv_bfloat16* w;     // slab 시작 (pinned 또는 device)
  const int64_t* blk_off;     // [E] expert 블록 원소 오프셋
  const int32_t* row_off;     // [E+1] k_pad 누적
  const uint16_t* kidx;       // [Σ k_pad] (패딩 → 0)
  long long out_off;          // out3d 열 오프셋 (노드 shard 시작 포함)
  long long n_cols;           // 이 slab의 N (노드 shard 행 수)
  int n_block, k_block;
};

struct PackedSparse {
  const float* a;       // [Σ k_pad] wn²
  const float* c;       // [Σ k_pad / 2]
  const float* thr_tab; // [E, ng]
  const float* topk_w;  // [M, top_k]
  float p, lam, pmax, grid;
  int ng, renorm_it;
};

__device__ __forceinline__ long long packed_tile_base(
    long long n, long long k, long long n_total, long long k_total, int n_block, int k_block) {
  const long long nb = (n / n_block) * n_block;
  const long long nbs = min(static_cast<long long>(n_block), n_total - nb);
  const long long kb = (k / k_block) * k_block;
  const long long kbs = min(static_cast<long long>(k_block), k_total - kb);
  const long long ns = ((n - nb) / kPackedStep) * kPackedStep;
  const long long ks = ((k - kb) / kPackedStep) * kPackedStep;
  return nb * k_total + kb * nbs + ns * kbs + ks * kPackedStep;
}

template <typename IdxT, bool SPARSE>
__global__ void __launch_bounds__(kPackedThreads) prism_gemv_packed(
    const __nv_bfloat16* __restrict__ x,
    const IdxT* __restrict__ topk,
    __nv_bfloat16* __restrict__ out,
    long long x_kx, long long out_row, long long top_k, int x_row_is_pair,
    PackedSlot s0, PackedSlot s1, PackedSparse sp0, PackedSparse sp1) {
  const bool slot1 = (blockIdx.z != 0);
  const PackedSlot s = slot1 ? s1 : s0;
  const PackedSparse& sp = slot1 ? sp1 : sp0;
  const long long pair = blockIdx.y;
  const long long m = pair / top_k;
  const long long e = static_cast<long long>(topk[pair]);
  const long long row = x_row_is_pair ? pair : m;
  const long long n0 = static_cast<long long>(blockIdx.x) * kPackedCols;
  const long long o0 = s.row_off[e];
  const int kr = static_cast<int>(s.row_off[e + 1] - o0);  // k_pad, 32의 배수
  const __nv_bfloat16* xr = x + row * x_kx;
  const uint16_t* ie = s.kidx + o0;
  const __nv_bfloat16* wblk = s.w + s.blk_off[e];

  const int tid = threadIdx.x;
  const int ns = tid >> 7;           // 타일 (0/1)
  const int half = (tid >> 6) & 1;   // n 절반
  const int kp = (tid >> 2) & 15;    // k-페어 슬롯
  const int ng = tid & 3;            // n 그룹 (4열)
  const long long nt = n0 + ns * kPackedStep;
  const long long ncol = nt + half * 16 + ng * 4;
  const bool active = ncol < s.n_cols;  // n_cols % 32 == 0 → 4열 단위 유효

  // 임계값 (SPARSE) — prism_gemv.cuh와 같은 식/반올림.
  float thr2 = 0.f;
  if constexpr (SPARSE) {
    constexpr int MAXK = 16;
    float sv[MAXK];
    const float* wj = sp.topk_w + m * top_k;
    float sum = 0.f;
    for (int i = 0; i < top_k; ++i) sum += wj[i];
    const float inv = 1.f / (sum > 1e-9f ? sum : 1e-9f);
    float gbar = 0.f;
    for (int i = 0; i < top_k; ++i) gbar += wj[i] * inv;
    gbar /= static_cast<float>(top_k);
    const float pmax = sp.pmax;
    auto clip = [pmax](float v) { return v < 0.f ? 0.f : (v > pmax ? pmax : v); };
    for (int i = 0; i < top_k; ++i) sv[i] = clip(sp.p - sp.lam * (wj[i] * inv - gbar));
    for (int it = 0; it < sp.renorm_it; ++it) {
      float mean = 0.f;
      for (int i = 0; i < top_k; ++i) mean += sv[i];
      mean /= static_cast<float>(top_k);
      if (mean < 1e-6f) mean = 1e-6f;
      const float scale = sp.p / mean;
      for (int i = 0; i < top_k; ++i) sv[i] = clip(sv[i] * scale);
    }
    long long gi = static_cast<long long>(rintf(sv[pair % top_k] / sp.grid));
    if (gi < 0) gi = 0;
    if (gi > static_cast<long long>(sp.ng) - 1) gi = static_cast<long long>(sp.ng) - 1;
    const float thr = sp.thr_tab[e * static_cast<long long>(sp.ng) + gi];
    thr2 = thr * thr;
  }

  __shared__ __nv_bfloat16 xs[kPackedKTile];
  __shared__ uint8_t keep[SPARSE ? kPackedKTile / 2 : 1];
  float acc[4] = {0.f, 0.f, 0.f, 0.f};

  for (int base = 0; base < kr; base += kPackedKTile) {
    const int cnt = min(kPackedKTile, kr - base);
    __syncthreads();
    for (int t = tid; t < cnt; t += kPackedThreads) xs[t] = xr[static_cast<long long>(ie[base + t])];
    __syncthreads();
    if constexpr (SPARSE) {
      const int np = cnt >> 1;
      for (int i = tid; i < np; i += kPackedThreads) {
        const float x0 = __bfloat162float(xs[2 * i]);
        const float x1 = __bfloat162float(xs[2 * i + 1]);
        const long long ar = o0 + base + 2 * i;
        float en = sp.a[ar] * x0 * x0 + sp.a[ar + 1] * x1 * x1 + 2.0f * sp.c[ar >> 1] * x0 * x1;
        if (en < 0.f) en = 0.f;
        keep[i] = (en >= thr2) ? uint8_t{1} : uint8_t{0};
      }
      __syncthreads();
    }
    if (active && nt < s.n_cols) {
      // 이 스테이지의 K 타일들: ks = base/32 .. ; 각 타일에서 페어 kp를 맡는다.
      for (int kk = 0; kk < cnt; kk += kPackedStep) {
        const int lk = kk + 2 * kp;  // 타일 내 k (스테이지 로컬)
        if constexpr (SPARSE) {
          if (!keep[lk >> 1]) continue;
        }
        const long long k_abs = base + kk;
        const long long tb = packed_tile_base(nt, k_abs, s.n_cols, kr, s.n_block, s.k_block);
        // 타일 안: half*512 + (kp*16 + i16)*2, i16 = half 안의 n (ng*4 .. +3)
        const uint4 v = *reinterpret_cast<const uint4*>(wblk + tb + half * 512 + (kp * 16 + ng * 4) * 2);
        const __nv_bfloat16* vv = reinterpret_cast<const __nv_bfloat16*>(&v);
        const float x0 = __bfloat162float(xs[lk]);
        const float x1 = __bfloat162float(xs[lk + 1]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          acc[q] += x0 * __bfloat162float(vv[2 * q]) + x1 * __bfloat162float(vv[2 * q + 1]);
        }
      }
    }
  }
  // kp 16개 리덕션 → smem [16][64 열]
  __shared__ float red[16][kPackedCols];
  const int col_local = ns * kPackedStep + half * 16 + ng * 4;
#pragma unroll
  for (int q = 0; q < 4; ++q) red[kp][col_local + q] = acc[q];
  __syncthreads();
  if (kp == 0 && active && nt < s.n_cols) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float t = 0.f;
#pragma unroll
      for (int r = 0; r < 16; ++r) t += red[r][col_local + q];
      out[pair * out_row + s.out_off + ncol + q] = __float2bfloat16(t);
    }
  }
}

struct PackedIn {
  tvm::ffi::TensorView w, blk_off, row_off, kidx;
  int64_t out_col_offset, n_cols, n_block, k_block;
};
struct PackedSparseIn {
  tvm::ffi::TensorView a, c, thr, topk_w;
  double p, lam, pmax, grid;
  int64_t ng, renorm_it;
};

inline PackedSlot make_slot(const PackedIn& in, const DLDevice& dev, int64_t E1) {
  using namespace host;
  auto E = SymbolicSize{"num_experts"};
  auto R = SymbolicSize{"total_rows"};
  auto S = SymbolicSize{"slab_elems"};
  auto cuda_device = SymbolicDevice{};
  cuda_device.set_value(dev);
  TensorMatcher({S}).with_dtype<bf16_t>().with_device<kDLCPU, kDLCUDAHost, kDLCUDA>().verify(in.w);
  TensorMatcher({E}).with_dtype<int64_t>().with_device(cuda_device).verify(in.blk_off);
  TensorMatcher({R}).with_dtype<uint16_t>().with_device(cuda_device).verify(in.kidx);
  RuntimeCheck(E.unwrap() + 1 == E1, "gemv_packed: blk_off must have E entries");
  RuntimeCheck(in.n_cols > 0 && in.n_cols % kPackedStep == 0, "gemv_packed: n_cols must be a multiple of 32, got ", in.n_cols);
  RuntimeCheck(in.n_block % kPackedStep == 0 && in.k_block % kPackedStep == 0 && in.n_block > 0 && in.k_block > 0,
               "gemv_packed: n_block/k_block must be positive multiples of 32");
  RuntimeCheck(reinterpret_cast<std::uintptr_t>(in.w.data_ptr()) % 16 == 0, "gemv_packed: slab must be 16-byte aligned");
  return PackedSlot{static_cast<const __nv_bfloat16*>(in.w.data_ptr()),
                    static_cast<const int64_t*>(in.blk_off.data_ptr()),
                    static_cast<const int32_t*>(in.row_off.data_ptr()),
                    static_cast<const uint16_t*>(in.kidx.data_ptr()),
                    in.out_col_offset, in.n_cols, static_cast<int>(in.n_block), static_cast<int>(in.k_block)};
}

inline void gemv_packed_impl(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView out,
    int64_t x_row_is_pair, const PackedIn& s0, const PackedIn* s1,
    const PackedSparseIn* sp0, const PackedSparseIn* sp1) {
  using namespace host;
  auto Rx = SymbolicSize{"x_rows"};
  auto Kx = SymbolicSize{"x_cols"};
  auto M = SymbolicSize{"num_tokens"};
  auto K = SymbolicSize{"top_k"};
  auto W_row = SymbolicSize{"out_row"};
  auto E1 = SymbolicSize{"num_experts_plus_one"};
  auto cuda_device = SymbolicDevice{};
  TensorMatcher({M, K}).with_dtype<int32_t, int64_t>().with_device<kDLCUDA>(cuda_device).verify(topk);
  TensorMatcher({Rx, Kx}).with_dtype<bf16_t>().with_device(cuda_device).verify(x);
  TensorMatcher({M, K, W_row}).with_dtype<bf16_t>().with_device(cuda_device).verify(out);
  TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(s0.row_off);
  if (s1) TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(s1->row_off);
  const int64_t m = M.unwrap(), top_k = K.unwrap(), out_row = W_row.unwrap();
  const int64_t x_rows = Rx.unwrap(), x_kx = Kx.unwrap();
  RuntimeCheck(x_row_is_pair ? (x_rows == m * top_k) : (x_rows == m), "gemv_packed: x rows mismatch");
  RuntimeCheck(x_kx <= 65536, "gemv_packed: x width exceeds uint16 index range");
  const DLDevice device = cuda_device.unwrap();
  PackedSlot a = make_slot(s0, device, E1.unwrap());
  PackedSlot b = s1 ? make_slot(*s1, device, E1.unwrap()) : a;
  RuntimeCheck(a.out_off >= 0 && a.out_off + a.n_cols <= out_row, "gemv_packed: out cols out of range");
  if (s1) RuntimeCheck(b.out_off >= 0 && b.out_off + b.n_cols <= out_row && b.n_cols == a.n_cols,
                       "gemv_packed: slot1 out cols out of range or n mismatch");
  PackedSparse sa{}, sb{};
  auto fill_sparse = [&](const PackedSparseIn& in, PackedSparse& d) {
    auto Ng = SymbolicSize{"ng"};
    auto E = SymbolicSize{"num_experts"};
    TensorMatcher({E, Ng}).with_dtype<float>().with_device(cuda_device).verify(in.thr);
    TensorMatcher({M, K}).with_dtype<float>().with_device(cuda_device).verify(in.topk_w);
    RuntimeCheck(Ng.unwrap() == in.ng && E.unwrap() + 1 == E1.unwrap(), "gemv_packed: thr shape mismatch");
    RuntimeCheck(top_k <= 16 && in.grid > 0.0, "gemv_packed: top_k <= 16 and grid > 0 required");
    d.a = static_cast<const float*>(in.a.data_ptr());
    d.c = static_cast<const float*>(in.c.data_ptr());
    d.thr_tab = static_cast<const float*>(in.thr.data_ptr());
    d.topk_w = static_cast<const float*>(in.topk_w.data_ptr());
    d.p = static_cast<float>(in.p); d.lam = static_cast<float>(in.lam);
    d.pmax = static_cast<float>(in.pmax); d.grid = static_cast<float>(in.grid);
    d.ng = static_cast<int>(in.ng); d.renorm_it = static_cast<int>(in.renorm_it);
  };
  if (sp0) fill_sparse(*sp0, sa);
  if (sp1) fill_sparse(*sp1, sb); else sb = sa;
  const dim3 grid(static_cast<unsigned int>(div_ceil(a.n_cols, static_cast<int64_t>(kPackedCols))),
                  static_cast<unsigned int>(m * top_k), s1 ? 2u : 1u);
  const dim3 block(kPackedThreads);
#define PRISM_PACKED_LAUNCH(IdxT, SP)                                                              \
  LaunchKernel(grid, block, device)(                                                            \
      prism_gemv_packed<IdxT, SP>, static_cast<const __nv_bfloat16*>(x.data_ptr()),            \
      static_cast<const IdxT*>(topk.data_ptr()), static_cast<__nv_bfloat16*>(out.data_ptr()),  \
      x_kx, out_row, top_k, static_cast<int>(x_row_is_pair), a, b, sa, sb)
  if (is_type<int32_t>(topk.dtype())) {
    if (sp0) PRISM_PACKED_LAUNCH(int32_t, true); else PRISM_PACKED_LAUNCH(int32_t, false);
  } else {
    if (sp0) PRISM_PACKED_LAUNCH(int64_t, true); else PRISM_PACKED_LAUNCH(int64_t, false);
  }
#undef PRISM_PACKED_LAUNCH
}

// dense, 슬롯 1개
void gemv_packed(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView out,
    tvm::ffi::TensorView w, tvm::ffi::TensorView blk_off, tvm::ffi::TensorView row_off,
    tvm::ffi::TensorView kidx, int64_t out_col_offset, int64_t n_cols, int64_t n_block,
    int64_t k_block, int64_t x_row_is_pair) {
  const PackedIn s0{w, blk_off, row_off, kidx, out_col_offset, n_cols, n_block, k_block};
  gemv_packed_impl(x, topk, out, x_row_is_pair, s0, nullptr, nullptr, nullptr);
}

// dense, gate+up 융합 (같은 노드 slab 둘)
void gemv_packed_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView out,
    tvm::ffi::TensorView w_g, tvm::ffi::TensorView blk_g, tvm::ffi::TensorView ro_g, tvm::ffi::TensorView ki_g,
    tvm::ffi::TensorView w_u, tvm::ffi::TensorView blk_u, tvm::ffi::TensorView ro_u, tvm::ffi::TensorView ki_u,
    int64_t out_col_g, int64_t out_col_u, int64_t n_cols, int64_t n_block, int64_t k_block,
    int64_t x_row_is_pair) {
  const PackedIn s0{w_g, blk_g, ro_g, ki_g, out_col_g, n_cols, n_block, k_block};
  const PackedIn s1{w_u, blk_u, ro_u, ki_u, out_col_u, n_cols, n_block, k_block};
  gemv_packed_impl(x, topk, out, x_row_is_pair, s0, &s1, nullptr, nullptr);
}

// sparse (k2wl2), 슬롯 1개
void gemv_packed_sparse(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView out,
    tvm::ffi::TensorView w, tvm::ffi::TensorView blk_off, tvm::ffi::TensorView row_off,
    tvm::ffi::TensorView kidx, tvm::ffi::TensorView a, tvm::ffi::TensorView c,
    tvm::ffi::TensorView thr, tvm::ffi::TensorView topk_w,
    int64_t out_col_offset, int64_t n_cols, int64_t n_block, int64_t k_block,
    int64_t x_row_is_pair, double p, double lam, double pmax, double grid,
    int64_t ng, int64_t renorm_it) {
  const PackedIn s0{w, blk_off, row_off, kidx, out_col_offset, n_cols, n_block, k_block};
  const PackedSparseIn sp{a, c, thr, topk_w, p, lam, pmax, grid, ng, renorm_it};
  gemv_packed_impl(x, topk, out, x_row_is_pair, s0, nullptr, &sp, nullptr);
}

// sparse, gate+up 융합
void gemv_packed_sparse_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView out,
    tvm::ffi::TensorView w_g, tvm::ffi::TensorView blk_g, tvm::ffi::TensorView ro_g, tvm::ffi::TensorView ki_g,
    tvm::ffi::TensorView w_u, tvm::ffi::TensorView blk_u, tvm::ffi::TensorView ro_u, tvm::ffi::TensorView ki_u,
    tvm::ffi::TensorView a_g, tvm::ffi::TensorView c_g, tvm::ffi::TensorView thr_g,
    tvm::ffi::TensorView a_u, tvm::ffi::TensorView c_u, tvm::ffi::TensorView thr_u,
    tvm::ffi::TensorView topk_w,
    int64_t out_col_g, int64_t out_col_u, int64_t n_cols, int64_t n_block, int64_t k_block,
    int64_t x_row_is_pair, double p_g, double lam_g, double p_u, double lam_u,
    double pmax, double grid, int64_t ng, int64_t renorm_it) {
  const PackedIn s0{w_g, blk_g, ro_g, ki_g, out_col_g, n_cols, n_block, k_block};
  const PackedIn s1{w_u, blk_u, ro_u, ki_u, out_col_u, n_cols, n_block, k_block};
  const PackedSparseIn sp0{a_g, c_g, thr_g, topk_w, p_g, lam_g, pmax, grid, ng, renorm_it};
  const PackedSparseIn sp1{a_u, c_u, thr_u, topk_w, p_u, lam_u, pmax, grid, ng, renorm_it};
  gemv_packed_impl(x, topk, out, x_row_is_pair, s0, &s1, &sp0, &sp1);
}

}  // namespace
