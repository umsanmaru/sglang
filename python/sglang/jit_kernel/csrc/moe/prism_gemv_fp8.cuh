#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

#include "prism_fp8.cuh"
#include "prism_sparse_common.cuh"

namespace {

using prism_sparse::SparseArgs;
using prism_sparse::SparseIn;

// Prism worklist GEMV — **FP8 e4m3 (128×128 블록 배율) 스토어** 판 (2026-08-29).
//
// 수학은 `prism_gemv.cuh`의 indexed worklist와 같다: pair p=(m,j)가 e=topk[p]의 W 행들(K
// 인덱스 kidx로 gather한 x와 곱)을 읽어 out[m, j, off + n]에 bf16으로 쓴다. 다른 것은 스토어
// 형식 하나다 —
//   codes  u8   [Σₑ k[e], N]        행 = k (원소 1 B)
//   scales fp32 [Σₑ k[e]/128, N/128] 행 = 128-k 블록, 열 = 128-n 블록
// row_off는 bf16 스토어와 같은 **k 단위**다 (로더가 k[e]·row_off[e]를 128 배수로 굽는다 —
// 배율이 원본 128행 블록에 걸려 있어 티어 K-인덱스가 블록을 쪼개면 "블록당 배율 1"이 깨진다).
//
// 스레드 배치 (block 256 = (8, 32), 열 타일 128):
//   tx(0..7) → 열 16개 (uint4 = 16 B = k 행의 16 열), 8 스레드가 한 행 128 B를 덮는다
//   ty(0..31) → 32-k 청크 c ≡ ty (mod 32) 를 통째로 (16 페어)
// **열 타일 128 = 배율 블록 하나**다. 그래서 배율 조회가 블록당 스칼라 하나(`blockIdx.x` 열)
// 이고 벡터 로드도, 배율 스트림도 없다 — mxfp4(32-k마다 E8M0 16 B)와 갈리는 유일한 지점이다.
//
// 누산: 청크의 부분합 accb[16] = Σ_{16 페어} (x0·v_even + x1·v_odd) (fp32, 정확한 곱) →
// acc[16] += accb · s(그 청크의 128-k 블록). ty 32개의 부분합을 smem 트리로 합친다 (순서 고정
// → 결정적). 정확표현 입력(작은 정수 x, 2의 거듭제곱 배율)에서 grouped 커널과 비트일치한다.
//
// SPARSE: 페어 마스크(k2wl2)는 bf16/mxfp4 커널과 **같은 함수**(prism_sparse_common.cuh)로
// 만들고, 죽은 페어의 두 행(각 128 B) 로드를 발행하지 않는다.
constexpr int kNCol = 128;       // 블록 열 타일 = 배율 블록 = 128 B/행
constexpr int kV = 16;           // 스레드당 열 (uint4)
constexpr int kNX = kNCol / kV;  // 8
constexpr int kNY = 32;          // 청크 슬롯
constexpr int kKTile = 2048;     // x 스테이징 폭 (k)
constexpr int kChunk = 32;       // ty 하나가 맡는 k 행 수 (16 페어)
constexpr int kBlk = prism_fp8::kBlk;  // 128

struct F8Slot {
  const uint8_t* codes;
  const float* scales;
  const int32_t* row_off;
  const uint16_t* kidx;
  long long out_off;
};

template <typename IdxT, bool SPARSE>
__global__ void __launch_bounds__(kNX* kNY) prism_gemv_fp8(
    const __nv_bfloat16* __restrict__ x,
    const IdxT* __restrict__ topk,
    __nv_bfloat16* __restrict__ out,
    long long x_kx, long long n_cols, long long out_row, long long top_k,
    int x_row_is_pair, F8Slot s0, F8Slot s1, SparseArgs sp0, SparseArgs sp1) {
  const bool slot1 = (blockIdx.z != 0);
  const F8Slot& s = slot1 ? s1 : s0;
  const SparseArgs& sp = slot1 ? sp1 : sp0;

  const long long pair = blockIdx.y;
  const long long m = pair / top_k;
  const long long e = static_cast<long long>(topk[pair]);
  const long long row = x_row_is_pair ? pair : m;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const long long nblock = blockIdx.x;                       // 배율 열 블록
  const long long n0 = nblock * kNCol + static_cast<long long>(tx) * kV;
  const bool active = n0 < n_cols;

  const long long o0 = static_cast<long long>(s.row_off[e]);
  const long long kr = static_cast<long long>(s.row_off[e + 1]) - o0;
  const __nv_bfloat16* xr = x + row * x_kx;
  const uint8_t* codes_e = s.codes + o0 * n_cols;
  const float* scales_e = s.scales + (o0 / kBlk) * (n_cols / kBlk);
  const uint16_t* ie = s.kidx + o0;

  float thr2 = 0.f;
  if constexpr (SPARSE) thr2 = prism_sparse::sparse_thr2(sp, m, pair, e, top_k);

  __shared__ __nv_bfloat16 xs[kKTile];
  __shared__ uint8_t keep[SPARSE ? kKTile / 2 : 1];
  __shared__ float red[kNY][kNCol];
  const int tid = ty * kNX + tx;
  constexpr int nthreads = kNX * kNY;

  float acc[kV];
#pragma unroll
  for (int j = 0; j < kV; ++j) acc[j] = 0.f;

  for (long long base = 0; base < kr; base += kKTile) {
    const int cnt = static_cast<int>(min(static_cast<long long>(kKTile), kr - base));
    __syncthreads();
    for (int t = tid; t < cnt; t += nthreads) xs[t] = xr[static_cast<long long>(ie[base + t])];
    __syncthreads();
    if constexpr (SPARSE) {
      const int np = cnt >> 1;
      for (int i = tid; i < np; i += nthreads) {
        const float x0 = __bfloat162float(xs[2 * i]);
        const float x1 = __bfloat162float(xs[2 * i + 1]);
        keep[i] = (prism_sparse::pair_energy(sp, o0 + base + 2 * i, x0, x1) >= thr2)
                      ? uint8_t{1} : uint8_t{0};
      }
      __syncthreads();
    }
    if (!active) continue;
    const int nchunk = cnt / kChunk;  // cnt는 128 배수 (kr·kKTile 모두 128 배수)
    for (int g = ty; g < nchunk; g += kNY) {
      float accb[kV];
#pragma unroll
      for (int j = 0; j < kV; ++j) accb[j] = 0.f;
      const long long krow = base + static_cast<long long>(g) * kChunk;
      const uint8_t* cp = codes_e + krow * n_cols + n0;
      bool any = false;
#pragma unroll 4
      for (int q = 0; q < kChunk / 2; ++q) {
        if constexpr (SPARSE) {
          if (!keep[(g * kChunk) / 2 + q]) continue;
        }
        any = true;
        const float x0 = __bfloat162float(xs[g * kChunk + 2 * q]);
        const float x1 = __bfloat162float(xs[g * kChunk + 2 * q + 1]);
        const uint4 c0 = *reinterpret_cast<const uint4*>(cp + static_cast<long long>(2 * q) * n_cols);
        const uint4 c1 = *reinterpret_cast<const uint4*>(cp + static_cast<long long>(2 * q + 1) * n_cols);
        const uint32_t w0[4] = {c0.x, c0.y, c0.z, c0.w};
        const uint32_t w1[4] = {c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int j = 0; j < kV; ++j) {
          const uint32_t b0 = (w0[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
          const uint32_t b1 = (w1[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
          accb[j] += x0 * prism_fp8::e4m3_val(b0) + x1 * prism_fp8::e4m3_val(b1);
        }
      }
      if (!any) continue;
      // 열 타일이 배율 블록 하나라 스칼라 1개다 (열 인덱스 = blockIdx.x).
      const float sc = scales_e[(krow / kBlk) * (n_cols / kBlk) + nblock];
#pragma unroll
      for (int j = 0; j < kV; ++j) acc[j] += accb[j] * sc;
    }
  }

  __syncthreads();  // xs/keep 소비 완료
#pragma unroll
  for (int j = 0; j < kV; ++j) red[ty][tx * kV + j] = active ? acc[j] : 0.f;
  __syncthreads();
  for (int stride = kNY / 2; stride > 0; stride >>= 1) {
    if (ty < stride) {
#pragma unroll
      for (int j = 0; j < kV; ++j) red[ty][tx * kV + j] += red[ty + stride][tx * kV + j];
    }
    __syncthreads();
  }
  if (ty == 0 && active) {
    // 16열 bf16 = 32 B → uint4 두 개로 쓴다 (out_off·n0가 16 배수 — host 검증).
    uint4 pk[2];
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(pk);
#pragma unroll
    for (int j = 0; j < kV / 2; ++j)
      h[j] = __floats2bfloat162_rn(red[0][tx * kV + 2 * j], red[0][tx * kV + 2 * j + 1]);
    __nv_bfloat16* op = out + pair * out_row + s.out_off + n0;
    *reinterpret_cast<uint4*>(op) = pk[0];
    *reinterpret_cast<uint4*>(op + 8) = pk[1];
  }
}

template <bool SPARSE>
inline void launch_gemv_fp8(const dim3& grid, const DLDevice& device,
                            tvm::ffi::TensorView topk,
                            const __nv_bfloat16* x, __nv_bfloat16* out,
                            int64_t x_kx, int64_t n_cols, int64_t out_row, int64_t top_k,
                            int x_row_is_pair, const F8Slot& s0, const F8Slot& s1,
                            const SparseArgs& sp0, const SparseArgs& sp1) {
  using namespace host;
  const dim3 block(kNX, kNY);
  if (is_type<int32_t>(topk.dtype())) {
    LaunchKernel(grid, block, device)(
        prism_gemv_fp8<int32_t, SPARSE>, x, static_cast<const int32_t*>(topk.data_ptr()),
        out, x_kx, n_cols, out_row, top_k, x_row_is_pair, s0, s1, sp0, sp1);
  } else {
    LaunchKernel(grid, block, device)(
        prism_gemv_fp8<int64_t, SPARSE>, x, static_cast<const int64_t*>(topk.data_ptr()),
        out, x_kx, n_cols, out_row, top_k, x_row_is_pair, s0, s1, sp0, sp1);
  }
}

inline bool aligned16(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

// 스토어 한 슬롯의 검증. R(k 행 수)은 kidx가 정하고 codes는 그와 같은 행 수, scales는
// R/128 × N/128이어야 한다 — 오프셋 테이블과 스토어가 어긋나면 조용히 남의 행을 읽는다.
inline int64_t verify_f8_store(tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
                               tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
                               host::SymbolicSize& E1, host::SymbolicSize& N,
                               host::SymbolicDevice& cuda_device, bool w_on_device,
                               const char* what) {
  using namespace host;
  auto R = SymbolicSize{"total_rows"};
  auto Rb = SymbolicSize{"total_k_blocks"};
  auto Nb = SymbolicSize{"n_blocks"};
  TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(row_off);
  TensorMatcher({R}).with_dtype<uint16_t>().with_device(cuda_device).verify(kidx);
  if (w_on_device) {
    TensorMatcher({R, N}).with_dtype<uint8_t>().with_device(cuda_device).verify(codes);
    TensorMatcher({Rb, Nb}).with_dtype<float>().with_device(cuda_device).verify(scales);
  } else {
    TensorMatcher({R, N}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>().verify(codes);
    TensorMatcher({Rb, Nb}).with_dtype<float>().with_device<kDLCPU, kDLCUDAHost>().verify(scales);
  }
  RuntimeCheck(Rb.unwrap() * kBlk == R.unwrap(), what, ": scales has ", Rb.unwrap(),
               " block rows but kidx has ", R.unwrap(), " rows (must be exactly 1/", kBlk, ")");
  RuntimeCheck(Nb.unwrap() * kBlk == N.unwrap(), what, ": scales has ", Nb.unwrap(),
               " block cols but the store has ", N.unwrap(), " columns (must be exactly 1/", kBlk, ")");
  RuntimeCheck(aligned16(codes.data_ptr()), what, ": codes must be 16-byte aligned");
  return R.unwrap();
}

inline SparseArgs fill_sparse(const SparseIn& sin, host::SymbolicSize& E, host::SymbolicSize& Ng,
                              host::SymbolicSize& M, host::SymbolicSize& K,
                              host::SymbolicDevice& cuda_device, int64_t rows, const char* what) {
  using namespace host;
  auto Ra = SymbolicSize{"a_rows"};
  auto Rc = SymbolicSize{"c_pairs"};
  TensorMatcher({Ra}).with_dtype<float>().with_device(cuda_device).verify(sin.a);
  TensorMatcher({Rc}).with_dtype<float>().with_device(cuda_device).verify(sin.c);
  TensorMatcher({E, Ng}).with_dtype<float>().with_device(cuda_device).verify(sin.thr);
  TensorMatcher({M, K}).with_dtype<float>().with_device(cuda_device).verify(sin.topk_w);
  RuntimeCheck(Ra.unwrap() == rows, what, ": wn² has ", Ra.unwrap(), " rows but the store has ", rows);
  RuntimeCheck(Rc.unwrap() * 2 == rows, what, ": pair_dot has ", Rc.unwrap(),
               " entries but the store has ", rows, " rows (must be exactly half)");
  RuntimeCheck(Ng.unwrap() == sin.ng, what, ": thr grid ", Ng.unwrap(), " != ng ", sin.ng);
  RuntimeCheck(sin.grid > 0.0, what, ": grid must be positive, got ", sin.grid);
  SparseArgs sp{};
  sp.a = static_cast<const float*>(sin.a.data_ptr());
  sp.c = static_cast<const float*>(sin.c.data_ptr());
  sp.thr_tab = static_cast<const float*>(sin.thr.data_ptr());
  sp.topk_w = static_cast<const float*>(sin.topk_w.data_ptr());
  sp.p = static_cast<float>(sin.p);
  sp.lam = static_cast<float>(sin.lam);
  sp.pmax = static_cast<float>(sin.pmax);
  sp.grid = static_cast<float>(sin.grid);
  sp.ng = static_cast<int>(sin.ng);
  sp.renorm_it = static_cast<int>(sin.renorm_it);
  return sp;
}

// 공통 검증 + launch. 슬롯 1(gate+up 융합)은 nullptr이면 grid.z=1.
inline void gemv_fp8_impl(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair,
    bool w_on_device, const SparseIn* sin,
    const tvm::ffi::TensorView* codes_up = nullptr,
    const tvm::ffi::TensorView* scales_up = nullptr,
    const tvm::ffi::TensorView* row_off_up = nullptr,
    const tvm::ffi::TensorView* kidx_up = nullptr,
    int64_t out_col_offset_up = 0, const SparseIn* sin_up = nullptr) {
  using namespace host;

  auto Rx = SymbolicSize{"x_rows"};
  auto Kx = SymbolicSize{"x_cols"};
  auto M = SymbolicSize{"num_tokens"};
  auto K = SymbolicSize{"top_k"};
  auto E1 = SymbolicSize{"num_experts_plus_one"};
  auto N = SymbolicSize{"n_cols"};
  auto W_row = SymbolicSize{"out_row"};
  auto cuda_device = SymbolicDevice{};

  TensorMatcher({M, K}).with_dtype<int32_t, int64_t>().with_device<kDLCUDA>(cuda_device).verify(topk);
  TensorMatcher({Rx, Kx}).with_dtype<bf16_t>().with_device(cuda_device).verify(x);
  TensorMatcher({M, K, W_row}).with_dtype<bf16_t>().with_device(cuda_device).verify(out);
  const int64_t rows0 = verify_f8_store(codes, scales, row_off, kidx, E1, N, cuda_device,
                                        w_on_device, "gemv_fp8");

  const int64_t m = M.unwrap(), top_k = K.unwrap();
  const int64_t n_cols = N.unwrap(), out_row = W_row.unwrap();
  const int64_t x_rows = Rx.unwrap(), x_kx = Kx.unwrap();

  RuntimeCheck(x_row_is_pair ? (x_rows == m * top_k) : (x_rows == m),
               "gemv_fp8: x rows (", x_rows, ") must be ", x_row_is_pair ? "M*top_k" : "M");
  RuntimeCheck(E1.unwrap() >= 2, "gemv_fp8: row_off must have E+1 >= 2 entries");
  RuntimeCheck(x_kx <= 65536, "gemv_fp8: x width ", x_kx, " exceeds the uint16 index range");
  RuntimeCheck(n_cols % kBlk == 0, "gemv_fp8: n_cols ", n_cols,
               " must be a multiple of the scale block ", kBlk);
  RuntimeCheck(out_col_offset >= 0 && out_col_offset % 8 == 0 && out_col_offset + n_cols <= out_row,
               "gemv_fp8: out cols [", out_col_offset, ",", out_col_offset + n_cols,
               ") must be 8-aligned and inside out width ", out_row);
  RuntimeCheck(out_row % 8 == 0 && aligned16(out.data_ptr()),
               "gemv_fp8: out must be 16-byte aligned with out_row % 8 == 0");
  RuntimeCheck(top_k <= 16, "gemv_fp8: top_k ", top_k, " exceeds the per-thread slot budget (16)");

  F8Slot s0{static_cast<const uint8_t*>(codes.data_ptr()),
            static_cast<const float*>(scales.data_ptr()),
            static_cast<const int32_t*>(row_off.data_ptr()),
            static_cast<const uint16_t*>(kidx.data_ptr()), out_col_offset};
  F8Slot s1 = s0;
  SparseArgs sp0{}, sp1{};
  auto E = SymbolicSize{"num_experts"};
  auto Ng = SymbolicSize{"sparsity_ng"};
  if (sin != nullptr) {
    sp0 = fill_sparse(*sin, E, Ng, M, K, cuda_device, rows0, "gemv_fp8_sparse");
    RuntimeCheck(E1.unwrap() == E.unwrap() + 1, "gemv_fp8_sparse: thr has ", E.unwrap(),
                 " experts but row_off implies ", E1.unwrap() - 1);
    sp1 = sp0;
  }
  const bool fused = (codes_up != nullptr);
  if (fused) {
    RuntimeCheck(scales_up && row_off_up && kidx_up, "gemv_fp8_gateup: up slot needs all four tensors");
    const int64_t rows1 = verify_f8_store(*codes_up, *scales_up, *row_off_up, *kidx_up, E1, N,
                                          cuda_device, w_on_device, "gemv_fp8_gateup(up)");
    RuntimeCheck(out_col_offset_up >= 0 && out_col_offset_up % 8 == 0 &&
                 out_col_offset_up + n_cols <= out_row,
                 "gemv_fp8_gateup: up out cols [", out_col_offset_up, ",",
                 out_col_offset_up + n_cols, ") invalid for out width ", out_row);
    s1 = F8Slot{static_cast<const uint8_t*>(codes_up->data_ptr()),
                static_cast<const float*>(scales_up->data_ptr()),
                static_cast<const int32_t*>(row_off_up->data_ptr()),
                static_cast<const uint16_t*>(kidx_up->data_ptr()), out_col_offset_up};
    if (sin != nullptr) {
      RuntimeCheck(sin_up != nullptr, "gemv_fp8_gateup: sparse fusion needs the up sparse spec");
      // proj별로 갈리는 것은 a/c/thr/p/lam 다섯 — 예산 스칼라와 topk_w는 슬롯 0과 공유.
      SparseArgs u = fill_sparse(*sin_up, E, Ng, M, K, cuda_device, rows1,
                                 "gemv_fp8_sparse_gateup(up)");
      sp1 = sp0;
      sp1.a = u.a; sp1.c = u.c; sp1.thr_tab = u.thr_tab; sp1.p = u.p; sp1.lam = u.lam;
    }
  }

  const DLDevice device = cuda_device.unwrap();
  const dim3 grid(static_cast<unsigned int>(div_ceil(n_cols, static_cast<int64_t>(kNCol))),
                  static_cast<unsigned int>(m * top_k), fused ? 2u : 1u);
  const __nv_bfloat16* xp = static_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* op = static_cast<__nv_bfloat16*>(out.data_ptr());
  if (sin != nullptr) {
    launch_gemv_fp8<true>(grid, device, topk, xp, op, x_kx, n_cols, out_row, top_k,
                          static_cast<int>(x_row_is_pair), s0, s1, sp0, sp1);
  } else {
    launch_gemv_fp8<false>(grid, device, topk, xp, op, x_kx, n_cols, out_row, top_k,
                           static_cast<int>(x_row_is_pair), s0, s1, sp0, sp1);
  }
}

// ── 진입점 8개: {device, pinned} × {dense, sparse} × {single, gateup} ─────────────
void gemv_fp8_indexed(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair) {
  gemv_fp8_impl(x, topk, codes, scales, row_off, kidx, out, out_col_offset, x_row_is_pair,
                true, nullptr);
}

void gemv_fp8_indexed_pinned(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair) {
  gemv_fp8_impl(x, topk, codes, scales, row_off, kidx, out, out_col_offset, x_row_is_pair,
                false, nullptr);
}

void gemv_fp8_indexed_sparse(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, tvm::ffi::TensorView a, tvm::ffi::TensorView c,
    tvm::ffi::TensorView thr, tvm::ffi::TensorView topk_w,
    int64_t out_col_offset, int64_t x_row_is_pair,
    double p, double lam, double pmax, double grid, int64_t ng, int64_t renorm_it) {
  const SparseIn sin{a, c, thr, topk_w, p, lam, pmax, grid, ng, renorm_it};
  gemv_fp8_impl(x, topk, codes, scales, row_off, kidx, out, out_col_offset, x_row_is_pair,
                true, &sin);
}

void gemv_fp8_indexed_pinned_sparse(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, tvm::ffi::TensorView a, tvm::ffi::TensorView c,
    tvm::ffi::TensorView thr, tvm::ffi::TensorView topk_w,
    int64_t out_col_offset, int64_t x_row_is_pair,
    double p, double lam, double pmax, double grid, int64_t ng, int64_t renorm_it) {
  const SparseIn sin{a, c, thr, topk_w, p, lam, pmax, grid, ng, renorm_it};
  gemv_fp8_impl(x, topk, codes, scales, row_off, kidx, out, out_col_offset, x_row_is_pair,
                false, &sin);
}

void gemv_fp8_indexed_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes_g, tvm::ffi::TensorView scales_g,
    tvm::ffi::TensorView row_off_g, tvm::ffi::TensorView kidx_g,
    tvm::ffi::TensorView codes_u, tvm::ffi::TensorView scales_u,
    tvm::ffi::TensorView row_off_u, tvm::ffi::TensorView kidx_u,
    tvm::ffi::TensorView out, int64_t out_col_offset_g, int64_t out_col_offset_u,
    int64_t x_row_is_pair) {
  gemv_fp8_impl(x, topk, codes_g, scales_g, row_off_g, kidx_g, out, out_col_offset_g,
                x_row_is_pair, true, nullptr, &codes_u, &scales_u, &row_off_u, &kidx_u,
                out_col_offset_u);
}

void gemv_fp8_indexed_pinned_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes_g, tvm::ffi::TensorView scales_g,
    tvm::ffi::TensorView row_off_g, tvm::ffi::TensorView kidx_g,
    tvm::ffi::TensorView codes_u, tvm::ffi::TensorView scales_u,
    tvm::ffi::TensorView row_off_u, tvm::ffi::TensorView kidx_u,
    tvm::ffi::TensorView out, int64_t out_col_offset_g, int64_t out_col_offset_u,
    int64_t x_row_is_pair) {
  gemv_fp8_impl(x, topk, codes_g, scales_g, row_off_g, kidx_g, out, out_col_offset_g,
                x_row_is_pair, false, nullptr, &codes_u, &scales_u, &row_off_u, &kidx_u,
                out_col_offset_u);
}

void gemv_fp8_indexed_sparse_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes_g, tvm::ffi::TensorView scales_g,
    tvm::ffi::TensorView row_off_g, tvm::ffi::TensorView kidx_g,
    tvm::ffi::TensorView codes_u, tvm::ffi::TensorView scales_u,
    tvm::ffi::TensorView row_off_u, tvm::ffi::TensorView kidx_u,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a_g, tvm::ffi::TensorView c_g, tvm::ffi::TensorView thr_g,
    tvm::ffi::TensorView a_u, tvm::ffi::TensorView c_u, tvm::ffi::TensorView thr_u,
    tvm::ffi::TensorView topk_w,
    int64_t out_col_offset_g, int64_t out_col_offset_u, int64_t x_row_is_pair,
    double p_g, double lam_g, double p_u, double lam_u,
    double pmax, double grid, int64_t ng, int64_t renorm_it) {
  const SparseIn sin{a_g, c_g, thr_g, topk_w, p_g, lam_g, pmax, grid, ng, renorm_it};
  const SparseIn sin_up{a_u, c_u, thr_u, topk_w, p_u, lam_u, pmax, grid, ng, renorm_it};
  gemv_fp8_impl(x, topk, codes_g, scales_g, row_off_g, kidx_g, out, out_col_offset_g,
                x_row_is_pair, true, &sin, &codes_u, &scales_u, &row_off_u, &kidx_u,
                out_col_offset_u, &sin_up);
}

void gemv_fp8_indexed_pinned_sparse_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes_g, tvm::ffi::TensorView scales_g,
    tvm::ffi::TensorView row_off_g, tvm::ffi::TensorView kidx_g,
    tvm::ffi::TensorView codes_u, tvm::ffi::TensorView scales_u,
    tvm::ffi::TensorView row_off_u, tvm::ffi::TensorView kidx_u,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a_g, tvm::ffi::TensorView c_g, tvm::ffi::TensorView thr_g,
    tvm::ffi::TensorView a_u, tvm::ffi::TensorView c_u, tvm::ffi::TensorView thr_u,
    tvm::ffi::TensorView topk_w,
    int64_t out_col_offset_g, int64_t out_col_offset_u, int64_t x_row_is_pair,
    double p_g, double lam_g, double p_u, double lam_u,
    double pmax, double grid, int64_t ng, int64_t renorm_it) {
  const SparseIn sin{a_g, c_g, thr_g, topk_w, p_g, lam_g, pmax, grid, ng, renorm_it};
  const SparseIn sin_up{a_u, c_u, thr_u, topk_w, p_u, lam_u, pmax, grid, ng, renorm_it};
  gemv_fp8_impl(x, topk, codes_g, scales_g, row_off_g, kidx_g, out, out_col_offset_g,
                x_row_is_pair, false, &sin, &codes_u, &scales_u, &row_off_u, &kidx_u,
                out_col_offset_u, &sin_up);
}

}  // namespace
