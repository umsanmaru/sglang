#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

#include "prism_mxfp4.cuh"
#include "prism_sparse_common.cuh"

namespace {

using prism_sparse::SparseArgs;
using prism_sparse::SparseIn;

// Prism worklist GEMV — **MXFP4 pair-row 스토어** 판 (2026-08-27).
//
// 수학은 `prism_gemv.cuh`의 indexed worklist와 같다: pair p=(m,j)가 e=topk[p]의 W
// 행들(K 인덱스 kidx로 gather한 x와 곱)을 읽어 out[m, j, off + n]에 bf16으로 쓴다.
// 다른 것은 스토어 형식 하나다 —
//   codes  u8 [Σₑ k[e]/2, N]   행 = k-페어 (하위 nibble = 짝수 k, 상위 = 홀수 k)
//   scales u8 [Σₑ k[e]/32, N]  행 = 32-k 블록의 E8M0 배율
// row_off는 bf16 스토어와 같은 **k 단위**다 (로더가 k[e]·row_off[e]를 32 배수로 굽는다 —
// 배율 블록이 원본 32행 블록이라 티어 K-인덱스가 블록을 쪼개지 않아야 하기 때문. 계약 ①의
// "정렬은 커널 키가 함의한다").
//
// 스레드 배치 (block 256 = (8, 32), 열 타일 128):
//   tx(0..7) → 열 16개 (uint4 = 16 B = 페어 행의 16 열), 8 스레드가 한 행 128 B를 덮는다
//   ty(0..31) → 32-k 블록 g ≡ ty (mod 32) 를 **통째로** (16 페어 + 배율 1행)
// 워프 = ty 4개 × tx 8개 → 페어 행 4개의 128 B 세그먼트 4개 — PCIe/L2 요청 단위(128 B)를
// 꽉 채운다. 블록당 in-flight: 256 스레드 × 16 B × (언롤 16 페어) = 64 KB.
//
// 누산: 블록 g의 부분합 accb[16] = Σ_{16 페어} (x0·2v_lo + x1·2v_hi) (fp32, 정확한 곱) →
// acc[16] += accb · 2^(e−128). ty 32개의 부분합을 smem 트리로 합친다 (순서 고정 → 결정적).
// 정확표현 입력(작은 정수 x, 배율 2^0)에서 grouped 커널과 비트일치한다 — 계약 ⑤의 exact
// 검출기가 여기에도 적용된다.
//
// SPARSE: 페어 마스크(k2wl2)는 bf16 커널과 **같은 함수**(prism_sparse_common.cuh)로 만들고,
// 죽은 페어의 codes 행(128 B) 로드를 발행하지 않는다. 배율 행은 블록에 산 페어가 하나라도
// 있으면 읽는다 (16 B/스레드, 코드 대비 1/16).
constexpr int kNCol = 128;      // 블록 열 타일 = 페어 행 128 B
constexpr int kV = 16;          // 스레드당 열 (uint4)
constexpr int kNX = kNCol / kV; // 8
constexpr int kNY = 32;         // 블록-of-32 슬롯
constexpr int kKTile = 2048;    // x 스테이징 폭 (k)
constexpr int kBlk = 32;        // 배율 블록 (k)

struct Mx4Slot {
  const uint8_t* codes;
  const uint8_t* scales;
  const int32_t* row_off;
  const uint16_t* kidx;
  long long out_off;
};

template <typename IdxT, bool SPARSE>
__global__ void __launch_bounds__(kNX* kNY) prism_gemv_mxfp4(
    const __nv_bfloat16* __restrict__ x,
    const IdxT* __restrict__ topk,
    __nv_bfloat16* __restrict__ out,
    long long x_kx, long long n_cols, long long out_row, long long top_k,
    int x_row_is_pair, Mx4Slot s0, Mx4Slot s1, SparseArgs sp0, SparseArgs sp1) {
  const bool slot1 = (blockIdx.z != 0);
  const Mx4Slot& s = slot1 ? s1 : s0;
  const SparseArgs& sp = slot1 ? sp1 : sp0;

  const long long pair = blockIdx.y;
  const long long m = pair / top_k;
  const long long e = static_cast<long long>(topk[pair]);
  const long long row = x_row_is_pair ? pair : m;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const long long n0 = static_cast<long long>(blockIdx.x) * kNCol + static_cast<long long>(tx) * kV;
  const bool active = n0 < n_cols;  // n_cols % 16 == 0 (host 검증) → 16열 단위 유효

  const long long o0 = static_cast<long long>(s.row_off[e]);
  const long long kr = static_cast<long long>(s.row_off[e + 1]) - o0;
  const __nv_bfloat16* xr = x + row * x_kx;
  const uint8_t* codes_e = s.codes + (o0 >> 1) * n_cols;
  const uint8_t* scales_e = s.scales + (o0 / kBlk) * n_cols;
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
    const int nblk = cnt / kBlk;  // cnt는 32 배수 (kr·kKTile 모두 32 배수)
    for (int g = ty; g < nblk; g += kNY) {
      float accb[kV];
#pragma unroll
      for (int j = 0; j < kV; ++j) accb[j] = 0.f;
      const long long prow = (base >> 1) + static_cast<long long>(g) * (kBlk / 2);
      const uint8_t* cp = codes_e + prow * n_cols + n0;
      bool any = false;
#pragma unroll 4
      for (int q = 0; q < kBlk / 2; ++q) {
        if constexpr (SPARSE) {
          if (!keep[g * (kBlk / 2) + q]) continue;
        }
        any = true;
        const float x0 = __bfloat162float(xs[g * kBlk + 2 * q]);
        const float x1 = __bfloat162float(xs[g * kBlk + 2 * q + 1]);
        const uint4 c = *reinterpret_cast<const uint4*>(cp + static_cast<long long>(q) * n_cols);
        const uint32_t w[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
        for (int j = 0; j < kV; ++j) {
          const uint32_t b = (w[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
          accb[j] += x0 * prism_mxfp4::fp4_val2(b & 0xFu) + x1 * prism_mxfp4::fp4_val2(b >> 4);
        }
      }
      if (!any) continue;
      const uint4 sc = *reinterpret_cast<const uint4*>(
          scales_e + ((base / kBlk) + g) * n_cols + n0);
      const uint32_t sw[4] = {sc.x, sc.y, sc.z, sc.w};
#pragma unroll
      for (int j = 0; j < kV; ++j) {
        const uint32_t eb = (sw[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
        acc[j] += accb[j] * prism_mxfp4::e8m0_half(eb);
      }
    }
  }

  __syncthreads();  // xs/keep 소비 완료 (red는 별 배열이지만 루프 재진입 대비)
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
inline void launch_gemv_mxfp4(const dim3& grid, const DLDevice& device,
                              tvm::ffi::TensorView topk,
                              const __nv_bfloat16* x, __nv_bfloat16* out,
                              int64_t x_kx, int64_t n_cols, int64_t out_row, int64_t top_k,
                              int x_row_is_pair, const Mx4Slot& s0, const Mx4Slot& s1,
                              const SparseArgs& sp0, const SparseArgs& sp1) {
  using namespace host;
  const dim3 block(kNX, kNY);
  if (is_type<int32_t>(topk.dtype())) {
    LaunchKernel(grid, block, device)(
        prism_gemv_mxfp4<int32_t, SPARSE>, x, static_cast<const int32_t*>(topk.data_ptr()),
        out, x_kx, n_cols, out_row, top_k, x_row_is_pair, s0, s1, sp0, sp1);
  } else {
    LaunchKernel(grid, block, device)(
        prism_gemv_mxfp4<int64_t, SPARSE>, x, static_cast<const int64_t*>(topk.data_ptr()),
        out, x_kx, n_cols, out_row, top_k, x_row_is_pair, s0, s1, sp0, sp1);
  }
}

inline bool aligned16(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

// 스토어 한 슬롯의 검증. R(k 행 수)은 kidx가 정하고 codes/scales는 그 절반/32분의 1이어야
// 한다 — 오프셋 테이블과 스토어가 어긋나면 조용히 남의 행을 읽으므로 여기서 잡는다.
inline int64_t verify_mx4_store(tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
                                tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
                                host::SymbolicSize& E1, host::SymbolicSize& N,
                                host::SymbolicDevice& cuda_device, bool w_on_device,
                                const char* what) {
  using namespace host;
  auto R = SymbolicSize{"total_rows"};
  auto Rp = SymbolicSize{"total_pairs"};
  auto Rg = SymbolicSize{"total_blocks"};
  TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(row_off);
  TensorMatcher({R}).with_dtype<uint16_t>().with_device(cuda_device).verify(kidx);
  if (w_on_device) {
    TensorMatcher({Rp, N}).with_dtype<uint8_t>().with_device(cuda_device).verify(codes);
    TensorMatcher({Rg, N}).with_dtype<uint8_t>().with_device(cuda_device).verify(scales);
  } else {
    TensorMatcher({Rp, N}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>().verify(codes);
    TensorMatcher({Rg, N}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>().verify(scales);
  }
  RuntimeCheck(Rp.unwrap() * 2 == R.unwrap(), what, ": codes has ", Rp.unwrap(),
               " pair rows but kidx has ", R.unwrap(), " rows (must be exactly half)");
  RuntimeCheck(Rg.unwrap() * kBlk == R.unwrap(), what, ": scales has ", Rg.unwrap(),
               " block rows but kidx has ", R.unwrap(), " rows (must be exactly 1/32)");
  RuntimeCheck(aligned16(codes.data_ptr()) && aligned16(scales.data_ptr()),
               what, ": codes/scales must be 16-byte aligned");
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
inline void gemv_mxfp4_impl(
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
  const int64_t rows0 = verify_mx4_store(codes, scales, row_off, kidx, E1, N, cuda_device,
                                         w_on_device, "gemv_mxfp4");

  const int64_t m = M.unwrap(), top_k = K.unwrap();
  const int64_t n_cols = N.unwrap(), out_row = W_row.unwrap();
  const int64_t x_rows = Rx.unwrap(), x_kx = Kx.unwrap();

  RuntimeCheck(x_row_is_pair ? (x_rows == m * top_k) : (x_rows == m),
               "gemv_mxfp4: x rows (", x_rows, ") must be ", x_row_is_pair ? "M*top_k" : "M");
  RuntimeCheck(E1.unwrap() >= 2, "gemv_mxfp4: row_off must have E+1 >= 2 entries");
  RuntimeCheck(x_kx <= 65536, "gemv_mxfp4: x width ", x_kx, " exceeds the uint16 index range");
  RuntimeCheck(n_cols % kV == 0, "gemv_mxfp4: n_cols ", n_cols, " must be a multiple of ", kV);
  RuntimeCheck(out_col_offset >= 0 && out_col_offset % 8 == 0 && out_col_offset + n_cols <= out_row,
               "gemv_mxfp4: out cols [", out_col_offset, ",", out_col_offset + n_cols,
               ") must be 8-aligned and inside out width ", out_row);
  RuntimeCheck(out_row % 8 == 0 && aligned16(out.data_ptr()),
               "gemv_mxfp4: out must be 16-byte aligned with out_row % 8 == 0");
  RuntimeCheck(top_k <= 16, "gemv_mxfp4: top_k ", top_k, " exceeds the per-thread slot budget (16)");

  Mx4Slot s0{static_cast<const uint8_t*>(codes.data_ptr()),
             static_cast<const uint8_t*>(scales.data_ptr()),
             static_cast<const int32_t*>(row_off.data_ptr()),
             static_cast<const uint16_t*>(kidx.data_ptr()), out_col_offset};
  Mx4Slot s1 = s0;
  SparseArgs sp0{}, sp1{};
  auto E = SymbolicSize{"num_experts"};
  auto Ng = SymbolicSize{"sparsity_ng"};
  if (sin != nullptr) {
    sp0 = fill_sparse(*sin, E, Ng, M, K, cuda_device, rows0, "gemv_mxfp4_sparse");
    RuntimeCheck(E1.unwrap() == E.unwrap() + 1, "gemv_mxfp4_sparse: thr has ", E.unwrap(),
                 " experts but row_off implies ", E1.unwrap() - 1);
    sp1 = sp0;
  }
  const bool fused = (codes_up != nullptr);
  if (fused) {
    RuntimeCheck(scales_up && row_off_up && kidx_up, "gemv_mxfp4_gateup: up slot needs all four tensors");
    const int64_t rows1 = verify_mx4_store(*codes_up, *scales_up, *row_off_up, *kidx_up, E1, N,
                                           cuda_device, w_on_device, "gemv_mxfp4_gateup(up)");
    RuntimeCheck(out_col_offset_up >= 0 && out_col_offset_up % 8 == 0 &&
                 out_col_offset_up + n_cols <= out_row,
                 "gemv_mxfp4_gateup: up out cols [", out_col_offset_up, ",",
                 out_col_offset_up + n_cols, ") invalid for out width ", out_row);
    s1 = Mx4Slot{static_cast<const uint8_t*>(codes_up->data_ptr()),
                 static_cast<const uint8_t*>(scales_up->data_ptr()),
                 static_cast<const int32_t*>(row_off_up->data_ptr()),
                 static_cast<const uint16_t*>(kidx_up->data_ptr()), out_col_offset_up};
    if (sin != nullptr) {
      RuntimeCheck(sin_up != nullptr, "gemv_mxfp4_gateup: sparse fusion needs the up sparse spec");
      // proj별로 갈리는 것은 a/c/thr/p/lam 다섯 — 예산 스칼라와 topk_w는 슬롯 0과 공유.
      SparseArgs u = fill_sparse(*sin_up, E, Ng, M, K, cuda_device, rows1,
                                 "gemv_mxfp4_sparse_gateup(up)");
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
    launch_gemv_mxfp4<true>(grid, device, topk, xp, op, x_kx, n_cols, out_row, top_k,
                            static_cast<int>(x_row_is_pair), s0, s1, sp0, sp1);
  } else {
    launch_gemv_mxfp4<false>(grid, device, topk, xp, op, x_kx, n_cols, out_row, top_k,
                             static_cast<int>(x_row_is_pair), s0, s1, sp0, sp1);
  }
}

// ── 진입점 8개: {device, pinned} × {dense, sparse} × {single, gateup} ─────────────
void gemv_mxfp4_indexed(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair) {
  gemv_mxfp4_impl(x, topk, codes, scales, row_off, kidx, out, out_col_offset, x_row_is_pair,
                  true, nullptr);
}

void gemv_mxfp4_indexed_pinned(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair) {
  gemv_mxfp4_impl(x, topk, codes, scales, row_off, kidx, out, out_col_offset, x_row_is_pair,
                  false, nullptr);
}

void gemv_mxfp4_indexed_sparse(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, tvm::ffi::TensorView a, tvm::ffi::TensorView c,
    tvm::ffi::TensorView thr, tvm::ffi::TensorView topk_w,
    int64_t out_col_offset, int64_t x_row_is_pair,
    double p, double lam, double pmax, double grid, int64_t ng, int64_t renorm_it) {
  const SparseIn sin{a, c, thr, topk_w, p, lam, pmax, grid, ng, renorm_it};
  gemv_mxfp4_impl(x, topk, codes, scales, row_off, kidx, out, out_col_offset, x_row_is_pair,
                  true, &sin);
}

void gemv_mxfp4_indexed_pinned_sparse(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, tvm::ffi::TensorView a, tvm::ffi::TensorView c,
    tvm::ffi::TensorView thr, tvm::ffi::TensorView topk_w,
    int64_t out_col_offset, int64_t x_row_is_pair,
    double p, double lam, double pmax, double grid, int64_t ng, int64_t renorm_it) {
  const SparseIn sin{a, c, thr, topk_w, p, lam, pmax, grid, ng, renorm_it};
  gemv_mxfp4_impl(x, topk, codes, scales, row_off, kidx, out, out_col_offset, x_row_is_pair,
                  false, &sin);
}

void gemv_mxfp4_indexed_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes_g, tvm::ffi::TensorView scales_g,
    tvm::ffi::TensorView row_off_g, tvm::ffi::TensorView kidx_g,
    tvm::ffi::TensorView codes_u, tvm::ffi::TensorView scales_u,
    tvm::ffi::TensorView row_off_u, tvm::ffi::TensorView kidx_u,
    tvm::ffi::TensorView out, int64_t out_col_offset_g, int64_t out_col_offset_u,
    int64_t x_row_is_pair) {
  gemv_mxfp4_impl(x, topk, codes_g, scales_g, row_off_g, kidx_g, out, out_col_offset_g,
                  x_row_is_pair, true, nullptr, &codes_u, &scales_u, &row_off_u, &kidx_u,
                  out_col_offset_u);
}

void gemv_mxfp4_indexed_pinned_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
    tvm::ffi::TensorView codes_g, tvm::ffi::TensorView scales_g,
    tvm::ffi::TensorView row_off_g, tvm::ffi::TensorView kidx_g,
    tvm::ffi::TensorView codes_u, tvm::ffi::TensorView scales_u,
    tvm::ffi::TensorView row_off_u, tvm::ffi::TensorView kidx_u,
    tvm::ffi::TensorView out, int64_t out_col_offset_g, int64_t out_col_offset_u,
    int64_t x_row_is_pair) {
  gemv_mxfp4_impl(x, topk, codes_g, scales_g, row_off_g, kidx_g, out, out_col_offset_g,
                  x_row_is_pair, false, nullptr, &codes_u, &scales_u, &row_off_u, &kidx_u,
                  out_col_offset_u);
}

void gemv_mxfp4_indexed_sparse_gateup(
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
  gemv_mxfp4_impl(x, topk, codes_g, scales_g, row_off_g, kidx_g, out, out_col_offset_g,
                  x_row_is_pair, true, &sin, &codes_u, &scales_u, &row_off_u, &kidx_u,
                  out_col_offset_u, &sin_up);
}

void gemv_mxfp4_indexed_pinned_sparse_gateup(
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
  gemv_mxfp4_impl(x, topk, codes_g, scales_g, row_off_g, kidx_g, out, out_col_offset_g,
                  x_row_is_pair, false, &sin, &codes_u, &scales_u, &row_off_u, &kidx_u,
                  out_col_offset_u, &sin_up);
}

}  // namespace
