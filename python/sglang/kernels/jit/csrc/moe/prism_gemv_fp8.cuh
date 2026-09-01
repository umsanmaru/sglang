#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
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
// 스레드 배치 — mxfp4 커널과 같은 (kV, kNX, kNY) 파라미터화 (2026-08-29):
//   tx(0..kNX-1) → 열 kV개 (kV B = k 행의 kV 열), ty(0..kNY-1) → 32-k 청크 c ≡ ty (mod kNY)
//   블록 = kNX·kNY 스레드, 열 타일 kNCol = kNX·kV, grid.x = ceil(N/kNCol)
//
// 타일을 고르는 근거는 `prism_gemv_mxfp4.cuh`의 주석과 같다 — m=1 decode에서 열 타일 128은
// 블록을 SM 수 아래로 떨어뜨려 GPU를 놀린다. 다만 **fp8은 원소가 1 B라 같은 치수에서 mxfp4의
// 2배를 읽어** 더 일찍 대역폭 영역에 들어가므로 kV를 4까지 내리지 않는 구간이 있다 (launcher
// 주석). 실측 (µs, 옛→새, RTX 5090, rotating id): q36 gate 26.9→10.1, q36 down 15.5→9.7,
// q3 gate 27.0→12.7, q3 down 16.5→10.5, dsv4 gate 49.8→37.5.
//
// 누산 순서: kV는 어느 열이 어느 스레드로 가느냐만 바꾸므로 비트가 그대로고, kNY는 어느 32-k
// 청크가 어느 ty로 가느냐를 바꾸므로 fp32 묶음이 갈린다 — kNY를 경로·슬롯과 무관하게 고르는
// 이유다 (launcher 주석).
//
// **배율 조회가 mxfp4와 갈리는 지점**: 배율 블록이 128×128이라 옛 타일에서는 열 타일이
// 배율 블록 하나와 정확히 겹쳐 `blockIdx.x`가 곧 배율 열 인덱스였다. 타일이 좁아지면 여러
// 블록이 한 배율 블록을 나눠 쓰므로 열 인덱스를 n0에서 다시 만든다 (`n0 / kBlk`). 타일이
// 128을 넘지 않고 128에 정렬돼 있으므로 블록 안에서는 여전히 값이 하나다 — 벡터 로드도,
// 배율 스트림도 없는 것은 그대로다.
//
// 누산: 청크의 부분합 accb[kV] = Σ_{16 페어} (x0·v_even + x1·v_odd) (fp32, 정확한 곱) →
// acc[kV] += accb · s(그 청크의 128-k 블록). ty의 부분합을 smem 트리로 합친다 (순서 고정
// → 결정적). 정확표현 입력(작은 정수 x, 2의 거듭제곱 배율)에서는 곱도 합도 정확하므로
// 어느 타일이든 같은 값이고, grouped 커널과 비트일치한다는 계약 ⑤가 그대로 성립한다.
//
// SPARSE: 페어 마스크(k2wl2)는 bf16/mxfp4 커널과 **같은 함수**(prism_sparse_common.cuh)로
// 만들고, 죽은 페어의 두 행 로드를 발행하지 않는다.
constexpr int kKTile = 2048;     // x 스테이징 폭 (k)
constexpr int kChunk = 32;       // ty 하나가 맡는 k 행 수 (16 페어)
constexpr int kBlk = prism_fp8::kBlk;  // 128

// kV 바이트를 한 번에 읽는 로드 타입 (mxfp4 커널의 mx_load와 같은 것).
template <int B> struct F8Load;
template <> struct F8Load<4>  { using T = uint32_t; };
template <> struct F8Load<8>  { using T = uint2; };
template <> struct F8Load<16> { using T = uint4; };

template <int B>
__device__ __forceinline__ void f8_load(const uint8_t* p, uint32_t (&w)[B / 4]) {
  union { typename F8Load<B>::T v; uint32_t w[B / 4]; } u;
  u.v = *reinterpret_cast<const typename F8Load<B>::T*>(p);
#pragma unroll
  for (int t = 0; t < B / 4; ++t) w[t] = u.w[t];
}

struct F8Slot {
  const uint8_t* codes;
  const float* scales;
  const int32_t* row_off;
  const uint16_t* kidx;
  long long out_off;
};

template <typename IdxT, bool SPARSE, int kV, int kNX, int kNY>
__global__ void __launch_bounds__(kNX* kNY) prism_gemv_fp8(
    const __nv_bfloat16* __restrict__ x,
    const IdxT* __restrict__ topk,
    __nv_bfloat16* __restrict__ out,
    long long x_kx, long long n_cols, long long out_row, long long top_k,
    int x_row_is_pair, F8Slot s0, F8Slot s1, SparseArgs sp0, SparseArgs sp1) {
  constexpr int kNCol = kNX * kV;  // 블록 열 타일 (≤ kBlk, kBlk에 정렬)
  static_assert(kNCol <= kBlk && kBlk % kNCol == 0, "열 타일은 배율 블록을 쪼개면 안 된다");
  const bool slot1 = (blockIdx.z != 0);
  const F8Slot& s = slot1 ? s1 : s0;
  const SparseArgs& sp = slot1 ? sp1 : sp0;

  const long long pair = blockIdx.y;
  const long long m = pair / top_k;
  const long long e = static_cast<long long>(topk[pair]);
  const long long row = x_row_is_pair ? pair : m;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const long long n0 = static_cast<long long>(blockIdx.x) * kNCol + static_cast<long long>(tx) * kV;
  const long long nblock = (static_cast<long long>(blockIdx.x) * kNCol) / kBlk;  // 배율 열 블록
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
#pragma unroll
      for (int q = 0; q < kChunk / 2; ++q) {
        if constexpr (SPARSE) {
          if (!keep[(g * kChunk) / 2 + q]) continue;
        }
        any = true;
        const float x0 = __bfloat162float(xs[g * kChunk + 2 * q]);
        const float x1 = __bfloat162float(xs[g * kChunk + 2 * q + 1]);
        uint32_t w0[kV / 4], w1[kV / 4];
        f8_load<kV>(cp + static_cast<long long>(2 * q) * n_cols, w0);
        f8_load<kV>(cp + static_cast<long long>(2 * q + 1) * n_cols, w1);
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
    // kV열 bf16 = 2·kV B (out_off 8 배수, n0 kV 배수 — host 검증).
    __nv_bfloat162 h[kV / 2];
#pragma unroll
    for (int j = 0; j < kV / 2; ++j)
      h[j] = __floats2bfloat162_rn(red[0][tx * kV + 2 * j], red[0][tx * kV + 2 * j + 1]);
    __nv_bfloat16* op = out + pair * out_row + s.out_off + n0;
    if constexpr (kV == 4) {
      *reinterpret_cast<uint2*>(op) = *reinterpret_cast<const uint2*>(h);
    } else if constexpr (kV == 8) {
      *reinterpret_cast<uint4*>(op) = *reinterpret_cast<const uint4*>(h);
    } else {
      *reinterpret_cast<uint4*>(op) = reinterpret_cast<const uint4*>(h)[0];
      *reinterpret_cast<uint4*>(op + 8) = reinterpret_cast<const uint4*>(h)[1];
    }
  }
}

// SM 수는 device당 한 번만 물어본다 (launch마다면 graph 캡처 중에도 host 시간이 든다).
inline uint32_t sm_count_of(int device_id) {
  constexpr int kMaxDev = 16;
  static uint32_t cache[kMaxDev] = {};
  if (device_id < 0 || device_id >= kMaxDev) return host::runtime::get_sm_count(device_id);
  if (cache[device_id] == 0) cache[device_id] = host::runtime::get_sm_count(device_id);
  return cache[device_id];
}

#define PRISM_F8_LAUNCH(V, NX, NY)                                                         \
  do {                                                                                     \
    const dim3 block((NX), (NY));                                                          \
    const dim3 grid(static_cast<unsigned int>(div_ceil(n_cols, static_cast<int64_t>((NX) * (V)))), \
                    static_cast<unsigned int>(pairs), static_cast<unsigned int>(slots));    \
    if (is_type<int32_t>(topk.dtype())) {                                                   \
      LaunchKernel(grid, block, device)(                                                    \
          prism_gemv_fp8<int32_t, SPARSE, (V), (NX), (NY)>, x,                              \
          static_cast<const int32_t*>(topk.data_ptr()), out, x_kx, n_cols, out_row, top_k,  \
          x_row_is_pair, s0, s1, sp0, sp1);                                                 \
    } else {                                                                                \
      LaunchKernel(grid, block, device)(                                                    \
          prism_gemv_fp8<int64_t, SPARSE, (V), (NX), (NY)>, x,                              \
          static_cast<const int64_t*>(topk.data_ptr()), out, x_kx, n_cols, out_row, top_k,  \
          x_row_is_pair, s0, s1, sp0, sp1);                                                 \
    }                                                                                       \
  } while (0)

// 타일 선택 — mxfp4와 같은 규칙. 어느 타일을 골라도 결과는 비트 동일하다.
template <bool SPARSE>
inline void launch_gemv_fp8(const DLDevice& device, tvm::ffi::TensorView topk,
                            const __nv_bfloat16* x, __nv_bfloat16* out,
                            int64_t x_kx, int64_t n_cols, int64_t out_row, int64_t top_k,
                            int x_row_is_pair, const F8Slot& s0, const F8Slot& s1,
                            const SparseArgs& sp0, const SparseArgs& sp1,
                            int64_t pairs, int64_t slots, bool w_on_device, int64_t avg_rows) {
  using namespace host;
  // ── 타일 선택 ────────────────────────────────────────────────────────────
  // **kNY는 수치 계약의 일부고 kV는 아니다.** 한 열은 언제나 한 스레드가 갖고, 그 스레드는
  // 자기 32-k 청크를 순서대로 누산한 뒤 청크 부분합을 ty 트리로 합친다. 어느 청크가 어느
  // ty로 가는지는 kNY만 정하므로 — kV를 바꿔도 더하는 항의 순서가 같지만 kNY를 바꾸면
  // 갈린다. 그래서 kNY는 **w_on_device·슬롯 수와 무관한 순수 함수**로 고른다:
  // 그래야 (a) pinned와 device가 비트 동일하고 (test_indexed_pinned_matches_device),
  // (b) gate+up 융합이 2회 launch와 비트 동일하다 (test_gemv_gateup_fusion_all_four_bitwise).
  // 둘 다 **일반 입력** 기준의 계약이라 "정확표현이면 순서 무관"으로는 못 넘어간다.
  //
  // **kV가 mxfp4와 갈리는 이유**: fp8은 원소가 1 B라 같은 치수에서 mxfp4의 2배를 읽는다.
  // 그만큼 먼저 대역폭 영역에 들어가고, 거기서는 스레드를 늘리는 것(kV↓)보다 요청을 넓게
  // 쓰는 것(kV↑)이 이긴다 — kV=4를 그대로 쓰면 dsv4 down이 오히려 느려진다 (37.8→41.6 µs).
  // SM당 읽는 바이트로 가른다 (실측: 49/74 KB/SM → kV=4, 296 KB/SM → kV=8).
  const int64_t sm = static_cast<int64_t>(sm_count_of(device.device_id));
  const int64_t bytes_per_sm = pairs * avg_rows * n_cols / sm;  // 슬롯 수를 세지 않는다
  const int64_t kv_dev = (bytes_per_sm >= (128 << 10)) ? 8 : 4;
  // 블록 수 추정도 같은 이유로 device 타일 기준 · 슬롯 무관이다 (pinned도 이 kNY를 쓴다).
  const int64_t blocks = div_ceil(n_cols, 8 * kv_dev) * pairs;
  const bool wide_ny = blocks < 2 * sm && avg_rows >= 2048;
  if (!w_on_device) {  // warm(pinned): 행 세그먼트 128 B 유지 — kV만 갈리므로 비트는 같다
    if (wide_ny) PRISM_F8_LAUNCH(16, 8, 64);
    else PRISM_F8_LAUNCH(16, 8, 32);
  } else if (kv_dev == 8) {
    if (wide_ny) PRISM_F8_LAUNCH(8, 8, 64);
    else PRISM_F8_LAUNCH(8, 8, 32);
  } else {
    if (wide_ny) PRISM_F8_LAUNCH(4, 8, 64);
    else PRISM_F8_LAUNCH(4, 8, 32);
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
  const int64_t pairs = m * top_k, slots = fused ? 2 : 1;
  const int64_t avg_rows = rows0 / (E1.unwrap() - 1);  // expert당 평균 k 행 (타일 heuristic)
  const __nv_bfloat16* xp = static_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* op = static_cast<__nv_bfloat16*>(out.data_ptr());
  if (sin != nullptr) {
    launch_gemv_fp8<true>(device, topk, xp, op, x_kx, n_cols, out_row, top_k,
                          static_cast<int>(x_row_is_pair), s0, s1, sp0, sp1,
                           pairs, slots, w_on_device, avg_rows);
  } else {
    launch_gemv_fp8<false>(device, topk, xp, op, x_kx, n_cols, out_row, top_k,
                           static_cast<int>(x_row_is_pair), s0, s1, sp0, sp1,
                           pairs, slots, w_on_device, avg_rows);
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
