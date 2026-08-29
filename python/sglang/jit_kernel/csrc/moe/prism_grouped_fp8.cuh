#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <mma.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

#include "prism_fp8.cuh"

namespace {

using namespace nvcuda;

// Prism grouped GEMM — **FP8 e4m3 (128×128 블록 배율)** 판 (prefill 형태, 2026-08-29).
//
// `prism_grouped.cuh`(bf16)·`prism_grouped_mxfp4.cuh`와 같은 구조다: pair를 expert로 묶어
// (grouping.py) 블록 = (expert, 토큰 타일, 열 타일)이 W 타일을 한 번 읽고 그 expert의 토큰
// 최대 kBM개에 곱한다. bf16 tensor core(wmma), fp32 누산, bf16 출력 (계약 ⑤). 갈리는 것은
// **B 타일 로더**다 — e4m3 바이트를 배율과 함께 smem에 bf16으로 푼다.
//
// **수치 계약**: 이 커널은 `bf16(code · scale)`을 tensor core에 넣는다 (= 같은 가중치를
// dequant해 bf16 스토어로 넣은 plan과 **같은 계산**). decode의 GEMV는 fp32 부분합에 fp32
// 배율을 곱하므로, 배율이 2의 거듭제곱이면 둘이 정확히 같고 아니면 W의 bf16 반올림
// 하나만큼(상대 2⁻⁹) 다르다 — fp8 양자화 오차(2⁻⁴)의 1/32이라 무해하고, tensor core가
// bf16을 요구하는 이상 prefill에서 이보다 정확해지려면 누산기를 두 벌 들어야 한다.
//
// 타일: kBM=128 pair × kBN=128 열 × kBK=64 k.
//   **kBN=128 = 배율 블록 하나**이고 kBK=64라 K 타일이 128-k 블록 하나 안에 들어간다 →
//   K 타일당 배율은 **스칼라 하나**다 (mxfp4의 배율 행 2개 + smem 경유가 여기서는 없다).
//   kr은 128 배수(fp8 정렬)라 부분 K 타일도 없다.
// smem: As 128×72 bf16 + Bs 64×136 bf16 → Cs로 재사용 (mxfp4와 동일).
//
// 레이아웃 둘:
//   ROWMAJOR — hot/warm 스토어. codes 행 = k(N 연속), 배율 fp32 [Σk/128, N/128].
//   KT_TILE8 — kt `GemmKernelTileK2FP8B128::BufferB`(cold slab): 32k×256n 타일, 2 KB 청크 =
//              16 페어 × 64 n × 2 k, 배율은 코드(64 B 올림) 뒤 전치 fp32 [K/128][N/128].
//              GPU 타일(64k × 128n) = 타일 컬럼 2 × n-그룹 2 = **2 KB 연속 청크 4개**.
//              바이트 오프셋(ktf8_off): (k&1) + (n&63)·2 + ((k>>1)&15)·128 + ((n>>6)&3)·2048
//              + (k>>5)·8192 + (n>>8)·8192·(K/32).
constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 64;
constexpr int kThreads = 256;
constexpr int kAld = kBK + 8;   // 72
constexpr int kBld = kBN + 8;   // 136
constexpr int kCld = kBN + 4;   // 132
constexpr int kCRows = kBM / 2;
constexpr int kSmemAB = kBM * kAld * 2 + kBK * kBld * 2;
constexpr int kSmemC = kCRows * kCld * 4;
constexpr int kSmemBytes = kSmemAB > kSmemC ? kSmemAB : kSmemC;
constexpr int kBlk = prism_fp8::kBlk;  // 128
constexpr int kColBytes = 8192;        // 타일 컬럼 = 32 k × 256 n × 1 B
constexpr int kGrpBytes = 2048;        // 16 페어 × 64 n × 2 k

struct F8Slot {
  const uint8_t* codes;   // ROWMAJOR: [Σ k, N] / KT_TILE8: slab 시작 (u8)
  const float* scales;    // ROWMAJOR: [Σ k/128, N/128] / KT_TILE8: 미사용 (slab 안)
  const int32_t* row_off; // [E+1] (k 단위)
  const uint16_t* kidx;   // [Σ k]
  long long out_off;
  const int64_t* blk_off; // KT_TILE8 전용: [E] expert 블록의 slab 내 **바이트** 오프셋
};

enum Layout : int { ROWMAJOR = 0, KT_TILE8 = 1 };

template <int LAYOUT>
__global__ void __launch_bounds__(kThreads) prism_grouped_gemm_fp8(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ pair_sorted,
    const int32_t* __restrict__ pair_off,
    const int32_t* __restrict__ tile_off,
    __nv_bfloat16* __restrict__ out,
    int num_experts, long long top_k, long long x_kx, int x_row_is_pair,
    long long n_cols, long long out_row, F8Slot s0, F8Slot s1) {
  const F8Slot s = (blockIdx.z != 0) ? s1 : s0;

  __shared__ __align__(128) unsigned char smem[kSmemBytes];
  __shared__ int sp[kBM];
  __shared__ int srow[kBM];
  __shared__ uint16_t skid[kBK];
  __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Bs = As + kBM * kAld;
  float* Cs = reinterpret_cast<float*>(smem);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  constexpr int WARPS_N = 4;
  const int wm = warp / WARPS_N;  // 0..1 → 행 wm*64
  const int wn = warp % WARPS_N;  // 0..3 → 열 wn*32
  constexpr int FM = 4, FN = 2;

  const long long n0 = static_cast<long long>(blockIdx.x) * kBN;
  // ROWMAJOR B 로더: 스레드 → (k 행 br 0..31, 열 청크 bc = 16열), 두 절반(h)으로 64행.
  const int br = tid >> 3;
  const int bc = (tid & 7) * 16;
  const bool b_active = (n0 + bc) < n_cols;
  // KT_TILE8 로더: cc = 타일 컬럼(0/1), gg = n-그룹(0/1), 청크 안 16 B = tid&63 (+1 KB 두 번째).
  const int t8_cc = tid >> 7, t8_gg = (tid >> 6) & 1, t8_q = tid & 63;
  const int t8_g = static_cast<int>(((n0 & 255) >> 6) + t8_gg);  // super 안의 n-그룹 (0..3)
  const bool t8_active = (n0 + t8_gg * 64) < n_cols;

  const int total_tiles = tile_off[num_experts];
  for (int t = blockIdx.y; t < total_tiles; t += gridDim.y) {
    __syncthreads();  // 이전 타일의 epilogue 완료 후 재사용
    int lo = 0, hi = num_experts - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (tile_off[mid] <= t) lo = mid; else hi = mid - 1;
    }
    const int e = lo;
    const int p0 = pair_off[e] + (t - tile_off[e]) * kBM;
    const int cnt = min(kBM, pair_off[e + 1] - p0);
    if (cnt <= 0) continue;
    const long long o0 = s.row_off[e];
    const long long kr = static_cast<long long>(s.row_off[e + 1]) - o0;
    const long long nblk = n_cols / kBlk;
    const uint8_t* codes_e = s.codes + o0 * n_cols + n0;
    const float* scales_e = s.scales + (o0 / kBlk) * nblk;
    // KT_TILE8: expert 블록 base와 그 안의 super/배율 테이블.
    const uint8_t* blk = (LAYOUT == KT_TILE8) ? s.codes + s.blk_off[e] : nullptr;
    const long long t8_super = (LAYOUT == KT_TILE8)
        ? (n0 >> 8) * static_cast<long long>(kColBytes) * (kr / 32) : 0;
    const float* blk_s = (LAYOUT == KT_TILE8)
        ? reinterpret_cast<const float*>(blk + (((n_cols * kr) + 63) & ~static_cast<long long>(63)))
        : nullptr;

    if (tid < kBM) {
      if (tid < cnt) {
        const int pr = pair_sorted[p0 + tid];
        sp[tid] = pr;
        srow[tid] = x_row_is_pair ? pr : static_cast<int>(pr / top_k);
      } else {
        sp[tid] = -1;
        srow[tid] = 0;
      }
    }

    uint4 breg[2] = {make_uint4(0u, 0u, 0u, 0u), make_uint4(0u, 0u, 0u, 0u)};
    float sreg = 0.f;
    // K 타일 하나(64 k × 128 n)를 레지스터로 선인출한다. 두 레이아웃 모두 스레드당 32 B.
    auto load_b = [&](long long kb) {
      if constexpr (LAYOUT == ROWMAJOR) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const long long k = kb + h * 32 + br;
          breg[h] = (b_active && k < kr)
              ? *reinterpret_cast<const uint4*>(codes_e + k * n_cols + bc)
              : make_uint4(0u, 0u, 0u, 0u);
        }
        sreg = (kb < kr) ? scales_e[(kb / kBlk) * nblk + (n0 / kBlk)] : 0.f;
      } else {
        const long long c = (kb >> 5) + t8_cc;  // 타일 컬럼
        const uint8_t* base = blk + t8_super + c * kColBytes + t8_g * kGrpBytes + t8_q * 16;
        const bool live = t8_active && c * 32 < kr;
#pragma unroll
        for (int h = 0; h < 2; ++h)
          breg[h] = live ? *reinterpret_cast<const uint4*>(base + h * 1024)
                         : make_uint4(0u, 0u, 0u, 0u);
        sreg = (kb < kr) ? blk_s[(kb / kBlk) * nblk + (n0 / kBlk)] : 0.f;
      }
    };
    // 배율은 K 타일당 스칼라 하나다 (kBN = 128 = 배율 블록, kBK = 64 ⊂ 128-k 블록).
    // B를 bf16으로 풀 때 그대로 곱한다 — 위 "수치 계약" 참조.
    auto store_b = [&]() {
      if constexpr (LAYOUT == ROWMAJOR) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t w[4] = {breg[h].x, breg[h].y, breg[h].z, breg[h].w};
          __nv_bfloat16* row = Bs + (h * 32 + br) * kBld + bc;
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            const uint32_t b = (w[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
            row[j] = __float2bfloat16(prism_fp8::e4m3_val(b) * sreg);
          }
        }
      } else {
        // 16 B = 페어 p × n 8개: 바이트 b → 페어 (t8_q·16 + b)>>7, n = ((...)&127)>>1, k 패리티 b&1.
        const int byte0 = t8_q * 16;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t w[4] = {breg[h].x, breg[h].y, breg[h].z, breg[h].w};
          const int b0 = byte0 + h * 1024;
          const int pidx = b0 >> 7;                 // 청크 안 페어 (0..15)
          const int nlo = (b0 & 127) >> 1;          // 그룹 안 첫 n (8개 연속)
          const int krow = t8_cc * 32 + 2 * pidx;   // 타일 안 짝수 k
          const int ncol = t8_gg * 64 + nlo;
          __nv_bfloat16* lo_row = Bs + krow * kBld + ncol;
          __nv_bfloat16* hi_row = lo_row + kBld;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const uint32_t even = (w[(2 * j) >> 2] >> (((2 * j) & 3) * 8)) & 0xFFu;
            const uint32_t odd = (w[(2 * j + 1) >> 2] >> (((2 * j + 1) & 3) * 8)) & 0xFFu;
            lo_row[j] = __float2bfloat16(prism_fp8::e4m3_val(even) * sreg);
            hi_row[j] = __float2bfloat16(prism_fp8::e4m3_val(odd) * sreg);
          }
        }
      }
    };
    load_b(0);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
      for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    for (long long kb = 0; kb < kr; kb += kBK) {
      __syncthreads();  // 이전 mma가 As/Bs 읽기를 끝냈다 (첫 회: sp/srow 가시)
      if (tid < kBK) skid[tid] = (kb + tid < kr) ? s.kidx[o0 + kb + tid] : uint16_t{0};
      __syncthreads();  // skid 가시 — 아래 A gather가 읽는다
      store_b();
      // A gather: 128 × 64 원소, 스레드당 32. 연속 스레드가 한 행의 연속 k를 맡는다.
#pragma unroll
      for (int st = 0; st < (kBM * kBK) / kThreads; ++st) {
        const int idx = tid + st * kThreads;
        const int i = idx >> 6;   // kBK == 64
        const int kk = idx & 63;
        __nv_bfloat16 v = __float2bfloat16(0.f);
        if (i < cnt && kb + kk < kr) v = x[static_cast<long long>(srow[i]) * x_kx + skid[kk]];
        As[i * kAld + kk] = v;
      }
      if (kb + kBK < kr) load_b(kb + kBK);  // 다음 타일 선인출 (PCIe 지연 은닉)
      __syncthreads();  // As/Bs/skid 가시
#pragma unroll
      for (int kk = 0; kk < kBK; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[FM];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b[FN];
#pragma unroll
        for (int i = 0; i < FM; ++i)
          wmma::load_matrix_sync(a[i], As + (wm * 64 + i * 16) * kAld + kk, kAld);
#pragma unroll
        for (int j = 0; j < FN; ++j)
          wmma::load_matrix_sync(b[j], Bs + kk * kBld + wn * 32 + j * 16, kBld);
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
          for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
    // epilogue: 행 절반(64행)씩 (mxfp4/bf16과 동일).
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      __syncthreads();
      if (wm == h) {
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
          for (int j = 0; j < FN; ++j)
            wmma::store_matrix_sync(Cs + (i * 16) * kCld + wn * 32 + j * 16, acc[i][j], kCld,
                                    wmma::mem_row_major);
      }
      __syncthreads();
#pragma unroll
      for (int st = 0; st < (kCRows * (kBN / 8)) / kThreads; ++st) {
        const int idx = tid + st * kThreads;
        const int i = idx >> 4;          // 16 청크/행
        const int c = (idx & 15) * 8;
        const int row = h * kCRows + i;
        if (row < cnt && n0 + c < n_cols) {
          const float* cp = Cs + i * kCld + c;
          uint4 pk;
          __nv_bfloat162* hh = reinterpret_cast<__nv_bfloat162*>(&pk);
          hh[0] = __floats2bfloat162_rn(cp[0], cp[1]);
          hh[1] = __floats2bfloat162_rn(cp[2], cp[3]);
          hh[2] = __floats2bfloat162_rn(cp[4], cp[5]);
          hh[3] = __floats2bfloat162_rn(cp[6], cp[7]);
          *reinterpret_cast<uint4*>(out + static_cast<long long>(sp[row]) * out_row + s.out_off +
                                    n0 + c) = pk;
        }
      }
    }
  }  // tile loop
}

inline bool aligned16(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

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
               " block cols but the store has ", N.unwrap(), " columns");
  RuntimeCheck(aligned16(codes.data_ptr()), what, ": codes must be 16-byte aligned");
  return R.unwrap();
}

// 공통 검증 + launch. `w_on_device`로 스토어의 거처 제약만 갈린다 (hot/warm 쌍둥이).
// max_blocks: launch 블록 수 상한 (0 = 없음).
inline void grouped_fp8_impl(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair,
    bool w_on_device, int64_t max_blocks,
    const tvm::ffi::TensorView* codes_up = nullptr,
    const tvm::ffi::TensorView* scales_up = nullptr,
    const tvm::ffi::TensorView* row_off_up = nullptr,
    const tvm::ffi::TensorView* kidx_up = nullptr,
    int64_t out_col_offset_up = 0,
    int layout = ROWMAJOR,
    const tvm::ffi::TensorView* blk_off = nullptr,
    const tvm::ffi::TensorView* blk_off_up = nullptr,
    int64_t cold_n_cols = 0) {
  using namespace host;

  auto Rx = SymbolicSize{"x_rows"};
  auto Kx = SymbolicSize{"x_cols"};
  auto M = SymbolicSize{"num_tokens"};
  auto K = SymbolicSize{"top_k"};
  auto P = SymbolicSize{"num_pairs"};
  auto E1 = SymbolicSize{"num_experts_plus_one"};
  auto N = SymbolicSize{"n_cols"};
  auto W_row = SymbolicSize{"out_row"};
  auto cuda_device = SymbolicDevice{};

  TensorMatcher({Rx, Kx}).with_dtype<bf16_t>().with_device<kDLCUDA>(cuda_device).verify(x);
  TensorMatcher({P}).with_dtype<int32_t>().with_device(cuda_device).verify(pair_sorted);
  TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(pair_off);
  TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(tile_off);
  TensorMatcher({M, K, W_row}).with_dtype<bf16_t>().with_device(cuda_device).verify(out);
  auto S = SymbolicSize{"slab_bytes"};
  if (layout == KT_TILE8) {
    // slab은 1-D u8(host-register된 kt 메모리)이고 길이는 expert 블록 합이라 N과 무관하다 —
    // n_cols는 인자로 받는다 (노드 N shard 행 수).
    RuntimeCheck(cold_n_cols % 256 == 0, "grouped_fp8_cold: n_cols ", cold_n_cols,
                 " must be a multiple of 256 (tile super)");
    TensorMatcher({S}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>().verify(codes);
    TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(row_off);
    auto R = SymbolicSize{"total_rows"};
    TensorMatcher({R}).with_dtype<uint16_t>().with_device(cuda_device).verify(kidx);
    RuntimeCheck(cold_n_cols > 0 && blk_off != nullptr, "grouped_fp8_cold: n_cols and blk_off required");
    auto E = SymbolicSize{"num_experts"};
    TensorMatcher({E}).with_dtype<int64_t>().with_device(cuda_device).verify(*blk_off);
    RuntimeCheck(E.unwrap() + 1 == E1.unwrap(), "grouped_fp8_cold: blk_off must have E entries");
    N.set_value(cold_n_cols);
  } else {
    verify_f8_store(codes, scales, row_off, kidx, E1, N, cuda_device, w_on_device, "grouped_fp8");
  }

  const int64_t m = M.unwrap(), top_k = K.unwrap(), p = P.unwrap();
  const int64_t n_cols = N.unwrap(), out_row = W_row.unwrap();
  const int64_t x_rows = Rx.unwrap(), x_kx = Kx.unwrap();
  const int64_t num_experts = E1.unwrap() - 1;

  RuntimeCheck(num_experts >= 1, "grouped_fp8: row_off needs E+1 >= 2 entries");
  RuntimeCheck(p == m * top_k, "grouped_fp8: pair_sorted has ", p,
               " entries but out implies M*top_k = ", m * top_k);
  RuntimeCheck(x_row_is_pair ? (x_rows == p) : (x_rows == m),
               "grouped_fp8: x rows (", x_rows, ") must be ", x_row_is_pair ? "M*top_k" : "M");
  RuntimeCheck(x_kx <= 65536, "grouped_fp8: x width ", x_kx, " exceeds the uint16 index range");
  RuntimeCheck(n_cols % kBlk == 0, "grouped_fp8: n_cols ", n_cols,
               " must be a multiple of the scale block ", kBlk);
  RuntimeCheck(out_row % 8 == 0 && out_col_offset % 8 == 0,
               "grouped_fp8: out_row (", out_row, ") and out_col_offset (", out_col_offset,
               ") must be multiples of 8");
  RuntimeCheck(out_col_offset >= 0 && out_col_offset + n_cols <= out_row,
               "grouped_fp8: out cols [", out_col_offset, ",", out_col_offset + n_cols,
               ") out of out width ", out_row);
  RuntimeCheck(aligned16(out.data_ptr()), "grouped_fp8: out must be 16-byte aligned");

  F8Slot s0{static_cast<const uint8_t*>(codes.data_ptr()),
            layout == KT_TILE8 ? nullptr : static_cast<const float*>(scales.data_ptr()),
            static_cast<const int32_t*>(row_off.data_ptr()),
            static_cast<const uint16_t*>(kidx.data_ptr()), out_col_offset,
            blk_off ? static_cast<const int64_t*>(blk_off->data_ptr()) : nullptr};
  F8Slot s1 = s0;
  const bool fused = (codes_up != nullptr);
  if (fused) {
    RuntimeCheck(row_off_up && kidx_up, "grouped_fp8_gateup: up slot needs its offset tensors");
    if (layout == KT_TILE8) {
      auto S2 = SymbolicSize{"slab_bytes_up"};
      TensorMatcher({S2}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>().verify(*codes_up);
      TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(*row_off_up);
      auto R2 = SymbolicSize{"total_rows_up"};
      TensorMatcher({R2}).with_dtype<uint16_t>().with_device(cuda_device).verify(*kidx_up);
      RuntimeCheck(blk_off_up != nullptr, "grouped_fp8_cold_gateup: blk_off_up required");
      auto E2 = SymbolicSize{"num_experts_up"};
      TensorMatcher({E2}).with_dtype<int64_t>().with_device(cuda_device).verify(*blk_off_up);
    } else {
      RuntimeCheck(scales_up != nullptr, "grouped_fp8_gateup: up slot needs scales");
      verify_f8_store(*codes_up, *scales_up, *row_off_up, *kidx_up, E1, N, cuda_device,
                      w_on_device, "grouped_fp8_gateup(up)");
    }
    RuntimeCheck(out_col_offset_up % 8 == 0 && out_col_offset_up >= 0 &&
                 out_col_offset_up + n_cols <= out_row,
                 "grouped_fp8_gateup: up out cols [", out_col_offset_up, ",",
                 out_col_offset_up + n_cols, ") invalid for out width ", out_row);
    s1 = F8Slot{static_cast<const uint8_t*>(codes_up->data_ptr()),
                layout == KT_TILE8 ? nullptr : static_cast<const float*>(scales_up->data_ptr()),
                static_cast<const int32_t*>(row_off_up->data_ptr()),
                static_cast<const uint16_t*>(kidx_up->data_ptr()), out_col_offset_up,
                blk_off_up ? static_cast<const int64_t*>(blk_off_up->data_ptr()) : nullptr};
  }

  const DLDevice device = cuda_device.unwrap();
  int64_t tiles_upper = div_ceil(p, static_cast<int64_t>(kBM)) + num_experts;
  const int64_t grid_x = div_ceil(n_cols, static_cast<int64_t>(kBN));
  const int64_t grid_z = fused ? 2 : 1;
  if (max_blocks > 0) {
    const int64_t cap = std::max<int64_t>(1, max_blocks / (grid_x * grid_z));
    if (cap < tiles_upper) tiles_upper = cap;
  }
  const dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(tiles_upper),
                  static_cast<unsigned int>(grid_z));
  const dim3 block(kThreads);
  auto launch = [&](auto kernel) {
    LaunchKernel(grid, block, device)(
        kernel,
        static_cast<const __nv_bfloat16*>(x.data_ptr()),
        static_cast<const int32_t*>(pair_sorted.data_ptr()),
        static_cast<const int32_t*>(pair_off.data_ptr()),
        static_cast<const int32_t*>(tile_off.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int>(num_experts), static_cast<long long>(top_k),
        static_cast<long long>(x_kx), static_cast<int>(x_row_is_pair),
        static_cast<long long>(n_cols), static_cast<long long>(out_row), s0, s1);
  };
  if (layout == KT_TILE8) launch(prism_grouped_gemm_fp8<KT_TILE8>);
  else launch(prism_grouped_gemm_fp8<ROWMAJOR>);
}

// KT_TILE8 (cold slab) 진입점 — kt fp8 타일 BufferB를 host 메모리(cudaHostRegister됨)에서
// 제자리 읽기. row_off/kidx는 타일 올림(128)된 값, blk_off는 expert 블록의 slab 내 바이트
// 오프셋. out_col_offset에는 노드 N shard 시작이 더해져 온다.
void grouped_fp8_cold(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView slab, tvm::ffi::TensorView blk_off,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair,
    int64_t max_blocks, int64_t n_cols, int64_t layout) {
  host::RuntimeCheck(layout == KT_TILE8, "grouped_fp8_cold: layout must be 1 (kt fp8 tile)");
  grouped_fp8_impl(x, pair_sorted, pair_off, tile_off, slab, slab, row_off, kidx, out,
                   out_col_offset, x_row_is_pair, false, max_blocks,
                   nullptr, nullptr, nullptr, nullptr, 0, KT_TILE8, &blk_off, nullptr, n_cols);
}

void grouped_fp8_cold_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView slab_g, tvm::ffi::TensorView blk_off_g,
    tvm::ffi::TensorView row_off_g, tvm::ffi::TensorView kidx_g,
    tvm::ffi::TensorView slab_u, tvm::ffi::TensorView blk_off_u,
    tvm::ffi::TensorView row_off_u, tvm::ffi::TensorView kidx_u,
    tvm::ffi::TensorView out, int64_t out_col_offset_g, int64_t out_col_offset_u,
    int64_t x_row_is_pair, int64_t max_blocks, int64_t n_cols, int64_t layout) {
  host::RuntimeCheck(layout == KT_TILE8, "grouped_fp8_cold_gateup: layout must be 1");
  grouped_fp8_impl(x, pair_sorted, pair_off, tile_off, slab_g, slab_g, row_off_g, kidx_g, out,
                   out_col_offset_g, x_row_is_pair, false, max_blocks,
                   &slab_u, nullptr, &row_off_u, &kidx_u, out_col_offset_u,
                   KT_TILE8, &blk_off_g, &blk_off_u, n_cols);
}

void grouped_fp8_indexed(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair, int64_t max_blocks) {
  grouped_fp8_impl(x, pair_sorted, pair_off, tile_off, codes, scales, row_off, kidx, out,
                   out_col_offset, x_row_is_pair, true, max_blocks);
}

void grouped_fp8_indexed_pinned(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair, int64_t max_blocks) {
  grouped_fp8_impl(x, pair_sorted, pair_off, tile_off, codes, scales, row_off, kidx, out,
                   out_col_offset, x_row_is_pair, false, max_blocks);
}

void grouped_fp8_indexed_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView codes_g, tvm::ffi::TensorView scales_g,
    tvm::ffi::TensorView row_off_g, tvm::ffi::TensorView kidx_g,
    tvm::ffi::TensorView codes_u, tvm::ffi::TensorView scales_u,
    tvm::ffi::TensorView row_off_u, tvm::ffi::TensorView kidx_u,
    tvm::ffi::TensorView out, int64_t out_col_offset_g, int64_t out_col_offset_u,
    int64_t x_row_is_pair, int64_t max_blocks) {
  grouped_fp8_impl(x, pair_sorted, pair_off, tile_off, codes_g, scales_g, row_off_g, kidx_g, out,
                   out_col_offset_g, x_row_is_pair, true, max_blocks,
                   &codes_u, &scales_u, &row_off_u, &kidx_u, out_col_offset_u);
}

void grouped_fp8_indexed_pinned_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView codes_g, tvm::ffi::TensorView scales_g,
    tvm::ffi::TensorView row_off_g, tvm::ffi::TensorView kidx_g,
    tvm::ffi::TensorView codes_u, tvm::ffi::TensorView scales_u,
    tvm::ffi::TensorView row_off_u, tvm::ffi::TensorView kidx_u,
    tvm::ffi::TensorView out, int64_t out_col_offset_g, int64_t out_col_offset_u,
    int64_t x_row_is_pair, int64_t max_blocks) {
  grouped_fp8_impl(x, pair_sorted, pair_off, tile_off, codes_g, scales_g, row_off_g, kidx_g, out,
                   out_col_offset_g, x_row_is_pair, false, max_blocks,
                   &codes_u, &scales_u, &row_off_u, &kidx_u, out_col_offset_u);
}

}  // namespace
