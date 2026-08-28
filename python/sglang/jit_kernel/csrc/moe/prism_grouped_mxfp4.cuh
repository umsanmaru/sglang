#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <mma.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

#include "prism_mxfp4.cuh"

namespace {

using namespace nvcuda;

// Prism grouped GEMM — **MXFP4 pair-row 스토어** 판 (prefill 형태, 2026-08-27).
//
// `prism_grouped.cuh`(bf16 ROWMAJOR)와 같은 구조다: pair를 expert로 묶어(grouping.py)
// 블록 = (expert, 토큰 타일, 열 타일)이 W 타일을 한 번 읽고 그 expert의 토큰 최대 kBM개에
// 곱한다. bf16 tensor core(wmma), fp32 누산, bf16 출력 (계약 ⑤). 갈리는 것은 **B 타일
// 로더**다 — fp4 코드 + E8M0 배율을 smem에 bf16으로 푼다 (e2m1 × 2^e는 bf16 격자에 있어
// 정확표현이므로 mma 이하는 W4A16 GEMV와 같은 값을 곱한다).
//
// 타일: kBM=128 pair × kBN=128 열 × kBK=64 k.
//   kBN=128 → 페어 행 하나가 128 B 연속 = PCIe/L2 요청 단위 (fp4는 열 하나가 0.5 B).
//   kBK=64  → 코드 타일 = 32 페어행 × 128 B = 4 KB = 256 스레드 × uint4 하나. 배율은 2 블록행
//             × 128 B = 16 uint4 (스레드 0..15). 다음 K 타일을 레지스터에 선인출한다.
// smem: As 128×72 bf16 (18 KB) + Bs 64×136 bf16 (17 KB) → Cs로 재사용 (64×132 fp32, 33 KB;
// 128행을 두 절반으로 나눠 쓴다 — 48 KB 정적 한도 안).
// warp 배치: 8 warp = wm(0..1) × wn(0..3) → warp 타일 64행 × 32열 (FM=4, FN=2).
//
// grid = (ceil(N/kBN), min(P/kBM + E, max_blocks/…), fused ? 2 : 1). 타일 축은 persistent
// (gridDim.y 스트라이드). expert별 K 행 수 kr은 32 배수(배율 블록)지만 64 배수는 아닐 수
// 있어 마지막 K 타일의 하위 32행은 0으로 채운다.
constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 64;
constexpr int kThreads = 256;
constexpr int kAld = kBK + 8;   // 72
constexpr int kBld = kBN + 8;   // 136
constexpr int kCld = kBN + 4;   // 132
constexpr int kCRows = kBM / 2; // epilogue 절반
constexpr int kSmemAB = kBM * kAld * 2 + kBK * kBld * 2;  // 18432 + 17408 = 35840
constexpr int kSmemC = kCRows * kCld * 4;                 // 33792
constexpr int kSmemBytes = kSmemAB > kSmemC ? kSmemAB : kSmemC;
constexpr int kBlk = 32;

struct Mx4Slot {
  const uint8_t* codes;   // PAIRROW: [Σ k/2, N] / KT_FP4: slab 시작 (u8)
  const uint8_t* scales;  // PAIRROW: [Σ k/32, N] / KT_FP4: 미사용
  const int32_t* row_off; // [E+1] (k 단위)
  const uint16_t* kidx;   // [Σ k]
  long long out_off;
  const int64_t* blk_off; // KT_FP4 전용: [E] expert 블록의 slab 내 **바이트** 오프셋
};

// B 타일 레이아웃.
//   PAIRROW — hot/warm 스토어 (codes 행 = k-페어, N 연속 / scales 행 = 32-k 블록).
//   KT_FP4  — kt `BufferBInt4KGroupImpl`(cold slab): expert 블록 = [n][k/2] nibble 행우선 +
//             fp32 d[n][k/32]. GPU가 cudaHostRegister된 slab을 **재배치 없이** 읽는다 —
//             cold weight가 CPU(kt fp4 커널)와 GPU 사이에서 한 벌이다. 행 하나의 64-k 조각
//             = 32 B 연속 (스레드 2개 × uint4); 128 행이 kr/2 B 간격이라 세그먼트가 짧다
//             (PCIe 효율은 PAIRROW보다 낮다 — cold prefill의 알려진 비용).
enum Layout : int { PAIRROW = 0, KT_FP4 = 1 };

template <int LAYOUT>
__global__ void __launch_bounds__(kThreads) prism_grouped_gemm_mxfp4(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ pair_sorted,
    const int32_t* __restrict__ pair_off,
    const int32_t* __restrict__ tile_off,
    __nv_bfloat16* __restrict__ out,
    int num_experts, long long top_k, long long x_kx, int x_row_is_pair,
    long long n_cols, long long out_row, Mx4Slot s0, Mx4Slot s1) {
  const Mx4Slot s = (blockIdx.z != 0) ? s1 : s0;

  __shared__ __align__(128) unsigned char smem[kSmemBytes];
  __shared__ int sp[kBM];
  __shared__ int srow[kBM];
  __shared__ uint16_t skid[kBK];
  __shared__ uint8_t ss[2][kBN];  // 이 K 타일의 배율 2행
  __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Bs = As + kBM * kAld;
  float* Cs = reinterpret_cast<float*>(smem);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  constexpr int WARPS_N = 4;
  const int wm = warp / WARPS_N;  // 0..1 → 행 wm*64
  const int wn = warp % WARPS_N;  // 0..3 → 열 wn*32
  constexpr int FM = 4, FN = 2;

  // B 로더 좌표: 스레드 → (페어 행 br 0..31, 열 청크 bc = 16열)
  const int br = tid >> 3;
  const int bc = (tid & 7) * 16;
  const long long n0 = static_cast<long long>(blockIdx.x) * kBN;
  const bool b_active = (n0 + bc) < n_cols;  // n_cols % 16 == 0 (host 검증)

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
    const uint8_t* codes_e = s.codes + (o0 >> 1) * n_cols + n0;
    const uint8_t* scales_e = s.scales + (o0 / kBlk) * n_cols + n0;
    // KT_FP4: expert 블록 base, 행 stride kr/2 B, 배율 fp32 테이블은 코드 뒤.
    const uint8_t* blk = (LAYOUT == KT_FP4) ? s.codes + s.blk_off[e] : nullptr;
    const float* blk_d = (LAYOUT == KT_FP4)
        ? reinterpret_cast<const float*>(blk + n_cols * (kr >> 1)) : nullptr;
    // KT_FP4 스레드 배치 (warp-연속 로드): lane l = tid & 7 이 한 행의 256-k 청크 안 16 B 조각
    // (k [l·32, l·32+32))을, 행 r = tid >> 3 (0..31) + 32·i (i=0..3) 네 행을 든다 → load 명령 하나에서
    // 인접 lane 8개가 같은 행의 인접 16 B = **128 B 연속**, warp가 행 4개. (행마다 lane을 갈랐던
    // 배치는 명령당 32 B 섹터 32개라 PCIe 실효 12–16 GB/s에 머물렀다, 2026-08-28.)
    const int kt_l = tid & 7;
    const int kt_r0 = tid >> 3;

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

    long long kb_cur = 0;
    uint4 breg = make_uint4(0u, 0u, 0u, 0u);
    uint4 sreg = make_uint4(0u, 0u, 0u, 0u);
    // KT_FP4: 256-k 청크 레지스터 **더블버퍼** — 청크 c를 푸는 4 스텝 동안 청크 c+1을 읽는다
    // (단일 버퍼는 청크 경계에서만 로드가 발행돼 PCIe 지연이 스텝마다 노출됐다: 32 GB/s).
    uint4 kt_c[2][4];
    float kt_s[2][4];
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int i = 0; i < 4; ++i) { kt_c[b][i] = make_uint4(0u, 0u, 0u, 0u); kt_s[b][i] = 0.f; }
    auto load_b = [&](long long kb) {
      if constexpr (LAYOUT == PAIRROW) {
        const long long k = kb + 2 * br;  // 이 페어 행의 짝수 k
        if (b_active && k < kr) {
          breg = *reinterpret_cast<const uint4*>(codes_e + ((kb >> 1) + br) * n_cols + bc);
        } else {
          breg = make_uint4(0u, 0u, 0u, 0u);
        }
        if (tid < 16) {
          const int g = tid >> 3;            // 배율 행 0/1
          const int c = (tid & 7) * 16;
          if ((n0 + c) < n_cols && kb + g * kBlk < kr) {
            sreg = *reinterpret_cast<const uint4*>(scales_e + ((kb / kBlk) + g) * n_cols + c);
          } else {
            sreg = make_uint4(0u, 0u, 0u, 0u);
          }
        }
      } else {
        // kb는 청크 시작(256 배수)이어야 한다 — 버퍼 (kb>>8)&1 에 넣는다.
        const int buf = static_cast<int>((kb >> 8) & 1);
        const long long k0 = kb + kt_l * kBlk;   // 이 lane의 32-k 조각
        const long long rowb = kr >> 1;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int r = kt_r0 + 32 * i;
          if ((n0 + r) < n_cols && k0 < kr) {
            kt_c[buf][i] = *reinterpret_cast<const uint4*>(blk + (n0 + r) * rowb + (k0 >> 1));
            kt_s[buf][i] = blk_d[(n0 + r) * (kr / kBlk) + (k0 / kBlk)];
          } else {
            kt_c[buf][i] = make_uint4(0u, 0u, 0u, 0u);
            kt_s[buf][i] = 0.f;
          }
        }
      }
    };
    // 배율을 smem에 (16 스레드) — 코드 dequant가 읽기 전에 sync가 필요하므로 두 단계다.
    auto store_s = [&]() {
      if (LAYOUT == PAIRROW && tid < 16) {
        const int g = tid >> 3;
        const int c = (tid & 7) * 16;
        *reinterpret_cast<uint4*>(&ss[g][c]) = sreg;
      }
    };
    auto store_b = [&]() {
      const uint32_t w[4] = {breg.x, breg.y, breg.z, breg.w};
      if constexpr (LAYOUT == PAIRROW) {
        const int g = br >> 4;  // 이 페어 행의 배율 블록 (0/1)
        __nv_bfloat16* lo_row = Bs + (2 * br) * kBld + bc;
        __nv_bfloat16* hi_row = lo_row + kBld;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const uint32_t b = (w[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
          const float sh = prism_mxfp4::e8m0_half(ss[g][bc + j]);
          lo_row[j] = prism_mxfp4::fp4_to_bf16(b & 0xFu, sh);
          hi_row[j] = prism_mxfp4::fp4_to_bf16(b >> 4, sh);
        }
      } else {
        // 이 K-스텝(kb, 64 k)은 lane l ∈ {2·sub, 2·sub+1} 이 든다 (sub = 청크 내 스텝 0..3):
        // l의 조각은 k [kb + (l&1)·32, +32). 한 스레드가 행 4개(r0 + 32i)를 푼다.
        (void)w;
        const int sub = static_cast<int>((kb_cur >> 6) & 3);
        const int buf = static_cast<int>((kb_cur >> 8) & 1);
        if ((kt_l >> 1) != sub) return;
        const int h = kt_l & 1;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 c = kt_c[buf][i];
          const uint32_t cw[4] = {c.x, c.y, c.z, c.w};
          const float sh = kt_s[buf][i] * 0.5f;  // fp32 2^e → ×0.5 (fp4_val2가 값×2)
          __nv_bfloat16* col = Bs + (h * kBlk) * kBld + kt_r0 + 32 * i;
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            const uint32_t b = (cw[j >> 2] >> ((j & 3) * 8)) & 0xFFu;
            col[(2 * j) * kBld] = prism_mxfp4::fp4_to_bf16(b & 0xFu, sh);
            col[(2 * j + 1) * kBld] = prism_mxfp4::fp4_to_bf16(b >> 4, sh);
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
      kb_cur = kb;
      __syncthreads();  // 이전 mma가 As/Bs/ss 읽기를 끝냈다 (첫 회: sp/srow 가시)
      if (tid < kBK) skid[tid] = (kb + tid < kr) ? s.kidx[o0 + kb + tid] : uint16_t{0};
      store_s();
      __syncthreads();  // ss·skid 가시
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
      if constexpr (LAYOUT == PAIRROW) {
        if (kb + kBK < kr) load_b(kb + kBK);  // 다음 타일 선인출 (PCIe 지연 은닉)
      } else {
        // 청크 시작 스텝에서 **다음 청크**를 다른 버퍼에 선인출 — 4 스텝의 지연 은닉.
        if ((kb & 255) == 0 && kb + 256 < kr) load_b(kb + 256);
      }
      __syncthreads();  // As/Bs 가시
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
    // epilogue: 행 절반(64행)씩 — half h는 wm == h 인 warp들이 fragment를 Cs에 내리고,
    // 전 스레드가 64행 × 128열을 bf16으로 쓴다 (행당 16청크 × 8열).
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      __syncthreads();  // As/Bs 소비 완료 (h=0) / 이전 절반의 out 쓰기 완료 (h=1)
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

// 공통 검증 + launch. `w_on_device`로 스토어의 거처 제약만 갈린다 (hot/warm 쌍둥이).
// max_blocks: launch 블록 수 상한 (0 = 없음) — PCIe 바운드 launch가 SM을 비워두게 하는 노브.
inline void grouped_mxfp4_impl(
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
    int layout = PAIRROW,
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
  if (layout == KT_FP4) {
    // slab은 1-D u8(host-register된 kt 메모리)이고 길이는 expert 블록 합(64 B 올림)이라 N과
    // 무관하다 — n_cols는 인자로 받는다 (노드 N shard 행 수). row_off/kidx는 그대로.
    TensorMatcher({S}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>().verify(codes);
    TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(row_off);
    auto R = SymbolicSize{"total_rows"};
    TensorMatcher({R}).with_dtype<uint16_t>().with_device(cuda_device).verify(kidx);
    RuntimeCheck(cold_n_cols > 0 && blk_off != nullptr, "grouped_mxfp4_cold: n_cols and blk_off required");
    auto E = SymbolicSize{"num_experts"};
    TensorMatcher({E}).with_dtype<int64_t>().with_device(cuda_device).verify(*blk_off);
    RuntimeCheck(E.unwrap() + 1 == E1.unwrap(), "grouped_mxfp4_cold: blk_off must have E entries");
    N.set_value(cold_n_cols);
  } else {
    verify_mx4_store(codes, scales, row_off, kidx, E1, N, cuda_device, w_on_device, "grouped_mxfp4");
  }

  const int64_t m = M.unwrap(), top_k = K.unwrap(), p = P.unwrap();
  const int64_t n_cols = N.unwrap(), out_row = W_row.unwrap();
  const int64_t x_rows = Rx.unwrap(), x_kx = Kx.unwrap();
  const int64_t num_experts = E1.unwrap() - 1;

  RuntimeCheck(num_experts >= 1, "grouped_mxfp4: row_off needs E+1 >= 2 entries");
  RuntimeCheck(p == m * top_k, "grouped_mxfp4: pair_sorted has ", p,
               " entries but out implies M*top_k = ", m * top_k);
  RuntimeCheck(x_row_is_pair ? (x_rows == p) : (x_rows == m),
               "grouped_mxfp4: x rows (", x_rows, ") must be ", x_row_is_pair ? "M*top_k" : "M");
  RuntimeCheck(x_kx <= 65536, "grouped_mxfp4: x width ", x_kx, " exceeds the uint16 index range");
  RuntimeCheck(n_cols % 16 == 0, "grouped_mxfp4: n_cols ", n_cols, " must be a multiple of 16");
  RuntimeCheck(out_row % 8 == 0 && out_col_offset % 8 == 0,
               "grouped_mxfp4: out_row (", out_row, ") and out_col_offset (", out_col_offset,
               ") must be multiples of 8");
  RuntimeCheck(out_col_offset >= 0 && out_col_offset + n_cols <= out_row,
               "grouped_mxfp4: out cols [", out_col_offset, ",", out_col_offset + n_cols,
               ") out of out width ", out_row);
  RuntimeCheck(aligned16(out.data_ptr()), "grouped_mxfp4: out must be 16-byte aligned");

  Mx4Slot s0{static_cast<const uint8_t*>(codes.data_ptr()),
             static_cast<const uint8_t*>(scales.data_ptr()),
             static_cast<const int32_t*>(row_off.data_ptr()),
             static_cast<const uint16_t*>(kidx.data_ptr()), out_col_offset,
             blk_off ? static_cast<const int64_t*>(blk_off->data_ptr()) : nullptr};
  Mx4Slot s1 = s0;
  const bool fused = (codes_up != nullptr);
  if (fused) {
    RuntimeCheck(scales_up && row_off_up && kidx_up, "grouped_mxfp4_gateup: up slot needs all four tensors");
    if (layout == KT_FP4) {
      auto S2 = SymbolicSize{"slab_bytes_up"};
      TensorMatcher({S2}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>().verify(*codes_up);
      TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(*row_off_up);
      auto R2 = SymbolicSize{"total_rows_up"};
      TensorMatcher({R2}).with_dtype<uint16_t>().with_device(cuda_device).verify(*kidx_up);
      RuntimeCheck(blk_off_up != nullptr, "grouped_mxfp4_cold_gateup: blk_off_up required");
      auto E2 = SymbolicSize{"num_experts_up"};
      TensorMatcher({E2}).with_dtype<int64_t>().with_device(cuda_device).verify(*blk_off_up);
    } else {
      verify_mx4_store(*codes_up, *scales_up, *row_off_up, *kidx_up, E1, N, cuda_device,
                       w_on_device, "grouped_mxfp4_gateup(up)");
    }
    RuntimeCheck(out_col_offset_up % 8 == 0 && out_col_offset_up >= 0 &&
                 out_col_offset_up + n_cols <= out_row,
                 "grouped_mxfp4_gateup: up out cols [", out_col_offset_up, ",",
                 out_col_offset_up + n_cols, ") invalid for out width ", out_row);
    s1 = Mx4Slot{static_cast<const uint8_t*>(codes_up->data_ptr()),
                 static_cast<const uint8_t*>(scales_up->data_ptr()),
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
  if (layout == KT_FP4) launch(prism_grouped_gemm_mxfp4<KT_FP4>);
  else launch(prism_grouped_gemm_mxfp4<PAIRROW>);
}

// KT_FP4 (cold slab) 진입점 — kt fp4 BufferB를 host 메모리(cudaHostRegister됨)에서 제자리 읽기.
// row_off는 k 단위(패딩 포함), kidx는 패딩 포함, blk_off는 expert 블록의 slab 내 바이트 오프셋.
// out_col_offset에는 노드 N shard 시작이 더해져 온다.
void grouped_mxfp4_cold(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView slab, tvm::ffi::TensorView blk_off,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair,
    int64_t max_blocks, int64_t n_cols) {
  grouped_mxfp4_impl(x, pair_sorted, pair_off, tile_off, slab, slab, row_off, kidx, out,
                     out_col_offset, x_row_is_pair, false, max_blocks,
                     nullptr, nullptr, nullptr, nullptr, 0, KT_FP4, &blk_off, nullptr, n_cols);
}

void grouped_mxfp4_cold_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView slab_g, tvm::ffi::TensorView blk_off_g,
    tvm::ffi::TensorView row_off_g, tvm::ffi::TensorView kidx_g,
    tvm::ffi::TensorView slab_u, tvm::ffi::TensorView blk_off_u,
    tvm::ffi::TensorView row_off_u, tvm::ffi::TensorView kidx_u,
    tvm::ffi::TensorView out, int64_t out_col_offset_g, int64_t out_col_offset_u,
    int64_t x_row_is_pair, int64_t max_blocks, int64_t n_cols) {
  grouped_mxfp4_impl(x, pair_sorted, pair_off, tile_off, slab_g, slab_g, row_off_g, kidx_g, out,
                     out_col_offset_g, x_row_is_pair, false, max_blocks,
                     &slab_u, &slab_u, &row_off_u, &kidx_u, out_col_offset_u,
                     KT_FP4, &blk_off_g, &blk_off_u, n_cols);
}

void grouped_mxfp4_indexed(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair, int64_t max_blocks) {
  grouped_mxfp4_impl(x, pair_sorted, pair_off, tile_off, codes, scales, row_off, kidx, out,
                     out_col_offset, x_row_is_pair, true, max_blocks);
}

void grouped_mxfp4_indexed_pinned(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView codes, tvm::ffi::TensorView scales,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair, int64_t max_blocks) {
  grouped_mxfp4_impl(x, pair_sorted, pair_off, tile_off, codes, scales, row_off, kidx, out,
                     out_col_offset, x_row_is_pair, false, max_blocks);
}

void grouped_mxfp4_indexed_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView codes_g, tvm::ffi::TensorView scales_g,
    tvm::ffi::TensorView row_off_g, tvm::ffi::TensorView kidx_g,
    tvm::ffi::TensorView codes_u, tvm::ffi::TensorView scales_u,
    tvm::ffi::TensorView row_off_u, tvm::ffi::TensorView kidx_u,
    tvm::ffi::TensorView out, int64_t out_col_offset_g, int64_t out_col_offset_u,
    int64_t x_row_is_pair, int64_t max_blocks) {
  grouped_mxfp4_impl(x, pair_sorted, pair_off, tile_off, codes_g, scales_g, row_off_g, kidx_g, out,
                     out_col_offset_g, x_row_is_pair, true, max_blocks,
                     &codes_u, &scales_u, &row_off_u, &kidx_u, out_col_offset_u);
}

void grouped_mxfp4_indexed_pinned_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView codes_g, tvm::ffi::TensorView scales_g,
    tvm::ffi::TensorView row_off_g, tvm::ffi::TensorView kidx_g,
    tvm::ffi::TensorView codes_u, tvm::ffi::TensorView scales_u,
    tvm::ffi::TensorView row_off_u, tvm::ffi::TensorView kidx_u,
    tvm::ffi::TensorView out, int64_t out_col_offset_g, int64_t out_col_offset_u,
    int64_t x_row_is_pair, int64_t max_blocks) {
  grouped_mxfp4_impl(x, pair_sorted, pair_off, tile_off, codes_g, scales_g, row_off_g, kidx_g, out,
                     out_col_offset_g, x_row_is_pair, false, max_blocks,
                     &codes_u, &scales_u, &row_off_u, &kidx_u, out_col_offset_u);
}

}  // namespace
