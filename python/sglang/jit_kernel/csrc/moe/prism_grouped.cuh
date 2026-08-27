#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <mma.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

namespace {

using namespace nvcuda;

// Prism grouped GEMM — prefill 형태의 GPU 티어 커널 (2026-08-26).
//
// worklist GEMV(prism_gemv.cuh)는 decode 형태다: 블록 하나가 pair (m, j) 하나를
// 맡아 그 expert의 W 슬랩을 **pair마다 다시 읽는다**. prefill에서 그 중복도는
// M·top_k/E (35B, M=2048: 64배)이고 warm에서는 그 전부가 PCIe다 — 실측 층당
// 12.9 GB를 읽어 250 ms/층, 스토어 전체(192 MiB)를 한 번 읽는 이론치(3.9 ms)의
// 64배다. 이 커널은 pair를 expert로 묶어 **W 타일을 한 번 읽고 그 expert의 모든
// 토큰에 곱한다** (grouped/segmented GEMM). bf16 tensor core(wmma), fp32 누산,
// bf16 출력 (계약 ⑤).
//
// hot과 warm은 같은 커널이다 — 다른 것은 `w`가 device냐 pinned(UVA)냐 하나뿐이고
// 그것은 host 검증(`w_on_device`)의 차이다 (계약 ①). cold packed 레이아웃을 읽는
// 변형은 B 타일 로더만 갈린다 (`prism_grouped_cold.cuh` 예정).
//
// 입력 그룹핑(grouping.py가 만든다 — 전부 device 상주, host sync 없음):
//   pair_sorted [P]   int32  expert 오름차순으로 정렬된 pair 번호 (p = m·top_k + j)
//   pair_off    [E+1] int32  expert e의 pair 구간 [pair_off[e], pair_off[e+1])
//   tile_off    [E+1] int32  expert e의 토큰 타일 구간 (타일 = kBM pair)
// grid = (ceil(N/kBN), ceil(P/kBM) + E, fused ? 2 : 1). grid.y는 상한이다 —
// 실제 타일 수 tile_off[E]는 device 값이고 host가 모르므로, Σₑ ceil(cₑ/BM) ≤
// P/BM + E 를 launch 모양으로 쓰고 초과 블록은 즉시 반환한다.
//
// 블록 (e, 토큰 타일 tt, 열 타일 bx): 이 expert의 pair 최대 kBM개 × 열 kBN개.
//   A = x[row(pair), kidx[o0 + k]]  — 인덱스 gather는 커널 안에서 (계약 ④: x는
//       L2 상주라 pre-gather 버퍼를 만들 이유가 없다)
//   B = W[o0 + k, n0 .. n0+kBN)     — 행마다 128 B 연속, 스레드 8개가 uint4로
//       읽는다. 다음 K 타일을 레지스터에 선인출해 PCIe 지연을 계산 아래 숨긴다.
// 누산 순서는 (wmma 내부 고정) 결정적이다. 정확표현 입력(작은 정수)에서 fp32
// 누산은 순서 무관 정확하므로 worklist 커널과 비트일치한다 — 그것이 계약 ⑤의
// exact 검출기가 이 커널에도 그대로 적용되는 이유다.
constexpr int kBM = 128;  // 타일당 pair 수 (= wmma 4 warp × 32 행)
constexpr int kBN = 64;   // 블록당 출력 열 (bf16 128 B — 캐시라인/PCIe 요청 단위)
constexpr int kBK = 32;   // smem에 올리는 K 행 수 (= wmma k 16 × 2)
constexpr int kThreads = 256;
constexpr int kAld = kBK + 8;  // smem A leading dim (bank 충돌 완화 + 32 B 정렬 유지)
constexpr int kBld = kBN + 8;  // smem B leading dim
constexpr int kCld = kBN + 4;  // smem C(fp32) leading dim
constexpr int kSmemAB = kBM * kAld * 2 + kBK * kBld * 2;  // 10240 + 4608
constexpr int kSmemC = kBM * kCld * 4;                    // 34816
constexpr int kSmemBytes = kSmemAB > kSmemC ? kSmemAB : kSmemC;

struct Slot {
  const __nv_bfloat16* w;    // ROWMAJOR: flat [Σₑ k[e], N] / COLD: packed slab 시작
  const int32_t* row_off;    // [E+1] — 두 레이아웃 모두 expert e의 K 행 수 = row_off[e+1]-row_off[e]
  const uint16_t* kidx;      // [Σₑ k[e]] (COLD: 타일 올림 패딩 포함, 패딩은 0을 가리킨다)
  long long out_off;         // out3d 열 오프셋 (COLD: 노드 N shard 시작이 더해져 온다)
  const int64_t* blk_off;    // COLD 전용: [E] expert 블록의 slab 내 원소 오프셋
};

// B 타일 레이아웃.
//   ROWMAJOR — hot/warm 스토어 [Σₑ k[e], N] K-major. 타일 행 = 128 B 연속.
//   COLD     — kt AMX `BufferBBF16Impl`의 packed 6D (n_block → k_block → n_step
//              → k_step → 32×32 타일, 타일 안은 16×16 dword 두 개가 transpose된
//              VNNI 배치). GPU는 이것을 **재배치하지 않고 제자리에서 읽는다** —
//              cold weight가 CPU AMX와 GPU 사이에서 한 벌이다.
enum Layout : int { ROWMAJOR = 0, COLD = 1 };

// kt GemmKernel224BF16의 타일 상수 (BufferBBF16Impl 참조). n_block/k_block은
// 커널 인자로 받고(TileK2는 128/7168), 타일 32×32와 그 내부 transpose는
// AMX 타일 규격이라 고정이다.
constexpr int kColdStep = 32;

// packed slab에서 (n, k) 원소가 속한 32×32 타일의 시작 오프셋 (원소 단위).
// pack_block의 주소식 그대로: n_block_begin*k + k_block_begin*n_block_size +
// n_begin*k_block_size + k_begin*N_STEP.
__device__ __forceinline__ long long cold_tile_base(
    long long n, long long k, long long n_total, long long k_total,
    int n_block, int k_block) {
  const long long nb = (n / n_block) * n_block;
  const long long nbs = min(static_cast<long long>(n_block), n_total - nb);
  const long long kb = (k / k_block) * k_block;
  const long long kbs = min(static_cast<long long>(k_block), k_total - kb);
  const long long ns = ((n - nb) / kColdStep) * kColdStep;
  const long long ks = ((k - kb) / kColdStep) * kColdStep;
  return nb * k_total + kb * nbs + ns * kbs + ks * kColdStep;
}

template <int LAYOUT>
__global__ void __launch_bounds__(kThreads) prism_grouped_gemm(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ pair_sorted,
    const int32_t* __restrict__ pair_off,
    const int32_t* __restrict__ tile_off,
    __nv_bfloat16* __restrict__ out,
    int num_experts, long long top_k, long long x_kx, int x_row_is_pair,
    long long n_cols, long long out_row, int n_block, int k_block,
    Slot s0, Slot s1) {
  // 슬롯 선택 (gate+up 융합) — blockIdx.z가 고른다. 블록 내 uniform.
  const Slot s = (blockIdx.z != 0) ? s1 : s0;

  // 타일 루프: gridDim.y가 총 타일 수보다 작으면 블록이 여러 타일을 순회한다
  // (persistent). warm은 PCIe 바운드라 SM을 다 채울 이유가 없고, 비워둔 SM에
  // hot 커널이 동시에 올라간다 — 그것이 hot∥warm 스트림 분리가 실제로 겹치는
  // 조건이다 (블록이 SM을 다 점유하면 스트림을 나눠도 직렬화된다).
  const int total_tiles = tile_off[num_experts];
  for (int t = blockIdx.y; t < total_tiles; t += gridDim.y) {
  __syncthreads();  // 이전 타일의 epilogue(Cs/sp 읽기)가 끝난 뒤 재사용
  // 이 타일의 expert: tile_off[e] <= t < tile_off[e+1] 인 e (이분 탐색).
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
  const long long n0 = static_cast<long long>(blockIdx.x) * kBN;

  __shared__ __align__(128) unsigned char smem[kSmemBytes];
  __shared__ int sp[kBM];         // 타일 행 → pair
  __shared__ int srow[kBM];       // 타일 행 → x 행
  __shared__ uint16_t skid[kBK];  // 이 K 타일의 kidx
  __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Bs = As + kBM * kAld;
  float* Cs = reinterpret_cast<float*>(smem);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int wm = warp >> 1;  // 0..3 → 행 wm*32
  const int wn = warp & 1;   // 0..1 → 열 wn*32

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

  // B 선인출 — 스레드당 uint4(8 bf16) 하나. 레이아웃이 소스 주소와 smem 배치를
  // 결정하고, 그 아래의 mma는 레이아웃을 모른다.
  //   ROWMAJOR: (행 br, 열 청크 bc) — 32행 × 8청크 = 256. 타일 행 = 128 B 연속.
  //   COLD: 32×32 타일 2개(n_step j=0,1)가 각각 2 KB 연속. 스레드 → (j, 청크 c):
  //     h = c/64 (n 절반), d = (c%64)*4 dword → k 페어 p = d/16, n16 = d%16.
  //     uint4 = dword 4개 = 같은 k 페어의 연속 n 4개 → smem에 (2p, n..n+3)/(2p+1, …)로 산개.
  const int br = tid >> 3;
  const int bc = (tid & 7) * 8;
  const bool b_active = (n0 + bc) < n_cols;  // n_cols % 8 == 0 → 청크 단위 유효
  const int cj = tid >> 7;            // COLD: n_step 절반 (0,1)
  const int cc = tid & 127;           // COLD: 타일 내 16 B 청크
  const int ch = cc >> 6;             // COLD: n 16-절반
  const int cd = (cc & 63) * 4;       // COLD: 절반 내 dword 시작
  const int cp = cd >> 4;             // COLD: k 페어 (0..15)
  const int cn = ch * 16 + (cd & 15); // COLD: 타일 내 n (0..31), 4개 연속
  const __nv_bfloat16* wblk = (LAYOUT == COLD) ? s.w + s.blk_off[e] : nullptr;
  uint4 breg = make_uint4(0u, 0u, 0u, 0u);
  auto load_b = [&](long long kb) {
    if constexpr (LAYOUT == ROWMAJOR) {
      const long long k = kb + br;
      if (b_active && k < kr) {
        breg = *reinterpret_cast<const uint4*>(s.w + (o0 + k) * n_cols + n0 + bc);
      } else {
        breg = make_uint4(0u, 0u, 0u, 0u);
      }
    } else {
      // COLD: kr은 32의 배수(kt K_STEP 올림), n_cols는 32의 배수(COL_GROUP) —
      // 타일 단위 유효성만 본다. kb+32 ≤ kr 은 루프 조건이 보장한다.
      const long long nt = n0 + cj * kColdStep;
      if (nt < n_cols) {
        const long long base = cold_tile_base(nt, kb, n_cols, kr, n_block, k_block);
        breg = *reinterpret_cast<const uint4*>(wblk + base + ch * 512 + cd * 2);
      } else {
        breg = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
  auto store_b = [&]() {
    if constexpr (LAYOUT == ROWMAJOR) {
      *reinterpret_cast<uint4*>(Bs + br * kBld + bc) = breg;
    } else {
      const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&breg);
      const int ncol = cj * kColdStep + cn;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        Bs[(2 * cp) * kBld + ncol + q] = v[2 * q];
        Bs[(2 * cp + 1) * kBld + ncol + q] = v[2 * q + 1];
      }
    }
  };
  load_b(0);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  for (long long kb = 0; kb < kr; kb += kBK) {
    __syncthreads();  // 이전 iteration의 mma가 As/Bs 읽기를 끝냈다 (첫 회: sp/srow)
    if (tid < kBK) {
      skid[tid] = (kb + tid < kr) ? s.kidx[o0 + kb + tid] : uint16_t{0};
    }
    store_b();
    __syncthreads();  // skid 가시
    // A gather: 128 × 32 원소, 스레드당 16. 연속 스레드가 한 행의 연속 k를 맡아
    // x 읽기가 kidx 정렬(로더가 오름차순으로 굽는다) 만큼 지역적이다.
#pragma unroll
    for (int st = 0; st < (kBM * kBK) / kThreads; ++st) {
      const int idx = tid + st * kThreads;
      const int i = idx >> 5;   // kBK == 32
      const int kk = idx & 31;
      __nv_bfloat16 v = __float2bfloat16(0.f);
      if (i < cnt && kb + kk < kr) {
        v = x[static_cast<long long>(srow[i]) * x_kx + skid[kk]];
      }
      As[i * kAld + kk] = v;
    }
    if (kb + kBK < kr) load_b(kb + kBK);  // 다음 타일 선인출 (PCIe 지연 은닉)
    __syncthreads();  // As 가시
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        wmma::load_matrix_sync(a[i], As + (wm * 32 + i * 16) * kAld + kk, kAld);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::load_matrix_sync(b[j], Bs + kk * kBld + wn * 32 + j * 16, kBld);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
  }
  __syncthreads();  // As/Bs 소비 완료 → Cs로 재사용
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(Cs + (wm * 32 + i * 16) * kCld + wn * 32 + j * 16,
                              acc[i][j], kCld, wmma::mem_row_major);
  __syncthreads();
  // 출력: 행당 8청크(각 8열) × 128행 = 1024 / 256 = 스레드당 4. rejoin 레이아웃
  // out[pair, off + n] 에 bf16 uint4로 쓴다 (128 B 연속 → coalesced).
#pragma unroll
  for (int st = 0; st < (kBM * (kBN / 8)) / kThreads; ++st) {
    const int idx = tid + st * kThreads;
    const int i = idx >> 3;
    const int c = (idx & 7) * 8;
    if (i < cnt && n0 + c < n_cols) {
      const float* cp = Cs + i * kCld + c;
      uint4 pk;
      __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&pk);
      h[0] = __floats2bfloat162_rn(cp[0], cp[1]);
      h[1] = __floats2bfloat162_rn(cp[2], cp[3]);
      h[2] = __floats2bfloat162_rn(cp[4], cp[5]);
      h[3] = __floats2bfloat162_rn(cp[6], cp[7]);
      *reinterpret_cast<uint4*>(out + static_cast<long long>(sp[i]) * out_row +
                                s.out_off + n0 + c) = pk;
    }
  }
  }  // tile loop
}


// ─── W-resident 변형 (2026-08-27) ──────────────────────────────────────────
//
// 위 커널은 블록 = (토큰 타일, 열 타일)이라 expert의 pair가 kBM(128)을 넘으면 토큰
// 타일마다 W를 **다시** 읽는다 — PCIe 바운드에서 그 재읽기가 그대로 바이트 낭비다
// (4096 청크는 평균 pair가 정확히 128이라 절반이 2타일; 라우팅이 몰리면 더).
// 여기서는 블록 = (expert, 열 타일)이고, 자기 W 슬라이스 [k_e, BN]을 smem에 **한 번**
// 올린 뒤 그 expert의 토큰 타일들을 순회한다 — W는 pair 수와 무관하게 정확히 1회.
// 갈아 끼우는 것은 A(x gather, L2 상주)뿐이다.
//
// smem: W 슬라이스 k_max × (BN+8) × 2 B (+ kidx k_max × 2 B)를 dynamic으로. cold
// gateup(k=1536)은 BN=32로 126 KB, warm(256)/down(384)은 BN=64로 들어간다. BN 선택은
// host가 k_max로 한다 (`wres_bn`). 1블록/SM이지만 PCIe 바운드라 충분하다 — 132 SM이
// 각자 100~200 KB를 동시에 끌어오면 in-flight는 남는다.
// K-chunking: smem에 W 전체가 안 들어가면(k_max·(BN+8)·2 B > 예산) K를 KC 행 조각으로
// 나눠 조각마다 토큰 타일들을 돌고, 부분 누산은 fp32 global scratch[pair, out_row]에
// 이어간다 (첫 조각은 store, 중간은 RMW, 마지막은 bf16 out). scratch 트래픽은 조각당
// pairs×BN×8 B — W(PCIe)에 비하면 작다. KC == k_max면 scratch를 건드리지 않는다.
template <int LAYOUT, int BN>
__global__ void __launch_bounds__(kThreads) prism_grouped_gemm_wres(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ pair_sorted,
    const int32_t* __restrict__ pair_off,
    const int32_t* __restrict__ tile_off,   // expert 마스크: tile_off[e+1]==tile_off[e] 면 이 launch가 맡지 않는 expert
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ scratch,   // [P, out_row] fp32 (KC < k_max일 때만 사용)
    int num_experts, long long top_k, long long x_kx, int x_row_is_pair,
    long long n_cols, long long out_row, int n_block, int k_block, int k_chunk,
    Slot s0, Slot s1) {
  constexpr int WARPS_M = (BN == 64) ? 4 : 8;
  constexpr int WARPS_N = 8 / WARPS_M;
  constexpr int ROWS_W = kBM / WARPS_M;   // warp당 행: 32 / 16
  constexpr int COLS_W = BN / WARPS_N;    // warp당 열: 32 / 16
  constexpr int FM = ROWS_W / 16, FN = COLS_W / 16;
  constexpr int WLD = BN + 8;             // smem W leading dim (32 B 정렬 유지)
  constexpr int CLD = BN + 4;
  extern __shared__ __align__(128) unsigned char dsmem[];
  __nv_bfloat16* Ws = reinterpret_cast<__nv_bfloat16*>(dsmem);            // [k_max][WLD]
  uint16_t* skid = reinterpret_cast<uint16_t*>(Ws + (size_t)k_chunk * WLD);  // [k_chunk]
  __shared__ __align__(128) __nv_bfloat16 As[kBM * kAld];
  __shared__ __align__(128) float Cs[kBM * CLD];
  __shared__ int sp[kBM];
  __shared__ int srow[kBM];

  const Slot s = (blockIdx.z != 0) ? s1 : s0;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int wm = warp / WARPS_N;
  const int wn = warp % WARPS_N;
  const long long n0 = static_cast<long long>(blockIdx.x) * BN;
  if (n0 >= n_cols) return;

  for (int e = blockIdx.y; e < num_experts; e += gridDim.y) {
    const int pbeg = pair_off[e];
    const int npairs = pair_off[e + 1] - pbeg;
    if (npairs <= 0) continue;
    // hybrid: grouping이 이 expert의 타일을 0으로 뒀으면(CPU 몫) 건너뛴다 — 스트리밍
    // 커널은 tile_off로 블록이 배정되어 자연히 지켜지지만, 여기는 expert 루프라 명시해야
    // 한다 (빠뜨리면 CPU 몫 행이 GPU에서도 계산되어 rejoin에서 이중 합산된다).
    if (tile_off[e + 1] == tile_off[e]) continue;
    const long long o0 = s.row_off[e];
    const int kr = static_cast<int>(s.row_off[e + 1] - o0);
    const int ntiles = (npairs + kBM - 1) / kBM;
    for (int kc0 = 0; kc0 < kr; kc0 += k_chunk) {
    const int kcnt = min(k_chunk, kr - kc0);         // 이 조각의 K 행 수 (32의 배수)
    const bool first = (kc0 == 0), last = (kc0 + kcnt >= kr);
    __syncthreads();  // 이전 조각/expert의 Ws/skid 읽기 완료

    // ── W 조각 [kc0, kc0+kcnt) 로드 (조각당 1회) ──
    if constexpr (LAYOUT == ROWMAJOR) {
      constexpr int CH = BN / 8;  // 행당 16 B 청크 수
      for (int idx = tid; idx < kcnt * CH; idx += kThreads) {
        const int r = idx / CH, c = (idx % CH) * 8;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (n0 + c < n_cols) v = *reinterpret_cast<const uint4*>(s.w + (o0 + kc0 + r) * n_cols + n0 + c);
        *reinterpret_cast<uint4*>(Ws + r * WLD + c) = v;
      }
    } else {
      // COLD: 32×32 타일(2 KB) 단위. 타일 (ks, ns) × 청크 c(0..127) — 청크 해석은 위와 같다.
      // BN=16이면 타일의 n 절반(h)만 필요하다: n0가 32 정렬이 아닐 수 있어 타일 시작은
      // n0를 32로 내림한 값이고, 그 안에서 우리 16열은 절반 ch = (n0 % 32) / 16 이다.
      constexpr int NS = (BN >= kColdStep) ? BN / kColdStep : 1;
      constexpr int CHUNKS = (BN >= kColdStep) ? 128 : 64;   // 타일당 읽을 16 B 청크 수
      const __nv_bfloat16* wblk = s.w + s.blk_off[e];
      const int ktiles = kcnt / kColdStep;
      const long long nbase = (BN >= kColdStep) ? n0 : (n0 / kColdStep) * kColdStep;
      const int hsel = (BN >= kColdStep) ? 0 : static_cast<int>((n0 - nbase) / 16);
      for (int idx = tid; idx < ktiles * NS * CHUNKS; idx += kThreads) {
        const int c0 = idx % CHUNKS;
        const int t = idx / CHUNKS;
        const int ks = t / NS, ns = t % NS;
        const int c = (BN >= kColdStep) ? c0 : (hsel * 64 + c0);
        const int ch = c >> 6, cd = (c & 63) * 4, cp = cd >> 4, cn = ch * 16 + (cd & 15);
        const long long nt = nbase + ns * kColdStep;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (nt < n_cols) {
          const long long base = cold_tile_base(nt, (long long)kc0 + (long long)ks * kColdStep, n_cols, kr, n_block, k_block);
          v = *reinterpret_cast<const uint4*>(wblk + base + ch * 512 + cd * 2);
        }
        const __nv_bfloat16* vv = reinterpret_cast<const __nv_bfloat16*>(&v);
        const int ncol = (BN >= kColdStep) ? (ns * kColdStep + cn) : (cn - hsel * 16);
        const int kk = ks * kColdStep + 2 * cp;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          Ws[kk * WLD + ncol + q] = vv[2 * q];
          Ws[(kk + 1) * WLD + ncol + q] = vv[2 * q + 1];
        }
      }
    }
    for (int t = tid; t < kcnt; t += kThreads) skid[t] = s.kidx[o0 + kc0 + t];
    __syncthreads();

    // ── 이 expert의 토큰 타일 순회 (이 K 조각에 대해) ──
    for (int tt = 0; tt < ntiles; ++tt) {
      const int p0 = pbeg + tt * kBM;
      const int cnt = min(kBM, npairs - tt * kBM);
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
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
      if (first) {
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
          for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);
      } else {
        // 이전 조각의 부분합을 scratch에서 이어받는다. 패딩 행(i ≥ cnt)은 pair가 없어
        // scratch 위치가 없으므로, 타일을 통째로 Cs에 올린 뒤 fragment로 읽는다.
        __syncthreads();
        constexpr int CH = BN / 8;
        for (int idx = tid; idx < kBM * CH; idx += kThreads) {
          const int i = idx / CH, c = (idx % CH) * 8;
          float4 v0 = make_float4(0.f, 0.f, 0.f, 0.f), v1 = v0;
          if (i < cnt && n0 + c < n_cols) {
            const float* sp_ = scratch + static_cast<long long>(sp[i]) * out_row + s.out_off + n0 + c;
            v0 = *reinterpret_cast<const float4*>(sp_);
            v1 = *reinterpret_cast<const float4*>(sp_ + 4);
          }
          *reinterpret_cast<float4*>(Cs + i * CLD + c) = v0;
          *reinterpret_cast<float4*>(Cs + i * CLD + c + 4) = v1;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
          for (int j = 0; j < FN; ++j)
            wmma::load_matrix_sync(acc[i][j], Cs + (wm * ROWS_W + i * 16) * CLD + wn * COLS_W + j * 16,
                                   CLD, wmma::mem_row_major);
      }
      for (int kb = 0; kb < kcnt; kb += kBK) {
        __syncthreads();  // 이전 K 스텝의 mma가 As를 다 읽었다 (첫 회: sp/srow 가시)
#pragma unroll
        for (int st = 0; st < (kBM * kBK) / kThreads; ++st) {
          const int idx = tid + st * kThreads;
          const int i = idx >> 5, kk = idx & 31;
          __nv_bfloat16 v = __float2bfloat16(0.f);
          if (i < cnt && kb + kk < kcnt) v = x[static_cast<long long>(srow[i]) * x_kx + skid[kb + kk]];
          As[i * kAld + kk] = v;
        }
        __syncthreads();
        const int ksteps = min(kBK, kcnt - kb);
        for (int kk = 0; kk < ksteps; kk += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[FM];
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b[FN];
#pragma unroll
          for (int i = 0; i < FM; ++i)
            wmma::load_matrix_sync(a[i], As + (wm * ROWS_W + i * 16) * kAld + kk, kAld);
#pragma unroll
          for (int j = 0; j < FN; ++j)
            wmma::load_matrix_sync(b[j], Ws + (kb + kk) * WLD + wn * COLS_W + j * 16, WLD);
#pragma unroll
          for (int i = 0; i < FM; ++i)
#pragma unroll
            for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
      }
      __syncthreads();  // As 소비 완료; Cs는 이전 타일의 out 쓰기가 끝난 뒤 (아래 sync)
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
          wmma::store_matrix_sync(Cs + (wm * ROWS_W + i * 16) * CLD + wn * COLS_W + j * 16,
                                  acc[i][j], CLD, wmma::mem_row_major);
      __syncthreads();
      constexpr int CH = BN / 8;
      for (int idx = tid; idx < kBM * CH; idx += kThreads) {
        const int i = idx / CH, c = (idx % CH) * 8;
        if (i < cnt && n0 + c < n_cols) {
          const float* cp = Cs + i * CLD + c;
          const long long o = static_cast<long long>(sp[i]) * out_row + s.out_off + n0 + c;
          if (last) {
            uint4 pk;
            __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&pk);
            h[0] = __floats2bfloat162_rn(cp[0], cp[1]);
            h[1] = __floats2bfloat162_rn(cp[2], cp[3]);
            h[2] = __floats2bfloat162_rn(cp[4], cp[5]);
            h[3] = __floats2bfloat162_rn(cp[6], cp[7]);
            *reinterpret_cast<uint4*>(out + o) = pk;
          } else {
            *reinterpret_cast<float4*>(scratch + o) = *reinterpret_cast<const float4*>(cp);
            *reinterpret_cast<float4*>(scratch + o + 4) = *reinterpret_cast<const float4*>(cp + 4);
          }
        }
      }
      __syncthreads();  // Cs/sp 읽기 완료 → 다음 타일
    }
    }  // K chunk
  }
}

// W-resident launch의 BN과 dynamic smem. 블록당 smem 상한은 디바이스마다 다르다 —
// H100 227 KB, Blackwell RTX PRO 6000(sm_120) 99 KB. 정적 smem(As 10 KB + Cs + 1 KB)을
// 포함해 들어가는 가장 큰 BN ∈ {64, 32, 16}을 고르고, 16으로도 안 들어가면 0(스트리밍
// 커널로 폴백). BN이 작을수록 W 행이 짧아져(32/64 B) PCIe 요청 효율이 조금 떨어진다.
inline size_t wres_dyn_smem(int64_t k_max, int bn) {
  return static_cast<size_t>(k_max * (bn + 8) * 2 + k_max * 2);
}
inline size_t wres_static_smem(int bn) {
  return static_cast<size_t>(kBM * kAld * 2 + kBM * (bn + 4) * 4 + 2 * kBM * 4);
}
inline int wres_smem_limit(int device_id) {
  static int limit = -1;
  if (limit < 0) {
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id) != cudaSuccess) v = 48 * 1024;
    limit = v;
  }
  return limit;
}
// BN은 64로 고정하고(A gather·PCIe 요청 효율), smem에 맞는 K 조각 크기를 고른다.
// 조각이 k_max보다 작으면 fp32 scratch로 누산을 이어간다.
inline int wres_k_chunk(int64_t k_max, int device_id) {
  const size_t budget = static_cast<size_t>(wres_smem_limit(device_id)) - wres_static_smem(64) - 1024;
  int64_t kc = static_cast<int64_t>(budget / ((64 + 8) * 2 + 2));
  kc = kc / kBK * kBK;
  if (kc <= 0) return 0;
  return static_cast<int>(std::min<int64_t>(kc, k_max));
}
template <int LAYOUT, int BN>
inline void wres_set_smem_attr(size_t bytes) {
  using namespace host;
  static size_t configured = 0;
  if (bytes > configured) {
    const cudaError_t err = cudaFuncSetAttribute(
        prism_grouped_gemm_wres<LAYOUT, BN>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(bytes));
    RuntimeCheck(err == cudaSuccess, "grouped_gemm(wres): cudaFuncSetAttribute(", bytes,
                 " B dynamic smem) failed: ", cudaGetErrorString(err));
    configured = bytes;
  }
}

// cold_async: CPU(kt)가 pinned 플래그에 seq를 쓰면 GPU stream이 그 값을 기다린다.
// 블로킹 host node(sync 콜백) 없이 CPU 완료를 stream 순서에 넣는 방법 — 콜백 스레드가
// 막히지 않으니 다른 stream의 host node와 직렬화되지 않는다. 스레드 1개 블록 1개.
__global__ void prism_wait_flag_kernel(const int32_t* flag, int32_t target) {
  const volatile int32_t* v = reinterpret_cast<const volatile int32_t*>(flag);
  while (*v < target) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(200);
#endif
  }
}

void prism_wait_flag(tvm::ffi::TensorView flag, int64_t target) {
  using namespace host;
  auto L = SymbolicSize{"len"};
  // 플래그는 pinned host(UVA)다 — CPU가 쓰고 GPU가 읽는다.
  TensorMatcher({L}).with_dtype<int32_t>().with_device<kDLCPU, kDLCUDAHost>().verify(flag);
  RuntimeCheck(L.unwrap() >= 1, "prism_wait_flag: empty flag");
  int dev = 0;
  cudaGetDevice(&dev);
  DLDevice device{kDLCUDA, dev};
  LaunchKernel(dim3(1), dim3(1), device)(prism_wait_flag_kernel,
                                          static_cast<const int32_t*>(flag.data_ptr()),
                                          static_cast<int32_t>(target));
}

inline bool aligned16(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

// 공통 검증 + launch. `w_on_device`로 W의 거처 제약만 갈린다 (hot/warm 쌍둥이).
// 슬롯 1(gate+up 융합)은 nullptr이면 grid.z=1.
// max_blocks: launch 블록 수 상한 (0 = 없음). PCIe 바운드 launch가 SM을 비워두게
// 하는 노브 — 실측(2026-08-27, H100) 128블록으로도 PCIe 51 GB/s를 포화하고 그때
// hot 커널이 완전히 겹친다 (블록을 더 주면 겹침이 사라진다).
// COLD 레이아웃은 blk_off(int64 [E], device)와 n_block/k_block을 요구한다.
inline void grouped_gemm_impl(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView w, tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair,
    bool w_on_device, int64_t max_blocks,
    const tvm::ffi::TensorView* w_up = nullptr,
    const tvm::ffi::TensorView* row_off_up = nullptr,
    const tvm::ffi::TensorView* kidx_up = nullptr,
    int64_t out_col_offset_up = 0,
    int layout = ROWMAJOR,
    const tvm::ffi::TensorView* blk_off = nullptr,
    const tvm::ffi::TensorView* blk_off_up = nullptr,
    int64_t n_block = 0, int64_t k_block = 0, int64_t cold_n_cols = 0,
    int64_t wres_k_max = 0, const tvm::ffi::TensorView* scratch = nullptr) {
  using namespace host;

  auto Rx = SymbolicSize{"x_rows"};
  auto Kx = SymbolicSize{"x_cols"};
  auto M = SymbolicSize{"num_tokens"};
  auto K = SymbolicSize{"top_k"};
  auto P = SymbolicSize{"num_pairs"};
  auto E1 = SymbolicSize{"num_experts_plus_one"};
  auto R = SymbolicSize{"total_rows"};
  auto R2 = SymbolicSize{"total_rows_up"};
  auto N = SymbolicSize{"n_cols"};
  auto W_row = SymbolicSize{"out_row"};
  auto cuda_device = SymbolicDevice{};

  TensorMatcher({Rx, Kx}).with_dtype<bf16_t>().with_device<kDLCUDA>(cuda_device).verify(x);
  TensorMatcher({P}).with_dtype<int32_t>().with_device(cuda_device).verify(pair_sorted);
  TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(pair_off);
  TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(tile_off);
  TensorMatcher({M, K, W_row}).with_dtype<bf16_t>().with_device(cuda_device).verify(out);
  TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(row_off);
  TensorMatcher({R}).with_dtype<uint16_t>().with_device(cuda_device).verify(kidx);
  // COLD slab은 1-D flat이고 길이가 expert 블록 합(64 B 올림)이라 N과 무관하다 —
  // n_cols는 인자로 받는다 (노드 N shard 행 수).
  auto S = SymbolicSize{"slab_elems"};
  if (layout == COLD) {
    TensorMatcher({S}).with_dtype<bf16_t>().with_device<kDLCPU, kDLCUDAHost>().verify(w);
    RuntimeCheck(cold_n_cols > 0, "grouped_gemm_cold: n_cols must be given");
    N.set_value(cold_n_cols);
  } else if (w_on_device) {
    TensorMatcher({R, N}).with_dtype<bf16_t>().with_device(cuda_device).verify(w);
  } else {
    TensorMatcher({R, N}).with_dtype<bf16_t>().with_device<kDLCPU, kDLCUDAHost>().verify(w);
  }

  const int64_t m = M.unwrap(), top_k = K.unwrap(), p = P.unwrap();
  const int64_t n_cols = N.unwrap(), out_row = W_row.unwrap();
  const int64_t x_rows = Rx.unwrap(), x_kx = Kx.unwrap();
  const int64_t num_experts = E1.unwrap() - 1;

  RuntimeCheck(num_experts >= 1, "grouped_gemm: row_off needs E+1 >= 2 entries");
  RuntimeCheck(p == m * top_k, "grouped_gemm: pair_sorted has ", p,
               " entries but out implies M*top_k = ", m * top_k);
  RuntimeCheck(x_row_is_pair ? (x_rows == p) : (x_rows == m),
               "grouped_gemm: x rows (", x_rows, ") must be ",
               x_row_is_pair ? "M*top_k" : "M");
  RuntimeCheck(x_kx <= 65536, "grouped_gemm: x width ", x_kx,
               " exceeds the uint16 index range");
  // uint4(8열) 로드/스토어 정렬. 계약 ①의 COL_GROUP=32가 N을 보증하고, 출력
  // 폭/오프셋은 inter/hidden 이라 같은 보증 아래 있다.
  RuntimeCheck(n_cols % 8 == 0, "grouped_gemm: n_cols ", n_cols, " must be a multiple of 8");
  RuntimeCheck(out_row % 8 == 0 && out_col_offset % 8 == 0,
               "grouped_gemm: out_row (", out_row, ") and out_col_offset (",
               out_col_offset, ") must be multiples of 8");
  RuntimeCheck(out_col_offset >= 0 && out_col_offset + n_cols <= out_row,
               "grouped_gemm: out cols [", out_col_offset, ",",
               out_col_offset + n_cols, ") out of out width ", out_row);
  RuntimeCheck(aligned16(w.data_ptr()) && aligned16(out.data_ptr()),
               "grouped_gemm: w and out must be 16-byte aligned");

  if (layout == COLD) {
    RuntimeCheck(blk_off != nullptr, "grouped_gemm_cold: blk_off required");
    TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(row_off);
    auto E = SymbolicSize{"num_experts"};
    TensorMatcher({E}).with_dtype<int64_t>().with_device(cuda_device).verify(*blk_off);
    RuntimeCheck(E.unwrap() + 1 == E1.unwrap(), "grouped_gemm_cold: blk_off must have E entries");
    RuntimeCheck(n_block > 0 && k_block > 0 && n_block % kColdStep == 0 &&
                 k_block % kColdStep == 0 && n_block % kBN == 0,
                 "grouped_gemm_cold: n_block/k_block must be multiples of 32 (and n_block of 64), got ",
                 n_block, "/", k_block);
    RuntimeCheck(n_cols % kColdStep == 0, "grouped_gemm_cold: n_cols ", n_cols,
                 " must be a multiple of 32 (AMX N_STEP)");
  }
  Slot s0{static_cast<const __nv_bfloat16*>(w.data_ptr()),
          static_cast<const int32_t*>(row_off.data_ptr()),
          static_cast<const uint16_t*>(kidx.data_ptr()), out_col_offset,
          blk_off ? static_cast<const int64_t*>(blk_off->data_ptr()) : nullptr};
  Slot s1 = s0;
  const bool fused = (w_up != nullptr);
  if (fused) {
    RuntimeCheck(row_off_up != nullptr && kidx_up != nullptr,
                 "grouped_gemm_gateup: up slot needs row_off and kidx");
    TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(*row_off_up);
    TensorMatcher({R2}).with_dtype<uint16_t>().with_device(cuda_device).verify(*kidx_up);
    if (layout == COLD) {
      auto S2 = SymbolicSize{"slab_elems_up"};
      TensorMatcher({S2}).with_dtype<bf16_t>().with_device<kDLCPU, kDLCUDAHost>().verify(*w_up);
    } else if (w_on_device) {
      TensorMatcher({R2, N}).with_dtype<bf16_t>().with_device(cuda_device).verify(*w_up);
    } else {
      TensorMatcher({R2, N}).with_dtype<bf16_t>().with_device<kDLCPU, kDLCUDAHost>().verify(*w_up);
    }
    RuntimeCheck(out_col_offset_up % 8 == 0 && out_col_offset_up >= 0 &&
                 out_col_offset_up + n_cols <= out_row,
                 "grouped_gemm_gateup: up out cols [", out_col_offset_up, ",",
                 out_col_offset_up + n_cols, ") invalid for out width ", out_row);
    RuntimeCheck(aligned16(w_up->data_ptr()), "grouped_gemm_gateup: w_up must be 16-byte aligned");
    if (layout == COLD) {
      RuntimeCheck(blk_off_up != nullptr, "grouped_gemm_cold_gateup: blk_off_up required");
      auto E = SymbolicSize{"num_experts"};
      TensorMatcher({E}).with_dtype<int64_t>().with_device(cuda_device).verify(*blk_off_up);
    }
    s1 = Slot{static_cast<const __nv_bfloat16*>(w_up->data_ptr()),
              static_cast<const int32_t*>(row_off_up->data_ptr()),
              static_cast<const uint16_t*>(kidx_up->data_ptr()), out_col_offset_up,
              blk_off_up ? static_cast<const int64_t*>(blk_off_up->data_ptr()) : nullptr};
  }

  const DLDevice device = cuda_device.unwrap();
  // grid.y 상한: Σₑ ceil(cₑ/BM) ≤ P/BM + E.
  int64_t tiles_upper = div_ceil(p, static_cast<int64_t>(kBM)) + num_experts;
  const int64_t grid_x = div_ceil(n_cols, static_cast<int64_t>(kBN));
  const int64_t grid_z = fused ? 2 : 1;
  // max_blocks: launch 전체 블록 수 상한 (0 = 없음). 타일 축(grid.y)에서 깎는다 —
  // 열 타일과 슬롯은 그대로 두어야 한 타일의 W가 한 번에 다 읽히기 때문이다.
  if (max_blocks > 0) {
    const int64_t cap = std::max<int64_t>(1, max_blocks / (grid_x * grid_z));
    if (cap < tiles_upper) tiles_upper = cap;
  }
  const dim3 grid(static_cast<unsigned int>(grid_x),
                  static_cast<unsigned int>(tiles_upper), static_cast<unsigned int>(grid_z));
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
        static_cast<long long>(n_cols), static_cast<long long>(out_row),
        static_cast<int>(n_block), static_cast<int>(k_block), s0, s1);
  };
  const int wres_kc = (wres_k_max > 0) ? wres_k_chunk(wres_k_max, device.device_id) : 0;
  if (wres_k_max > 0 && wres_kc > 0) {
    // W-resident: 블록 = (expert, 열 타일). grid.y = E (max_blocks로 persistent 상한).
    RuntimeCheck(wres_k_max % kBK == 0, "grouped_gemm(wres): k_max must be a multiple of 32, got ", wres_k_max);
    constexpr int bn = 64;
    const size_t dyn = wres_dyn_smem(wres_kc, bn);
    float* scratch_ptr = nullptr;
    if (wres_kc < wres_k_max) {
      RuntimeCheck(scratch != nullptr, "grouped_gemm(wres): K-chunked run needs an fp32 scratch [P, out_row]");
      TensorMatcher({P, W_row}).with_dtype<float>().with_device(cuda_device).verify(*scratch);
      scratch_ptr = static_cast<float*>(scratch->data_ptr());
    }
    int64_t gy = num_experts;
    const int64_t gx = div_ceil(n_cols, static_cast<int64_t>(bn));
    if (max_blocks > 0) gy = std::max<int64_t>(1, std::min<int64_t>(gy, max_blocks / (gx * grid_z)));
    const dim3 wgrid(static_cast<unsigned int>(gx), static_cast<unsigned int>(gy), static_cast<unsigned int>(grid_z));
    auto wlaunch = [&](auto kernel) {
      LaunchKernel(wgrid, block, device, dyn)(
          kernel,
          static_cast<const __nv_bfloat16*>(x.data_ptr()),
          static_cast<const int32_t*>(pair_sorted.data_ptr()),
          static_cast<const int32_t*>(pair_off.data_ptr()),
          static_cast<const int32_t*>(tile_off.data_ptr()),
          static_cast<__nv_bfloat16*>(out.data_ptr()), scratch_ptr,
          static_cast<int>(num_experts), static_cast<long long>(top_k),
          static_cast<long long>(x_kx), static_cast<int>(x_row_is_pair),
          static_cast<long long>(n_cols), static_cast<long long>(out_row),
          static_cast<int>(n_block), static_cast<int>(k_block), wres_kc, s0, s1);
    };
    if (layout == COLD) { wres_set_smem_attr<COLD, 64>(dyn); wlaunch(prism_grouped_gemm_wres<COLD, 64>); }
    else { wres_set_smem_attr<ROWMAJOR, 64>(dyn); wlaunch(prism_grouped_gemm_wres<ROWMAJOR, 64>); }
    return;
  }
  if (layout == COLD) launch(prism_grouped_gemm<COLD>);
  else launch(prism_grouped_gemm<ROWMAJOR>);
}

void grouped_gemm_indexed(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView w, tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair,
    int64_t max_blocks, int64_t wres_k_max, tvm::ffi::TensorView scratch) {
  const tvm::ffi::TensorView* sc = scratch.numel() > 0 ? &scratch : nullptr;
  grouped_gemm_impl(x, pair_sorted, pair_off, tile_off, w, row_off, kidx, out,
                    out_col_offset, x_row_is_pair, true, max_blocks,
                    nullptr, nullptr, nullptr, 0, ROWMAJOR, nullptr, nullptr, 0, 0, 0, wres_k_max, sc);
}

void grouped_gemm_indexed_pinned(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView w, tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair,
    int64_t max_blocks, int64_t wres_k_max, tvm::ffi::TensorView scratch) {
  const tvm::ffi::TensorView* sc = scratch.numel() > 0 ? &scratch : nullptr;
  grouped_gemm_impl(x, pair_sorted, pair_off, tile_off, w, row_off, kidx, out,
                    out_col_offset, x_row_is_pair, false, max_blocks,
                    nullptr, nullptr, nullptr, 0, ROWMAJOR, nullptr, nullptr, 0, 0, 0, wres_k_max, sc);
}

// COLD 레이아웃 — kt packed slab을 host 메모리(cudaHostRegister됨)에서 제자리 읽기.
// row_off는 타일 올림된 k_pad 누적, kidx는 패딩 포함, blk_off는 expert 블록의
// slab 내 원소 오프셋. out_col_offset에는 노드 N shard 시작이 더해져 온다.
void grouped_gemm_cold(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView w, tvm::ffi::TensorView blk_off,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair,
    int64_t max_blocks, int64_t n_block, int64_t k_block, int64_t n_cols, int64_t wres_k_max,
    tvm::ffi::TensorView scratch) {
  const tvm::ffi::TensorView* sc = scratch.numel() > 0 ? &scratch : nullptr;
  grouped_gemm_impl(x, pair_sorted, pair_off, tile_off, w, row_off, kidx, out,
                    out_col_offset, x_row_is_pair, false, max_blocks,
                    nullptr, nullptr, nullptr, 0, COLD, &blk_off, nullptr,
                    n_block, k_block, n_cols, wres_k_max, sc);
}

void grouped_gemm_cold_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView w_gate, tvm::ffi::TensorView blk_off_gate,
    tvm::ffi::TensorView row_off_gate, tvm::ffi::TensorView kidx_gate,
    tvm::ffi::TensorView w_up, tvm::ffi::TensorView blk_off_up,
    tvm::ffi::TensorView row_off_up, tvm::ffi::TensorView kidx_up,
    tvm::ffi::TensorView out, int64_t out_col_offset_gate,
    int64_t out_col_offset_up, int64_t x_row_is_pair,
    int64_t max_blocks, int64_t n_block, int64_t k_block, int64_t n_cols, int64_t wres_k_max,
    tvm::ffi::TensorView scratch) {
  const tvm::ffi::TensorView* sc = scratch.numel() > 0 ? &scratch : nullptr;
  grouped_gemm_impl(x, pair_sorted, pair_off, tile_off, w_gate, row_off_gate,
                    kidx_gate, out, out_col_offset_gate, x_row_is_pair, false,
                    max_blocks, &w_up, &row_off_up, &kidx_up, out_col_offset_up,
                    COLD, &blk_off_gate, &blk_off_up, n_block, k_block, n_cols, wres_k_max, sc);
}

void grouped_gemm_indexed_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView w_gate, tvm::ffi::TensorView row_off_gate,
    tvm::ffi::TensorView kidx_gate,
    tvm::ffi::TensorView w_up, tvm::ffi::TensorView row_off_up,
    tvm::ffi::TensorView kidx_up,
    tvm::ffi::TensorView out, int64_t out_col_offset_gate,
    int64_t out_col_offset_up, int64_t x_row_is_pair, int64_t max_blocks, int64_t wres_k_max,
    tvm::ffi::TensorView scratch) {
  const tvm::ffi::TensorView* sc = scratch.numel() > 0 ? &scratch : nullptr;
  grouped_gemm_impl(x, pair_sorted, pair_off, tile_off, w_gate, row_off_gate,
                    kidx_gate, out, out_col_offset_gate, x_row_is_pair, true,
                    max_blocks, &w_up, &row_off_up, &kidx_up, out_col_offset_up,
                    ROWMAJOR, nullptr, nullptr, 0, 0, 0, wres_k_max, sc);
}

void grouped_gemm_indexed_pinned_gateup(
    tvm::ffi::TensorView x, tvm::ffi::TensorView pair_sorted,
    tvm::ffi::TensorView pair_off, tvm::ffi::TensorView tile_off,
    tvm::ffi::TensorView w_gate, tvm::ffi::TensorView row_off_gate,
    tvm::ffi::TensorView kidx_gate,
    tvm::ffi::TensorView w_up, tvm::ffi::TensorView row_off_up,
    tvm::ffi::TensorView kidx_up,
    tvm::ffi::TensorView out, int64_t out_col_offset_gate,
    int64_t out_col_offset_up, int64_t x_row_is_pair, int64_t max_blocks, int64_t wres_k_max,
    tvm::ffi::TensorView scratch) {
  const tvm::ffi::TensorView* sc = scratch.numel() > 0 ? &scratch : nullptr;
  grouped_gemm_impl(x, pair_sorted, pair_off, tile_off, w_gate, row_off_gate,
                    kidx_gate, out, out_col_offset_gate, x_row_is_pair, false,
                    max_blocks, &w_up, &row_off_up, &kidx_up, out_col_offset_up,
                    ROWMAJOR, nullptr, nullptr, 0, 0, 0, wres_k_max, sc);
}

}  // namespace
