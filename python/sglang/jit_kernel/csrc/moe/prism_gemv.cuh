#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// Prism worklist GEMV: pair p=(m,j)가 e=topk[m,j]의 W 밴드를 store에서 직접
// 읽어 rejoin 레이아웃 out[m, j, off:off+N]에 쓴다. 누산 fp32, 출력 bf16
// (계약 ⑤). 리덕션 순서는 (ty-strided 루프 → smem 4-way 합) 고정 —
// 정확표현 입력에서 결정적/비트재현.
//
// grid = (ceil_div(N, 64), M×top_k), block = (64, 4).
// thread(tx, ty): col n = bx*64+tx, r ∈ {ty, ty+4, ...} 부분합 → smem 합산.
//
// INDEXED(계약 ① 2026-08-25): 티어 멤버십이 밴드에서 **가변 per-expert 인덱스**로
// 바뀐 형태. 달라지는 것은 두 줄뿐이다 —
//   - K 구간이 `(k_offset, k_rows)` 상수에서 `row_off[e] ..= row_off[e+1]`로
//   - activation 접근이 `xr[k_offset + r]`에서 `xr[kidx[o0 + r]]`로
// grid는 그대로다: k는 grid 차원이 아니라 루프라서 expert마다 길이가 달라도
// launch 모양이 변하지 않는다. `k_rows[e] == 0`이면 루프가 0회 돌고 acc=0을
// 기록하는데, 그것이 그 티어의 정확한 부분합이므로 "이 expert는 이 티어에
// 없음"이 분기 없이 처리된다.
//
// W는 INDEXED에서 flat `[Σₑ k[e], N]`이고 **row_off를 인덱스와 공유한다**
// (둘 다 expert당 k[e]개). 비인덱스 경로의 `e * k_rows`가 `row_off[e]`로
// 일반화된 것이라, 연속 인덱스(밴드 퇴화형)에서는 읽는 원소도 누산 순서도
// 완전히 같다 — 그래서 밴드 경로와 **비트일치**한다.
//
// 스펙 편차: W 로드(`we[r * n_cols + n]`)는 스칼라 bf16(스레드당 2B)이고
// 정렬 RuntimeCheck도 없다 — 스펙(2026-08-25-prism-gemv-worklist-design.md
// §2)이 요구한 uint4/uint2 벡터화(및 수반 정렬 체크)를 이 1차 구현은 안 한다.
// 디바이스 경로 실측 10.7us vs gather+bmm 17.5us로 이미 충분해 벡터화를
// 미뤘다; warm/UVA 대역폭이 임계로 확인되면 벡터 로드 + 정렬 RuntimeCheck를
// 후속으로 추가한다.
// SPARSE(계약 ①의 k2wl2, 2026-08-26): 티어가 자기 행에 대해 페어 마스크를
// 만들고 **죽은 페어의 W 로드를 아예 발행하지 않는다**. warm에서 그 로드는
// PCIe이므로 건너뛴 만큼이 그대로 대역폭 절약이다 (hot은 VRAM이라 이득이
// 작고, 그래서 hot은 dense로 둔다 — tiers.py의 SPARSE_TIERS).
//
// 임계값을 **CPU에서 건네받지 않고 여기서 다시 계산한다**는 것이 설계의
// 핵심이다. thr은 (라우터 가중, p, λ, pmax, grid, ng, 곡선)의 순수 함수이고
// 그 전부가 이미 GPU에 있다. 건네받으려면 스텝마다 device sync가 생겨 CUDA
// graph 캡처가 깨진다 — 같은 순수 함수를 양쪽이 독립적으로 계산하면 sync
// 없이 같은 값에 도달한다. 식과 반올림은 kt의 slot_sparsity/thr_of/
// build_pair_mask를 그대로 옮긴 것이다 (rintf ↔ lrint 모두
// round-half-to-even).
//
// 비트일치 주의: 점수식의 FMA 융합 순서는 nvcc가 정하므로 CPU와 마지막 비트가
// 갈릴 수 있다. 임계에 정확히 걸린 페어 하나가 한쪽에서 뒤집히는 것이 최악인데,
// **페어는 정확히 한 티어에만 속하므로**(ROW_GROUP % PAIR_GROUP == 0) 이중계산도
// 누락도 아니고 그 페어 하나의 정확도 차이다 — rejoin 정확성 문제가 아니다.
struct SparseArgs {
  const float* a;        // [Σₑ k[e]]      = wn² (weight와 같은 오프셋)
  const float* c;        // [Σₑ k[e]/2]    = 인접열 내적
  const float* thr_tab;  // [E, ng]        = sparsity → threshold 곡선
  const float* topk_w;   // [M, top_k]     = 라우터 가중
  float p, lam, pmax, grid;
  int ng, renorm_it;
};

template <typename IdxT, bool INDEXED, int V, bool SPARSE>
__global__ void prism_gemv_worklist(
    const __nv_bfloat16* __restrict__ x,
    const IdxT* __restrict__ topk,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    const int32_t* __restrict__ row_off,   // INDEXED: [E+1] / else nullptr
    const uint16_t* __restrict__ kidx,     // INDEXED: [Σ k[e]] / else nullptr
    long long x_kx,          // x row 길이 (Kx)
    long long k_offset,      // 비인덱스 경로 전용
    long long k_rows,        // 비인덱스 경로 전용 (expert 공통)
    long long n_cols,        // N
    long long out_row,       // out3d의 마지막 축 길이 (W_row)
    long long out_off,       // out_col_offset
    long long top_k,
    int x_row_is_pair,
    SparseArgs sp) {
  // 점수는 gather된 activation을 읽으므로 인덱스 경로만이 마스킹을 표현할 수
  // 있다 (밴드 경로는 전환기 잔재다).
  static_assert(!SPARSE || INDEXED, "SPARSE requires the indexed path");
  const long long pair = blockIdx.y;
  const long long m = pair / top_k;
  const long long e = static_cast<long long>(topk[pair]);
  const long long row = x_row_is_pair ? pair : m;
  // 블록의 열 타일은 V와 무관하게 64로 고정한다. V를 키우면 blockDim이
  // (64/V, 4V)로 재배치되므로 **블록 수와 타일 기하가 불변**이다 — 단순히
  // 열을 V배 맡게 하면 grid.x가 1/V로 붕괴한다 (gate는 N=512, V=8에서 8블록).
  constexpr int NCOL = 64;
  constexpr int NY = 4 * V;  // = blockDim.y
  const long long n0 = static_cast<long long>(blockIdx.x) * NCOL +
                       static_cast<long long>(threadIdx.x) * V;
  const bool active = n0 < n_cols;

  long long o0, kr;
  if constexpr (INDEXED) {
    o0 = static_cast<long long>(row_off[e]);
    kr = static_cast<long long>(row_off[e + 1]) - o0;
  } else {
    o0 = e * k_rows;
    kr = k_rows;
  }
  const __nv_bfloat16* xr = x + row * x_kx;
  const __nv_bfloat16* we = w + o0 * n_cols;
  const uint16_t* ie = INDEXED ? kidx + o0 : nullptr;

  // 이 (m, j)의 임계값. 전 스레드가 중복 계산한다 — top_k(보통 8)짜리 루프라
  // syncthreads 한 번보다 싸고, 공유 상태가 없어 결정적이다.
  float thr2 = 0.f;
  if constexpr (SPARSE) {
    constexpr int MAXK = 16;  // host RuntimeCheck가 top_k <= MAXK를 보증
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
    // 재정규화가 s의 평균을 쓰므로 슬롯 하나만 따로 구할 수 없다 (kt와 동일).
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
    thr2 = thr * thr;  // kt도 제곱 비교다 (sqrt를 양쪽 다 생략)
  }

  // 이 블록이 쓰는 activation 조각을 smem에 한 번 모은다.
  //
  // 인덱스 경로에서 이게 없으면 내부 루프가 `idx[r] → x[idx[r]]` **의존 로드
  // 사슬**을 원소마다 탄다. gate/up 치수(k=768, N=512)는 grid가 (8, 8) = 64
  // 블록뿐이라 188 SM에서 순수 지연 바운드이고, 그 사슬이 1:1로 벽시계에
  // 드러난다 (2026-08-25 실측: 10.5 → 19.3 µs, 1.83배). 블록당 1회 협력
  // gather로 바꾸면 내부 루프는 의존성 없는 smem 읽기가 된다.
  //
  // K를 KTILE 단위로 끊는 이유는 smem을 상수로 묶기 위해서다 (k[e]가
  // 가변이라 최대치를 host가 모른다). KTILE이 blockDim.y(4)의 배수이므로
  // 스레드별 누산 순서는 청킹 전과 **완전히 같다** — 연속 인덱스에서 밴드
  // 경로와의 비트일치가 이 변경으로 깨지지 않는 이유다.
  //
  // 스테이징은 인덱스 경로에만 건다: 밴드 경로는 x를 순차로 읽어 끊을 사슬이
  // 없고, syncthreads만 늘어 순손실이었다 (실측 bs=1 10.5 → 12.0 µs).
  constexpr int KTILE = 2048;

  float acc[V];
#pragma unroll
  for (int j = 0; j < V; ++j) acc[j] = 0.f;

  // W 로드 폭. bf16 스칼라(2 B/스레드)로는 피크 대역폭을 채울 만큼 요청을
  // 띄우지 못한다 (실측 실효 524 GB/s). V개 열이 W에서 연속이므로 uint2(8 B)/
  // uint4(16 B) 한 번으로 가져와 스레드당 in-flight 바이트를 늘린다.
  // 정렬은 계약 ①의 COL_GROUP=32가 보증한다 (n_cols % 8 == 0).
  auto fma_row = [&](const __nv_bfloat16* wp, float xv) {
    if constexpr (V == 1) {
      acc[0] += xv * __bfloat162float(wp[0]);
    } else if constexpr (V == 4) {
      const uint2 v = *reinterpret_cast<const uint2*>(wp);
      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] += xv * __bfloat162float(wb[j]);
    } else {
      const uint4 v = *reinterpret_cast<const uint4*>(wp);
      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += xv * __bfloat162float(wb[j]);
    }
  };

  if constexpr (INDEXED) {
    __shared__ __nv_bfloat16 xs[KTILE];
    // 페어 마스크. 한 페어를 정확히 한 스레드가 판정하고 전 스레드가 읽는다.
    __shared__ uint8_t keep[SPARSE ? KTILE / 2 : 1];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    for (long long base = 0; base < kr; base += KTILE) {
      const int cnt = static_cast<int>(min(static_cast<long long>(KTILE), kr - base));
      __syncthreads();
      for (int t = tid; t < cnt; t += nthreads) {
        xs[t] = xr[static_cast<long long>(ie[base + t])];
      }
      __syncthreads();
      int np = 0;
      if constexpr (SPARSE) {
        // 페어가 타일을 가로지르지 않는다: KTILE이 짝수이고 row_off가
        // ROW_GROUP=2 정렬이라 base가 항상 짝수다. 따라서 gather된 짝수/홀수
        // 위치가 같은 원본 페어의 두 반쪽이고 (index.py의 페어 무결성 검사),
        // c의 첨자는 절대 행의 절반이다 — kt의 `c + row_base(e)/2`와 같은 규약.
        np = cnt >> 1;
        for (int i = tid; i < np; i += nthreads) {
          const float x0 = __bfloat162float(xs[2 * i]);
          const float x1 = __bfloat162float(xs[2 * i + 1]);
          const long long ar = o0 + base + 2 * i;
          float en = sp.a[ar] * x0 * x0 + sp.a[ar + 1] * x1 * x1 +
                     2.0f * sp.c[ar >> 1] * x0 * x1;
          if (en < 0.f) en = 0.f;
          keep[i] = (en >= thr2) ? uint8_t{1} : uint8_t{0};
        }
        __syncthreads();
      }
      if (active) {
        // 루프 모양이 dense와 **같다** (스레드 y가 행 y, y+NY, …). 압축
        // 리스트로 바꾸면 누산 순서가 달라져 p=0에서도 dense와 비트가 갈리고,
        // 순서 재현이 계약 ⑤의 요구다. 건너뛰는 것은 W 로드 발행 자체이므로
        // 대역폭 절약은 압축과 동일하다 — 잃는 것은 NY 스레드 간 부하 균형뿐이고,
        // 마스크가 페어 단위라 그 편차는 스레드마다 같은 비율이다.
        for (int r = threadIdx.y; r < cnt; r += NY) {
          if constexpr (SPARSE) {
            // (r >> 1) == np는 cnt가 홀수인 경우의 반쪽 페어다 — 불변식상
            // 도달 불가지만, 도달하면 마스킹하지 않고 계산한다 (느릴 뿐 안전).
            if ((r >> 1) < np && !keep[r >> 1]) continue;
          }
          fma_row(we + (base + r) * n_cols + n0, __bfloat162float(xs[r]));
        }
      }
    }
  } else {
    // 밴드 경로는 x를 순차로 읽어 의존 사슬이 없다 — 스테이징이 순손실이라
    // (실측 bs=1 10.5 → 12.0 µs) 원래 루프를 그대로 둔다.
    if (active) {
      for (long long r = threadIdx.y; r < kr; r += NY) {
        fma_row(we + r * n_cols + n0, __bfloat162float(xr[k_offset + r]));
      }
    }
  }
  __shared__ float red[NY][NCOL];
#pragma unroll
  for (int j = 0; j < V; ++j) {
    red[threadIdx.y][threadIdx.x * V + j] = active ? acc[j] : 0.f;
  }
  __syncthreads();
  if constexpr (V == 1) {
    // 기존 순서 그대로 — 밴드 경로의 비트 재현을 건드리지 않는다.
    if (threadIdx.y == 0 && active) {
      const float t = red[0][threadIdx.x] + red[1][threadIdx.x] +
                      red[2][threadIdx.x] + red[3][threadIdx.x];
      out[pair * out_row + out_off + n0] = __float2bfloat16(t);
    }
  } else {
    for (int stride = NY / 2; stride > 0; stride >>= 1) {
      if (threadIdx.y < stride) {
#pragma unroll
        for (int j = 0; j < V; ++j) {
          red[threadIdx.y][threadIdx.x * V + j] +=
              red[threadIdx.y + stride][threadIdx.x * V + j];
        }
      }
      __syncthreads();
    }
    if (threadIdx.y == 0 && active) {
#pragma unroll
      for (int j = 0; j < V; ++j) {
        out[pair * out_row + out_off + n0 + j] =
            __float2bfloat16(red[0][threadIdx.x * V + j]);
      }
    }
  }
}

// topk dtype 디스패치. 나머지 인자는 두 경로가 공유한다 (비인덱스는
// row_off/kidx가 nullptr, 인덱스는 k_offset/k_rows가 무시된다).
template <bool INDEXED, int V, bool SPARSE>
inline void launch_gemv_worklist(
    const dim3& grid, const DLDevice& device,
    tvm::ffi::TensorView topk,
    const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out,
    const int32_t* row_off, const uint16_t* kidx,
    int64_t x_kx, int64_t k_offset, int64_t k_rows, int64_t n_cols,
    int64_t out_row, int64_t out_col_offset, int64_t top_k, int x_row_is_pair,
    const SparseArgs& sp) {
  using namespace host;
  const dim3 block(64 / V, 4 * V);  // 블록당 열 타일 64 고정 (커널 주석 참조)
  if (is_type<int32_t>(topk.dtype())) {
    LaunchKernel(grid, block, device)(
        prism_gemv_worklist<int32_t, INDEXED, V, SPARSE>, x,
        static_cast<const int32_t*>(topk.data_ptr()), w, out, row_off, kidx,
        x_kx, k_offset, k_rows, n_cols, out_row, out_col_offset, top_k,
        x_row_is_pair, sp);
  } else {
    LaunchKernel(grid, block, device)(
        prism_gemv_worklist<int64_t, INDEXED, V, SPARSE>, x,
        static_cast<const int64_t*>(topk.data_ptr()), w, out, row_off, kidx,
        x_kx, k_offset, k_rows, n_cols, out_row, out_col_offset, top_k,
        x_row_is_pair, sp);
  }
}

// W 로드 폭 선택. uint2/uint4 정렬 조건은 (a) n_cols가 V의 배수, (b) 스토어
// 시작 주소가 V*2 바이트 정렬. (a)는 계약 ①의 COL_GROUP=32가 보증하고,
// row_off[e]는 임의여도 된다 — 행 시작이 n_cols 원소의 배수이므로 (a)가
// 곧 행 정렬이다. hint > 0이면 강제(벤치/디버그용), 0이면 자동.
inline int choose_vec(int64_t hint, int64_t n_cols, const void* w) {
  using namespace host;
  auto ok = [&](int v) {
    return n_cols % v == 0 &&
           reinterpret_cast<std::uintptr_t>(w) % (static_cast<std::size_t>(v) * 2) == 0;
  };
  if (hint > 0) {
    RuntimeCheck(hint == 1 || hint == 4 || hint == 8,
                 "gemv_worklist_indexed: vec must be 0(auto), 1, 4 or 8, got ", hint);
    RuntimeCheck(ok(static_cast<int>(hint)),
                 "gemv_worklist_indexed: vec=", hint,
                 " needs n_cols (", n_cols, ") divisible by it and a ",
                 hint * 2, "-byte aligned store");
    return static_cast<int>(hint);
  }
  if (ok(8)) return 8;
  if (ok(4)) return 4;
  return 1;
}

// 공통 검증+launch. src_on_device로 W의 device 제약만 갈라진다
// (gather_bands_from_pinned/device 쌍둥이와 같은 이유 — pinned 변형은
// UVA 플랫폼 가정의 회귀 가드를 겸한다).
inline void gemv_worklist_impl(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView w,
    tvm::ffi::TensorView out, int64_t k_offset, int64_t out_col_offset,
    int64_t x_row_is_pair, bool w_on_device) {
  using namespace host;

  auto Rx = SymbolicSize{"x_rows"};
  auto Kx = SymbolicSize{"x_cols"};
  auto M = SymbolicSize{"num_tokens"};
  auto K = SymbolicSize{"top_k"};
  auto E = SymbolicSize{"num_experts"};
  auto R = SymbolicSize{"k_rows"};
  auto N = SymbolicSize{"n_cols"};
  auto W_row = SymbolicSize{"out_row"};
  auto cuda_device = SymbolicDevice{};

  TensorMatcher({M, K}).with_dtype<int32_t, int64_t>()
      .with_device<kDLCUDA>(cuda_device).verify(topk);
  TensorMatcher({Rx, Kx}).with_dtype<bf16_t>().with_device(cuda_device).verify(x);
  TensorMatcher({M, K, W_row}).with_dtype<bf16_t>().with_device(cuda_device).verify(out);
  if (w_on_device) {
    TensorMatcher({E, R, N}).with_dtype<bf16_t>().with_device(cuda_device).verify(w);
  } else {
    TensorMatcher({E, R, N}).with_dtype<bf16_t>().with_device<kDLCPU, kDLCUDAHost>().verify(w);
  }

  const int64_t m = M.unwrap(), top_k = K.unwrap();
  const int64_t k_rows = R.unwrap(), n_cols = N.unwrap();
  const int64_t x_rows = Rx.unwrap(), x_kx = Kx.unwrap();
  const int64_t out_row = W_row.unwrap();

  RuntimeCheck(x_row_is_pair ? (x_rows == m * top_k) : (x_rows == m),
               "gemv_worklist: x rows (", x_rows, ") must be ",
               x_row_is_pair ? "M*top_k" : "M");
  RuntimeCheck(k_offset >= 0 && k_offset + k_rows <= x_kx,
               "gemv_worklist: band [", k_offset, ",", k_offset + k_rows,
               ") out of x width ", x_kx);
  RuntimeCheck(out_col_offset >= 0 && out_col_offset + n_cols <= out_row,
               "gemv_worklist: out cols [", out_col_offset, ",",
               out_col_offset + n_cols, ") out of out width ", out_row);

  const DLDevice device = cuda_device.unwrap();
  const dim3 block(64, 4);
  const dim3 grid(static_cast<unsigned int>(div_ceil(n_cols, static_cast<int64_t>(64))),
                  static_cast<unsigned int>(m * top_k));

  // 밴드 경로는 V=1 고정 — 인덱스 전환과 함께 폐기될 경로라 건드리지 않는다.
  launch_gemv_worklist<false, 1, false>(
      grid, device, topk,
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<const __nv_bfloat16*>(w.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      nullptr, nullptr,
      x_kx, k_offset, k_rows, n_cols, out_row, out_col_offset, top_k,
      static_cast<int>(x_row_is_pair), SparseArgs{});
}

// 인덱스 변형: W가 flat [Σₑ k[e], N]이고 K 구간은 row_off가, activation 열은
// kidx가 준다. row_off/kidx는 **항상 device 상주**다 (W가 pinned인 warm 변형
// 에서도) — tiny하고, 그래프 캡처가 주소를 baked해야 하므로 로드 타임에
// device에 올라간 것을 그대로 쓴다.
// sparse 변형은 여기에 4개 텐서(a, c, thr, topk_weights)와 예산 스칼라를 더
// 얹는다. dense 진입점은 손대지 않는다 — 템플릿 bool로 갈라지므로 dense
// codegen이 움직이지 않는 것이 이 구조의 요점이다.
struct SparseIn {
  tvm::ffi::TensorView a, c, thr, topk_w;
  double p, lam, pmax, grid;
  int64_t ng, renorm_it;
};

inline void gemv_worklist_indexed_impl(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView w,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair,
    int64_t vec, bool w_on_device, const SparseIn* sin) {
  using namespace host;

  auto Rx = SymbolicSize{"x_rows"};
  auto Kx = SymbolicSize{"x_cols"};
  auto M = SymbolicSize{"num_tokens"};
  auto K = SymbolicSize{"top_k"};
  auto R = SymbolicSize{"total_rows"};
  auto E1 = SymbolicSize{"num_experts_plus_one"};
  auto N = SymbolicSize{"n_cols"};
  auto W_row = SymbolicSize{"out_row"};
  auto cuda_device = SymbolicDevice{};

  TensorMatcher({M, K}).with_dtype<int32_t, int64_t>()
      .with_device<kDLCUDA>(cuda_device).verify(topk);
  TensorMatcher({Rx, Kx}).with_dtype<bf16_t>().with_device(cuda_device).verify(x);
  TensorMatcher({M, K, W_row}).with_dtype<bf16_t>().with_device(cuda_device).verify(out);
  // R을 w와 kidx가 공유한다 — 길이 불일치가 여기서 잡힌다 (오프셋 테이블과
  // 스토어가 어긋나면 조용히 남의 행을 읽는다).
  TensorMatcher({E1}).with_dtype<int32_t>().with_device(cuda_device).verify(row_off);
  TensorMatcher({R}).with_dtype<uint16_t>().with_device(cuda_device).verify(kidx);
  if (w_on_device) {
    TensorMatcher({R, N}).with_dtype<bf16_t>().with_device(cuda_device).verify(w);
  } else {
    TensorMatcher({R, N}).with_dtype<bf16_t>().with_device<kDLCPU, kDLCUDAHost>().verify(w);
  }

  const int64_t m = M.unwrap(), top_k = K.unwrap();
  const int64_t n_cols = N.unwrap(), out_row = W_row.unwrap();
  const int64_t x_rows = Rx.unwrap(), x_kx = Kx.unwrap();

  RuntimeCheck(x_row_is_pair ? (x_rows == m * top_k) : (x_rows == m),
               "gemv_worklist_indexed: x rows (", x_rows, ") must be ",
               x_row_is_pair ? "M*top_k" : "M");
  RuntimeCheck(E1.unwrap() >= 2,
               "gemv_worklist_indexed: row_off must have E+1 >= 2 entries, got ",
               E1.unwrap());
  // uint16 인덱스가 x의 열을 가리킬 수 있어야 한다. 값 자체는 host에서 볼 수
  // 없으므로(디바이스 메모리) 표현 범위만 막는다 — 값 검증은 로드 타임의
  // 순열 검사(index.py)가 담당한다.
  RuntimeCheck(x_kx <= 65536,
               "gemv_worklist_indexed: x width ", x_kx,
               " exceeds the uint16 index range");
  RuntimeCheck(out_col_offset >= 0 && out_col_offset + n_cols <= out_row,
               "gemv_worklist_indexed: out cols [", out_col_offset, ",",
               out_col_offset + n_cols, ") out of out width ", out_row);

  SparseArgs sp{};
  if (sin != nullptr) {
    auto Ng = SymbolicSize{"sparsity_ng"};
    auto E = SymbolicSize{"num_experts"};
    // a/c는 weight와 **같은 오프셋 테이블**을 쓴다 — R을 공유시켜 길이 불일치를
    // 여기서 잡는다 (어긋나면 조용히 남의 페어 점수로 마스킹한다).
    TensorMatcher({R}).with_dtype<float>().with_device(cuda_device).verify(sin->a);
    // SymbolicSize에 산술이 없으므로 페어 축은 별도 기호로 받고 R과의 관계를
    // RuntimeCheck로 묶는다.
    auto Rp = SymbolicSize{"total_pairs"};
    TensorMatcher({Rp}).with_dtype<float>().with_device(cuda_device).verify(sin->c);
    TensorMatcher({E, Ng}).with_dtype<float>().with_device(cuda_device).verify(sin->thr);
    TensorMatcher({M, K}).with_dtype<float>().with_device(cuda_device).verify(sin->topk_w);
    RuntimeCheck(Rp.unwrap() * 2 == R.unwrap(),
                 "gemv_worklist_indexed_sparse: pair_dot has ", Rp.unwrap(),
                 " entries but the store has ", R.unwrap(),
                 " rows (must be exactly half)");
    RuntimeCheck(E1.unwrap() == E.unwrap() + 1,
                 "gemv_worklist_indexed_sparse: thr has ", E.unwrap(),
                 " experts but row_off implies ", E1.unwrap() - 1);
    RuntimeCheck(Ng.unwrap() == sin->ng,
                 "gemv_worklist_indexed_sparse: thr grid ", Ng.unwrap(),
                 " != ng ", sin->ng);
    RuntimeCheck(top_k <= 16,
                 "gemv_worklist_indexed_sparse: top_k ", top_k,
                 " exceeds the per-thread slot budget (16)");
    RuntimeCheck(sin->grid > 0.0,
                 "gemv_worklist_indexed_sparse: grid must be positive, got ",
                 sin->grid);
    sp.a = static_cast<const float*>(sin->a.data_ptr());
    sp.c = static_cast<const float*>(sin->c.data_ptr());
    sp.thr_tab = static_cast<const float*>(sin->thr.data_ptr());
    sp.topk_w = static_cast<const float*>(sin->topk_w.data_ptr());
    sp.p = static_cast<float>(sin->p);
    sp.lam = static_cast<float>(sin->lam);
    sp.pmax = static_cast<float>(sin->pmax);
    sp.grid = static_cast<float>(sin->grid);
    sp.ng = static_cast<int>(sin->ng);
    sp.renorm_it = static_cast<int>(sin->renorm_it);
  }

  const DLDevice device = cuda_device.unwrap();
  const dim3 block(64, 4);
  const dim3 grid(static_cast<unsigned int>(div_ceil(n_cols, static_cast<int64_t>(64))),
                  static_cast<unsigned int>(m * top_k));

#define PRISM_LAUNCH_INDEXED(V, SP)                                          \
  launch_gemv_worklist<true, V, SP>(                                         \
      grid, device, topk,                                                    \
      static_cast<const __nv_bfloat16*>(x.data_ptr()),                       \
      static_cast<const __nv_bfloat16*>(w.data_ptr()),                       \
      static_cast<__nv_bfloat16*>(out.data_ptr()),                           \
      static_cast<const int32_t*>(row_off.data_ptr()),                       \
      static_cast<const uint16_t*>(kidx.data_ptr()),                         \
      x_kx, 0, 0, n_cols, out_row, out_col_offset, top_k,                    \
      static_cast<int>(x_row_is_pair), sp)
#define PRISM_LAUNCH_INDEXED_V(V)                                            \
  do {                                                                       \
    if (sin != nullptr) PRISM_LAUNCH_INDEXED(V, true);                        \
    else PRISM_LAUNCH_INDEXED(V, false);                                      \
  } while (0)

  switch (choose_vec(vec, n_cols, w.data_ptr())) {
    case 8: PRISM_LAUNCH_INDEXED_V(8); break;
    case 4: PRISM_LAUNCH_INDEXED_V(4); break;
    default: PRISM_LAUNCH_INDEXED_V(1); break;
  }
#undef PRISM_LAUNCH_INDEXED_V
#undef PRISM_LAUNCH_INDEXED
}

void gemv_worklist(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView w,
    tvm::ffi::TensorView out, int64_t k_offset, int64_t out_col_offset,
    int64_t x_row_is_pair) {
  gemv_worklist_impl(x, topk, w, out, k_offset, out_col_offset, x_row_is_pair, true);
}

void gemv_worklist_pinned(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView w,
    tvm::ffi::TensorView out, int64_t k_offset, int64_t out_col_offset,
    int64_t x_row_is_pair) {
  gemv_worklist_impl(x, topk, w, out, k_offset, out_col_offset, x_row_is_pair, false);
}

void gemv_worklist_indexed(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView w,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair,
    int64_t vec) {
  gemv_worklist_indexed_impl(x, topk, w, row_off, kidx, out, out_col_offset,
                             x_row_is_pair, vec, true, nullptr);
}

void gemv_worklist_indexed_pinned(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView w,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair,
    int64_t vec) {
  gemv_worklist_indexed_impl(x, topk, w, row_off, kidx, out, out_col_offset,
                             x_row_is_pair, vec, false, nullptr);
}

// sparse 쌍둥이. 인자가 넷 늘고(a, c, thr, topk_weights) 예산 스칼라가 붙는
// 것 외에 dense와 같다. 별도 진입점으로 둔 이유: optional 텐서로 하나에
// 합치면 dense 경로가 매 호출 nullable 검사를 지나고, 무엇보다 "sparse인지"가
// 호출부에서 안 보이게 된다.
void gemv_worklist_indexed_sparse(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView w,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, tvm::ffi::TensorView a, tvm::ffi::TensorView c,
    tvm::ffi::TensorView thr, tvm::ffi::TensorView topk_w,
    int64_t out_col_offset, int64_t x_row_is_pair, int64_t vec,
    double p, double lam, double pmax, double grid,
    int64_t ng, int64_t renorm_it) {
  const SparseIn sin{a, c, thr, topk_w, p, lam, pmax, grid, ng, renorm_it};
  gemv_worklist_indexed_impl(x, topk, w, row_off, kidx, out, out_col_offset,
                             x_row_is_pair, vec, true, &sin);
}

void gemv_worklist_indexed_pinned_sparse(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView w,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, tvm::ffi::TensorView a, tvm::ffi::TensorView c,
    tvm::ffi::TensorView thr, tvm::ffi::TensorView topk_w,
    int64_t out_col_offset, int64_t x_row_is_pair, int64_t vec,
    double p, double lam, double pmax, double grid,
    int64_t ng, int64_t renorm_it) {
  const SparseIn sin{a, c, thr, topk_w, p, lam, pmax, grid, ng, renorm_it};
  gemv_worklist_indexed_impl(x, topk, w, row_off, kidx, out, out_col_offset,
                             x_row_is_pair, vec, false, &sin);
}

}  // namespace
