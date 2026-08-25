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
template <typename IdxT, bool INDEXED>
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
    int x_row_is_pair) {
  const long long pair = blockIdx.y;
  const long long m = pair / top_k;
  const long long e = static_cast<long long>(topk[pair]);
  const long long row = x_row_is_pair ? pair : m;
  const long long n = static_cast<long long>(blockIdx.x) * 64 + threadIdx.x;

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

  float acc = 0.f;
  if constexpr (INDEXED) {
    __shared__ __nv_bfloat16 xs[KTILE];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    for (long long base = 0; base < kr; base += KTILE) {
      const int cnt = static_cast<int>(min(static_cast<long long>(KTILE), kr - base));
      __syncthreads();
      for (int t = tid; t < cnt; t += nthreads) {
        xs[t] = xr[static_cast<long long>(ie[base + t])];
      }
      __syncthreads();
      if (n < n_cols) {
        for (int r = threadIdx.y; r < cnt; r += 4) {
          acc += __bfloat162float(xs[r]) *
                 __bfloat162float(we[(base + r) * n_cols + n]);
        }
      }
    }
  } else {
    // 밴드 경로는 x를 순차로 읽어 의존 사슬이 없다 — 스테이징이 순손실이라
    // (실측 bs=1 10.5 → 12.0 µs) 원래 루프를 그대로 둔다.
    if (n < n_cols) {
      for (long long r = threadIdx.y; r < kr; r += 4) {
        acc += __bfloat162float(xr[k_offset + r]) *
               __bfloat162float(we[r * n_cols + n]);
      }
    }
  }
  __shared__ float red[4][64];
  red[threadIdx.y][threadIdx.x] = acc;
  __syncthreads();
  if (threadIdx.y == 0 && n < n_cols) {
    const float s = red[0][threadIdx.x] + red[1][threadIdx.x] +
                    red[2][threadIdx.x] + red[3][threadIdx.x];
    out[pair * out_row + out_off + n] = __float2bfloat16(s);
  }
}

// topk dtype 디스패치. 나머지 인자는 두 경로가 공유한다 (비인덱스는
// row_off/kidx가 nullptr, 인덱스는 k_offset/k_rows가 무시된다).
template <bool INDEXED>
inline void launch_gemv_worklist(
    const dim3& grid, const dim3& block, const DLDevice& device,
    tvm::ffi::TensorView topk,
    const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out,
    const int32_t* row_off, const uint16_t* kidx,
    int64_t x_kx, int64_t k_offset, int64_t k_rows, int64_t n_cols,
    int64_t out_row, int64_t out_col_offset, int64_t top_k, int x_row_is_pair) {
  using namespace host;
  if (is_type<int32_t>(topk.dtype())) {
    LaunchKernel(grid, block, device)(
        prism_gemv_worklist<int32_t, INDEXED>, x,
        static_cast<const int32_t*>(topk.data_ptr()), w, out, row_off, kidx,
        x_kx, k_offset, k_rows, n_cols, out_row, out_col_offset, top_k, x_row_is_pair);
  } else {
    LaunchKernel(grid, block, device)(
        prism_gemv_worklist<int64_t, INDEXED>, x,
        static_cast<const int64_t*>(topk.data_ptr()), w, out, row_off, kidx,
        x_kx, k_offset, k_rows, n_cols, out_row, out_col_offset, top_k, x_row_is_pair);
  }
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

  launch_gemv_worklist<false>(
      grid, block, device, topk,
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<const __nv_bfloat16*>(w.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      nullptr, nullptr,
      x_kx, k_offset, k_rows, n_cols, out_row, out_col_offset, top_k,
      static_cast<int>(x_row_is_pair));
}

// 인덱스 변형: W가 flat [Σₑ k[e], N]이고 K 구간은 row_off가, activation 열은
// kidx가 준다. row_off/kidx는 **항상 device 상주**다 (W가 pinned인 warm 변형
// 에서도) — tiny하고, 그래프 캡처가 주소를 baked해야 하므로 로드 타임에
// device에 올라간 것을 그대로 쓴다.
inline void gemv_worklist_indexed_impl(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView w,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair,
    bool w_on_device) {
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

  const DLDevice device = cuda_device.unwrap();
  const dim3 block(64, 4);
  const dim3 grid(static_cast<unsigned int>(div_ceil(n_cols, static_cast<int64_t>(64))),
                  static_cast<unsigned int>(m * top_k));

  launch_gemv_worklist<true>(
      grid, block, device, topk,
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<const __nv_bfloat16*>(w.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<const int32_t*>(row_off.data_ptr()),
      static_cast<const uint16_t*>(kidx.data_ptr()),
      x_kx, 0, 0, n_cols, out_row, out_col_offset, top_k,
      static_cast<int>(x_row_is_pair));
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
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair) {
  gemv_worklist_indexed_impl(x, topk, w, row_off, kidx, out, out_col_offset,
                             x_row_is_pair, true);
}

void gemv_worklist_indexed_pinned(
    tvm::ffi::TensorView x, tvm::ffi::TensorView topk, tvm::ffi::TensorView w,
    tvm::ffi::TensorView row_off, tvm::ffi::TensorView kidx,
    tvm::ffi::TensorView out, int64_t out_col_offset, int64_t x_row_is_pair) {
  gemv_worklist_indexed_impl(x, topk, w, row_off, kidx, out, out_col_offset,
                             x_row_is_pair, false);
}

}  // namespace
