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
// 스펙 편차: W 로드(`we[r * n_cols + n]`)는 스칼라 bf16(스레드당 2B)이고
// 정렬 RuntimeCheck도 없다 — 스펙(2026-08-25-prism-gemv-worklist-design.md
// §2)이 요구한 uint4/uint2 벡터화(및 수반 정렬 체크)를 이 1차 구현은 안 한다.
// 디바이스 경로 실측 10.7us vs gather+bmm 17.5us로 이미 충분해 벡터화를
// 미뤘다; warm/UVA 대역폭이 임계로 확인되면 벡터 로드 + 정렬 RuntimeCheck를
// 후속으로 추가한다.
template <typename IdxT>
__global__ void prism_gemv_worklist(
    const __nv_bfloat16* __restrict__ x,
    const IdxT* __restrict__ topk,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    long long x_kx,          // x row 길이 (Kx)
    long long k_offset,
    long long k_rows,
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

  const __nv_bfloat16* xr = x + row * x_kx + k_offset;
  const __nv_bfloat16* we = w + e * k_rows * n_cols;

  float acc = 0.f;
  if (n < n_cols) {
    for (long long r = threadIdx.y; r < k_rows; r += 4) {
      acc += __bfloat162float(xr[r]) * __bfloat162float(we[r * n_cols + n]);
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

  if (is_type<int32_t>(topk.dtype())) {
    LaunchKernel(grid, block, device)(
        prism_gemv_worklist<int32_t>,
        static_cast<const __nv_bfloat16*>(x.data_ptr()),
        static_cast<const int32_t*>(topk.data_ptr()),
        static_cast<const __nv_bfloat16*>(w.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()),
        x_kx, k_offset, k_rows, n_cols, out_row, out_col_offset, top_k,
        static_cast<int>(x_row_is_pair));
  } else {
    LaunchKernel(grid, block, device)(
        prism_gemv_worklist<int64_t>,
        static_cast<const __nv_bfloat16*>(x.data_ptr()),
        static_cast<const int64_t*>(topk.data_ptr()),
        static_cast<const __nv_bfloat16*>(w.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()),
        x_kx, k_offset, k_rows, n_cols, out_row, out_col_offset, top_k,
        static_cast<int>(x_row_is_pair));
  }
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

}  // namespace
