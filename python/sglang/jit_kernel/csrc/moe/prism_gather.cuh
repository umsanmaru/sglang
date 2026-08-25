#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// Prism warm-band gather: dst[g] = pinned_src[sel[g]].
//
// `sel` lives on the GPU (int32, device tensor) so a CUDA-graph capture of
// this launch needs no per-token host repatch -- the graph replays whatever
// indices `sel` holds at replay time. `src` points at pinned host memory
// (torch `.pin_memory()`), read directly by the GPU over UVA (this kernel's
// test is the regression guard for that platform assumption). Ported from
// planir's `WarmBandGatherPinnedKernel` (kernels.cu:114); the only
// intentional deviation is that `sel` is an int32 device tensor here (planir
// used the same convention already, so semantics are preserved exactly).
//
// grid = (ceil_div(stride16, blockDim.x), g), block = 256.
// stride16 = (rows * N * dtype_bytes) / 16, i.e. number of 16-byte (uint4)
// words per band row-major [rows, N] slab.
__global__ void prism_gather_bands(
    const uint4* __restrict__ src,
    const int32_t* __restrict__ sel,
    uint4* __restrict__ dst,
    long long stride16) {
  const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= stride16) return;
  const long long g = blockIdx.y;
  dst[g * stride16 + t] = src[static_cast<long long>(sel[g]) * stride16 + t];
}

constexpr uint32_t kBlockSize = 256;

// gather_bands_from_pinned(pinned_src[E, rows, N] bf16 (pinned CPU),
//                           sel[g] int32 (CUDA), dst[g, rows, N] bf16 (CUDA))
//
// Stream contract: this launcher does NOT accept an explicit stream argument.
// Following this JIT module's established convention (see
// `cuda_wait_value.cuh` / `hicache.cuh`), the kernel launches on whatever
// stream is "current" for `dst`'s CUDA device at call time, resolved via
// `LaunchKernel::resolve_device` -> `TVMFFIEnvGetStream`. The caller MUST
// select the target stream on the Python side (e.g. `with
// torch.cuda.stream(stream):`) before invoking this function.
void gather_bands_from_pinned(
    tvm::ffi::TensorView pinned_src, tvm::ffi::TensorView sel, tvm::ffi::TensorView dst) {
  using namespace host;

  auto E = SymbolicSize{"num_experts"};
  auto R = SymbolicSize{"rows"};
  auto N = SymbolicSize{"cols"};
  auto G = SymbolicSize{"num_gathered"};
  auto cuda_device = SymbolicDevice{};

  // pinned_src is host memory (torch pin_memory()); it may be reported as
  // plain CPU or as the CUDA-host device type depending on the DLPack
  // producer, so accept either.
  TensorMatcher({E, R, N})  //
      .with_dtype<bf16_t>()
      .with_device<kDLCPU, kDLCUDAHost>()
      .verify(pinned_src);

  TensorMatcher({G})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(cuda_device)
      .verify(sel);

  TensorMatcher({G, R, N})  //
      .with_dtype<bf16_t>()
      .with_device(cuda_device)
      .verify(dst);

  const int64_t rows = R.unwrap();
  const int64_t cols = N.unwrap();
  const int64_t g = G.unwrap();
  const int64_t band_bytes = rows * cols * static_cast<int64_t>(sizeof(bf16_t));

  RuntimeCheck(
      band_bytes % 16 == 0,
      "prism_gather_bands: rows*N*dtype_size (",
      band_bytes,
      " bytes) must be a multiple of 16 bytes for uint4-vectorized gather");

  RuntimeCheck(g > 0, "prism_gather_bands: sel/dst must be non-empty (g > 0)");

  const int64_t stride16 = band_bytes / 16;
  const DLDevice device = cuda_device.unwrap();

  const dim3 block(kBlockSize);
  const dim3 grid(static_cast<unsigned int>(div_ceil(stride16, static_cast<int64_t>(kBlockSize))),
                   static_cast<unsigned int>(g));

  LaunchKernel(grid, block, device)(
      prism_gather_bands,
      static_cast<const uint4*>(pinned_src.data_ptr()),
      static_cast<const int32_t*>(sel.data_ptr()),
      static_cast<uint4*>(dst.data_ptr()),
      stride16);
}

// gather_bands_from_device(device_src[E, rows, N] bf16 (CUDA),
//                          sel[g] int32 (CUDA), dst[g, rows, N] bf16 (CUDA))
//
// Twin of gather_bands_from_pinned with a VRAM-resident source: same kernel,
// same launch geometry, same stream contract (launches on the caller's
// current stream) -- only the source-device constraint differs. src, sel and
// dst must live on the same CUDA device. Used by the prism HOT tier, whose
// store is device-resident: the generic torch index_select kernel it replaces
// is latency-bound (2-byte scalar accesses, sequential index loop), while
// this one moves 16 bytes per thread with all g slabs in flight at once.
void gather_bands_from_device(
    tvm::ffi::TensorView device_src, tvm::ffi::TensorView sel, tvm::ffi::TensorView dst) {
  using namespace host;

  auto E = SymbolicSize{"num_experts"};
  auto R = SymbolicSize{"rows"};
  auto N = SymbolicSize{"cols"};
  auto G = SymbolicSize{"num_gathered"};
  auto cuda_device = SymbolicDevice{};

  TensorMatcher({G})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(cuda_device)
      .verify(sel);

  TensorMatcher({E, R, N})  //
      .with_dtype<bf16_t>()
      .with_device(cuda_device)
      .verify(device_src);

  TensorMatcher({G, R, N})  //
      .with_dtype<bf16_t>()
      .with_device(cuda_device)
      .verify(dst);

  const int64_t rows = R.unwrap();
  const int64_t cols = N.unwrap();
  const int64_t g = G.unwrap();
  const int64_t band_bytes = rows * cols * static_cast<int64_t>(sizeof(bf16_t));

  RuntimeCheck(
      band_bytes % 16 == 0,
      "gather_bands_from_device: rows*N*dtype_size (",
      band_bytes,
      " bytes) must be a multiple of 16 bytes for uint4-vectorized gather");

  RuntimeCheck(g > 0, "gather_bands_from_device: sel/dst must be non-empty (g > 0)");

  const int64_t stride16 = band_bytes / 16;
  const DLDevice device = cuda_device.unwrap();

  const dim3 block(kBlockSize);
  const dim3 grid(static_cast<unsigned int>(div_ceil(stride16, static_cast<int64_t>(kBlockSize))),
                   static_cast<unsigned int>(g));

  LaunchKernel(grid, block, device)(
      prism_gather_bands,
      static_cast<const uint4*>(device_src.data_ptr()),
      static_cast<const int32_t*>(sel.data_ptr()),
      static_cast<uint4*>(dst.data_ptr()),
      stride16);
}

}  // namespace
