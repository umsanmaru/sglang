#pragma once

// Host-side helpers shared by the Prism kernel entry points.

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace prism_host {

// Cached per device: querying on every launch costs host time, also during graph capture.
inline uint32_t sm_count(int device_id) {
  constexpr int kMaxDev = 16;
  static uint32_t cache[kMaxDev] = {};
  if (device_id < 0 || device_id >= kMaxDev) return host::runtime::get_sm_count(device_id);
  if (cache[device_id] == 0) cache[device_id] = host::runtime::get_sm_count(device_id);
  return cache[device_id];
}

inline bool aligned16(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}


struct GemvIo {
  int64_t m, top_k, out_row, x_cols;
};

// Activation-side inputs shared by the Prism GEMVs: topk [M, K], x bf16 with M or M*K rows
// (x_row_is_pair) and out bf16 [M, K, out_row], all on one CUDA device. Binds M, K and W_row.
inline GemvIo verify_gemv_io(tvm::ffi::TensorView x, tvm::ffi::TensorView topk,
                             tvm::ffi::TensorView out, host::SymbolicSize& M,
                             host::SymbolicSize& K, host::SymbolicSize& W_row,
                             host::SymbolicDevice& cuda_device, int64_t x_row_is_pair,
                             const char* what) {
  using namespace host;
  auto Rx = SymbolicSize{"x_rows"};
  auto Kx = SymbolicSize{"x_cols"};
  TensorMatcher({M, K}).with_dtype<int32_t, int64_t>().with_device<kDLCUDA>(cuda_device).verify(topk);
  TensorMatcher({Rx, Kx}).with_dtype<bf16_t>().with_device(cuda_device).verify(x);
  TensorMatcher({M, K, W_row}).with_dtype<bf16_t>().with_device(cuda_device).verify(out);
  const int64_t m = M.unwrap(), top_k = K.unwrap(), x_rows = Rx.unwrap();
  RuntimeCheck(x_row_is_pair ? (x_rows == m * top_k) : (x_rows == m),
               what, ": x rows (", x_rows, ") must be ", x_row_is_pair ? "M*top_k" : "M");
  return {m, top_k, W_row.unwrap(), Kx.unwrap()};
}

}  // namespace prism_host

}  // namespace sglang
