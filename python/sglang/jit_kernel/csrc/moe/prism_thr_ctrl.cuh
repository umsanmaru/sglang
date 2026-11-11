#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// Online threshold controller step for one layer (driver: srt/layers/moe/prism/thr_ctrl.py).
// Launched once per masked decode step after the layer's rejoin, so the GPU warm kernels and the
// kt cold job of this step have finished adding to their counters. Every `window` steps it turns
// the window's mask counts into a realized full-K sparsity d per proj, updates
// alpha <- clamp(alpha * (1 - mu * (d - target))), and rewrites both live threshold tables as
// calib * alpha. Single block: the token counter needs no cross-block agreement, and the table
// rewrite only happens once per window.
constexpr int kProj = 3;  // gate, up, down; the order shared with thr_ctrl.py and kt keep_ctr
constexpr int kThreads = 256;

struct CtrlParams {
  long long window;
  long long table_elems;  // E * ng of one proj's table
  long long log_windows;  // ring length of the per-window log
  float mu, alpha_min, alpha_max;
  float target[kProj];
  double units_per_token[kProj];  // full-K mask units per decode token: top_k * K (/2 for pairs)
};

__global__ void prism_thr_ctrl_step(
    float* __restrict__ live_dev, const float* __restrict__ calib_dev, float* live_host,
    const long long* __restrict__ gpu_ctr, const long long* cpu_ctr, long long* __restrict__ snap,
    long long* __restrict__ tokens, float* __restrict__ alpha, float* log, long long* windows_host,
    CtrlParams prm) {
  __shared__ long long s_t;
  __shared__ float s_alpha[kProj];
  if (threadIdx.x == 0) s_t = ++tokens[0];
  __syncthreads();
  const long long t = s_t;
  if (t % prm.window != 0) return;

  if (threadIdx.x < kProj) {
    const int p = threadIdx.x;
    // cpu_ctr is host memory written by the kt job between launches; __ldcv skips stale cache lines.
    const long long kept = gpu_ctr[2 * p] + __ldcv(cpu_ctr + 2 * p);
    const long long total = gpu_ctr[2 * p + 1] + __ldcv(cpu_ctr + 2 * p + 1);
    const long long d_kept = kept - snap[2 * p];
    const long long d_total = total - snap[2 * p + 1];
    snap[2 * p] = kept;
    snap[2 * p + 1] = total;
    float a = alpha[p];
    float d = __int_as_float(0x7fc00000);  // NaN: no masked rows in this window, alpha unchanged
    const double units = static_cast<double>(prm.window) * prm.units_per_token[p];
    if (d_total > 0 && units > 0.0) {
      d = static_cast<float>(static_cast<double>(d_total - d_kept) / units);
      a = fminf(fmaxf(a * (1.0f - prm.mu * (d - prm.target[p])), prm.alpha_min), prm.alpha_max);
      alpha[p] = a;
    }
    s_alpha[p] = a;
    const long long slot = (t / prm.window - 1) % prm.log_windows;
    log[(slot * kProj + p) * 2] = a;
    log[(slot * kProj + p) * 2 + 1] = d;
  }
  __syncthreads();
  // Published after the log entries, so a host reader never sees a count ahead of its records.
  if (threadIdx.x == 0) {
    __threadfence_system();
    *reinterpret_cast<volatile long long*>(windows_host) = t / prm.window;
  }

  const long long n = kProj * prm.table_elems;
  for (long long i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = calib_dev[i] * s_alpha[i / prm.table_elems];
    live_dev[i] = v;
    live_host[i] = v;
  }
  // The next reader of live_host is the CPU (the kt job of the next step).
  __threadfence_system();
}

void thr_ctrl_step(
    tvm::ffi::TensorView live_dev, tvm::ffi::TensorView calib_dev, tvm::ffi::TensorView live_host,
    tvm::ffi::TensorView gpu_ctr, tvm::ffi::TensorView cpu_ctr, tvm::ffi::TensorView snap,
    tvm::ffi::TensorView tokens, tvm::ffi::TensorView alpha, tvm::ffi::TensorView log,
    tvm::ffi::TensorView windows_host, int64_t window, double mu, double alpha_min, double alpha_max,
    double target_gate, double target_up, double target_down,
    double units_gate, double units_up, double units_down) {
  using namespace host;

  auto E = SymbolicSize{"num_experts"};
  auto Ng = SymbolicSize{"ng"};
  auto R = SymbolicSize{"log_windows"};
  auto cuda_device = SymbolicDevice{};
  TensorMatcher({kProj, E, Ng}).with_dtype<float>().with_device<kDLCUDA>(cuda_device).verify(live_dev);
  TensorMatcher({kProj, E, Ng}).with_dtype<float>().with_device(cuda_device).verify(calib_dev);
  TensorMatcher({kProj, E, Ng}).with_dtype<float>().with_device<kDLCPU, kDLCUDAHost>().verify(live_host);
  TensorMatcher({kProj, 2}).with_dtype<int64_t>().with_device(cuda_device).verify(gpu_ctr);
  TensorMatcher({kProj, 2}).with_dtype<int64_t>().with_device<kDLCPU, kDLCUDAHost>().verify(cpu_ctr);
  TensorMatcher({kProj, 2}).with_dtype<int64_t>().with_device(cuda_device).verify(snap);
  TensorMatcher({1}).with_dtype<int64_t>().with_device(cuda_device).verify(tokens);
  TensorMatcher({kProj}).with_dtype<float>().with_device(cuda_device).verify(alpha);
  TensorMatcher({R, kProj, 2}).with_dtype<float>().with_device<kDLCPU, kDLCUDAHost>().verify(log);
  TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCPU, kDLCUDAHost>().verify(windows_host);
  RuntimeCheck(window >= 1, "thr_ctrl_step: window must be >= 1, got ", window);
  RuntimeCheck(alpha_min > 0.0 && alpha_min <= alpha_max,
               "thr_ctrl_step: bad alpha range [", alpha_min, ", ", alpha_max, "]");

  CtrlParams prm{};
  prm.window = window;
  prm.table_elems = E.unwrap() * Ng.unwrap();
  prm.log_windows = R.unwrap();
  prm.mu = static_cast<float>(mu);
  prm.alpha_min = static_cast<float>(alpha_min);
  prm.alpha_max = static_cast<float>(alpha_max);
  prm.target[0] = static_cast<float>(target_gate);
  prm.target[1] = static_cast<float>(target_up);
  prm.target[2] = static_cast<float>(target_down);
  prm.units_per_token[0] = units_gate;
  prm.units_per_token[1] = units_up;
  prm.units_per_token[2] = units_down;

  LaunchKernel(dim3(1), dim3(kThreads), cuda_device.unwrap())(
      prism_thr_ctrl_step, static_cast<float*>(live_dev.data_ptr()),
      static_cast<const float*>(calib_dev.data_ptr()), static_cast<float*>(live_host.data_ptr()),
      static_cast<const long long*>(gpu_ctr.data_ptr()), static_cast<const long long*>(cpu_ctr.data_ptr()),
      static_cast<long long*>(snap.data_ptr()), static_cast<long long*>(tokens.data_ptr()),
      static_cast<float*>(alpha.data_ptr()), static_cast<float*>(log.data_ptr()),
      static_cast<long long*>(windows_host.data_ptr()), prm);
}

}  // namespace
