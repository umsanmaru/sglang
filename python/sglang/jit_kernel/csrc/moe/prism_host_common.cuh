#pragma once

// Host-side helpers shared by the Prism kernel entry points.

#include <sgl_kernel/runtime.cuh>

#include <cstdint>

namespace prism_host {

// Cached per device: querying on every launch costs host time, also during graph capture.
inline uint32_t sm_count(int device_id) {
  constexpr int kMaxDev = 16;
  static uint32_t cache[kMaxDev] = {};
  if (device_id < 0 || device_id >= kMaxDev) return host::runtime::get_sm_count(device_id);
  if (cache[device_id] == 0) cache[device_id] = host::runtime::get_sm_count(device_id);
  return cache[device_id];
}

}  // namespace prism_host
