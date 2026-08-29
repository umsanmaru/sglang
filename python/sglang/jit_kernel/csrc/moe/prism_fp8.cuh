#pragma once

#include <cuda_bf16.h>

#include <cstdint>

// FP8 e4m3 (128×128 블록 fp32 배율) 디코드 — GEMV·grouped 두 커널의 공용 device 헬퍼.
//
// 체크포인트 규약 (DeepSeek blockwise fp8, kt `fp8-moe.hpp`와 같은 것): 원소 하나 = e4m3
// 바이트 (bit7 부호, bit6:3 지수(bias 7), bit2:0 가수), 배율은 fp32 `scale_inv` 하나가
// 원본 128 n × 128 k 블록을 덮고 `w = q · scale_inv`다.
//
// Prism 스토어: codes u8 [Σₑ k[e], N] — 행 = k, 열 n 연속; scales fp32 [Σₑ k[e]/128, N/128]
// — 행 = 128-k 블록, 열 = 128-n 블록. bf16 스토어 `[Σₖ, N]`의 fp8 판이라 row_off(k 단위)/
// k_index/페어 마스크 기계가 그대로 맞는다 (k[e]·row_off[e]는 128 배수 — 계약 ①의 정렬을
// 커널 키가 함의한다).
//
// 수치(W8A16): 32-k 청크의 부분합 Σ x_k·v_k 를 fp32로 모은 뒤 × 블록 배율. e4m3 → 값 변환은
// **정확**하다 (가수 3비트가 bf16 격자 안) — 참조(dequant fp32 GEMV)와의 차이는 fp32 누산
// 순서뿐이다. cpu-mm `ktf8_fp8_32`(AVX-512)와 같은 비트 산술이라 CPU cold 커널과도 같은 값을
// 곱한다.
//
// **denormal/NaN 규약**: 지수·가수가 모두 0인 코드만 0이다. e == 0, m != 0(denormal)과
// 0x7F/0xFF(NaN)는 이 산술이 유한한 다른 값으로 디코드한다 — kt/cpu-mm 인코더가 그런 코드를
// 만들지 않는다는 전제이고, CPU 쪽과 **같은 전제·같은 결과**다 (양쪽이 같이 틀리는 것이
// 조용히 갈리는 것보다 낫다).

namespace prism_fp8 {

// e4m3 바이트 → bf16 비트 패턴. 가수 3비트를 bf16 가수 상위로 올리고 지수를 bias 7 → 127로
// 다시 태운다 (+120 << 7 = 0x3C00), 부호는 bit 15.
__device__ __forceinline__ uint32_t e4m3_bf16_bits(uint32_t b) {
  const uint32_t mag = (b & 0x7Fu) << 4;
  return ((b & 0x80u) << 8) | (mag ? (mag + 0x3C00u) : 0u);
}

// e4m3 바이트 → float (정확). bf16 패턴을 fp32 상위 16비트에 놓는 것이 곧 변환이다.
__device__ __forceinline__ float e4m3_val(uint32_t b) {
  return __uint_as_float(e4m3_bf16_bits(b) << 16);
}

// e4m3 바이트 → bf16 (grouped 커널의 B 타일이 tensor core에 넣는 형태).
__device__ __forceinline__ __nv_bfloat16 e4m3_to_bf16(uint32_t b) {
  const uint16_t bits = static_cast<uint16_t>(e4m3_bf16_bits(b));
  __nv_bfloat16 out;
  *reinterpret_cast<uint16_t*>(&out) = bits;
  return out;
}

constexpr int kBlk = 128;  // 배율 블록 (n·k 양 축)

}  // namespace prism_fp8
