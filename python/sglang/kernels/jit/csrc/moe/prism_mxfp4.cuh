#pragma once

#include <cuda_bf16.h>

#include <cstdint>

// MXFP4 (e2m1 코드 + E8M0 32-블록 배율) 디코드 — GEMV·grouped 두 커널의 공용 device 헬퍼.
//
// 체크포인트 규약 (DeepSeek-V4-Flash, `inference/convert.py`): 바이트 하나 = k 짝수(하위
// nibble) + k 홀수(상위 nibble), nibble = s·e2m1 (bit3 부호, bit2:1 지수, bit0 가수).
// 값 = {0, .5, 1, 1.5, 2, 3, 4, 6} × ±1. 배율 E8M0 바이트 e → 2^(e−127).
//
// Prism 스토어 (pair-row): codes u8 [Σₑ k[e]/2, N] — 행 p = k-페어, 열 n 연속;
// scales u8 [Σₑ k[e]/32, N] — 행 g = 32-k 블록. bf16 스토어 `[Σₖ, N]`의 fp4 판이라
// row_off(k 단위)/k_index/페어 마스크 기계가 그대로 맞는다 (k[e]·row_off[e]는 32 배수).
//
// 수치(W4A16): 블록 부분합 Σ_{k∈g} (2·v_k)·x_k 를 fp32로 모은 뒤 × 2^(e−128) (= 배율/2).
// 2·v_k ∈ {0,±1,…,±12}는 정수라 bf16 x와의 곱이 fp32에서 정확하고, 배율은 2의 거듭제곱이라
// 곱이 정확하다 — 참조(dequant fp32 GEMV)와의 차이는 fp32 누산 순서뿐이다.

namespace prism_mxfp4 {

// e2m1 가수·지수 3비트 → |값|×2 (정수 0..12). 8개 nibble 테이블을 uint32 하나에 담아
// 레지스터 shift로 조회한다 (constant/local 메모리 없음).
//   idx 0..7 → 0, 1, 2, 3, 4, 6, 8, 12  (=0x0,1,2,3,4,6,8,C)
__device__ __forceinline__ int fp4_mag2(uint32_t nib) {
  return static_cast<int>((0xC8643210u >> ((nib & 7u) * 4u)) & 0xFu);
}

// nibble → 값×2 (float, 정확).
__device__ __forceinline__ float fp4_val2(uint32_t nib) {
  const float m = static_cast<float>(fp4_mag2(nib));
  return (nib & 8u) ? -m : m;
}

// E8M0 바이트 → 2^(e−127) / 2 = 2^(e−128). e=0이면 fp32 denormal(2^−128)이라 무해.
__device__ __forceinline__ float e8m0_half(uint32_t e) {
  return __uint_as_float(e << 23) * 0.5f;
}

// 코드 nibble + 배율 → bf16 (정확표현: e2m1 × 2^e는 bf16 격자에 있다).
__device__ __forceinline__ __nv_bfloat16 fp4_to_bf16(uint32_t nib, float scale_half) {
  return __float2bfloat16_rn(fp4_val2(nib) * scale_half);
}

}  // namespace prism_mxfp4
