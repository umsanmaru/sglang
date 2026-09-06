#pragma once

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

// k2wl2 sparsity의 커널 측 공용 부품 — bf16 worklist(prism_gemv.cuh)와 mxfp4
// worklist(prism_gemv_mxfp4.cuh)가 **같은 임계값·같은 페어 에너지 식**을 쓴다는 것을
// 코드로 강제한다. 식과 반올림은 kt의 slot_sparsity/thr_of/build_pair_mask를 그대로
// 옮긴 것이다 (rintf ↔ lrint 모두 round-half-to-even). 자세한 설계 근거는
// prism_gemv.cuh 상단 주석.

namespace prism_sparse {

struct SparseArgs {
  const float* a;        // [Σₑ k[e]]      = wn² (weight와 같은 오프셋)
  const float* c;        // [Σₑ k[e]/2]    = 인접열 내적
  const float* thr_tab;  // [E, ng]        = sparsity → threshold 곡선
  const float* topk_w;   // [M, top_k]     = 라우터 가중
  float p, lam, pmax, grid;
  int ng, renorm_it;
};

// host 진입점이 받는 sparse 인자 묶음 (검증 전).
struct SparseIn {
  tvm::ffi::TensorView a, c, thr, topk_w;
  double p, lam, pmax, grid;
  int64_t ng, renorm_it;
  // score k1 (per-k, 2026-09-05): 마스크가 |x_k| >= thr 하나라 a/c를 쓰지 않는다 — 호출자는
  // a/c 자리에 아무 텐서(관례상 thr)를 넣고 per_k=true를 준다. fill 쪽이 a/c 검증을 건너뛴다.
  bool per_k = false;
};

// 이 (m, j)의 임계값(비제곱). 전 스레드가 중복 계산한다 — top_k(≤16)짜리 루프라
// syncthreads 한 번보다 싸고, 공유 상태가 없어 결정적이다. host RuntimeCheck가
// top_k ≤ 16을 보증한다. k2wl2는 thr²와 페어 에너지를, k1은 thr와 |x_k|를 비교한다
// (kt build_k_mask와 같은 비교 — bf16→fp32는 정확하므로 양쪽 마스크가 비트 동일).
__device__ __forceinline__ float sparse_thr(const SparseArgs& sp, long long m,
                                            long long pair, long long e,
                                            long long top_k) {
  constexpr int MAXK = 16;
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
  return sp.thr_tab[e * static_cast<long long>(sp.ng) + gi];
}

// k2wl2용 임계값 제곱 (kt도 제곱 비교다 — sqrt를 양쪽 다 생략).
__device__ __forceinline__ float sparse_thr2(const SparseArgs& sp, long long m,
                                             long long pair, long long e,
                                             long long top_k) {
  const float thr = sparse_thr(sp, m, pair, e, top_k);
  return thr * thr;
}

// score k1: keep_k = |x_k| >= thr (kt k_mask.hpp와 같은 판정).
__device__ __forceinline__ bool keep_k(float x, float thr) { return fabsf(x) >= thr; }

// 페어 (절대 행 ar = 2·pair_id)의 에너지: a[ar]·x0² + a[ar+1]·x1² + 2·c[ar/2]·x0·x1, 음수는 0.
__device__ __forceinline__ float pair_energy(const SparseArgs& sp, long long ar,
                                             float x0, float x1) {
  float en = sp.a[ar] * x0 * x0 + sp.a[ar + 1] * x1 * x1 +
             2.0f * sp.c[ar >> 1] * x0 * x1;
  return en < 0.f ? 0.f : en;
}

}  // namespace prism_sparse
