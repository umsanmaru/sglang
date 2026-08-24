"""입력기반 sparsity의 GPU-측 몫 — threshold 격자 조회.

계약 ① (CONTRACTS.md): 마스크는 각 proj의 K(contraction) 축에 걸리고 점수는
그 proj의 입력에서만 나온다. **마스크 계산과 적용은 cold(kt)가 전담한다** —
warm 티어는 dense로 계산한다 (2026-08-24 결정).

이 모듈이 남기는 것은 예산 → threshold 변환 하나다:

    s   = clip(p − lam·(g_e − ḡ), 0, pmax)      # renorm_it회 재정규화
    thr = table[layer, expert, round(s / grid)]

이 값이 pinned staging을 거쳐 kt로 내려가고, kt가 자기 cold 밴드에서
`imp >= thr`을 판정한다 (kt `build_pair_mask`). 예산·격자 조회를 한 곳에 둔
이유는 grid/ng가 s의 정의와 어긋나면 threshold가 조용히 포화하기 때문이다.

warm dense 결정의 근거 (실측):
- warm GEMM은 latency·occupancy 바운드다 (35B-A3B warm gate: g=8, M=1,
  k_rows=192, N=512 → 출력 4096 원소). 작업량을 7.5배 줄여도 시간은 24%만
  줄었고, 어떤 마스킹 경로든 dense bmm(3.23 µs)보다 1.6~2.0배 느렸다.
- warm 밴드는 이 모델에서 gate/up K의 9.4%뿐이고 down은 warm 밴드가 없다
  (I=512에 warm-frac 0.1이면 ROW_GROUP 아래로 떨어진다).

전 연산이 device-side다 (host 동기화·.item() 없음) — graph 캡처에 안전하다.
"""

from __future__ import annotations

from typing import Mapping

import torch

from sglang.srt.layers.moe.prism.plan import Plan, PlanError, Proj
from sglang.srt.layers.moe.prism.weights import PreparedWeights


class LayerSparsity:
    """한 레이어의 device 상주 threshold 곡선 + 예산. 상태 없음."""

    def __init__(
        self,
        thr: Mapping[Proj, torch.Tensor],
        budget: Mapping[Proj, tuple],
        *,
        pmax: float,
        grid: float,
        ng: int,
        renorm_it: int,
    ):
        self._thr = dict(thr)
        self._budget = dict(budget)
        self.pmax, self.grid, self.ng, self.renorm_it = pmax, grid, ng, renorm_it

    @classmethod
    def from_prepared(
        cls, plan: Plan, layer_idx: int, prepared: PreparedWeights,
        device: torch.device,
    ) -> "LayerSparsity":
        """Stage 2 산출물의 threshold 곡선을 device로 올린다 (레이어당 1회).

        점수 재료(wn²/pair_dot)는 올리지 않는다 — 그것을 쓰는 쪽은 kt이고,
        kt는 host 상주 테이블을 NUMA-local로 읽는다.
        """
        spec = plan.sparsity
        if spec is None or prepared.thr is None:
            raise PlanError(f"layer {layer_idx}: prepared weights carry no calib")
        ep = plan.expert(layer_idx, 0)
        thr, budget = {}, {}
        for proj in Proj:
            pp = ep.proj(proj)
            if pp.sparsity_p is None or pp.sparsity_lambda is None:
                raise PlanError(f"layer {layer_idx} {proj.value}: no (p, lambda)")
            budget[proj] = (float(pp.sparsity_p), float(pp.sparsity_lambda))
            thr[proj] = prepared.thr[proj].to(device=device, dtype=torch.float32)
        return cls(
            thr, budget, pmax=spec.pmax, grid=spec.grid, ng=spec.ng,
            renorm_it=spec.renorm_it,
        )

    def slot_thr(
        self, proj: Proj, topk_ids: torch.Tensor, twn: torch.Tensor
    ) -> torch.Tensor:
        """[M, k] fp32 threshold. twn은 **정규화된** 라우터 가중이다.

        s_mat과 격자 조회를 한 몸으로 둔 이유: 둘 사이에 다른 코드가 끼면
        grid/ng가 s의 정의와 어긋날 수 있고, 그것은 조용한 포화가 된다.
        """
        p, lam = self._budget[proj]
        gbar = twn.mean(1, keepdim=True)
        s = (p - lam * (twn - gbar)).clamp(0.0, self.pmax)
        for _ in range(self.renorm_it):
            s = (s * (p / s.mean(1, keepdim=True).clamp_min(1e-6))).clamp(
                0.0, self.pmax
            )
        idx = (s / self.grid).round().long().clamp_(0, self.ng - 1)
        return self._thr[proj][topk_ids, idx]


def normalized_router_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    """[M, k] fp32, 토큰별 합 1. 이미 정규화돼 있어도 무해(멱등)하다.

    sglang topk_config.renormalize의 기본값은 True지만 모델 config에 딸린
    값이라, calib(twn = tw / tw.sum)과 어긋날 여지를 남기지 않기 위해 여기서
    한 번 더 나눈다 — λ와 ḡ의 스케일이 곧 마스크 세기다.
    """
    w = topk_weights.float()
    return w / w.sum(-1, keepdim=True).clamp_min(1e-9)
