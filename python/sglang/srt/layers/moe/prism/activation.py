"""MoE gate/up 활성화의 기술자 — rejoin#1이 fp32 합 뒤에 적용하는 함수.

prism은 활성화를 **한 곳**(rejoin.py `rejoin_gateup`)에서만 적용한다: 모든 티어(hot/warm GPU,
cold CPU kt, cold GPU)는 pre-activation partial을 내고, K-split 부분합을 fp32로 더한 뒤 여기
기술된 함수를 한 번 건다. 그래서 모델의 활성화가 바뀌어도 커널·kt·plan은 그대로고 이 객체와
rejoin 커널의 분기만 늘어난다.

지원:
  silu  : act = silu(g) · u.  `limit`(DSV4-Flash/GLM-5.3 swiglu_limit=10)이 있으면 그 전에
          u ∈ [−L, L], g ≤ L 로 hard clamp (참조 `Expert.forward` / glm5_next `swiglu_clamped`).
  situ  : Kimi K3 (`SituAndMul`): act = α·tanh(g/α)·σ(g) · u,  `limit`(linear_beta)이 있으면
          u ← L·tanh(u/L) soft clip.  K3: α=4.0, L=25.0.

MoeRunnerConfig에서의 유도(`from_runner_config`): 모델 파일이 FusedMoE에 넘긴
`activation` / `swiglu_limit` / `gemm1_alpha` / `gemm1_clamp_limit`. 모르는 조합은 즉사한다 —
조용히 silu로 계산하면 "돌아가는데 다른 모델"이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.moe.prism.plan import PlanError

# rejoin 커널의 ACT constexpr 코드
ACT_SILU = 0
ACT_SITU = 1


@dataclass(frozen=True)
class Activation:
    kind: str                       # "silu" | "situ"
    limit: Optional[float] = None   # silu: hard clamp L / situ: soft clip L (linear_beta)
    alpha: Optional[float] = None   # situ: β (tanh saturation). silu에서는 None

    def __post_init__(self):
        if self.kind not in ("silu", "situ"):
            raise PlanError(f"unsupported MoE activation {self.kind!r} (prism: silu, situ)")
        if self.kind == "situ" and (self.alpha is None or self.alpha <= 0):
            raise PlanError(f"situ needs alpha (β) > 0, got {self.alpha}")
        if self.kind == "silu" and self.alpha is not None:
            raise PlanError("silu takes no alpha (swigluoai-style alpha is not supported)")
        if self.limit is not None and self.limit <= 0:
            raise PlanError(f"activation limit must be > 0, got {self.limit}")

    # ── 생성 ────────────────────────────────────────────────────────────
    @staticmethod
    def silu(limit: Optional[float] = None) -> "Activation":
        return Activation("silu", None if limit is None else float(limit))

    @staticmethod
    def situ(alpha: float, limit: Optional[float]) -> "Activation":
        return Activation("situ", None if limit is None else float(limit), float(alpha))

    @staticmethod
    def from_runner_config(cfg) -> "Activation":
        """FusedMoE의 MoeRunnerConfig → Activation. 필드는 fused_moe_triton/layer.py가
        모델 인자를 그대로 옮긴 것 (activation, swiglu_limit, gemm1_alpha, gemm1_clamp_limit)."""
        kind = getattr(cfg, "activation", "silu") or "silu"
        alpha = getattr(cfg, "gemm1_alpha", None)
        clamp = getattr(cfg, "gemm1_clamp_limit", None)
        swiglu_limit = getattr(cfg, "swiglu_limit", None)
        if kind == "silu":
            if alpha is not None or clamp is not None:
                raise PlanError(
                    f"silu with gemm1_alpha={alpha}/gemm1_clamp_limit={clamp} is not supported by prism")
            return Activation.silu(swiglu_limit)
        if kind == "situ":
            if swiglu_limit is not None:
                raise PlanError("situ does not take swiglu_limit (use gemm1_clamp_limit)")
            # SituAndMul(beta=1.0 기본) — 모델이 alpha를 안 주면 1.0
            return Activation.situ(1.0 if alpha is None else alpha, clamp)
        raise PlanError(f"unsupported MoE activation {kind!r} (prism: silu, situ)")

    # ── 커널 인자 ────────────────────────────────────────────────────────
    @property
    def act_code(self) -> int:
        return ACT_SILU if self.kind == "silu" else ACT_SITU

    @property
    def has_limit(self) -> bool:
        return self.limit is not None

    def kernel_scalars(self) -> tuple[float, float]:
        """(limit, alpha) — 없는 쪽은 0.0 (constexpr가 읽지 않는다)."""
        return (float(self.limit or 0.0), float(self.alpha or 0.0))

    # ── 참조 ────────────────────────────────────────────────────────────
    def reference(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """torch 참조 (fp32 계산). 테스트·프로파일러가 rejoin 커널과 대조한다.
        sglang의 SiluAndMul / SituAndMul.forward_native 및 참조 Expert.forward와 같은 식."""
        g = gate.float()
        u = up.float()
        if self.kind == "silu":
            if self.limit is not None:
                u = u.clamp(-self.limit, self.limit)
                g = g.clamp(max=self.limit)
            return torch.nn.functional.silu(g) * u
        g = self.alpha * torch.tanh(g / self.alpha) * torch.sigmoid(g)
        if self.limit is not None:
            u = self.limit * torch.tanh(u / self.limit)
        return g * u

    def reference_gateup(self, gate_up: torch.Tensor) -> torch.Tensor:
        """[..., 2I] (gate 앞, up 뒤) → [..., I] fp32."""
        d = gate_up.shape[-1] // 2
        return self.reference(gate_up[..., :d], gate_up[..., d:])

    def __str__(self) -> str:
        if self.kind == "silu":
            return "silu" if self.limit is None else f"silu(limit={self.limit})"
        return f"situ(alpha={self.alpha}, limit={self.limit})"
