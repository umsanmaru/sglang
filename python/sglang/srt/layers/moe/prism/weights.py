"""Stage 2: weight 절단·변환·배치. 산출물은 PreparedWeights (계약 ③).

소유권 (계약 ③):
- hot  → Python 소유 device 텐서 (P0: None)
- warm → Python 소유 pinned 텐서 + (proj) → k_offset
- cold → C++ MOE 객체 핸들. 단, cold backend(K3/S5)가 붙기 전까지는
  PendingColdTensors가 슬라이스본을 임시 소유한다 — backend 접속 시
  텐서를 넘기고 핸들로 대체되는 것이 P0의 로딩 흐름이다.

Stage 2 종료 후 full-K 텐서는 어디에도 존재하지 않는다: 이 모듈은 입력
w13/w2에 대한 참조를 보관하지 않으며, 호출자는 반환 즉시 원본을 놓는다.

레이아웃:
- ckpt 방향: w13 [E, 2·inter, hidden] (gate가 앞 절반, up이 뒤 절반 —
  fused_moe_triton/layer.py:431-432가 명시하는 sglang w13 관례),
  w2 [E, hidden, inter].
- 주의: quant method가 `load_up_proj_weight_first=True`(layer.py:434,
  trtllm cutlass 계열)를 세팅하면 w13 내부 순서가 뒤집힌다. 이 로더는
  기본 순서를 가정하므로, method 통합(S7)에서 해당 플래그가 False임을
  assert해야 한다 — 위반 시 gate/up이 조용히 뒤바뀐다.
- warm store는 [E, k_rows, N] (K-major) — transpose 표기 없이 GEMM에
  들어가는 no-transpose 정준 방향을 로드 시점에 한 번 고정한 것
  (kernels.py의 warm GEMM 계약이 이 방향 하나만 가정; N-major도 GEMM
  자체는 가능하지만 커널 계약 단일화 + P1 persistent GEMV의 스트리밍
  패턴 + cuBLAS T-경로 회피를 위해 통일). cold 슬라이스는 ckpt 방향
  [E, N, k_cold] 유지 — 소비자인 kt-kernel pack이 기대하는 방향이다.

P0 loader capability gap (스키마 제약이 아님, NotImplementedError로 명시):
- HOT 밴드 미지원
- 티어당 다중 밴드 미지원
- 레이어 내 expert 간 기하 불일치 미지원 (store가 [E, ...] 균일 적층이므로)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.moe.prism.plan import (
    ExpertProjPlan,
    Plan,
    PlanError,
    Proj,
    Tier,
)


@dataclass
class WarmBand:
    """한 proj의 warm 밴드: pinned bf16 [E, k_rows, N] (K-major, no-transpose 정준 방향)."""

    k_offset: int
    weights: torch.Tensor

    @property
    def k_rows(self) -> int:
        return self.weights.shape[1]


@dataclass
class WarmStore:
    gate: Optional[WarmBand]
    up: Optional[WarmBand]
    down: Optional[WarmBand]

    def band(self, proj: Proj) -> Optional[WarmBand]:
        return {Proj.GATE: self.gate, Proj.UP: self.up, Proj.DOWN: self.down}[proj]


@dataclass
class ColdBand:
    """한 proj의 cold 밴드: CPU bf16 [E, N, k_rows] (ckpt 방향)."""

    k_offset: int
    weights: torch.Tensor

    @property
    def k_rows(self) -> int:
        return self.weights.shape[2]


@dataclass
class PendingColdTensors:
    """cold backend 접속 전까지의 임시 소유자. backend가 붙으면 이 텐서들은
    C++로 pack되어 넘어가고 PreparedWeights.cold는 핸들로 대체된다."""

    gate: Optional[ColdBand]
    up: Optional[ColdBand]
    down: Optional[ColdBand]

    def band(self, proj: Proj) -> Optional[ColdBand]:
        return {Proj.GATE: self.gate, Proj.UP: self.up, Proj.DOWN: self.down}[proj]


@dataclass
class PreparedWeights:
    """Stage 2의 유일한 산출물이자 weight lifetime owner (계약 ③)."""

    hot: None  # P0: hot = ∅
    warm: WarmStore
    cold: PendingColdTensors  # S5 이후: ColdHandle


def _uniform_proj_plan(plan: Plan, layer_idx: int, proj: Proj) -> ExpertProjPlan:
    """레이어 내 모든 expert의 proj 기하가 동일함을 요구하고 그것을 반환.

    P0 store가 [E, ...] 균일 적층이라서다. 가변 k_warm[e]가 오면 store가
    offset 테이블 기반으로 바뀌면서 이 요구가 사라진다.
    """
    first = plan.expert(layer_idx, 0).proj(proj)
    for expert in range(1, plan.dims.num_experts):
        other = plan.expert(layer_idx, expert).proj(proj)
        if other is not first and other != first:
            raise NotImplementedError(
                f"P0 loader requires uniform geometry across experts; "
                f"layer {layer_idx} {proj.value} differs at expert {expert}"
            )
    return first


def _single_band(pp: ExpertProjPlan, tier: Tier, where: str):
    bands = [b for b in pp.bands if b.tier is tier]
    if not bands:
        return None
    if len(bands) > 1:
        raise NotImplementedError(f"{where}: P0 loader supports one {tier.value} band")
    return bands[0]


def _proj_source(w13: torch.Tensor, w2: torch.Tensor, inter: int, proj: Proj):
    """proj의 ckpt-방향 소스 [E, N, K] 뷰."""
    if proj is Proj.GATE:
        return w13[:, :inter, :]
    if proj is Proj.UP:
        return w13[:, inter:, :]
    return w2


def prepare_layer_weights(
    layer_idx: int,
    w13: torch.Tensor,
    w2: torch.Tensor,
    plan: Plan,
    *,
    pin_memory: bool = True,
) -> PreparedWeights:
    """한 레이어의 full weight를 Plan대로 절단·변환·배치한다.

    process_weights_after_loading 훅(rank 0)에서 호출된다. 반환 후 호출자는
    w13/w2 참조를 놓아야 한다 (full 텐서 소멸 계약).

    pin_memory=False는 CUDA 없는 테스트용 탈출구다.
    """
    dims = plan.dims
    expected_w13 = (dims.num_experts, 2 * dims.intermediate_size, dims.hidden_size)
    expected_w2 = (dims.num_experts, dims.hidden_size, dims.intermediate_size)
    if tuple(w13.shape) != expected_w13 or tuple(w2.shape) != expected_w2:
        raise PlanError(
            f"layer {layer_idx}: weight shape mismatch vs plan dims: "
            f"w13 {tuple(w13.shape)} (expected {expected_w13}), "
            f"w2 {tuple(w2.shape)} (expected {expected_w2}) — "
            f"plan이 다른 모델에 적용되고 있을 가능성"
        )

    warm_bands: dict[Proj, Optional[WarmBand]] = {}
    cold_bands: dict[Proj, Optional[ColdBand]] = {}

    for proj in Proj:
        pp = _uniform_proj_plan(plan, layer_idx, proj)
        where = f"layer {layer_idx} {proj.value}"
        if pp.has_tier(Tier.HOT):
            raise NotImplementedError(f"{where}: P0 loader does not support HOT bands")

        src = _proj_source(w13, w2, dims.intermediate_size, proj)  # [E, N, K]

        warm = _single_band(pp, Tier.WARM, where)
        if warm is None:
            warm_bands[proj] = None
        else:
            # [E, N, k] → GEMM-ready [E, k, N], pinned
            sliced = src[:, :, warm.start : warm.end].transpose(1, 2)
            store = torch.empty(
                sliced.shape, dtype=w13.dtype, pin_memory=pin_memory
            )
            store.copy_(sliced)
            warm_bands[proj] = WarmBand(k_offset=warm.start, weights=store)

        cold = _single_band(pp, Tier.COLD, where)
        if cold is None:
            cold_bands[proj] = None
        else:
            # ckpt 방향 유지 (kt pack 입력)
            cold_bands[proj] = ColdBand(
                k_offset=cold.start,
                weights=src[:, :, cold.start : cold.end].contiguous(),
            )

        got = (warm.end - warm.start if warm else 0) + (
            cold.end - cold.start if cold else 0
        )
        if got != dims.k_of(proj):
            # validate_static이 커버리지를 이미 보증하므로, 여기 도달은
            # loader 자체의 결함이다 — 조용히 지나가면 이중계산/누락.
            raise AssertionError(
                f"{where}: loader dropped rows ({got} != K={dims.k_of(proj)})"
            )

    return PreparedWeights(
        hot=None,
        warm=WarmStore(
            gate=warm_bands[Proj.GATE], up=warm_bands[Proj.UP], down=warm_bands[Proj.DOWN]
        ),
        cold=PendingColdTensors(
            gate=cold_bands[Proj.GATE], up=cold_bands[Proj.UP], down=cold_bands[Proj.DOWN]
        ),
    )
