"""티어 인덱스 — K축 소유 행의 표현·생성·검증 (계약 ① 2026-08-25 개정).

밴드 `[start, end)`가 티어 멤버십을 표현하던 자리를 **가변 per-expert 인덱스**가
대체한다. 배치 기준(어느 행이 중요한가)이 실제로 만들어내는 것은 행의 집합이지
연속 구간이 아니고, expert마다 다른 개수를 줄 수 있어야 hot 예산을 전역 스칼라가
아니라 expert별로 쓸 수 있다.

표현은 **flat + offset**이다 — `[E, k, N]` 균일 적층이 가변 길이를 표현할 수
없으므로 스토어가 flat이 되고, 인덱스는 그 스토어와 **같은 오프셋 테이블**을
공유한다 (둘 다 expert당 `k[e]`개니까). 오프셋 하나가 weight·인덱스·점수
테이블 셋을 서비스한다.

이 모듈이 소유하는 것:

- `TierIndex` / `LayerIndex` — 표현
- `from_bands` — 기존 밴드 Plan → 연속 인덱스. **전환기의 등가 다리**다: 기존
  plan 40개가 그대로 새 경로를 타고 "밴드 경로와 비트일치"라는 검증 기준이
  유지된다. 밴드는 인덱스가 연속인 퇴화형일 뿐이라는 계약 문장의 실물.
- `validate_layer` — 순열·페어·오프셋 검증. plan.py의 `_validate_bands`가 하던
  일의 대체물이며, 인덱스 시대에 이중계산/누락을 잡는 **두 방어선 중 하나**다
  (다른 하나는 셔플 인덱스에서의 정수 비트일치 — 계약 ⑤-5).

**자산 로더는 아직 여기 없다.** 계약 ①이 "Plan=정책 / 자산=기하"로 경계를
그었지만 자산 생성기가 이 저장소 밖에 아직 존재하지 않는다 — 생산자가 없는 파일
포맷을 먼저 발명하지 않는다. 생성기가 생기는 시점에 `TierIndex.from_rows`를
입구로 하는 로더가 이 모듈에 붙는다 (calib.py와 같은 형태: 경로 + sha256 대조 →
논리명 노출).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch

from sglang.srt.layers.moe.prism.plan import (
    PAIR_GROUP,
    ModelDims,
    Plan,
    PlanError,
    Proj,
    Tier,
)

# 인덱스 dtype·상한·run 판정은 공유 코어가 소유한다 (2026-08-31 승격) — dense가
# 같은 표현을 쓰므로 두 곳에 두면 드리프트가 곧 무증상 오답이다. 여기서 re-export
# 하는 것은 기존 import 경로(`moe.prism.index.IDX_DTYPE`)를 살리기 위해서다.
from sglang.srt.layers.prism.store import (  # noqa: F401
    IDX_DTYPE,
    MAX_K,
    OFF_DTYPE,
)
from sglang.srt.layers.prism.store import is_row_run as _is_run  # noqa: F401


@dataclass(frozen=True)
class TierIndex:
    """한 (proj, tier)가 소유하는 K행 — expert별 가변 길이, flat + offset.

    `row_off`: int32 [E+1], `idx`: uint16 [row_off[-1]].
    expert e의 행은 `idx[row_off[e] : row_off[e+1]]`이고, **같은 구간이 weight
    스토어와 점수 테이블의 구간이기도 하다** (오프셋 공유).
    """

    row_off: torch.Tensor
    idx: torch.Tensor
    # 모든 expert의 행이 단위 stride 오름차순 구간인가 (= 밴드 퇴화형).
    # 참이면 소비자가 gather를 건너뛰고 포인터 오프셋으로 대체할 수 있다 —
    # 기존 밴드 경로와의 비트일치가 이 플래그 위에서 성립한다.
    contiguous: bool

    @property
    def num_experts(self) -> int:
        return int(self.row_off.numel()) - 1

    @property
    def total_rows(self) -> int:
        return int(self.row_off[-1])

    def k_rows(self, expert: int) -> int:
        return int(self.row_off[expert + 1]) - int(self.row_off[expert])

    def for_expert(self, expert: int) -> torch.Tensor:
        """expert의 행 번호 [k[e]] (uint16). 빈 티어면 길이 0."""
        return self.idx[int(self.row_off[expert]) : int(self.row_off[expert + 1])]

    def start_of(self, expert: int) -> int:
        """`contiguous`일 때 expert 구간의 시작 행. 아니면 의미 없음."""
        if not self.contiguous:
            raise ValueError("start_of is only meaningful for a contiguous index")
        return int(self.idx[int(self.row_off[expert])])

    def to(self, device) -> "TierIndex":
        return TierIndex(
            row_off=self.row_off.to(device),
            idx=self.idx.to(device),
            contiguous=self.contiguous,
        )

    # ── 생성 ─────────────────────────────────────────────────────────────
    @classmethod
    def from_rows(
        cls, per_expert: Sequence[Sequence[int]]
    ) -> Optional["TierIndex"]:
        """expert별 행 목록 → TierIndex. 전 expert가 비면 None.

        None을 돌려주는 이유: "이 티어는 이 레이어에 없다"를 길이 0 텐서가 아니라
        부재로 표현해야 스토어(`HotStore.gate = None`)와 어휘가 맞는다.
        자산 로더도 이 입구를 쓴다.
        """
        rows = [torch.as_tensor(r, dtype=torch.int64).reshape(-1) for r in per_expert]
        total = sum(int(r.numel()) for r in rows)
        if total == 0:
            return None
        off = torch.zeros(len(rows) + 1, dtype=OFF_DTYPE)
        acc = 0
        for e, r in enumerate(rows):
            acc += int(r.numel())
            off[e + 1] = acc
        flat = torch.cat(rows) if rows else torch.empty(0, dtype=torch.int64)
        if int(flat.numel()) and int(flat.max()) > MAX_K:
            raise PlanError(
                f"K row index {int(flat.max())} exceeds {MAX_K} — "
                f"uint16 인덱스로 표현 불가 (dtype을 올려야 한다)"
            )
        if int(flat.numel()) and int(flat.min()) < 0:
            raise PlanError(f"negative K row index {int(flat.min())}")
        return cls(
            row_off=off,
            idx=flat.to(IDX_DTYPE),
            contiguous=all(_is_run(r) for r in rows),
        )


@dataclass(frozen=True)
class LayerIndex:
    """한 레이어의 (proj, tier) → TierIndex. 없는 티어는 None."""

    tiers: Mapping[tuple, Optional[TierIndex]]

    def get(self, proj: Proj, tier: Tier) -> Optional[TierIndex]:
        return self.tiers.get((proj, tier))

    def to(self, device) -> "LayerIndex":
        return LayerIndex(
            {k: (v.to(device) if v is not None else None) for k, v in self.tiers.items()}
        )


# ---------------------------------------------------------------------------
# 밴드 → 인덱스 (전환기의 등가 다리)
# ---------------------------------------------------------------------------


def from_bands(plan: Plan, layer_idx: int) -> LayerIndex:
    """밴드 Plan의 한 레이어를 연속 인덱스로 옮긴다.

    per-expert 밴드가 서로 달라도 그대로 따라간다 (Plan의 overrides가 이미
    per-(layer, expert)이므로 가변 길이가 여기서 자연히 나온다). 티어당 밴드가
    여러 개면 밴드 순서대로 이어 붙인다 — 인덱스 표현에는 "티어당 단일 밴드"
    제약이 없다.
    """
    dims = plan.dims
    tiers: dict[tuple, Optional[TierIndex]] = {}
    for proj in Proj:
        K = dims.k_of(proj)
        if K > MAX_K:
            raise PlanError(
                f"{proj.value}: K={K} exceeds uint16 index range {MAX_K}"
            )
        per_tier: dict[Tier, list[list[int]]] = {t: [] for t in Tier}
        for expert in range(dims.num_experts):
            pp = plan.expert(layer_idx, expert).proj(proj)
            for tier in Tier:
                rows: list[int] = []
                for band in pp.bands:
                    if band.tier is tier:
                        rows.extend(range(band.start, band.end))
                per_tier[tier].append(rows)
        for tier in Tier:
            tiers[(proj, tier)] = TierIndex.from_rows(per_tier[tier])
    return LayerIndex(tiers)


# ---------------------------------------------------------------------------
# 검증 — plan.py의 밴드 검증을 대체한다
# ---------------------------------------------------------------------------


def _validate_structure(ti: TierIndex, num_experts: int, where: str) -> None:
    if ti.row_off.dtype != OFF_DTYPE or ti.idx.dtype != IDX_DTYPE:
        raise PlanError(
            f"{where}: dtypes must be ({OFF_DTYPE}, {IDX_DTYPE}), "
            f"got ({ti.row_off.dtype}, {ti.idx.dtype})"
        )
    if ti.num_experts != num_experts:
        raise PlanError(
            f"{where}: row_off has {ti.num_experts} experts, expected {num_experts}"
        )
    off = ti.row_off.to(torch.int64)
    if int(off[0]) != 0:
        raise PlanError(f"{where}: row_off[0] must be 0, got {int(off[0])}")
    if bool((off[1:] < off[:-1]).any()):
        raise PlanError(f"{where}: row_off must be non-decreasing")
    if int(off[-1]) != int(ti.idx.numel()):
        raise PlanError(
            f"{where}: row_off[-1]={int(off[-1])} != idx length {int(ti.idx.numel())}"
        )


def _validate_pairs(rows: torch.Tensor, where: str) -> None:
    """인덱스는 페어 단위로 움직인다 (계약 ① 정렬 규칙).

    원본에서 인접한 `(2p, 2p+1)`이 같은 티어에 있고 gather 후에도 인접해야 한다.
    쪼개지면 어느 티어도 k2wl2 점수를 재구성할 수 없고, VNNI 패킹의 skip 단위도
    깨진다. 페어 **안에서의** 순서는 자유다 — `wn`이 같은 인덱스로 동행하므로
    점수식이 대칭이다.
    """
    n = int(rows.numel())
    if n % PAIR_GROUP:
        raise PlanError(
            f"{where}: row count {n} is not a multiple of PAIR_GROUP={PAIR_GROUP}"
        )
    if n == 0:
        return
    pairs = rows.view(-1, PAIR_GROUP)
    lo, hi = pairs[:, 0], pairs[:, 1]
    if not bool(((lo // PAIR_GROUP) == (hi // PAIR_GROUP)).all()) or bool(
        (lo == hi).any()
    ):
        bad = int(((lo // PAIR_GROUP) != (hi // PAIR_GROUP)).nonzero()[0][0]) if not bool(
            ((lo // PAIR_GROUP) == (hi // PAIR_GROUP)).all()
        ) else int((lo == hi).nonzero()[0][0])
        raise PlanError(
            f"{where}: masking pair split at position {bad * PAIR_GROUP} "
            f"({int(lo[bad])}, {int(hi[bad])}) — 한 페어의 두 채널은 같은 티어에서 "
            f"인접해야 한다"
        )


def validate_layer(
    layer_index: LayerIndex, dims: ModelDims, layer_idx: int
) -> None:
    """한 레이어의 인덱스 정합성. 위반은 전부 PlanError (로드 시 hard error).

    검사 셋 중 **순열**이 최우선이다: hot ∪ warm ∪ cold가 `[0, K)`를 정확히 한
    번씩 덮지 않으면 조용한 이중계산 또는 누락이 되고, 밴드 시절의 disjoint+커버
    검증이 사라진 자리를 이것이 메운다.
    """
    for proj in Proj:
        K = dims.k_of(proj)
        for tier in Tier:
            ti = layer_index.get(proj, tier)
            if ti is not None:
                _validate_structure(
                    ti, dims.num_experts, f"layer {layer_idx} {proj.value} {tier.value}"
                )
        for expert in range(dims.num_experts):
            rows: list[torch.Tensor] = []
            for tier in Tier:
                ti = layer_index.get(proj, tier)
                if ti is None:
                    continue
                r = ti.for_expert(expert).to(torch.int64)
                _validate_pairs(
                    r, f"layer {layer_idx} expert {expert} {proj.value} {tier.value}"
                )
                rows.append(r)
            where = f"layer {layer_idx} expert {expert} {proj.value}"
            joined = (
                torch.cat(rows) if rows else torch.empty(0, dtype=torch.int64)
            )
            if int(joined.numel()) != K:
                raise PlanError(
                    f"{where}: tiers own {int(joined.numel())} rows but K={K} — "
                    f"커버리지 위반 (이중계산 또는 누락)"
                )
            counts = torch.bincount(joined, minlength=K)
            if bool((counts != 1).any()):
                dup = int((counts > 1).nonzero()[0][0]) if bool((counts > 1).any()) else None
                missing = int((counts == 0).nonzero()[0][0]) if bool((counts == 0).any()) else None
                raise PlanError(
                    f"{where}: tier 인덱스가 [0, {K})의 순열이 아니다 "
                    f"(중복 행 {dup}, 누락 행 {missing}) — 이중계산/누락"
                )
