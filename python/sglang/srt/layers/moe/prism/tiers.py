"""GPU 티어 — 계약 ④의 `GpuTier`와 두 구현.

계약 ①이 "hot과 warm의 계산 계약은 완전히 동일하다"고 못박은 것을 타입으로
표현한다. 두 구현은 스토어가 device냐 pinned냐 하나로만 갈리고 커널·출력
레이아웃·호출 규약이 같다 — `gemv_worklist_indexed`와 그 pinned 쌍둥이의
차이가 곧 이 두 클래스의 차이 전부다.

warm이 "전송"되던 시절에는 `stage → arena → GEMM`이라는 별도 사슬이 필요했다.
제자리 UVA 읽기가 되면서 그 사슬이 포인터 종류 하나로 축소됐고, 그래서
`DeviceArena`·stager·grouping이 함께 사라졌다.

**cold는 이 Protocol에 들어오지 않는다.** submit/sync 2-phase에 CPU 완료 대기가
있어 `run` 하나로 접히지 않고, 억지로 통합하면 "submit은 no-op, sync에서 다 함"
같은 거짓 구현이 생긴다 (계약 ④).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol

import torch

from sglang.srt.layers.moe.prism.plan import Proj, Tier
from sglang.srt.layers.moe.prism.weights import PreparedWeights, TierShard


class GpuTier(Protocol):
    """한 (layer, proj)의 GPU 티어 하나.

    `run`은 current stream에 launch만 하고 즉시 반환한다 — sync point가 없다
    (계약 ④의 표). 출력은 호출자 소유이고, 티어는 로드 타임에 소유된 스토어·
    인덱스만 읽는다 (영구 할당 금지 규칙).
    """

    shard: TierShard

    def run(self, x2d: torch.Tensor, topk_ids: torch.Tensor,
            out3d: torch.Tensor, out_col_off: int, *,
            x_row_is_pair: bool) -> None: ...


@dataclass(frozen=True)
class _IndexedTier:
    """두 구현의 공통 본체 — 다른 것은 어떤 커널 래퍼를 부르느냐뿐이다."""

    shard: TierShard

    def _fn(self):
        raise NotImplementedError

    def run(self, x2d, topk_ids, out3d, out_col_off, *, x_row_is_pair) -> None:
        s = self.shard
        self._fn()(
            x2d, topk_ids, s.w_flat, s.row_off, s.k_index, out3d,
            out_col_off, x_row_is_pair, torch.cuda.current_stream(),
        )


@dataclass(frozen=True)
class ResidentTier(_IndexedTier):
    """HOT — 스토어가 VRAM 상주. GPU가 device 포인터로 읽는다."""

    def _fn(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed

        return gemv_worklist_indexed


@dataclass(frozen=True)
class PinnedDirectTier(_IndexedTier):
    """WARM — 스토어가 pinned host 상주. GPU가 UVA로 제자리 읽는다.

    배치 안에 같은 expert가 두 번 나오면 그만큼 PCIe를 두 번 탄다 (bs=1에서는
    발생 불가). 재사용이 이득이 되는 구간이 실측되면 그때 세 번째 구현
    (select → device 버퍼 → 재사용)이 이 Protocol에 붙는다.
    """

    def _fn(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed_pinned

        return gemv_worklist_indexed_pinned


def build_layer_tiers(
    prepared: PreparedWeights,
) -> Mapping[tuple, Optional[GpuTier]]:
    """Stage 2 산출물 → (proj, tier) → GpuTier. 없는 티어는 항목 자체가 없다."""
    tiers: dict[tuple, GpuTier] = {}
    for proj in Proj:
        hot = None if prepared.hot is None else prepared.hot.band(proj)
        if hot is not None:
            tiers[(proj, Tier.HOT)] = ResidentTier(hot)
        warm = prepared.warm.band(proj)
        if warm is not None:
            tiers[(proj, Tier.WARM)] = PinnedDirectTier(warm)
    return tiers
