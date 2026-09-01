"""dense executor — 티어 partial을 모아 rejoin (계약 ④).

MoE executor(659줄)와 비교하면 **없는 것이 차이의 대부분**이다:

| | MoE | dense |
|---|---|---|
| phase | 2 (gateup → down) | **1** — gate_up과 down이 별개 `LinearBase`라 한 호출에 하나 |
| grouping | pair를 expert로 묶는다 | **자명** — E=1이라 전 토큰이 expert 0 |
| rejoin | Σ→act, Σ→라우터 가중합 | Σ 하나 |
| graph 분기 | cold expert_ids 조달이 갈린다 | (cold 미배선) |

제어 흐름::

    cold submit (있으면) ∥ hot/warm GEMV
    cold sync → partial 회수
    rejoin: fp32 Σ → bf16

**출력 레이아웃.** 커널은 `out3d [M, 1, N_total]`에 쓰고, 조각은 자기 열 범위에만
쓴다. `[M, 1, N]`은 `[M, N]`의 뷰이므로 reshape이 공짜다 — 그리고 분할된
`gate_up_proj`에서 gate가 앞 열, up이 뒤 열이 되어 sglang의 `SiluAndMul`이 기대하는
`[M, 2I]` 레이아웃이 **그대로 나온다**.

**버퍼 0 초기화.** 한 티어를 안 쓰는 조각이 있으면 그 열이 안 채워지므로 0으로
할당한다 (gate는 hot인데 up은 warm인 plan에서 실제로 생긴다). 모든 조각이 그
티어를 쓰면 `empty`로 족하다 — 전 열이 덮인다.

cold는 아직 배선되지 않았다 (`cold_backend` 다음 단계). cold 행이 있는 plan을
주면 `register()`에서 즉사한다 — 조용히 그 행을 빼고 계산하면 **결과가 틀린다**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from sglang.srt.layers.prism.geometry import PlanError, Tier
from sglang.srt.layers.prism.linear.rejoin import rejoin
from sglang.srt.layers.prism.linear.tiers import LinearGpuTier, build_part_tiers
from sglang.srt.layers.prism.linear.weights import PreparedLinear


@dataclass
class _LayerProj:
    """등록된 한 (layer, proj)의 실행 상태."""

    prepared: PreparedLinear
    # 티어 → 그 티어를 쓰는 조각들의 어댑터 (조각마다 열 오프셋이 다르다)
    tiers: Mapping[Tier, tuple]
    # 그 티어를 **모든** 조각이 쓰는가 (아니면 버퍼를 0으로)
    writes_all: Mapping[Tier, bool]


class LinearExecutor:
    """proj 상태와 공유 리소스로 한 호출을 조율한다.

    MoE executor와 달리 상태가 거의 없다 — 모드 결정(graph/stream/hybrid)이 전부
    cold에 딸린 것이었고 cold가 아직 없다.
    """

    # grouped GEMM(prefill 형태)으로 갈아타는 최소 M. worklist는 pair마다 W를 다시
    # 읽는데 **dense는 E=1이라 중복도가 곧 M**이다 (MoE는 M·k/E라 32배 덜 나쁘다).
    # warm은 그 재읽기가 전부 PCIe라 큰 M에서 치명적이다 — Qwen3.8 M=2048에서 forward
    # 한 번에 89.7 TB, 약 30분이었다 (2026-09-01, prefill 캡처가 안 끝나서 발견).
    # grouped는 W를 한 번만 읽지만 타일 launch 고정비가 있어 작은 M에서 불리하다.
    # MoE의 교차점(16)을 그대로 쓴다 — dense는 재읽기가 더 심하니 더 낮아야 하면
    # 낮아지지 높아지지 않는다. 실측으로 정할 항목이다 (TODO).
    GROUPED_MIN_M = 16

    def __init__(self, *, max_tokens: int = 4096, device: Optional[torch.device] = None,
                 grouped_min_m: Optional[int] = None):
        self._projs: dict[tuple, _LayerProj] = {}
        self._max_tokens = int(max_tokens)
        self._device = device
        self._grouped_min_m = (self.GROUPED_MIN_M if grouped_min_m is None
                               else int(grouped_min_m))
        # E=1 grouping은 자명하다 (전 토큰이 expert 0) — 스텝마다 정렬할 이유가 없어
        # 최대 크기로 한 번 만들고 앞을 잘라 쓴다. `pair_sorted`는 항등 순열이고
        # `pair_off`/`tile_off`만 M에 따라 달라진다.
        self._group_cache: dict[int, object] = {}
        # E=1 퇴화 인자 — 스텝마다 만들면 할당이 생기므로 최대 크기로 한 번 잡고
        # 앞을 잘라 쓴다. 값이 상수(0 / 1.0)라 슬라이스가 곧 정답이다.
        self._ids: Optional[torch.Tensor] = None
        self._ones: Optional[torch.Tensor] = None

    # ── 등록 ─────────────────────────────────────────────────────────────
    def register(self, layer_idx: int, name: str, prepared: PreparedLinear,
                 specs: Optional[Mapping] = None) -> None:
        key = (layer_idx, name)
        if key in self._projs:
            raise RuntimeError(f"{key} already registered")
        if any(p.cold is not None for p in prepared.parts):
            raise NotImplementedError(
                f"layer {layer_idx} proj '{name}': COLD rows are not wired yet — "
                f"cold backend가 붙기 전까지 cold 밴드가 없는 plan만 실행할 수 있다"
            )
        per_tier: dict[Tier, list] = {}
        for part in prepared.parts:
            for tier, adapter in build_part_tiers(part, specs).items():
                per_tier.setdefault(tier, []).append(adapter)
        if not per_tier:
            raise PlanError(f"layer {layer_idx} proj '{name}': no GPU tier rows")
        n_parts = len(prepared.parts)
        self._projs[key] = _LayerProj(
            prepared=prepared,
            tiers={t: tuple(v) for t, v in per_tier.items()},
            writes_all={t: len(v) == n_parts for t, v in per_tier.items()},
        )

    def registered(self) -> frozenset:
        return frozenset(self._projs)

    def warmup(self, device) -> None:
        """지연 할당을 startup으로 앞당긴다 (캡처 안에서 처음 잡히면 graph pool에 들어간다).

        `_degenerate`의 두 상수 버퍼가 그 대상이다 — 값이 0/1.0으로 고정이라 미리
        만들어도 되고, 그래야 캡처가 그것을 자기 풀에 넣지 않는다.
        """
        self._degenerate(1, device)

    # ── step ─────────────────────────────────────────────────────────────
    def _degenerate(self, m: int, device):
        """E=1 퇴화 인자 (topk_ids=0, weights=1.0)를 앞 m행만 잘라 준다."""
        if self._ids is None or self._ids.device != device:
            self._ids = torch.zeros(self._max_tokens, 1, dtype=torch.int64, device=device)
            self._ones = torch.ones(self._max_tokens, 1, dtype=torch.float32, device=device)
        if m > self._max_tokens:
            raise ValueError(
                f"M={m} exceeds max_tokens={self._max_tokens} — "
                f"SGLANG_PRISM_LINEAR_MAX_TOKENS를 올려야 한다"
            )
        return self._ids[:m], self._ones[:m]

    def _grouping(self, m: int, device):
        """E=1의 자명한 Grouping. pair p = m·1 + 0 = m 이라 정렬이 항등이다."""
        from sglang.srt.layers.moe.prism.grouping import Grouping
        from sglang.jit_kernel.prism_grouped import TILE_M

        g = self._group_cache.get(m)
        if g is None:
            if self._ids is None or self._ids.device != device:
                self._degenerate(m, device)
            pair = torch.arange(m, dtype=torch.int32, device=device)
            pair_off = torch.tensor([0, m], dtype=torch.int32, device=device)
            tiles = (m + TILE_M - 1) // TILE_M
            tile_off = torch.tensor([0, tiles], dtype=torch.int32, device=device)
            g = Grouping(pair_sorted=pair, pair_off=pair_off, tile_off=tile_off,
                         tile_m=TILE_M)
            self._group_cache[m] = g
        return g

    def run(self, layer_idx: int, name: str, x: torch.Tensor, *,
            masking: bool = False) -> torch.Tensor:
        """x `[M, K]` bf16 cuda → out `[M, N]` bf16 cuda.

        `masking`은 **스텝별** 결정이다 (계약 ①: sparsity는 decode 전용). 티어의
        종류는 로드 타임에 고정되지만 마스킹 여부는 매 스텝 갈리므로, sparse 티어도
        masking=False면 dense 커널을 부른다.
        """
        st = self._projs.get((layer_idx, name))
        if st is None:
            raise KeyError(f"layer {layer_idx} proj '{name}' is not registered")
        prep = st.prepared
        m = x.shape[0]
        if x.shape[1] != prep.k:
            raise ValueError(f"x has K={x.shape[1]} but proj expects {prep.k}")
        ids, ones = self._degenerate(m, x.device)
        # prefill 형태: W를 조각당 한 번만 읽는다. masking은 decode 전용이라
        # (계약 ①) 둘이 동시에 켜지지 않는다.
        grouping = self._grouping(m, x.device) if (
            m >= self._grouped_min_m and not masking) else None

        parts = []
        for tier, adapters in st.tiers.items():
            alloc = torch.empty if st.writes_all[tier] else torch.zeros
            buf = alloc(m, 1, prep.n, dtype=torch.bfloat16, device=x.device)
            for adapter in adapters:
                adapter.run(x, ids, ones, buf, masking=masking, grouping=grouping)
            parts.append(buf.view(m, prep.n))
        return rejoin(parts)
