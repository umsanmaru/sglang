"""스테이징 전략 — pinned store → arena 이동의 메커니즘만 다르고 의미는 동일.

공통 규약: wait_event가 있으면 warm_stream에서 먼저 기다린 뒤 쓰기 시작
(arena aliasing WAR — 이벤트 체인 자체는 executor 소유), 반환 event는
이 밴드의 전송 완료 표지. 어느 구현도 영구 메모리를 할당하지 않는다(계약 ④
— 스크래치는 ExecutionResources 소유).
"""
from __future__ import annotations

from typing import Optional, Protocol, Sequence

import torch

from sglang.jit_kernel.prism_gather import gather_bands_from_pinned
from sglang.srt.layers.moe.prism.plan import Proj
from sglang.srt.layers.moe.prism.weights import WarmBand

# env 이름만 여기서 소유 — 값을 읽는 것은 조립 지점(method.py)의 몫이다.
# (이 모듈은 hidden input 없이 override 인자만 받는다.)
ENV_STAGER = "SGLANG_PRISM_STAGER"


class Stager(Protocol):
    def stage(self, band: WarmBand, group_ids: Sequence[int], arena_view: torch.Tensor,
              warm_stream: torch.cuda.Stream, wait_event: Optional[torch.cuda.Event],
              proj: Proj) -> torch.cuda.Event: ...


class PerSlotCopyStager:
    """기준선: slot당 copy_ (기존 executor.stage 이동 — 등가성 테스트의 준거)."""

    def stage(self, band, group_ids, arena_view, warm_stream, wait_event, proj):
        with torch.cuda.stream(warm_stream):
            if wait_event is not None:
                warm_stream.wait_event(wait_event)
            for slot, e in enumerate(group_ids):
                arena_view[slot].copy_(band.weights[e], non_blocking=True)
        evt = torch.cuda.Event()
        evt.record(warm_stream)
        return evt


class BatchedCopyStager:
    """eager: host index_select로 [G, rows, N] 한 덩이 결집 → H2D 1회.

    NVTX 실측 근거(2026-08-20): slot당 copy_ 24회/층이 H2D 26조각 + dispatch를
    만들었다. CPU-측 결집(memcpy 수 MB, ~수십 µs)과 launch 1회로 치환.

    scratch는 proj당 단일 버퍼가 아니라 더블버퍼(resources.stage_scratch의
    slot 0/1) — H2D는 enqueue만 하고 즉시 반환하므로(비동기), 다음 stage()
    호출의 host index_select가 "아직 이전 H2D가 읽는 중인" 같은 물리
    버퍼를 덮어쓰면 조용한 weight corruption이 난다 (Critical review
    finding, 2026-08-20). 고정: 호출마다 버퍼를 flip하고, 그 버퍼를 마지막에
    쓴 H2D의 완료 이벤트를 가드로 들고 있다가 재사용 직전에 host-wait한다.

    self._flip/self._guard는 인스턴스 상태이지만 "영구 메모리"가 아니다 —
    flip은 정수 카운터, guard는 재사용 가능한 transient cuda.Event 핸들일
    뿐 buffer/storage를 할당하지 않는다 (계약 ④는 storage에 대한 것).
    버퍼 identity 자체는 여전히 ExecutionResources 소유.
    """

    def __init__(self, resources):
        self._res = resources
        self._flip: dict = {}
        self._guard: dict = {}

    def stage(self, band, group_ids, arena_view, warm_stream, wait_event, proj):
        g = len(group_ids)
        idx = self._res.stage_index()[:g]
        idx.copy_(torch.as_tensor(group_ids, dtype=torch.int64))

        b = self._flip.get(proj, 0)
        self._flip[proj] = 1 - b
        guard = self._guard.get((proj, b))
        if guard is not None:
            # 이 버퍼를 마지막으로 읽은 H2D가 끝날 때까지 host-wait —
            # 그 전에 index_select로 덮어쓰면 WAR corruption.
            guard.synchronize()

        scratch = self._res.stage_scratch(proj, b)[:g]
        torch.index_select(band.weights, 0, idx, out=scratch)  # host memcpy
        with torch.cuda.stream(warm_stream):
            if wait_event is not None:
                warm_stream.wait_event(wait_event)
            arena_view[:g].copy_(scratch, non_blocking=True)  # H2D 1회
        evt = torch.cuda.Event()
        evt.record(warm_stream)
        self._guard[(proj, b)] = evt
        return evt


class GatherKernelStager:
    """device 인덱스 + UVA gather 커널 1 launch (env override
    `SGLANG_PRISM_STAGER=gather` 또는 graph_mode 선택).

    sel(선택 인덱스) 버퍼는 BatchedCopyStager의 scratch와 같은 WAR 계급이다:
    host copy_ → 비동기 H2D → gather 커널이 warm_stream에 순서대로 올라가므로,
    이전 H2D가 sel_pinned를 아직 읽는 중에 다음 stage() 호출의 host copy_가
    같은 물리 버퍼를 덮어쓰면 조용한 index corruption이 난다. 다만 sel은
    proj별 내용이 아니라 한 그룹의 gate/up/down 3회 stage() 호출에 동일한
    group_ids로 재사용되므로, BatchedCopyStager처럼 (proj, slot) 키가 아니라
    호출 순서에만 매인 단일 flip 카운터 + 2-버퍼 guard로 충분하다
    (resources.sel_pinned(slot)). sel_device는 디바이스측 쓰기라 스트림
    순서로 자동 직렬화되므로 단일 버퍼로 둔다 (resources.sel_device()).
    """

    def __init__(self, resources):
        self._res = resources
        self._flip = 0
        self._guard: dict = {}  # buf slot(0/1) -> 마지막으로 그 버퍼를 읽은 H2D의 완료 event

    def stage(self, band, group_ids, arena_view, warm_stream, wait_event, proj):
        g = len(group_ids)
        b = self._flip
        self._flip = 1 - b
        guard = self._guard.get(b)
        if guard is not None:
            # 이 버퍼를 마지막으로 읽은 H2D가 끝날 때까지 host-wait —
            # 그 전에 sel_pinned를 덮어쓰면 WAR corruption.
            guard.synchronize()

        sel_p = self._res.sel_pinned(b)[:g]
        sel_p.copy_(torch.as_tensor(group_ids, dtype=torch.int32))
        with torch.cuda.stream(warm_stream):
            if wait_event is not None:
                warm_stream.wait_event(wait_event)
            sel_d = self._res.sel_device()[:g]
            sel_d.copy_(sel_p, non_blocking=True)               # 32B H2D
            gather_bands_from_pinned(band.weights, sel_d, arena_view[:g], warm_stream)
        evt = torch.cuda.Event()
        evt.record(warm_stream)
        self._guard[b] = evt
        return evt


def select_stager(resources, graph_mode: bool, override: Optional[str] = None) -> Stager:
    """staging 전략 선택. override(통상 env ENV_STAGER의 값 — 읽기는 호출자
    몫)가 있으면 그것이 우선, 없으면 graph_mode에 따라 갈린다: eager 기본값은
    batched, graph_mode=True면 gather (device 인덱스라 graph 캡처와 호환).
    """
    if override == "gather":
        return GatherKernelStager(resources)
    if override == "batched":
        return BatchedCopyStager(resources)
    if override == "per_slot":
        return PerSlotCopyStager()
    if override is not None:
        raise ValueError(f"unknown {ENV_STAGER}={override!r}")
    return GatherKernelStager(resources) if graph_mode else BatchedCopyStager(resources)
