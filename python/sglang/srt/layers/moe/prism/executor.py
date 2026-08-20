"""Prism eager executor — 계약 ④의 primitive들과 2-phase 조율 (P0).

제어 흐름 (CONTRACTS.md / P0 스펙):

    topk D2H(S1) → dedup·그룹
    cold gateup submit ∥ warm: [stage → GEMM] 그룹 직렬 루프
    cold sync(S3) → rejoin#1 (fp32 합 → act → bf16)
    cold down submit ∥ warm down
    cold sync → rejoin#2 (fp32 합 → router 가중 expert합 → bf16)

원칙:
- primitive는 영구 메모리를 할당하지 않는다 — arena/staging은
  ExecutionResources 소유 (계약 ④). eager의 일시 출력만 신규 할당.
- 누산은 전부 fp32, 재료화는 bf16 (계약 ⑤).
- 그룹 직렬 루프의 arena WAR은 이벤트 체인으로 보장: 그룹 g+1의 stage는
  그룹 g의 GEMM 완료 이벤트를 warm stream에서 기다린다.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional, Sequence

import torch

# nsys 구간 태그 (SGLANG_PRISM_NVTX=1일 때만 활성 — 평시 no-op).
# 여기 구간은 host-측 push/pop이다: GPU 실측 시간은 CUDA HW row에서 읽고,
# cold↔GPU 겹침은 cold.*.window(submit 반환~sync 반환)와 커널들의 교차로 읽는다.
_NVTX = os.environ.get("SGLANG_PRISM_NVTX") == "1"


@contextmanager
def _nvtx(msg: str):
    if not _NVTX:
        yield
        return
    torch.cuda.nvtx.range_push(msg)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _nvtx_push(msg: str) -> None:
    if _NVTX:
        torch.cuda.nvtx.range_push(msg)


def _nvtx_pop() -> None:
    if _NVTX:
        torch.cuda.nvtx.range_pop()

from sglang.srt.layers.moe.prism.cold_backend import ColdBackend
from sglang.srt.layers.moe.prism.kernels import WarmGemmFn
from sglang.srt.layers.moe.prism.plan import Plan, Proj, Tier
from sglang.srt.layers.moe.prism.resources import ExecutionResources
from sglang.srt.layers.moe.prism.weights import PreparedWeights, WarmBand


def expert_groups(unique_ids: Sequence[int], n_slots: int) -> list[list[int]]:
    """distinct expert들을 arena slot 수 단위로 절단 (P0: 직렬 루프)."""
    return [list(unique_ids[i : i + n_slots]) for i in range(0, len(unique_ids), n_slots)]


def stage(band: WarmBand, group_ids: Sequence[int], arena_view: torch.Tensor,
          warm_stream: torch.cuda.Stream, wait_event: Optional[torch.cuda.Event]) -> torch.cuda.Event:
    """warm 밴드 이동 (pinned store → arena slot). 반환 event = 전송 완료 표지.

    wait_event: 직전 그룹의 GEMM 완료 — arena 재사용 WAR 보호.
    """
    with torch.cuda.stream(warm_stream):
        if wait_event is not None:
            warm_stream.wait_event(wait_event)
        for slot, e in enumerate(group_ids):
            arena_view[slot].copy_(band.weights[e], non_blocking=True)
    evt = torch.cuda.Event()
    evt.record(warm_stream)
    return evt


class PrismExecutor:
    """레이어 상태(warm store, cold 여부)와 공유 리소스로 2-phase를 조율."""

    def __init__(self, plan: Plan, resources: ExecutionResources,
                 cold: Optional[ColdBackend], gpu_warm_kernel: WarmGemmFn):
        self._plan = plan
        self._res = resources
        self._cold = cold
        self._warm_gemm = gpu_warm_kernel
        self._layers: dict[int, PreparedWeights] = {}
        self._layer_has_cold: dict[int, bool] = {}
        # cold task가 나중에 읽는 qlen — 주소 고정 멤버 (계약 ④의 포인터 경유)
        self._qlen_pin = torch.zeros(1, dtype=torch.int32)

    def register_layer(self, layer_idx: int, prepared: PreparedWeights) -> None:
        """Stage 2 산출물 등록. cold 밴드가 있으면 backend에 이미 load_layer된
        상태여야 한다 (로딩 순서는 method/loader의 책임)."""
        ep = self._plan.expert(layer_idx, 0)
        has_cold = any(ep.proj(p).has_tier(Tier.COLD) for p in Proj)
        if has_cold and self._cold is None:
            raise RuntimeError(f"layer {layer_idx} has COLD bands but no cold backend")
        self._layers[layer_idx] = prepared
        self._layer_has_cold[layer_idx] = has_cold

    # ── 본체 ──────────────────────────────────────────────────────────────
    def run_layer(self, layer_idx: int, hidden: torch.Tensor,
                  topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        """hidden [M, H] bf16 cuda, topk_ids [M, k] int64, topk_weights [M, k].
        반환 [M, H] bf16 cuda (router 가중 expert 합 완료)."""
        prepared = self._layers[layer_idx]
        has_cold = self._layer_has_cold[layer_idx]
        res, dims = self._res, self._plan.dims
        m, k = hidden.shape[0], topk_ids.shape[1]
        inter, h = dims.intermediate_size, dims.hidden_size

        _nvtx_push(f"prism.L{layer_idx}")

        # S1: topk D2H (P0 유일의 host 블록) + dedup
        with _nvtx("s1.topk_d2h+dedup"):
            ids_cpu = topk_ids.to("cpu")
            unique_ids = torch.unique(ids_cpu).tolist()
            groups = expert_groups(unique_ids, res.spec.n_slots)

        # ── Phase 1: gateup ──────────────────────────────────────────────
        if has_cold:
            with _nvtx("cold.gu.fill_x(D2H-block)"):
                res.staging.fill_x(hidden)        # blocking D2H (P0 strict)
                res.staging.fill_expert_ids(ids_cpu)
                self._qlen_pin[0] = m
            with _nvtx("cold.gu.submit"):
                self._cold.submit_gateup(         # enqueue-only, 즉시 반환
                    layer_idx, self._qlen_pin.data_ptr(), k,
                    res.staging._expert_ids.data_ptr(), res.staging._x.data_ptr(),
                    res.staging._partial_gateup.data_ptr(),
                )
            _nvtx_push("cold.gu.window")          # CPU expert 연산 재실 구간

        warm_gu = None
        gate_band, up_band = prepared.warm.band(Proj.GATE), prepared.warm.band(Proj.UP)
        if gate_band is not None:
            warm_gu = torch.zeros(m, k, 2 * inter, dtype=torch.float32, device=hidden.device)
            # WAR 시드: 이전 레이어의 down GEMM이 같은 arena 바이트를 읽는 중일 수
            # 있다 (down이 gate/up storage를 alias + 레이어 간 재사용). 첫 stage가
            # current stream의 기왕 작업 완료를 기다리게 한다.
            prev_done = torch.cuda.Event()
            prev_done.record(torch.cuda.current_stream())
            for gi, group in enumerate(groups):
                with _nvtx(f"warm.gu.g{gi}x{len(group)}"):
                    with _nvtx("stage.gate+up(H2D)"):
                        evt_g = stage(gate_band, group, self._res.arena.view(Proj.GATE), res.warm_stream, prev_done)
                        evt_u = stage(up_band, group, self._res.arena.view(Proj.UP), res.warm_stream, None)
                    cur = torch.cuda.current_stream()
                    cur.wait_event(evt_g)
                    cur.wait_event(evt_u)
                    g = len(group)
                    with _nvtx("gemm.gate"):
                        gate_out = self._warm_gemm(hidden, self._res.arena.view(Proj.GATE)[:g], gate_band.k_offset)
                    with _nvtx("gemm.up"):
                        up_out = self._warm_gemm(hidden, self._res.arena.view(Proj.UP)[:g], up_band.k_offset)
                    with _nvtx("scatter.gu"):
                        self._scatter_gateup(warm_gu, topk_ids, group, gate_out, up_out, inter)
                    prev_done = torch.cuda.Event()
                    prev_done.record(cur)

        cold_gu = None
        if has_cold:
            with _nvtx("cold.gu.sync(host-block)"):
                self._cold.sync()                 # S3: CPU 완료 블록
            _nvtx_pop()                           # cold.gu.window
            with _nvtx("cold.gu.h2d_out"):
                cold_gu = res.staging.gateup_out(m).to(hidden.device)  # H2D

        # rejoin#1: fp32 합 → act → bf16 (계약 ⑤)
        with _nvtx("rejoin1.acc+silu"):
            gu = self._accumulate(warm_gu, cold_gu, (m, k, 2 * inter), hidden.device)
            gate, up = gu.split(inter, dim=2)
            act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)  # [M, k, inter]

        # ── Phase 2: down ────────────────────────────────────────────────
        if has_cold:
            with _nvtx("cold.dn.fill_act(D2H-block)"):
                res.staging.fill_act(act)
            with _nvtx("cold.dn.submit"):
                self._cold.submit_down(
                    layer_idx, self._qlen_pin.data_ptr(), k,
                    res.staging._expert_ids.data_ptr(), res.staging._act.data_ptr(),
                    res.staging._partial_down.data_ptr(),
                )
            _nvtx_push("cold.dn.window")

        warm_down = None
        down_band = prepared.warm.band(Proj.DOWN)
        if down_band is not None:
            warm_down = torch.zeros(m, k, h, dtype=torch.float32, device=hidden.device)
            act_band = act[:, :, down_band.k_offset : down_band.k_offset + down_band.k_rows].float()
            # WAR 시드: down arena는 gate/up storage를 alias — 첫 down stage는
            # gateup GEMM(current stream)이 arena를 다 읽은 뒤에만 덮어야 한다.
            # (이전의 prev_done=None은 잠복 레이스였고, slot당 host 동기화가
            #  우연히 직렬화해 숨겨져 있었다 — sync-free 전환에서 발현, 2026-08-20)
            prev_done = torch.cuda.Event()
            prev_done.record(torch.cuda.current_stream())
            for gi, group in enumerate(groups):
                with _nvtx(f"warm.dn.g{gi}x{len(group)}"):
                    with _nvtx("stage.down(H2D)"):
                        evt = stage(down_band, group, self._res.arena.view(Proj.DOWN), res.warm_stream, prev_done)
                    cur = torch.cuda.current_stream()
                    cur.wait_event(evt)
                    w = self._res.arena.view(Proj.DOWN)
                    with _nvtx("gemm+where.dn"):
                        for slot, e in enumerate(group):
                            # sync-free: 전 (m, j) 좌표를 계산하고 where로 선별 —
                            # 데이터 의존 host 분기/nonzero 없음 (위 _scatter_gateup 참조).
                            # slot당 여분 계산은 decode에서 무시 가능(수 µs GEMM).
                            mask = (topk_ids == e).unsqueeze(-1)          # [M, k, 1]
                            contrib = act_band @ w[slot].float()          # [M, k, H] fp32 누산 (계약 ⑤)
                            warm_down = torch.where(mask, contrib, warm_down)
                    prev_done = torch.cuda.Event()
                    prev_done.record(cur)

        cold_down = None
        if has_cold:
            with _nvtx("cold.dn.sync(host-block)"):
                self._cold.sync()
            _nvtx_pop()                           # cold.dn.window
            with _nvtx("cold.dn.h2d_out"):
                cold_down = res.staging.down_out(m).to(hidden.device)

        # rejoin#2: fp32 합 → router 가중 expert합 → bf16
        with _nvtx("rejoin2.acc+wsum"):
            down = self._accumulate(warm_down, cold_down, (m, k, h), hidden.device)
            out = (down * topk_weights.to(torch.float32).unsqueeze(-1)).sum(dim=1)
        _nvtx_pop()                               # prism.L{layer_idx}
        return out.to(torch.bfloat16)

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _accumulate(warm: Optional[torch.Tensor], cold: Optional[torch.Tensor],
                    shape, device) -> torch.Tensor:
        """티어 partial들의 fp32 합 (없는 티어는 0 기여)."""
        if warm is None and cold is None:
            return torch.zeros(shape, dtype=torch.float32, device=device)
        total = warm if warm is not None else torch.zeros(shape, dtype=torch.float32, device=device)
        if cold is not None:
            total = total + cold.to(torch.float32)
        return total

    @staticmethod
    def _scatter_gateup(warm_gu: torch.Tensor, topk_ids: torch.Tensor, group: Sequence[int],
                        gate_out: torch.Tensor, up_out: torch.Tensor, inter: int) -> None:
        """그룹 GEMM 결과 [G, M, inter]를 (m, slot) 좌표의 [M, k, 2*inter]로 산란.

        sync-free 규칙: host 동기화를 유발하는 연산(nonzero, bool(mask.any()),
        item, 데이터 의존 shape) 금지 — slot당 동기화가 층당 수 ms를 차지했던
        실측(2026-08-20)의 재발 방지. broadcast where는 전부 device에 머문다.
        """
        for slot, e in enumerate(group):
            mask = (topk_ids == e).unsqueeze(-1)       # [M, k, 1] device bool
            g = gate_out[slot].float().unsqueeze(1)    # [M, 1, inter] → k로 broadcast
            u = up_out[slot].float().unsqueeze(1)
            warm_gu[:, :, :inter] = torch.where(mask, g, warm_gu[:, :, :inter])
            warm_gu[:, :, inter:] = torch.where(mask, u, warm_gu[:, :, inter:])
