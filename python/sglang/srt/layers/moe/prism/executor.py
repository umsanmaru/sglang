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

from typing import Optional, Sequence

import torch

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

        # S1: topk D2H (P0 유일의 host 블록) + dedup
        ids_cpu = topk_ids.to("cpu")
        unique_ids = torch.unique(ids_cpu).tolist()
        groups = expert_groups(unique_ids, res.spec.n_slots)

        # ── Phase 1: gateup ──────────────────────────────────────────────
        if has_cold:
            res.staging.fill_x(hidden)            # blocking D2H (P0 strict)
            res.staging.fill_expert_ids(ids_cpu)
            self._qlen_pin[0] = m
            self._cold.submit_gateup(             # enqueue-only, 즉시 반환
                layer_idx, self._qlen_pin.data_ptr(), k,
                res.staging._expert_ids.data_ptr(), res.staging._x.data_ptr(),
                res.staging._partial_gateup.data_ptr(),
            )

        warm_gu = None
        gate_band, up_band = prepared.warm.band(Proj.GATE), prepared.warm.band(Proj.UP)
        if gate_band is not None:
            warm_gu = torch.zeros(m, k, 2 * inter, dtype=torch.float32, device=hidden.device)
            prev_done: Optional[torch.cuda.Event] = None
            for group in groups:
                evt_g = stage(gate_band, group, self._res.arena.view(Proj.GATE), res.warm_stream, prev_done)
                evt_u = stage(up_band, group, self._res.arena.view(Proj.UP), res.warm_stream, None)
                cur = torch.cuda.current_stream()
                cur.wait_event(evt_g)
                cur.wait_event(evt_u)
                g = len(group)
                gate_out = self._warm_gemm(hidden, self._res.arena.view(Proj.GATE)[:g], gate_band.k_offset)
                up_out = self._warm_gemm(hidden, self._res.arena.view(Proj.UP)[:g], up_band.k_offset)
                self._scatter_gateup(warm_gu, topk_ids, group, gate_out, up_out, inter)
                prev_done = torch.cuda.Event()
                prev_done.record(cur)

        cold_gu = None
        if has_cold:
            self._cold.sync()                     # S3: CPU 완료 블록
            cold_gu = res.staging.gateup_out(m).to(hidden.device)  # H2D

        # rejoin#1: fp32 합 → act → bf16 (계약 ⑤)
        gu = self._accumulate(warm_gu, cold_gu, (m, k, 2 * inter), hidden.device)
        gate, up = gu.split(inter, dim=2)
        act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)  # [M, k, inter]

        # ── Phase 2: down ────────────────────────────────────────────────
        if has_cold:
            res.staging.fill_act(act)
            self._cold.submit_down(
                layer_idx, self._qlen_pin.data_ptr(), k,
                res.staging._expert_ids.data_ptr(), res.staging._act.data_ptr(),
                res.staging._partial_down.data_ptr(),
            )

        warm_down = None
        down_band = prepared.warm.band(Proj.DOWN)
        if down_band is not None:
            warm_down = torch.zeros(m, k, h, dtype=torch.float32, device=hidden.device)
            act_band = act[:, :, down_band.k_offset : down_band.k_offset + down_band.k_rows].float()
            prev_done = None
            for group in groups:
                evt = stage(down_band, group, self._res.arena.view(Proj.DOWN), res.warm_stream, prev_done)
                cur = torch.cuda.current_stream()
                cur.wait_event(evt)
                w = self._res.arena.view(Proj.DOWN)
                for slot, e in enumerate(group):
                    mask = topk_ids == e            # [M, k] bool (GPU)
                    if not bool(mask.any()):
                        continue
                    a = act_band[mask]              # [n, rows] fp32
                    warm_down[mask] = a @ w[slot].float()  # fp32 누산 (계약 ⑤)
                prev_done = torch.cuda.Event()
                prev_done.record(cur)

        cold_down = None
        if has_cold:
            self._cold.sync()
            cold_down = res.staging.down_out(m).to(hidden.device)

        # rejoin#2: fp32 합 → router 가중 expert합 → bf16
        down = self._accumulate(warm_down, cold_down, (m, k, h), hidden.device)
        out = (down * topk_weights.to(torch.float32).unsqueeze(-1)).sum(dim=1)
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
        """그룹 GEMM 결과 [G, M, inter]를 (m, slot) 좌표의 [M, k, 2*inter]로 산란."""
        for slot, e in enumerate(group):
            mask = topk_ids == e                       # [M, k] bool
            if not bool(mask.any()):
                continue
            m_idx = mask.any(dim=1).nonzero(as_tuple=True)[0]
            # 토큰 m이 expert e를 여러 slot에서 뽑는 일은 없음(topk 무중복)
            j_idx = mask[m_idx].float().argmax(dim=1)
            warm_gu[m_idx, j_idx, :inter] = gate_out[slot, m_idx].float()
            warm_gu[m_idx, j_idx, inter:] = up_out[slot, m_idx].float()
