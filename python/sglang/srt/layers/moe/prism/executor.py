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

graph-safe 경로 (Task 8, M==1 전용 — 캡처 중 자동 선택):
- per-token host 결정 제거: ids_cpu D2H 없이 그룹은 위치 표지 [0..k) 절단,
  expert 선택은 stager가 device topk_ids에서 직접(sel_device) 나른다.
- cold submit/sync는 current stream 경유(kt cudaLaunchHostFunc host node)로
  캡처되고, staging D2H/H2D는 전부 non_blocking stream op — replay마다
  재실행되어 그 시점의 topk/hidden을 나른다.
- eager에서도 cold stream 통합만 opt-in 가능 (생성자 cold_stream — env
  SGLANG_PRISM_COLD_STREAM 읽기는 조립 지점 method.py의 몫).

sparsity (계약 ①, 선택): plan.sparsity가 있으면 **cold만** 마스킹된다.
warm/hot은 dense로 계산한다 (warm GEMM이 latency 바운드라 마스킹이 순손실이
었다 — CONTRACTS.md ①). sparsity 수식 전체가 kt에 있고, 이 모듈이 하는 일은
라우터 가중을 staging에 내려 포인터를 넘기는 것뿐이다.
**M==1(decode)에서만** 적용된다 (prefill-dense/decode-sparse).

이 모듈은 env/외부 시스템을 직접 읽지 않는다: 모드 결정 입력(cold_stream,
capture_mode_fn)은 전부 생성자 주입이고, 호출별 모드는 _plan_flow()가
_LayerFlow 값 객체로 1회 확정한다.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

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
from sglang.srt.layers.moe.prism.grouping import GroupingStrategy, select_grouping
from sglang.srt.layers.moe.prism.kernels import WarmGemmFn
from sglang.srt.layers.moe.prism.plan import Plan, Proj, Tier
from sglang.srt.layers.moe.prism.resources import ExecutionResources
from sglang.srt.layers.moe.prism.stagers import GatherKernelStager, Stager
from sglang.srt.layers.moe.prism.weights import PreparedWeights


def _band_slice(x: torch.Tensor, band) -> torch.Tensor:
    """x[..., k_offset : k_offset + k_rows] — 마스킹 대상 밴드 구간."""
    return x[..., band.k_offset : band.k_offset + band.k_rows]


@dataclass(frozen=True)
class _LayerFlow:
    """run_layer 한 호출의 실행 모드 — _plan_flow()가 상단에서 1회 확정하는
    값 객체. (모드 플래그들이 함수 본문과 helper 인자를 관통하며 흩어지는 것을
    막는다 — 실행 모드가 늘면 여기와 _plan_flow만 늘어난다.)"""

    graph_flow: bool                    # graph-safe 경로인가 (캡처/워밍업/강제)
    use_cold_stream: bool               # cold 호출을 stream host node로 보낼 것인가
    stream_arg: Optional[int]           # cold submit/sync의 cudaStream_t (None = host 경로)
    qlen_ptr: int                       # cold task가 역참조할 qlen 버퍼 주소
    grouping: GroupingStrategy
    groups: list[list[int]]
    flat_ids: Optional[torch.Tensor]    # graph 경로 sel 원천 (device int64 [k]) / eager는 None
    ids_cpu: Optional[torch.Tensor]     # eager 경로 expert_ids 원천 / graph는 None


class PrismExecutor:
    """레이어 상태(warm store, cold 여부)와 공유 리소스로 2-phase를 조율."""

    def __init__(self, plan: Plan, resources: ExecutionResources,
                 cold: Optional[ColdBackend], gpu_warm_kernel: WarmGemmFn,
                 stager: Stager, *,
                 cold_stream: bool = False,
                 force_graph_path: bool = False,
                 capture_mode_fn: Optional[Callable[[], bool]] = None):
        """cold_stream: eager에서도 cold submit/sync를 stream 통합으로 (opt-in).
        force_graph_path: 캡처 없이 graph-safe 경로 강제 (테스트/디버그).
        capture_mode_fn: sglang CudaGraphRunner의 capture 구간(캡처 전 워밍업
        포함) 신호 — 조립 지점이 주입한다. None이면 실제 stream 캡처 중에만
        graph 경로를 탄다 (워밍업의 lazy init이 캡처 안으로 밀리므로, sglang
        통합 실행에서는 반드시 주입할 것 — method.py 참조)."""
        self._plan = plan
        self._res = resources
        self._cold = cold
        self._warm_gemm = gpu_warm_kernel
        self._stager = stager
        self._cold_stream = cold_stream
        self._force_graph_path = force_graph_path
        self._capture_mode_fn = capture_mode_fn or (lambda: False)
        self._layers: dict[int, PreparedWeights] = {}
        self._layer_has_cold: dict[int, bool] = {}
        self._sparse = plan.sparsity is not None
        # cold task가 나중에 읽는 qlen — 주소 고정 멤버 (계약 ④의 포인터 경유)
        self._qlen_pin = torch.zeros(1, dtype=torch.int32)
        # graph 경로 전용 qlen 버퍼 — eager의 _qlen_pin과 절대 공유하지 않는다.
        # graph flow는 M==1만 허용(아래 guard)하므로 상수 1이며, 캡처 후에도
        # 아무도 다시 쓰지 않는다. 공유하면: 캡처가 baked하는 포인터가 나중의
        # eager prefill(self._qlen_pin[0] = L)에 노출되고, 그 후의 모든 graph
        # replay가 cold를 L토큰짜리로 돌려 perf가 붕괴한다 (review Finding A,
        # 실측 30B decode graph 328ms/tok vs eager 106ms/tok — 원인이 바로
        # 이 stale 공유 포인터였다).
        self._qlen_pin_graph = torch.ones(1, dtype=torch.int32)
        self._graph_stager: Optional[GatherKernelStager] = None

    def register_layer(self, layer_idx: int, prepared: PreparedWeights) -> None:
        """Stage 2 산출물 등록. cold 밴드가 있으면 backend에 이미 load_layer된
        상태여야 한다 (로딩 순서는 method/loader의 책임)."""
        ep = self._plan.expert(layer_idx, 0)
        has_cold = any(ep.proj(p).has_tier(Tier.COLD) for p in Proj)
        if has_cold and self._cold is None:
            raise RuntimeError(f"layer {layer_idx} has COLD bands but no cold backend")
        self._layers[layer_idx] = prepared
        self._layer_has_cold[layer_idx] = has_cold

    # ── 모드 결정 ──────────────────────────────────────────────────────────
    def _plan_flow(self, m: int, k: int, topk_ids: torch.Tensor) -> _LayerFlow:
        """호출별 실행 모드를 1회 확정. graph-safe 경로는 캡처 중 불법 연산
        (pageable D2H, tolist, event.synchronize, blocking copy)을 전부
        우회해야 하므로, 여기서 갈라진 결정이 본문의 유일한 분기 원천이다."""
        graph_flow = (
            torch.cuda.is_current_stream_capturing()
            or self._force_graph_path
            or self._capture_mode_fn()
        )
        # cold stream 통합: graph 경로는 필수(호출이 kt host node로 캡처돼야
        # 함), eager는 생성자 opt-in. None이면 기존 host 경로 (P0 그대로).
        use_cold_stream = graph_flow or self._cold_stream
        stream_arg = (
            torch.cuda.current_stream().cuda_stream if use_cold_stream else None
        )
        if graph_flow:
            # 그룹 조성에 host 결정 없음: M==1 + slot-order 전제에서 그룹은
            # 항상 위치 표지 [0..k) 절단 — ids_cpu D2H 자체가 사라진다.
            # qlen도 격리된 상수 버퍼(1)를 쓴다 (Finding A).
            if m != 1:
                raise RuntimeError(
                    f"prism graph path requires M==1 (bs=1), got M={m} — "
                    f"capture with cuda_graph_bs=[1] (and cuda_graph_max_bs=1)"
                )
            grouping = select_grouping(1)  # SlotOrderGrouping 싱글턴
            return _LayerFlow(
                graph_flow=True, use_cold_stream=True, stream_arg=stream_arg,
                qlen_ptr=self._qlen_pin_graph.data_ptr(),
                grouping=grouping,
                groups=grouping.make_groups_for_graph(k, self._res.spec.n_slots),
                flat_ids=topk_ids.view(-1),  # device int64 [k] — stager의 sel 원천
                ids_cpu=None,
            )
        # S1: topk D2H (P0 유일의 host 블록) + dedup
        with _nvtx("s1.topk_d2h+dedup"):
            ids_cpu = topk_ids.to("cpu")
            grouping = select_grouping(m)
            groups = grouping.make_groups(ids_cpu, self._res.spec.n_slots)
        return _LayerFlow(
            graph_flow=False, use_cold_stream=use_cold_stream, stream_arg=stream_arg,
            qlen_ptr=self._qlen_pin.data_ptr(),
            grouping=grouping, groups=groups, flat_ids=None, ids_cpu=ids_cpu,
        )

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
        flow = self._plan_flow(m, k, topk_ids)

        # sparsity: M==1(decode)에서만 마스킹한다 (prefill-dense/decode-sparse).
        # threshold는 세 proj 모두 topk_weights만으로 정해지므로 여기서 한 번에
        # 구한다 — 전부 device 연산이라 graph 캡처에 안전하다.
        # sparsity: cold만 마스킹한다 (warm/hot은 dense). 라우터 가중만
        # 내려보내면 kt가 예산·격자 조회·마스크 판정을 전부 처리한다.
        masking = self._sparse and m == 1
        flat_ids = topk_ids.view(-1)

        # ── Phase 1: gateup ──────────────────────────────────────────────
        if has_cold:
            with _nvtx("cold.gu.fill_x(D2H-block)"):
                # stream 통합 시 non_blocking: kt host node가 같은 stream에
                # 순서대로 실행되므로 host-측 완료 보장이 불필요 (Task 8 —
                # CONTRACTS ④ 개정). 기본은 P0 blocking 그대로.
                res.staging.fill_x(hidden, non_blocking=flow.use_cold_stream)
                if flow.graph_flow:
                    # device topk_ids → pinned int64 async D2H (캡처 가능;
                    # kt는 pinned를 읽는다). ids_cpu는 이 경로에 존재하지 않음.
                    res.staging.fill_expert_ids(topk_ids, non_blocking=True)
                    # qlen_pin_graph는 상수 1(M==1 guard가 보장) — 아무도
                    # 다시 쓰지 않는다. 격리가 깨지지 않았는지만 방어적으로
                    # 확인 (host 메모리 읽기라 capture-safe).
                    assert int(self._qlen_pin_graph[0]) == 1, (
                        f"qlen_pin_graph={int(self._qlen_pin_graph[0])} != 1 — "
                        f"graph path must never write this buffer"
                    )
                else:
                    res.staging.fill_expert_ids(flow.ids_cpu)
                    self._qlen_pin[0] = m
            # sparsity: 라우터 가중을 한 번 내리면 gateup/down 양쪽이 쓴다
            # (레이어 안에서 다시 쓰지 않으므로 버퍼 하나로 충분하다).
            w_ptr = 0
            if masking:
                with _nvtx("cold.fill_topk_w"):
                    res.staging.fill_topk_w(
                        topk_weights, non_blocking=flow.use_cold_stream)
                w_ptr = res.staging.topk_w_ptr()
            with _nvtx("cold.gu.submit"):
                self._cold.submit_gateup(         # enqueue-only, 즉시 반환
                    layer_idx, flow.qlen_ptr, k,
                    res.staging.expert_ids_ptr(), res.staging.x_ptr(),
                    res.staging.partial_gateup_ptr(),
                    cuda_stream=flow.stream_arg,
                    weights_ptr=w_ptr,
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
            for gi, group in enumerate(flow.groups):
                with _nvtx(f"warm.gu.g{gi}x{len(group)}"):
                    with _nvtx("stage.gate+up(H2D)"):
                        evt_g = self._stage(gate_band, group, gi, prev_done, Proj.GATE, flow)
                        evt_u = self._stage(up_band, group, gi, None, Proj.UP, flow)
                    cur = torch.cuda.current_stream()
                    cur.wait_event(evt_g)
                    cur.wait_event(evt_u)
                    g = len(group)
                    with _nvtx("gemm.gate"):
                        gate_out = self._warm_gemm(hidden, self._res.arena.view(Proj.GATE)[:g], gate_band.k_offset)
                    with _nvtx("gemm.up"):
                        up_out = self._warm_gemm(hidden, self._res.arena.view(Proj.UP)[:g], up_band.k_offset)
                    with _nvtx("scatter.gu"):
                        flow.grouping.scatter_gateup(warm_gu, topk_ids, group, gi, res.spec.n_slots,
                                                     gate_out, up_out, inter)
                    prev_done = torch.cuda.Event()
                    prev_done.record(cur)

        cold_gu = None
        if has_cold:
            with _nvtx("cold.gu.sync(host-block)"):
                # stream 통합 시 sync는 stream에 올라간 host node — host는
                # 즉시 반환하고, 이후 consumer들이 stream 순서로 보호된다.
                self._cold.sync(cuda_stream=flow.stream_arg)  # S3: CPU 완료 블록
            _nvtx_pop()                           # cold.gu.window
            with _nvtx("cold.gu.h2d_out"):
                # pinned → cuda H2D. stream 통합 시 non_blocking (기본 .to()의
                # 말미 streamSynchronize는 캡처 불가). 이 pinned 버퍼의 다음
                # writer(다음 레이어의 cold task)도 같은 stream의 host node라
                # WAR는 stream 순서로 보호된다.
                cold_gu = res.staging.gateup_out(m).to(
                    hidden.device, non_blocking=flow.use_cold_stream
                )

        # rejoin#1: fp32 합 → act → bf16 (계약 ⑤)
        with _nvtx("rejoin1.acc+silu"):
            gu = self._accumulate(warm_gu, cold_gu, (m, k, 2 * inter), hidden.device)
            gate, up = gu.split(inter, dim=2)
            act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)  # [M, k, inter]

        # ── Phase 2: down ────────────────────────────────────────────────
        if has_cold:
            with _nvtx("cold.dn.fill_act(D2H-block)"):
                res.staging.fill_act(act, non_blocking=flow.use_cold_stream)
            with _nvtx("cold.dn.submit"):
                self._cold.submit_down(
                    layer_idx, flow.qlen_ptr, k,
                    res.staging.expert_ids_ptr(), res.staging.act_ptr(),
                    res.staging.partial_down_ptr(),
                    cuda_stream=flow.stream_arg,
                    weights_ptr=w_ptr,
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
            for gi, group in enumerate(flow.groups):
                with _nvtx(f"warm.dn.g{gi}x{len(group)}"):
                    with _nvtx("stage.down(H2D)"):
                        evt = self._stage(down_band, group, gi, prev_done, Proj.DOWN, flow)
                    cur = torch.cuda.current_stream()
                    cur.wait_event(evt)
                    w = self._res.arena.view(Proj.DOWN)
                    with _nvtx("gemm+where.dn"):
                        warm_down = flow.grouping.down_apply(warm_down, topk_ids, group, gi,
                                                             res.spec.n_slots, act_band, w)
                    prev_done = torch.cuda.Event()
                    prev_done.record(cur)

        cold_down = None
        if has_cold:
            with _nvtx("cold.dn.sync(host-block)"):
                self._cold.sync(cuda_stream=flow.stream_arg)
            _nvtx_pop()                           # cold.dn.window
            with _nvtx("cold.dn.h2d_out"):
                cold_down = res.staging.down_out(m).to(
                    hidden.device, non_blocking=flow.use_cold_stream
                )

        # rejoin#2: fp32 합 → router 가중 expert합 → bf16
        with _nvtx("rejoin2.acc+wsum"):
            down = self._accumulate(warm_down, cold_down, (m, k, h), hidden.device)
            out = (down * topk_weights.to(torch.float32).unsqueeze(-1)).sum(dim=1)
        _nvtx_pop()                               # prism.L{layer_idx}
        return out.to(torch.bfloat16)

    # ── helpers ──────────────────────────────────────────────────────────
    def _stage(self, band, group: Sequence[int], gi: int,
               wait_event: Optional[torch.cuda.Event], proj: Proj,
               flow: _LayerFlow) -> torch.cuda.Event:
        """스테이징 디스패치: eager는 주입된 stager(group_ids 경유), graph는
        GatherKernelStager.stage_from_device(device topk 슬라이스 경유).
        graph 그룹은 위치 표지이므로 sel은 flat_ids의 같은 위치 절단이다.
        graph 그룹은 위치 표지이므로 sel은 flat_ids의 같은 위치 절단이다."""
        arena_view = self._res.arena.view(proj)
        if flow.graph_flow:
            j0 = gi * self._res.spec.n_slots
            sel = flow.flat_ids[j0 : j0 + len(group)]
            return self._graph_stager_inst().stage_from_device(
                band, sel, arena_view, self._res.warm_stream, wait_event, proj
            )
        return self._stager.stage(
            band, group, arena_view, self._res.warm_stream, wait_event, proj
        )

    def _graph_stager_inst(self) -> GatherKernelStager:
        """graph 경로 전용 stager (지연 생성). eager stager가 이미 gather면
        재사용 — stage_from_device는 flip/guard 상태를 안 쓰므로 무해하다."""
        if self._graph_stager is None:
            if isinstance(self._stager, GatherKernelStager):
                self._graph_stager = self._stager
            else:
                self._graph_stager = GatherKernelStager(self._res)
        return self._graph_stager

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
