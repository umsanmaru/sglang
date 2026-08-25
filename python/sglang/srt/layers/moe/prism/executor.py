"""Prism eager executor — 계약 ④의 primitive들과 2-phase 조율 (P0).

제어 흐름 (CONTRACTS.md / P0 스펙):

    topk D2H(S1) → dedup·그룹
    cold gateup submit ∥ hot: [gather → GEMM] ∥ warm: [stage → GEMM] 그룹 직렬 루프
    cold sync(S3) → rejoin#1 (fp32 합 → act → bf16)
    cold down submit ∥ warm down
    cold sync → rejoin#2 (fp32 합 → router 가중 expert합 → bf16)

hot 티어 (계약 ①): VRAM 상주라 stager(PCIe 전송)를 타지 않는다 — warm 경로에서
호스트 교차만 빠진 형태이고, GEMM 커널·grouping 계약은 warm과 **완전히 동일**하다
(hot store가 warm과 같은 [E, k_rows, N] K-major이기 때문). gather 커널도 warm과
같은 uint4 커널(device-src 변형)이며 목적지만 전용 hot arena다 — bmm이 연속 배치
축을 요구해 VRAM→VRAM 복사 한 번은 남는다 (torch index_select였을 때 지연 바운드
~40 µs/proj → gather 커널로 대역폭 수준; 복사 자체의 제거는 grouped GEMM TODO).
그룹 루프를 warm과 공유하는 이유는 slot 제약 때문이 아니라(hot arena는 current
stream 순서로 자급) scatter_gateup/down_apply의 (m, j) 좌표 복원 규약을 하나로
유지하기 위해서다.
발행 순서는 `WAR 시드 record → hot GEMM → warm 루프`다: hot은 warm arena를
건드리지 않으므로 warm의 첫 H2D가 hot GEMM 뒤로 밀릴 이유가 없고, 이 순서에서만
warm 전송이 hot 연산 뒤에 겹친다.

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

from sglang.jit_kernel.prism_gather import gather_bands_from_device
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


def _hot_band(prepared: PreparedWeights, proj: Proj):
    """hot store가 아예 없는 PreparedWeights(구 테스트 픽스처)도 받아준다."""
    return None if prepared.hot is None else prepared.hot.band(proj)


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
    worklist: bool = False              # decode worklist 모드 (spec 2026-08-25)


class PrismExecutor:
    """레이어 상태(warm store, cold 여부)와 공유 리소스로 2-phase를 조율."""

    def __init__(self, plan: Plan, resources: ExecutionResources,
                 cold: Optional[ColdBackend], gpu_warm_kernel: WarmGemmFn,
                 stager: Stager, *,
                 cold_stream: bool = False,
                 force_graph_path: bool = False,
                 capture_mode_fn: Optional[Callable[[], bool]] = None,
                 worklist_kernels=None,
                 worklist_max_m: int = 32):
        """cold_stream: eager에서도 cold submit/sync를 stream 통합으로 (opt-in).
        force_graph_path: 캡처 없이 graph-safe 경로 강제 (테스트/디버그).
        capture_mode_fn: sglang CudaGraphRunner의 capture 구간(캡처 전 워밍업
        포함) 신호 — 조립 지점이 주입한다. None이면 실제 stream 캡처 중에만
        graph 경로를 탄다 (워밍업의 lazy init이 캡처 안으로 밀리므로, sglang
        통합 실행에서는 반드시 주입할 것 — method.py 참조).
        worklist_kernels: (device_fn, pinned_fn) 쌍 (kernels.resolve_worklist_kernels
        반환) 또는 None. None이면 worklist 경로가 아예 존재하지 않는다 (기존
        dedup/hot-gather-scatter 경로만).
        worklist_max_m: worklist 경로를 태우는 M 상한 (decode 전용, eager) —
        이보다 크면(prefill) 기존 dedup 경로로 폴백한다."""
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
        # worklist GEMV (decode 전용). None이면 기존 경로만 존재한다.
        self._worklist_fns = worklist_kernels          # (device_fn, pinned_fn)
        self._worklist_max_m = worklist_max_m
        # graph qlen pin — bs(M)별로 격리 (Finding A의 일반화: 절대 공유 금지).
        # 위 _qlen_pin_graph 주석의 stale-pointer 사고가 재발하지 않으려면,
        # bs가 다른 replay가 같은 버퍼를 절대 나눠 쓰지 않아야 한다 — worklist가
        # M>1 graph를 열면서 상수 1 버퍼 하나로는 부족해져 dict로 일반화한다.
        self._qlen_pins_graph: dict[int, torch.Tensor] = {
            1: self._qlen_pin_graph  # 기존 상수 1 버퍼를 bs=1 항목으로 흡수
        }

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
        worklist = self._worklist_fns is not None and m <= self._worklist_max_m
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
            # worklist가 없으면 여전히 M==1만 허용(P0 그대로); worklist가
            # 있으면 M<=worklist_max_m까지 열어준다 (pair-native라 그룹/슬롯
            # 제약이 없다).
            if m != 1 and not worklist:
                raise RuntimeError(
                    f"prism graph path requires M==1 (bs=1) without a worklist "
                    f"kernel, got M={m} — use kernels.gpu_warm='gemv_worklist' "
                    f"or capture with cuda_graph_bs=[1]"
                )
            # qlen pin — bs(M)별로 격리 (Finding A, 절대 공유 금지: 위
            # _qlen_pins_graph 주석 참조). bs=1은 생성자가 이미 심어둔 항목.
            qlen_pin = self._qlen_pins_graph.get(m)
            if qlen_pin is None:
                # 캡처 전 워밍업 호출(capture_mode_fn)에서 생성된다 — 실제
                # 캡처 중 신규 bs가 나타나면 이 할당은 캡처 밖 host alloc이라
                # 안전하지만, sglang 워밍업 순서상 도달하지 않는 것이 정상.
                qlen_pin = torch.full((1,), m, dtype=torch.int32)
                self._qlen_pins_graph[m] = qlen_pin
            grouping = select_grouping(1)  # SlotOrderGrouping 싱글턴
            return _LayerFlow(
                graph_flow=True, use_cold_stream=True, stream_arg=stream_arg,
                qlen_ptr=qlen_pin.data_ptr(),
                grouping=grouping,
                groups=[] if worklist else grouping.make_groups_for_graph(k, self._res.spec.n_slots),
                flat_ids=topk_ids.view(-1),  # device int64 [k] — stager의 sel 원천
                ids_cpu=None,
                worklist=worklist,
            )
        # S1: topk D2H (P0 유일의 host 블록). cold submit이 expert_ids를
        # ids_cpu에서 채우므로 worklist 모드에서도 여전히 계산한다 — dedup
        # (make_groups)만 건너뛴다 (worklist는 그룹/gather/scatter가 없다).
        with _nvtx("s1.topk_d2h"):
            ids_cpu = topk_ids.to("cpu")
        if worklist:
            grouping, groups = select_grouping(1), []
        else:
            with _nvtx("s1.dedup"):
                grouping = select_grouping(m)
                groups = grouping.make_groups(ids_cpu, self._res.spec.n_slots)
        return _LayerFlow(
            graph_flow=False, use_cold_stream=use_cold_stream, stream_arg=stream_arg,
            qlen_ptr=self._qlen_pin.data_ptr(),
            grouping=grouping, groups=groups, flat_ids=None, ids_cpu=ids_cpu,
            worklist=worklist,
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
                    # qlen_pins_graph[m]은 상수 m(위 guard가 보장) — 아무도
                    # 다시 쓰지 않는다. 격리가 깨지지 않았는지만 방어적으로
                    # 확인 (host 메모리 읽기라 capture-safe). bs별 버퍼가
                    # 절대 공유되지 않아야 한다는 Finding A의 불변식을 여기서
                    # per-bs로 재확인한다.
                    pin = self._qlen_pins_graph[m]
                    assert int(pin[0]) == m, (
                        f"qlen_pin_graph[{m}]={int(pin[0])} != {m} — "
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

        # WAR 시드: 이전 레이어의 down GEMM이 같은 arena 바이트를 읽는 중일 수
        # 있다 (down이 gate/up storage를 alias + 레이어 간 재사용). 첫 stage가
        # current stream의 기왕 작업 완료를 기다리게 한다.
        # **hot GEMM보다 먼저** 기록한다 — hot은 arena를 안 건드리므로 warm의
        # 첫 H2D가 hot 연산을 기다릴 이유가 없다 (순서가 뒤집히면 hot이 클수록
        # warm 전송이 통째로 직렬화된다).
        # worklist 모드는 arena를 건드리지 않으므로 WAR 시드가 불필요하다.
        if not flow.worklist:
            prev_done = torch.cuda.Event()
            prev_done.record(torch.cuda.current_stream())

        hot_gu = None
        hot_g, hot_u = _hot_band(prepared, Proj.GATE), _hot_band(prepared, Proj.UP)
        if hot_g is not None and flow.worklist:
            # worklist: pair (m,j)가 rejoin 좌표 — 그룹/gather/scatter 없음.
            # 두 호출이 [:, :, :inter]/[inter:]를 완전히 덮으므로 empty로 충분.
            dev_fn, _ = self._worklist_fns
            hot_gu = torch.empty(m, k, 2 * inter, dtype=torch.bfloat16, device=hidden.device)
            with _nvtx("hot.gu.worklist"):
                cur = torch.cuda.current_stream()
                dev_fn(hidden, topk_ids, hot_g.weights, hot_gu, hot_g.k_offset, 0, False, cur)
                dev_fn(hidden, topk_ids, hot_u.weights, hot_gu, hot_u.k_offset, inter, False, cur)
        elif hot_g is not None:
            hot_gu = torch.zeros(m, k, 2 * inter, dtype=torch.float32, device=hidden.device)
            for gi, group in enumerate(flow.groups):
                with _nvtx(f"hot.gu.g{gi}x{len(group)}"):
                    with _nvtx("gather.gate+up"):
                        sel_d = self._hot_sel(group, gi, flow)
                        wg = self._hot_gather(hot_g, Proj.GATE, sel_d)
                        wu = self._hot_gather(hot_u, Proj.UP, sel_d)
                    with _nvtx("gemm.gate"):
                        gate_out = self._warm_gemm(hidden, wg, hot_g.k_offset)
                    with _nvtx("gemm.up"):
                        up_out = self._warm_gemm(hidden, wu, hot_u.k_offset)
                    with _nvtx("scatter.gu"):
                        flow.grouping.scatter_gateup(hot_gu, topk_ids, group, gi,
                                                     res.spec.n_slots, gate_out, up_out, inter)

        warm_gu = None
        gate_band, up_band = prepared.warm.band(Proj.GATE), prepared.warm.band(Proj.UP)
        if gate_band is not None and flow.worklist:
            # warm worklist: W가 pinned CPU라 pinned_fn(UVA 직접 읽기)을 쓴다.
            # stager/warm_stream/이벤트 없이 current stream 직렬 (spec §3) —
            # 배치 내 중복 expert는 PCIe 재전송이라는 설계 트레이드오프.
            _, pin_fn = self._worklist_fns
            warm_gu = torch.empty(m, k, 2 * inter, dtype=torch.bfloat16, device=hidden.device)
            with _nvtx("warm.gu.worklist"):
                cur = torch.cuda.current_stream()
                pin_fn(hidden, topk_ids, gate_band.weights, warm_gu, gate_band.k_offset, 0, False, cur)
                pin_fn(hidden, topk_ids, up_band.weights, warm_gu, up_band.k_offset, inter, False, cur)
        elif gate_band is not None:
            warm_gu = torch.zeros(m, k, 2 * inter, dtype=torch.float32, device=hidden.device)
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
            gu = self._accumulate((hot_gu, warm_gu, cold_gu),
                                  (m, k, 2 * inter), hidden.device)
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

        # WAR 시드: down arena는 gate/up storage를 alias — 첫 down stage는
        # gateup GEMM(current stream)이 arena를 다 읽은 뒤에만 덮어야 한다.
        # (이전의 prev_done=None은 잠복 레이스였고, slot당 host 동기화가
        #  우연히 직렬화해 숨겨져 있었다 — sync-free 전환에서 발현, 2026-08-20)
        # worklist 모드는 arena를 건드리지 않으므로 WAR 시드가 불필요하다.
        if not flow.worklist:
            prev_done = torch.cuda.Event()
            prev_done.record(torch.cuda.current_stream())

        hot_down = None
        hot_d = _hot_band(prepared, Proj.DOWN)
        if hot_d is not None and flow.worklist:
            dev_fn, _ = self._worklist_fns
            hot_down = torch.empty(m, k, h, dtype=torch.bfloat16, device=hidden.device)
            with _nvtx("hot.dn.worklist"):
                dev_fn(act.reshape(m * k, inter), topk_ids, hot_d.weights, hot_down,
                       hot_d.k_offset, 0, True, torch.cuda.current_stream())
        elif hot_d is not None:
            hot_down = torch.zeros(m, k, h, dtype=torch.float32, device=hidden.device)
            hot_act = _band_slice(act, hot_d).float()
            for gi, group in enumerate(flow.groups):
                with _nvtx(f"hot.dn.g{gi}x{len(group)}"):
                    with _nvtx("gather.down"):
                        sel_d = self._hot_sel(group, gi, flow)
                        wd = self._hot_gather(hot_d, Proj.DOWN, sel_d)
                    with _nvtx("gemm+where.dn"):
                        hot_down = flow.grouping.down_apply(
                            hot_down, topk_ids, group, gi, res.spec.n_slots, hot_act, wd)

        warm_down = None
        down_band = prepared.warm.band(Proj.DOWN)
        if down_band is not None and flow.worklist:
            _, pin_fn = self._worklist_fns
            warm_down = torch.empty(m, k, h, dtype=torch.bfloat16, device=hidden.device)
            with _nvtx("warm.dn.worklist"):
                pin_fn(act.reshape(m * k, inter), topk_ids, down_band.weights, warm_down,
                       down_band.k_offset, 0, True, torch.cuda.current_stream())
        elif down_band is not None:
            warm_down = torch.zeros(m, k, h, dtype=torch.float32, device=hidden.device)
            act_band = act[:, :, down_band.k_offset : down_band.k_offset + down_band.k_rows].float()
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
            down = self._accumulate((hot_down, warm_down, cold_down),
                                    (m, k, h), hidden.device)
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

    def _hot_sel(self, group: Sequence[int], gi: int,
                 flow: _LayerFlow) -> torch.Tensor:
        """그룹의 expert 인덱스를 hot 전용 device int32 버퍼에 올린다.

        sel 원천은 _stage와 같은 규약이다: graph 경로의 그룹은 위치 표지라
        flat_ids의 같은 위치 절단(device cast-copy — 캡처 가능), eager 그룹은
        expert id 목록 그 자체(소형 H2D)다. 반환 버퍼는 current stream에서만
        읽고 쓰이므로 (gather·GEMM과 같은 stream) 그룹/phase 간 재사용 WAR가
        stream 순서로 자동 충족된다 — warm sel처럼 더블버퍼가 필요 없다.
        """
        g = len(group)
        sel_d = self._res.hot_sel_device()[:g]
        if flow.graph_flow:
            j0 = gi * self._res.spec.n_slots
            sel_d.copy_(flow.flat_ids[j0 : j0 + g])
        else:
            sel_d.copy_(torch.as_tensor(group, dtype=torch.int32))
        return sel_d

    def _hot_gather(self, band, proj: Proj, sel_d: torch.Tensor) -> torch.Tensor:
        """hot store에서 그룹의 [g, k_rows, N] GEMM 입력을 hot arena로 모은다.

        bmm이 연속 배치 축을 요구하므로 복사 자체는 피할 수 없다 (제거는
        grouped GEMM 몫 — TODO). 이전의 `index_select`는 torch 범용 커널
        (indexSelectSmallIndex)이라 지연 바운드(2 B 스칼라 접근 + 인덱스 순차
        루프)로 slab 크기와 무관하게 ~40 µs가 들었다. warm과 같은 uint4
        gather 커널의 device-src 변형으로 g개 slab을 한 웨이브에 옮긴다
        (2026-08-25 nsys, h125 35B: proj당 42 µs → HBM 대역폭 수준).

        목적지는 영구 hot arena — warm arena와 같은 레이어-최대 크기 가정이라
        band.k_rows가 spec.k_hot_of(proj)와 다르면 커널 shape 검증이 즉사한다
        (uniform plan에서는 항상 일치; warm 경로와 같은 제약).
        """
        dst = self._res.hot_view(proj)[: sel_d.numel()]
        gather_bands_from_device(
            band.weights, sel_d, dst, torch.cuda.current_stream()
        )
        return dst

    @staticmethod
    def _accumulate(parts: Sequence[Optional[torch.Tensor]],
                    shape, device) -> torch.Tensor:
        """티어 partial들의 fp32 합 (없는 티어는 0 기여)."""
        total: Optional[torch.Tensor] = None
        for part in parts:
            if part is None:
                continue
            p32 = part if part.dtype == torch.float32 else part.to(torch.float32)
            total = p32 if total is None else total + p32
        if total is None:
            return torch.zeros(shape, dtype=torch.float32, device=device)
        return total
