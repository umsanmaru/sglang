"""Prism executor — 계약 ④의 primitive들과 2-phase 조율.

제어 흐름:

    cold gateup submit ∥ hot/warm GEMV (pair-native)
    cold sync → rejoin#1 (fp32 합 → act → bf16)
    cold down submit ∥ hot/warm GEMV
    cold sync → rejoin#2 (fp32 합 → router 가중 expert합 → bf16)

GPU 티어는 `tiers.py`의 `GpuTier` 구현이 전담한다 — hot과 warm의 차이는
스토어가 device냐 pinned냐 하나뿐이고, 이 모듈은 그 차이를 모른다 (계약 ①).
pair (m, j)가 곧 rejoin 좌표이므로 **그룹·arena·stager·scatter가 없다**:
블록이 `topk[pair]`에서 expert를 스스로 읽고 결과를 out의 제 위치에 직접 쓴다.

decode와 prefill이 **같은 경로**다. 밴드 시절의 bmm 폴백은 (a) 가변 per-expert
K를 원리적으로 표현할 수 없고(연속 배치 축 요구), (b) 라우팅되지 않은
(토큰, expert) 쌍까지 계산하는 낭비가 있었다. 실측(2026-08-25, 35B gate 치수)
에서 worklist가 prefill M=1024~4096 구간에서 bmm 대비 1.6~1.9배인데, 그 대가로
경로가 하나가 되고 prefill이 인덱스를 지원하게 된다. 더 줄이려면 grouped GEMM
(토큰을 expert로 묶어 tensor core GEMM)이고 그건 별도 커널이다 — TODO.

graph-safe 경로: GPU 티어에는 host 결정이 아예 없어 eager와 캡처가 **같은
호출**이다. 갈리는 것은 cold의 expert_ids 조달 하나뿐이다 (eager는 ids_cpu
host copy, graph는 device→pinned async D2H).

sparsity (계약 ①): plan.sparsity가 있으면 **cold만** 마스킹된다. 이 모듈이 하는
일은 라우터 가중을 staging에 내려 포인터를 넘기는 것뿐이고, 예산·곡선·점수·
마스크는 전부 cold 커널 안에 있다. **M==1(decode)에서만** 적용된다.

이 모듈은 env/외부 시스템을 직접 읽지 않는다: 모드 결정 입력(cold_stream,
capture_mode_fn)은 전부 생성자 주입이고, 호출별 모드는 _plan_flow()가
_LayerFlow 값 객체로 1회 확정한다.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

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
from sglang.srt.layers.moe.prism.grouping import Grouping, build_grouping
from sglang.srt.layers.moe.prism.plan import Plan, Proj, Tier
from sglang.srt.layers.moe.prism.rejoin import rejoin_down, rejoin_gateup
from sglang.srt.layers.moe.prism.resources import ExecutionResources
from sglang.srt.layers.moe.prism.tiers import LayerTiers, build_layer_tiers
from sglang.srt.layers.moe.prism.weights import PreparedWeights


@dataclass(frozen=True)
class _LayerFlow:
    """run_layer 한 호출의 실행 모드 — _plan_flow()가 상단에서 1회 확정하는
    값 객체. GPU 티어는 여기 없다(모드가 없으므로) — 남은 것은 전부 cold 조달
    방식이다."""

    graph_flow: bool                  # graph-safe 경로인가 (캡처/워밍업/강제)
    use_cold_stream: bool             # cold 호출을 stream host node로 보낼 것인가
    stream_arg: Optional[int]         # cold submit/sync의 cudaStream_t (None = host 경로)
    qlen_ptr: int                     # cold task가 역참조할 qlen 버퍼 주소
    ids_cpu: Optional[torch.Tensor]   # eager 경로 expert_ids 원천 / graph는 None


class PrismExecutor:
    """레이어 상태(티어, cold 여부)와 공유 리소스로 2-phase를 조율."""

    # grouped GEMM(prefill 형태)으로 갈아타는 최소 M. worklist는 pair마다 W를
    # 다시 읽으므로(중복도 M·k/E) M이 커질수록 불리하고, grouped는 expert당
    # 타일 launch 고정비가 있어 작은 M에서 불리하다. 교차점은 실측으로 정한다
    # (2026-08-26, 35B dims H100: M=8 worklist 우세, M=128 warm은 grouped 4배
    # 우세·hot은 근소 열세 — warm이 PCIe라 결정 변수다).
    GROUPED_MIN_M = 16
    # cold를 GPU가 읽는 최소 M (cold_gpu.py). GPU 비용은 M 무관 상수(PCIe로 cold
    # 스토어 1회), CPU 비용은 M에 비례 — 교차점은 실측이다 (2026-08-27, 35B
    # h12.5/w12.5/cold75, H100 PCIe Gen5 ~51 GB/s, CPU 14스레드 AMX 2노드: 엔진
    # TTFT 1344tok 동률, 2688tok GPU −15%, 672tok CPU −11%). 하드웨어·스레드 수에
    # 따라 움직이므로 planner 항목이다 (TODO.md).
    COLD_GPU_MIN_M = 1536
    # hybrid 비용 모델의 보정점: frac 실측(0.65/0.85 @ 6thr)이 이 pair 수(M=2688, k=8)의
    # 균등 라우팅에서 나왔다. 다른 M/분포에는 여기서 환산한 c/g 상수를 그대로 쓴다.
    HYBRID_CALIB_PAIRS = 2688 * 8

    def __init__(self, plan: Plan, resources: ExecutionResources,
                 cold: Optional[ColdBackend], *,
                 cold_stream: bool = False,
                 force_graph_path: bool = False,
                 capture_mode_fn: Optional[Callable[[], bool]] = None,
                 grouped_min_m: Optional[int] = None,
                 split_streams: bool = False,
                 cold_gpu_min_m: Optional[int] = None,
                 cold_hybrid_frac=None, hybrid_local_node: int = 0,
                 warm_cpu_min_m: Optional[int] = None,
                 cold_async: bool = False):
        """cold_stream: eager에서도 cold submit/sync를 stream 통합으로 (opt-in).
        force_graph_path: 캡처 없이 graph-safe 경로 강제 (테스트/디버그).
        capture_mode_fn: sglang CudaGraphRunner의 capture 구간(캡처 전 워밍업
        포함) 신호 — 조립 지점이 주입한다.
        grouped_min_m: 이 M 이상이면 GPU 티어가 grouped GEMM을 탄다 (None =
        GROUPED_MIN_M). 0/None 대신 큰 값을 주면 worklist 강제 (벤치용).
        split_streams: grouped 경로에서 warm을 resources.warm_stream에 발행해
        hot(compute)과 warm(PCIe)을 겹친다.
        cold_gpu_min_m: 이 M 이상이면 cold를 CPU 대신 **GPU가 packed slab을 제자리
        읽어** 계산한다 (cold_gpu.py; warm 스트림 뒤에 이어 발행). None = 끔.
        cold backend가 gpu_view를 내지 않는 레이어는 값과 무관하게 CPU다.
        cold_hybrid_frac: cold GPU 조건(m ≥ cold_gpu_min_m)에서 cold를 **전부** GPU로
        보내는 대신 expert의 이 비율만 GPU가, 나머지는 CPU가 **동시에** 계산한다.
        float 하나 또는 (gateup, down) 둘 — 두 phase의 CPU/GPU 비용 비가 달라(down은
        CPU export 고정비가 커 GPU 몫이 더 커야 한다) 균형점이 phase마다 다르다.
        backend.hybrid_mask(kt skip 마스크)가 필요하고, eager·비-stream 호출에서만
        켠다 (host가 앞서 달리는 모드에선 마스크 쓰기가 kt 읽기와 경쟁한다)."""
        self._plan = plan
        self._res = resources
        self._cold = cold
        self._cold_stream = cold_stream
        self._grouped_min_m = (self.GROUPED_MIN_M if grouped_min_m is None
                               else int(grouped_min_m))
        self._split_streams = split_streams
        # None = 끔 (backend가 gpu view를 안 만든 경우와 동일). 기본값은 조립 지점이
        # COLD_GPU_MIN_M을 넘긴다 — 여기서 암묵 기본을 두면 view 없는 executor가
        # 조용히 CPU로 돌아 "켰는데 안 켜진" 상태가 된다.
        self._cold_gpu_min_m = cold_gpu_min_m
        self._layer_cold_gpu: dict[int, bool] = {}
        # warm-kt 모드에서 이 M 이상의 prefill은 warm 행을 GPU 대신 **CPU(warm-kt 인스턴스)**가
        # 계산한다 — hot=0이면 kt 네이티브와 같은 CPU FLOPs. None = 항상 GPU.
        self._warm_cpu_min_m = warm_cpu_min_m
        # cold_async (2026-08-27): cold 경로를 **전용 stream**에 두고 완료를 블로킹 sync
        # 콜백 대신 pinned 플래그 wait 커널로 받는다 — host 스레드도 CUDA 콜백 스레드도
        # 어디서도 블록되지 않아, hot/warm stream과 직렬화되지 않는다. eager 전용
        # (graph 캡처 경로는 기존 host node sync 유지). hybrid 마스크 즉시쓰기와는 공존 불가.
        # ⚠ cold_async는 미완이다 (2026-08-27): 첫 호출에서 partial H2D가 flag wait보다
        # 앞서 실행돼 결과가 틀리고, wait_flag 스핀 커널의 stream 오배치로 hang한다.
        # 기본 꺼짐. 켜지 말 것 — cuStreamWaitValue32 재구현 후 활성화 (TODO.md).
        self._cold_async = cold_async
        if cold_async:
            import warnings
            warnings.warn("SGLANG_PRISM_COLD_ASYNC is experimental and known-broken "
                          "(incorrect first-call output + possible GPU hang); do not use.",
                          RuntimeWarning)
        self._layer_warm_kt: dict[int, bool] = {}
        self._hybrid_frac = cold_hybrid_frac
        # phase별 (host u8 패턴 → kt 마스크에 복사, device bool → 제한 그룹핑)
        self._hybrid_masks: dict = {}
        if cold_hybrid_frac is not None:
            if cold is None or getattr(cold, "hybrid_masks", None) is None:
                raise ValueError("cold_hybrid_frac requires a cold backend built with hybrid_mask=True")
            # GPU-local 노드(plan shard 인덱스)만 expert 분할; 나머지 노드는 CPU 전부.
            self._hybrid_local_node = hybrid_local_node
            fr = (cold_hybrid_frac if isinstance(cold_hybrid_frac, (tuple, list))
                  else (cold_hybrid_frac, cold_hybrid_frac))
            self._hybrid_fracs = {"gu": float(fr[0]), "dn": float(fr[1])}
        self._force_graph_path = force_graph_path
        self._capture_mode_fn = capture_mode_fn or (lambda: False)
        self._layers: dict[int, PreparedWeights] = {}
        self._tiers: dict[int, LayerTiers] = {}
        self._layer_has_cold: dict[int, bool] = {}
        self._sparse = plan.sparsity is not None
        # cold task가 나중에 읽는 qlen — 주소 고정 멤버 (계약 ④의 포인터 경유)
        self._qlen_pin = torch.zeros(1, dtype=torch.int32)
        # graph 경로 qlen 버퍼는 **bs별로 격리**한다 (Finding A). 캡처가 baked하는
        # 포인터를 eager의 _qlen_pin과 공유하면 나중의 eager prefill 쓰기에
        # 노출돼 replay마다 cold가 L토큰짜리로 돌아 perf가 붕괴한다 (실측 30B
        # decode graph 328ms/tok vs eager 106ms/tok). bs가 다른 replay끼리도
        # 절대 공유하지 않는다.
        self._qlen_pins_graph: dict[int, torch.Tensor] = {}

    def register_layer(self, layer_idx: int, prepared: PreparedWeights) -> None:
        """Stage 2 산출물 등록. cold 행이 있으면 backend에 이미 load_layer된
        상태여야 한다 (로딩 순서는 method/loader의 책임)."""
        ep = self._plan.expert(layer_idx, 0)
        has_cold = any(ep.proj(p).has_tier(Tier.COLD) for p in Proj)
        if has_cold and self._cold is None:
            raise RuntimeError(f"layer {layer_idx} has COLD rows but no cold backend")
        self._layers[layer_idx] = prepared
        cold_gpu = None
        if has_cold and self._cold_gpu_min_m is not None:
            cold_gpu = getattr(self._cold, "gpu_view", lambda _i: None)(layer_idx)
            if cold_gpu is not None and self._hybrid_frac is not None:
                # hybrid: GPU는 GPU-local 노드 shard만 읽는다 (원격은 CPU 전부).
                from sglang.srt.layers.moe.prism.cold_gpu import ColdGpuLayer
                ln = self._hybrid_local_node
                cold_gpu = ColdGpuLayer(
                    gate=tuple(sl for sl in cold_gpu.gate if sl.node == ln),
                    up=tuple(sl for sl in cold_gpu.up if sl.node == ln),
                    down=tuple(sl for sl in cold_gpu.down if sl.node == ln))
        warm_kt = None
        warm_kt_calib = None
        if prepared.warm_kt is not None:
            warm_kt = getattr(self._cold, "warm_view", lambda _i: None)(layer_idx)
            if warm_kt is None:
                raise RuntimeError(f"layer {layer_idx}: warm_kt tensors but backend has no warm view "
                                   f"(load_warm_layer not called)")
            wk = prepared.warm_kt
            if wk.gate.calib is not None:
                warm_kt_calib = {Proj.GATE: wk.gate.calib, Proj.UP: wk.up.calib, Proj.DOWN: wk.down.calib}
        self._tiers[layer_idx] = build_layer_tiers(
            prepared, self._plan, layer_idx, cold_gpu=cold_gpu,
            warm_kt=warm_kt, warm_kt_calib=warm_kt_calib)
        self._layer_has_cold[layer_idx] = has_cold
        self._layer_cold_gpu[layer_idx] = cold_gpu is not None
        self._layer_warm_kt[layer_idx] = warm_kt is not None

    # ── 모드 결정 ──────────────────────────────────────────────────────────
    def _plan_flow(self, m: int, topk_ids: torch.Tensor) -> _LayerFlow:
        """호출별 실행 모드를 1회 확정. graph-safe 경로는 캡처 중 불법 연산
        (pageable D2H, tolist, event.synchronize, blocking copy)을 전부
        우회해야 하므로, 여기서 갈라진 결정이 본문의 유일한 분기 원천이다."""
        graph_flow = (
            torch.cuda.is_current_stream_capturing()
            or self._force_graph_path
            or self._capture_mode_fn()
        )
        # cold stream 통합: graph 경로는 필수(호출이 kt host node로 캡처돼야
        # 함), eager는 생성자 opt-in.
        use_cold_stream = graph_flow or self._cold_stream
        stream_arg = (
            torch.cuda.current_stream().cuda_stream if use_cold_stream else None
        )
        # graph **또는** stream 경로: cold의 expert_ids를 device→pinned **async** D2H로
        # 내리고 kt를 stream host node로 submit한다 (kt native와 동일). blocking
        # `topk_ids.to("cpu")`가 없어 host가 앞으로 달린다 — 그 blocking이 작은 M에서
        # 층당 4.7 ms의 노출된 파이프라인 대기로 찍혔다 (2026-08-27 nsys). qlen은
        # 매 스텝 host 쓰기(stream 순서 밖) 대신 **고정 pinned 상수 버퍼**를 쓴다.
        if graph_flow or self._cold_stream:
            qlen_pin = self._qlen_pins_graph.get(m)
            if qlen_pin is None:
                qlen_pin = torch.full((1,), m, dtype=torch.int32)
                self._qlen_pins_graph[m] = qlen_pin
            return _LayerFlow(
                graph_flow=graph_flow, use_cold_stream=True, stream_arg=stream_arg,
                qlen_ptr=qlen_pin.data_ptr(), ids_cpu=None,
            )
        # 순수 eager(스트림 통합 없음): cold가 expert_ids를 host에서 읽으므로 blocking
        # D2H 1회. 그 blocking이 뒤따르는 host 쓰기(qlen_pin)의 사실상 throttle이다
        # (계약 ④ Task8-3). cold_async는 자기 stream에서 async로 내린다.
        if self._cold_async:
            ids_cpu = None
        else:
            with _nvtx("s1.topk_d2h"):
                ids_cpu = topk_ids.to("cpu")
        return _LayerFlow(
            graph_flow=False, use_cold_stream=False,
            stream_arg=stream_arg, qlen_ptr=self._qlen_pin.data_ptr(),
            ids_cpu=ids_cpu,
        )

    # ── 본체 ──────────────────────────────────────────────────────────────
    def run_layer(self, layer_idx: int, hidden: torch.Tensor,
                  topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                  swiglu_limit: Optional[float] = None) -> torch.Tensor:
        """hidden [M, H] bf16 cuda, topk_ids [M, k] int64, topk_weights [M, k].
        swiglu_limit: 모델의 SwiGLU clamp (DSV4-Flash 10.0; None = 없음) — rejoin#1에서 적용.
        반환 [M, H] bf16 cuda (router 가중 expert 합 완료)."""
        has_cold = self._layer_has_cold[layer_idx]
        tiers = self._tiers[layer_idx]
        res, dims = self._res, self._plan.dims
        m, k = hidden.shape[0], topk_ids.shape[1]
        inter, h = dims.intermediate_size, dims.hidden_size

        _nvtx_push(f"prism.L{layer_idx}")
        flow = self._plan_flow(m, topk_ids)
        # sparsity는 cold만, decode에서만 (prefill-dense).
        masking = self._sparse and m == 1
        # prefill 형태: pair를 expert로 묶어 GPU 티어가 W를 expert당 한 번 읽는다.
        # 레이어당 1회 만들어 phase·티어가 공유한다 (topk가 같다).
        grouping: Optional[Grouping] = (
            build_grouping(topk_ids, dims.num_experts)
            if m >= self._grouped_min_m else None
        )
        # cold를 GPU가 읽는가 (대배치 prefill). 이 분기 하나가 "cold partial을
        # CPU와 GPU가 둘 다 내는" 이중계산을 막는다 — 아래 has_cold 블록 전부가
        # CPU 경로이고, GPU 경로는 _run_gateup/_run_down의 Tier.COLD 항목이다.
        cold_gpu = bool(
            has_cold and grouping is not None
            and self._cold_gpu_min_m is not None and m >= self._cold_gpu_min_m
            and self._layer_cold_gpu[layer_idx]
        )
        # hybrid: GPU는 마스크된 expert만(제한 그룹핑), CPU는 나머지(kt skip 마스크).
        # 두 partial은 (m, j) 행이 서로소이고 각자 남의 행을 0으로 두므로 합이 곧 전체다.
        hybrid = bool(cold_gpu and self._hybrid_frac is not None
                      and not flow.graph_flow and not flow.use_cold_stream
                      and not self._cold_async)   # 마스크 즉시쓰기가 host 선행과 충돌
        cold_grouping_gu = cold_grouping_dn = grouping
        if self._cold is not None and getattr(self._cold, "hybrid_masks", None) is not None:
            # host 즉시쓰기: 각 phase의 cold submit 직전에 그 phase의 패턴을 쓴다 —
            # 이전 phase/층의 kt 작업은 sync로 끝났다 (eager). graph/stream 모드는
            # 항상 0이라 쓰기가 값을 바꾸지 않는다. GPU-local 노드 마스크만 채우고
            # 원격 노드 마스크는 항상 0(그 shard는 CPU 전부).
            if hybrid:
                self._hybrid_masks = self._balance_hybrid(flow.ids_cpu, dims.num_experts,
                                                          hidden.device)
                self._cold.hybrid_masks[self._hybrid_local_node].copy_(self._hybrid_masks["gu"][0])
                cold_grouping_gu = build_grouping(topk_ids, dims.num_experts,
                                                  expert_mask=self._hybrid_masks["gu"][1])
                cold_grouping_dn = build_grouping(topk_ids, dims.num_experts,
                                                  expert_mask=self._hybrid_masks["dn"][1])
            else:
                for mk in self._cold.hybrid_masks:
                    mk.zero_()
        if cold_gpu and not hybrid:
            has_cold = False
        # warm-kt CPU 계산 (prefill): warm GPU 티어를 건너뛰고 warm-kt 인스턴스를 cold와 나란히
        # submit한다. eager·비-stream·grouped 호출에서만 (cold와 같은 이유).
        warm_cpu = bool(
            self._warm_cpu_min_m is not None and m >= self._warm_cpu_min_m
            and self._layer_warm_kt.get(layer_idx, False) and grouping is not None
            and not flow.graph_flow and not flow.use_cold_stream
        )
        # GPU sparse 티어와 rejoin이 같은 fp32 텐서를 쓴다 — 캐스팅이 두 번
        # 일어나면 스텝마다 할당이 하나 더 생긴다.
        w32 = topk_weights if topk_weights.dtype is torch.float32 \
            else topk_weights.to(torch.float32)
        cold_async = bool(self._cold_async and (has_cold or warm_cpu) and not flow.graph_flow
                          and not hybrid and res.cold_stream is not None)
        async_has_cold, async_warm_cpu = has_cold, warm_cpu
        if cold_async:
            has_cold = False   # 아래 host-block 경로를 전부 건너뛴다
            warm_cpu_legacy = False
        else:
            warm_cpu_legacy = warm_cpu

        # ── Phase 1: gateup ──────────────────────────────────────────────
        w_ptr = 0
        cold_gu = warm_gu = None
        if cold_async:
            cold_gu, warm_gu = self._cold_phase_async(
                layer_idx, "gateup", hidden, topk_ids, w32, m, k, masking,
                async_has_cold, async_warm_cpu)
        if has_cold or warm_cpu_legacy:
            with _nvtx("cold.gu.fill_x(D2H-block)"):
                # stream 통합 시 non_blocking: kt host node가 같은 stream에
                # 순서대로 실행되므로 host-측 완료 보장이 불필요하다.
                res.staging.fill_x(hidden, non_blocking=flow.use_cold_stream)
                if flow.use_cold_stream:
                    # device topk_ids → pinned int64 async D2H (graph·stream 공통).
                    # qlen은 고정 pinned 버퍼라 여기서 쓰지 않는다.
                    res.staging.fill_expert_ids(topk_ids, non_blocking=True)
                    pin = self._qlen_pins_graph[m]
                    assert int(pin[0]) == m, (
                        f"qlen_pin_graph[{m}]={int(pin[0])} != {m} — "
                        f"stream/graph path must never write this buffer"
                    )
                else:
                    res.staging.fill_expert_ids(flow.ids_cpu)
                    self._qlen_pin[0] = m
            if masking:
                with _nvtx("cold.fill_topk_w"):
                    res.staging.fill_topk_w(
                        topk_weights, non_blocking=flow.use_cold_stream)
                w_ptr = res.staging.topk_w_ptr()
            if has_cold:
                with _nvtx("cold.gu.submit"):
                    self._cold.submit_gateup(         # enqueue-only, 즉시 반환
                        layer_idx, flow.qlen_ptr, k,
                        res.staging.expert_ids_ptr(), res.staging.x_ptr(),
                        res.staging.partial_gateup_ptr(),
                        cuda_stream=flow.stream_arg,
                        weights_ptr=w_ptr,
                    )
            if warm_cpu_legacy:
                with _nvtx("warm.gu.submit"):
                    self._cold.submit_warm_gateup(
                        layer_idx, flow.qlen_ptr, k,
                        res.staging.expert_ids_ptr(), res.staging.x_ptr(),
                        res.staging.warm_partial_gateup_ptr(),
                        cuda_stream=flow.stream_arg, weights_ptr=0,
                    )
            _nvtx_push("cold.gu.window")          # CPU expert 연산 재실 구간

        gu_parts = self._run_gateup(tiers, hidden, topk_ids, w32, m, k, inter,
                                    masking, grouping, cold_gpu, cold_grouping_gu, hybrid,
                                    skip_warm=warm_cpu)
        if cold_async:
            torch.cuda.current_stream().wait_stream(res.cold_stream)   # partial H2D 완료

        if has_cold or warm_cpu_legacy:
            with _nvtx("cold.gu.sync(host-block)"):
                self._cold.sync(cuda_stream=flow.stream_arg)   # 두 인스턴스 모두 (같은 풀)
            _nvtx_pop()                           # cold.gu.window
            with _nvtx("cold.gu.h2d_out"):
                if has_cold:
                    cold_gu = res.staging.gateup_out(m).to(
                        hidden.device, non_blocking=flow.use_cold_stream)
                if warm_cpu_legacy:
                    warm_gu = res.staging.warm_gateup_out(m).to(
                        hidden.device, non_blocking=flow.use_cold_stream)

        # rejoin#1: fp32 합 → act → bf16 (계약 ⑤)
        # 융합 커널 한 launch: Σ partial(fp32) → silu·up → bf16 (rejoin.py). torch
        # 사슬(캐스팅 3 + add 2 + split/silu/mul/cast)은 prefill에서 88 MB 텐서를
        # ~10번 왕복해 층당 2.5 ms였다 (2026-08-27 nsys).
        with _nvtx("rejoin1.acc+silu"):
            act = rejoin_gateup(gu_parts + [cold_gu, warm_gu], inter, swiglu_limit)

        # ── Phase 2: down ────────────────────────────────────────────────
        if hybrid:
            self._cold.hybrid_masks[self._hybrid_local_node].copy_(self._hybrid_masks["dn"][0])  # down의 GPU 몫
        cold_down = warm_down = None
        if cold_async:
            cold_down, warm_down = self._cold_phase_async(
                layer_idx, "down", act, topk_ids, w32, m, k, masking,
                async_has_cold, async_warm_cpu)
        if has_cold or warm_cpu_legacy:
            with _nvtx("cold.dn.fill_act(D2H-block)"):
                res.staging.fill_act(act, non_blocking=flow.use_cold_stream)
            if has_cold:
                with _nvtx("cold.dn.submit"):
                    self._cold.submit_down(
                        layer_idx, flow.qlen_ptr, k,
                        res.staging.expert_ids_ptr(), res.staging.act_ptr(),
                        res.staging.partial_down_ptr(),
                        cuda_stream=flow.stream_arg,
                        weights_ptr=w_ptr,
                    )
            if warm_cpu_legacy:
                with _nvtx("warm.dn.submit"):
                    self._cold.submit_warm_down(
                        layer_idx, flow.qlen_ptr, k,
                        res.staging.expert_ids_ptr(), res.staging.act_ptr(),
                        res.staging.warm_partial_down_ptr(),
                        cuda_stream=flow.stream_arg, weights_ptr=0,
                    )
            _nvtx_push("cold.dn.window")

        down_parts = self._run_down(tiers, act, topk_ids, w32, m, k, inter, h,
                                    masking, grouping, cold_gpu, cold_grouping_dn, hybrid,
                                    skip_warm=warm_cpu)
        if cold_async:
            torch.cuda.current_stream().wait_stream(res.cold_stream)

        if has_cold or warm_cpu_legacy:
            with _nvtx("cold.dn.sync(host-block)"):
                self._cold.sync(cuda_stream=flow.stream_arg)
            _nvtx_pop()                           # cold.dn.window
            with _nvtx("cold.dn.h2d_out"):
                if has_cold:
                    cold_down = res.staging.down_out(m).to(
                        hidden.device, non_blocking=flow.use_cold_stream)
                if warm_cpu_legacy:
                    warm_down = res.staging.warm_down_out(m).to(
                        hidden.device, non_blocking=flow.use_cold_stream)

        # rejoin#2: fp32 합 → router 가중 expert합 → bf16
        with _nvtx("rejoin2.acc+wsum"):
            out = rejoin_down(down_parts + [cold_down, warm_down], w32)
        _nvtx_pop()                               # prism.L{layer_idx}
        return out

    def _balance_hybrid(self, ids_cpu: torch.Tensor, num_experts: int, device) -> dict:
        """phase별 GPU expert 마스크 — **비용 모델**로 나눈다.

        GPU cold 비용은 expert 개수에 비례한다 (expert당 W 바이트를 PCIe로 1회, pair 수
        무관). CPU 비용은 pair 수에 비례한다 (FLOPs; AMX는 expert당 M이 클수록 효율).
        따라서 pair가 적은 expert(꼬리)를 GPU에, 몰린 expert를 CPU에 주는 것이 맞다 —
        반대로 하면 CPU가 작은 M의 expert 수백 개를 돌며 병목이 된다 (엔진 실측 8064tok
        4.0 → 7.3 s). 비율 상수 c/g는 균등 라우팅에서 실측한 균형 frac f로부터:
        f·E·g = (1−f)·P·c ⇒ c/g = f·E / ((1−f)·P). 실제 분포에서는 count 오름차순으로
        GPU 집합 S를 늘려 |S|·g ≥ c·Σ_{e∉S} count_e 가 되는 최소 |S|를 고른다.
        """
        counts = torch.bincount(ids_cpu.reshape(-1), minlength=num_experts)
        total = int(counts.sum())
        active = counts > 0
        n_active = int(active.sum())
        order = torch.argsort(counts, descending=False)  # 0-count 먼저, 그다음 꼬리부터
        cum = torch.cumsum(counts[order], 0)
        out = {}
        for phase, f in self._hybrid_fracs.items():
            f = min(max(f, 1e-3), 1 - 1e-3)
            # c/g는 **하드웨어 상수**다 — frac f가 실측된 보정점(균등 라우팅, E expert,
            # HYBRID_CALIB_PAIRS pair)에서 한 번 환산하고 현재 분포에는 그대로 적용한다.
            # 현재 n_active/total로 환산하면 M이 작을 때 ratio가 커져 거의 전부를 GPU로
            # 보내는 오류가 난다 (2026-08-27 small-M 실측: hybrid가 CPU보다 느렸다).
            ratio = f * num_experts / ((1.0 - f) * self.HYBRID_CALIB_PAIRS)
            n_zero = num_experts - n_active
            # n = GPU가 맡는 active expert 수. 조건: n ≥ ratio·(total − cum_active(n))
            n_gpu = n_active
            for n in range(0, n_active + 1):
                covered = int(cum[n_zero + n - 1]) if n > 0 else 0
                if n >= ratio * (total - covered):
                    n_gpu = n
                    break
            pattern = torch.zeros(num_experts, dtype=torch.uint8)
            pattern[order[: n_zero + n_gpu]] = 1
            out[phase] = (pattern, pattern.bool().to(device))
        return out

    # ── cold_async ──────────────────────────────────────────────────────
    def _cold_phase_async(self, layer_idx, phase, x_or_act, topk_ids, w32, m, k, masking,
                          has_cold: bool, warm_cpu: bool):
        """cold(+warm-kt) 한 phase를 cold stream에 **enqueue만** 하고 partial(device)을
        돌려준다. 순서: [입력 D2H] → [submit host node] → [signal host node] →
        [wait_flag 커널] → [partial H2D]. 호출자는 rejoin 전에 current stream이
        cold stream을 기다리게 한다. host는 어디서도 블록되지 않는다."""
        res, cs = self._res, self._res.cold_stream
        main = torch.cuda.current_stream()
        cs.wait_stream(main)  # 입력(hidden/act, topk) 준비 완료
        qlen_pin = self._qlen_pins_graph.get(m)
        if qlen_pin is None:
            qlen_pin = torch.full((1,), m, dtype=torch.int32)
            self._qlen_pins_graph[m] = qlen_pin
        with torch.cuda.stream(cs), _nvtx(f"cold.{phase}.async"):
            st = res.staging
            if phase == "gateup":
                st.fill_x(x_or_act, non_blocking=True)
                st.fill_expert_ids(topk_ids, non_blocking=True)   # device → pinned async
                if masking:
                    st.fill_topk_w(w32, non_blocking=True)
            else:
                st.fill_act(x_or_act, non_blocking=True)
            w_ptr = st.topk_w_ptr() if masking else 0
            sarg = cs.cuda_stream
            if has_cold:
                if phase == "gateup":
                    self._cold.submit_gateup(layer_idx, qlen_pin.data_ptr(), k, st.expert_ids_ptr(),
                                             st.x_ptr(), st.partial_gateup_ptr(),
                                             cuda_stream=sarg, weights_ptr=w_ptr)
                else:
                    self._cold.submit_down(layer_idx, qlen_pin.data_ptr(), k, st.expert_ids_ptr(),
                                           st.act_ptr(), st.partial_down_ptr(),
                                           cuda_stream=sarg, weights_ptr=w_ptr)
            if warm_cpu:
                if phase == "gateup":
                    self._cold.submit_warm_gateup(layer_idx, qlen_pin.data_ptr(), k, st.expert_ids_ptr(),
                                                  st.x_ptr(), st.warm_partial_gateup_ptr(),
                                                  cuda_stream=sarg, weights_ptr=0)
                else:
                    self._cold.submit_warm_down(layer_idx, qlen_pin.data_ptr(), k, st.expert_ids_ptr(),
                                                st.act_ptr(), st.warm_partial_down_ptr(),
                                                cuda_stream=sarg, weights_ptr=0)
            res.cold_seq += 1
            self._cold.submit_signal(sarg, res.cold_flag.data_ptr(), res.cold_seq)
            from sglang.jit_kernel.prism_grouped import wait_flag
            wait_flag(res.cold_flag, res.cold_seq, cs)
            dev = x_or_act.device
            if phase == "gateup":
                cold_p = st.gateup_out(m).to(dev, non_blocking=True) if has_cold else None
                warm_p = st.warm_gateup_out(m).to(dev, non_blocking=True) if warm_cpu else None
            else:
                cold_p = st.down_out(m).to(dev, non_blocking=True) if has_cold else None
                warm_p = st.warm_down_out(m).to(dev, non_blocking=True) if warm_cpu else None
        return cold_p, warm_p

    # ── GPU 티어 ─────────────────────────────────────────────────────────
    def _warm_stream(self, grouping) -> Optional[torch.cuda.Stream]:
        """hot ∥ warm 스트림 분리를 이 호출에 적용할 것인가.

        grouped(prefill) 경로에서만이다: hot은 compute(tensor core), warm은
        PCIe 대역폭에 각각 묶여 있어 겹치면 둘의 합이 아니라 max가 된다.
        decode(worklist)는 graph 캡처 경로에 cold host node가 current stream에
        얽혀 있어 여기서 건드리지 않는다.
        """
        ws = self._res.warm_stream
        if not (self._split_streams and grouping is not None and ws is not None):
            return None
        ws.wait_stream(torch.cuda.current_stream())  # x·topk·grouping 준비 완료
        return ws

    @staticmethod
    def _tier_order(ws, cold_gpu: bool) -> tuple:
        # 분리 시 PCIe 소비자(warm, 그 뒤에 cold)를 먼저 발행한다 — 임계경로라
        # 먼저 시작해야 하고, cold는 warm 스트림 **뒤에 이어** 붙는다.
        pcie = (Tier.WARM, Tier.COLD) if cold_gpu else (Tier.WARM,)
        return pcie + (Tier.HOT,) if ws is not None else (Tier.HOT,) + pcie

    @staticmethod
    def _on_side(tier: Tier) -> bool:
        return tier in (Tier.WARM, Tier.COLD)

    @staticmethod
    def _join(ws) -> None:
        if ws is not None:
            torch.cuda.current_stream().wait_stream(ws)

    def _run_gateup(self, tiers, hidden, topk_ids, topk_weights, m, k, inter,
                    masking, grouping=None, cold_gpu: bool = False,
                    cold_grouping=None, cold_partial: bool = False,
                    skip_warm: bool = False) -> list:
        """티어별로 [M, k, 2·inter] partial 하나. gate가 앞 절반, up이 뒤 절반.

        티어마다 버퍼가 따로인 이유: 셋 다 **같은 열**에 쓰므로(계약 ②의
        overwrite 의미론) 한 버퍼를 나눠 쓸 수 없고, 티어 간 합산은 rejoin의
        fp32 누산 1회다 (계약 ⑤).

        버퍼는 current(main) stream에서 할당하고 warm은 side stream에서 쓴다 —
        소비(rejoin)와 해제가 join 뒤 main에서 일어나므로 할당자 재사용이
        side의 쓰기를 앞지를 수 없다.
        """
        ws = self._warm_stream(grouping)
        parts = []
        for tier in self._tier_order(ws, cold_gpu):
            gu = tiers.gateup.get(tier)
            if gu is None or (skip_warm and tier is Tier.WARM):
                continue
            # 두 절반을 다 덮는 구현이면 empty로 족하다 (계약 ② overwrite). hybrid의
            # cold GPU는 자기 expert 행만 쓰므로 나머지 행이 0이어야 한다.
            is_cold = tier is Tier.COLD
            alloc = torch.zeros if (not gu.writes_all or (is_cold and cold_partial)) else torch.empty
            buf = alloc(m, k, 2 * inter, dtype=torch.bfloat16, device=hidden.device)
            ctx = torch.cuda.stream(ws) if (ws is not None and self._on_side(tier)) \
                else nullcontext()
            with ctx, _nvtx(f"{tier.value}.gu"):
                gu.run(hidden, topk_ids, topk_weights, buf, inter, masking=masking,
                       grouping=(cold_grouping if is_cold and cold_grouping is not None else grouping))
            parts.append(buf)
        self._join(ws)
        return parts

    def _run_down(self, tiers, act, topk_ids, topk_weights, m, k, inter, h,
                  masking, grouping=None, cold_gpu: bool = False,
                  cold_grouping=None, cold_partial: bool = False,
                  skip_warm: bool = False) -> list:
        """act는 expert별 값이라 행이 pair (m, j)다 — x_row_is_pair=True."""
        act2d = act.reshape(m * k, inter)
        ws = self._warm_stream(grouping)
        parts = []
        for tier in self._tier_order(ws, cold_gpu):
            td = tiers.down.get(tier)
            if td is None or (skip_warm and tier is Tier.WARM):
                continue
            is_cold = tier is Tier.COLD
            alloc = torch.zeros if (is_cold and cold_partial) else torch.empty
            buf = alloc(m, k, h, dtype=torch.bfloat16, device=act.device)
            ctx = torch.cuda.stream(ws) if (ws is not None and self._on_side(tier)) \
                else nullcontext()
            with ctx, _nvtx(f"{tier.value}.dn"):
                td.run(act2d, topk_ids, topk_weights, buf, 0, x_row_is_pair=True,
                       masking=masking,
                       grouping=(cold_grouping if is_cold and cold_grouping is not None else grouping))
            parts.append(buf)
        self._join(ws)
        return parts
