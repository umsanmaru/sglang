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
from contextlib import contextmanager
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
from sglang.srt.layers.moe.prism.plan import Plan, Proj, Tier
from sglang.srt.layers.moe.prism.resources import ExecutionResources
from sglang.srt.layers.moe.prism.tiers import GpuTier, build_layer_tiers
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

    def __init__(self, plan: Plan, resources: ExecutionResources,
                 cold: Optional[ColdBackend], *,
                 cold_stream: bool = False,
                 force_graph_path: bool = False,
                 capture_mode_fn: Optional[Callable[[], bool]] = None):
        """cold_stream: eager에서도 cold submit/sync를 stream 통합으로 (opt-in).
        force_graph_path: 캡처 없이 graph-safe 경로 강제 (테스트/디버그).
        capture_mode_fn: sglang CudaGraphRunner의 capture 구간(캡처 전 워밍업
        포함) 신호 — 조립 지점이 주입한다."""
        self._plan = plan
        self._res = resources
        self._cold = cold
        self._cold_stream = cold_stream
        self._force_graph_path = force_graph_path
        self._capture_mode_fn = capture_mode_fn or (lambda: False)
        self._layers: dict[int, PreparedWeights] = {}
        self._tiers: dict[int, Mapping[tuple, GpuTier]] = {}
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
        self._tiers[layer_idx] = build_layer_tiers(prepared, self._plan, layer_idx)
        self._layer_has_cold[layer_idx] = has_cold

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
        if graph_flow:
            qlen_pin = self._qlen_pins_graph.get(m)
            if qlen_pin is None:
                # 캡처 전 워밍업에서 생성된다 — 실제 캡처 중 신규 bs가 나타나면
                # 이 할당은 캡처 밖 host alloc이라 안전하지만, sglang 워밍업
                # 순서상 도달하지 않는 것이 정상.
                qlen_pin = torch.full((1,), m, dtype=torch.int32)
                self._qlen_pins_graph[m] = qlen_pin
            return _LayerFlow(
                graph_flow=True, use_cold_stream=True, stream_arg=stream_arg,
                qlen_ptr=qlen_pin.data_ptr(), ids_cpu=None,
            )
        # eager: cold가 expert_ids를 host에서 읽으므로 D2H 1회.
        with _nvtx("s1.topk_d2h"):
            ids_cpu = topk_ids.to("cpu")
        return _LayerFlow(
            graph_flow=False, use_cold_stream=use_cold_stream,
            stream_arg=stream_arg, qlen_ptr=self._qlen_pin.data_ptr(),
            ids_cpu=ids_cpu,
        )

    # ── 본체 ──────────────────────────────────────────────────────────────
    def run_layer(self, layer_idx: int, hidden: torch.Tensor,
                  topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        """hidden [M, H] bf16 cuda, topk_ids [M, k] int64, topk_weights [M, k].
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
        # GPU sparse 티어와 rejoin이 같은 fp32 텐서를 쓴다 — 캐스팅이 두 번
        # 일어나면 스텝마다 할당이 하나 더 생긴다.
        w32 = topk_weights if topk_weights.dtype is torch.float32 \
            else topk_weights.to(torch.float32)

        # ── Phase 1: gateup ──────────────────────────────────────────────
        w_ptr = 0
        if has_cold:
            with _nvtx("cold.gu.fill_x(D2H-block)"):
                # stream 통합 시 non_blocking: kt host node가 같은 stream에
                # 순서대로 실행되므로 host-측 완료 보장이 불필요하다.
                res.staging.fill_x(hidden, non_blocking=flow.use_cold_stream)
                if flow.graph_flow:
                    # device topk_ids → pinned int64 async D2H (캡처 가능).
                    res.staging.fill_expert_ids(topk_ids, non_blocking=True)
                    pin = self._qlen_pins_graph[m]
                    assert int(pin[0]) == m, (
                        f"qlen_pin_graph[{m}]={int(pin[0])} != {m} — "
                        f"graph path must never write this buffer"
                    )
                else:
                    res.staging.fill_expert_ids(flow.ids_cpu)
                    self._qlen_pin[0] = m
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

        gu_parts = self._run_gateup(tiers, hidden, topk_ids, w32, m, k, inter,
                                    masking)

        cold_gu = None
        if has_cold:
            with _nvtx("cold.gu.sync(host-block)"):
                self._cold.sync(cuda_stream=flow.stream_arg)
            _nvtx_pop()                           # cold.gu.window
            with _nvtx("cold.gu.h2d_out"):
                cold_gu = res.staging.gateup_out(m).to(
                    hidden.device, non_blocking=flow.use_cold_stream
                )

        # rejoin#1: fp32 합 → act → bf16 (계약 ⑤)
        with _nvtx("rejoin1.acc+silu"):
            gu = self._accumulate(gu_parts + [cold_gu], (m, k, 2 * inter),
                                  hidden.device)
            gate, up = gu.split(inter, dim=2)
            act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)

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

        down_parts = self._run_down(tiers, act, topk_ids, w32, m, k, inter, h,
                                    masking)

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
            down = self._accumulate(down_parts + [cold_down], (m, k, h),
                                    hidden.device)
            out = (down * w32.unsqueeze(-1)).sum(dim=1)
        _nvtx_pop()                               # prism.L{layer_idx}
        return out.to(torch.bfloat16)

    # ── GPU 티어 ─────────────────────────────────────────────────────────
    @staticmethod
    def _run_gateup(tiers, hidden, topk_ids, topk_weights, m, k, inter,
                    masking) -> list:
        """티어별로 [M, k, 2·inter] partial 하나. gate가 앞 절반, up이 뒤 절반.

        티어마다 버퍼가 따로인 이유: 셋 다 **같은 열**에 쓰므로(계약 ②의
        overwrite 의미론) 한 버퍼를 나눠 쓸 수 없고, 티어 간 합산은 rejoin의
        fp32 누산 1회다 (계약 ⑤).
        """
        parts = []
        for tier in (Tier.HOT, Tier.WARM):
            tg, tu = tiers.get((Proj.GATE, tier)), tiers.get((Proj.UP, tier))
            if tg is None and tu is None:
                continue
            # 둘 다 있으면 두 호출이 [:, :, :inter]/[inter:]를 완전히 덮는다.
            alloc = torch.empty if (tg is not None and tu is not None) else torch.zeros
            buf = alloc(m, k, 2 * inter, dtype=torch.bfloat16, device=hidden.device)
            with _nvtx(f"{tier.value}.gu"):
                if tg is not None:
                    tg.run(hidden, topk_ids, topk_weights, buf, 0,
                           x_row_is_pair=False, masking=masking)
                if tu is not None:
                    tu.run(hidden, topk_ids, topk_weights, buf, inter,
                           x_row_is_pair=False, masking=masking)
            parts.append(buf)
        return parts

    @staticmethod
    def _run_down(tiers, act, topk_ids, topk_weights, m, k, inter, h,
                  masking) -> list:
        """act는 expert별 값이라 행이 pair (m, j)다 — x_row_is_pair=True."""
        act2d = act.reshape(m * k, inter)
        parts = []
        for tier in (Tier.HOT, Tier.WARM):
            td = tiers.get((Proj.DOWN, tier))
            if td is None:
                continue
            buf = torch.empty(m, k, h, dtype=torch.bfloat16, device=act.device)
            with _nvtx(f"{tier.value}.dn"):
                td.run(act2d, topk_ids, topk_weights, buf, 0,
                       x_row_is_pair=True, masking=masking)
            parts.append(buf)
        return parts

    @staticmethod
    def _accumulate(parts, shape, device) -> torch.Tensor:
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
