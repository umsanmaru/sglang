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

읽는 순서: `CpuDispatch`(CPU가 cold/warm을 어떻게 조달하는가) → `_LayerFlow`
(한 호출의 실행 계획) → `_plan_flow`(그 계획을 세우는 곳) → `run_layer`(두
phase가 같은 네 걸음을 밟는 본체) → `_cpu_partials_begin/end`(dispatch별 구현).
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

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
from sglang.srt.layers.moe.prism.activation import Activation
from sglang.srt.layers.moe.prism.rejoin import rejoin_down, rejoin_gateup
from sglang.srt.layers.moe.prism.resources import ExecutionResources
from sglang.srt.layers.moe.prism.tiers import LayerTiers, build_layer_tiers
from sglang.srt.layers.moe.prism.weights import PreparedWeights


class CpuDispatch(Enum):
    """한 호출에서 **CPU가 cold/warm partial을 어떻게 조달하는가** — 이 값 하나가
    run_layer의 유일한 CPU-측 분기다.

    NONE 이외의 셋은 "언제 누가 블록되는가"만 다르다. 셋 모두 같은 두 primitive
    (submit → sync, 계약 ④)를 쓰고 결과 partial의 의미도 같다.
    """

    #: CPU 작업 없음 — cold 행이 없거나 GPU가 cold slab을 통째로 읽는다(cold_on_gpu).
    NONE = "none"
    #: submit → GPU 티어 발행 → 호출 스레드가 블로킹 sync. 기본 경로.
    HOST_BLOCK = "host_block"
    #: submit/sync host node를 **곁 스트림**(resources.cold_stream)에 얹는다. sync가
    #: main의 hot/warm 커널을 기다리지 않는다 (cold 전용 — warm/hybrid와 공존 불가).
    SIDE_STREAM = "side_stream"
    #: 전부 cold stream에 enqueue하고 pinned flag wait 커널로 합류 — host가 어디서도
    #: 블록되지 않는다. ⚠ 실험적·미완 (PrismExecutor.__init__의 경고 참조).
    ASYNC_FLAG = "async_flag"


#: run_layer의 두 phase 이름. staging 버퍼와 submit 진입점을 고르는 키다.
GATEUP = "gateup"
DOWN = "down"


@dataclass(frozen=True, slots=True)
class _ColdChannel:
    """cold 호출을 **어느 통로로** 내보내는가 — graph 캡처 가능성이 정하는 부분.

    graph-safe 경로는 캡처 중 불법 연산(pageable D2H, tolist, event.synchronize,
    blocking copy)을 전부 우회해야 하므로, 여기서 갈린 결정이 아래 모든 cold
    호출의 인자(cuda_stream, qlen_ptr, expert_ids 원천)를 결정한다.
    """

    graph_flow: bool                  # graph-safe 경로인가 (캡처/워밍업/강제)
    use_cold_stream: bool             # cold 호출을 stream host node로 보낼 것인가
    stream_arg: Optional[int]         # cold submit/sync의 cudaStream_t (None = host 경로)
    qlen_ptr: int                     # cold task가 역참조할 qlen 버퍼 주소
    ids_cpu: Optional[torch.Tensor]   # eager 경로 expert_ids 원천 / graph는 None


@dataclass(frozen=True, slots=True)
class _LayerFlow:
    """run_layer 한 호출의 실행 계획 — `_plan_flow()`가 상단에서 **1회** 확정하는
    값 객체. 본문에는 여기서 정해진 것 말고 다른 분기가 없다.

    네 묶음이다: ① cold 조달 통로, ② 이 호출의 형태, ③ 티어 배치(cold/warm을
    CPU와 GPU에 어떻게 나눌 것인가), ④ 관측.
    """

    # ① 통로 (_ColdChannel에서 평탄화 — 본문에서 가장 자주 읽힌다)
    graph_flow: bool
    use_cold_stream: bool
    stream_arg: Optional[int]
    qlen_ptr: int
    ids_cpu: Optional[torch.Tensor]
    # ② 이 호출의 형태
    m: int                                    # 토큰 수 (decode=1)
    k: int                                    # top_k
    masking: bool                             # cold sparsity 적용 (decode 전용, 계약 ①)
    topk_w32: torch.Tensor                    # 라우터 가중 fp32 — 커널과 rejoin이 공유
    grouping: Optional[Grouping]              # prefill 형태(expert별 묶음) / None = worklist
    # ③ 티어 배치
    cold_on_gpu: bool                         # GPU 티어가 cold slab을 제자리 읽는가
    hybrid: bool                              # cold를 GPU(expert 일부) ∥ CPU(나머지)로 쪼개는가
    cpu_dispatch: CpuDispatch                 # CPU 조달 방식 — 본문의 유일한 CPU 분기
    cpu_cold: bool                            # CPU가 cold partial을 계산하는가
    cpu_warm: bool                            # CPU(kt warm 인스턴스)가 warm partial을 계산하는가
    skip_gpu_warm: bool                       # GPU warm 티어를 건너뛰는가 (CPU가 대신 낸다)
    cold_grouping_gateup: Optional[Grouping]  # hybrid에서 GPU가 맡은 expert로 제한된 그룹핑
    cold_grouping_down: Optional[Grouping]
    # ④ 관측 (mask_stats; graph 경로에선 파이썬이 replay되지 않아 항상 꺼진다)
    probe_on: bool                            # NaN 진단 프로브 (prefill 포함)
    stats_on: bool                            # effective sparsity 실측 (decode·eager)


@dataclass(frozen=True, slots=True)
class _PhasePorts:
    """한 phase(gateup/down)의 staging 버퍼와 cold submit 진입점 묶음.

    gateup과 down은 **글자만 다른 같은 절차**다(입력을 내리고 submit하고 partial을
    올린다). 그 차이를 여기 한 번 적어두면 아래 세 dispatch 경로가 phase를 몰라도
    된다. staging도 cold backend도 생성 뒤 바뀌지 않으므로 executor당 한 벌만 만든다.
    """

    name: str                                     # "gateup" | "down"
    tag: str                                      # nvtx 약칭 ("gu" | "dn")
    is_gateup: bool
    fill_label: str                               # nvtx용 입력 D2H 이름
    fill_inputs: Callable[..., None]              # hidden→_x(+expert_ids) / act→_act
    input_ptr: Callable[[], int]
    submit_cold_name: str                         # cold backend 진입점 이름 — 늦게 꺾는다
    submit_warm_name: str
    cold_partial_ptr: Callable[[], int]
    warm_partial_ptr: Callable[[], int]
    cold_out: Callable[[int], torch.Tensor]
    warm_out: Callable[[int], torch.Tensor]


def _build_phase_ports(staging) -> dict:
    """phase 이름 → `_PhasePorts`."""

    def fill_gateup(x, expert_ids, non_blocking: bool) -> None:
        staging.fill_x(x, non_blocking=non_blocking)
        staging.fill_expert_ids(expert_ids, non_blocking=non_blocking)

    def fill_down(act, _expert_ids, non_blocking: bool) -> None:
        # expert_ids는 gateup에서 이미 내려갔고 두 phase가 같은 버퍼를 읽는다.
        staging.fill_act(act, non_blocking=non_blocking)

    return {
        GATEUP: _PhasePorts(
            name=GATEUP, tag="gu", is_gateup=True, fill_label="fill_x",
            fill_inputs=fill_gateup, input_ptr=staging.x_ptr,
            submit_cold_name="submit_gateup", submit_warm_name="submit_warm_gateup",
            cold_partial_ptr=staging.partial_gateup_ptr,
            warm_partial_ptr=staging.warm_partial_gateup_ptr,
            cold_out=staging.gateup_out, warm_out=staging.warm_gateup_out,
        ),
        DOWN: _PhasePorts(
            name=DOWN, tag="dn", is_gateup=False, fill_label="fill_act",
            fill_inputs=fill_down, input_ptr=staging.act_ptr,
            submit_cold_name="submit_down", submit_warm_name="submit_warm_down",
            cold_partial_ptr=staging.partial_down_ptr,
            warm_partial_ptr=staging.warm_partial_down_ptr,
            cold_out=staging.down_out, warm_out=staging.warm_down_out,
        ),
    }


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
                 warm_cpu_max_m: Optional[int] = None,
                 cold_async: bool = False,
                 cold_split: bool = False,
                 mask_stats=None):
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
        # warm_cpu_max_m: 1 < M < max is CPU prefill (exclusive upper bound).
        # warm_cpu_min_m retains the legacy explicit large-M CPU override.
        self._warm_cpu_min_m = warm_cpu_min_m
        self._warm_cpu_max_m = warm_cpu_max_m
        # cold_async (2026-08-27): cold 경로를 **전용 stream**에 두고 완료를 블로킹 sync
        # 콜백 대신 pinned 플래그 wait 커널로 받는다 — host 스레드도 CUDA 콜백 스레드도
        # 어디서도 블록되지 않아, hot/warm stream과 직렬화되지 않는다. eager 전용
        # (graph 캡처 경로는 기존 host node sync 유지). hybrid 마스크 즉시쓰기와는 공존 불가.
        # ⚠ cold_async는 미완이다 (2026-08-27): 첫 호출에서 partial H2D가 flag wait보다
        # 앞서 실행돼 결과가 틀리고, wait_flag 스핀 커널의 stream 오배치로 hang한다.
        # 기본 꺼짐. 켜지 말 것 — cuStreamWaitValue32 재구현 후 활성화 (TODO.md).
        # cold_split (2026-08-31): graph/stream 경로에서 cold의 submit/sync host node를
        # **곁 스트림**(res.cold_stream)에 얹는다. 신호 방식은 그대로 두고 stream만
        # 가른다 — cold_async와 달리 플래그도 wait 커널도 쓰지 않으므로 그 미완 경로에
        # 기대지 않는다.
        #
        # 왜: 지금은 sync host node가 main stream에 있어 **hot/warm 커널 완료까지**
        # 기다린다. sync가 실제로 기다려야 하는 것은 CPU 완료뿐이라 그 의존은 가짜다.
        # 프로파일 ④ 실측(E=128, top_k=8, fp8): down phase가 71.4 → 49.5 µs (−31%),
        # GPU 39 µs가 통째로 숨는다. gateup은 이미 cold가 훨씬 길어 변화 없음(−0.06 µs).
        # 4라운드 반복에서 재현됨.
        self._cold_split = cold_split
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
        # effective sparsity 실측기 (mask_stats.MaskStats; None = 끔). decode·eager에서만 관측.
        self._mask_stats = mask_stats
        # cold task가 나중에 읽는 qlen — 주소 고정 멤버 (계약 ④의 포인터 경유)
        self._qlen_pin = torch.zeros(1, dtype=torch.int32)
        # graph/stream 경로의 bs별 qlen 버퍼 — 격리 이유는 `_qlen_pin_for` 참조.
        self._qlen_pins_graph: dict[int, torch.Tensor] = {}
        # phase(gateup/down)별 staging·submit 진입점 (CPU 경로 전용 어댑터).
        self._ports = _build_phase_ports(resources.staging)

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
        if self._mask_stats is not None:
            self._mask_stats.register_layer(layer_idx)
        self._layer_cold_gpu[layer_idx] = cold_gpu is not None
        self._layer_warm_kt[layer_idx] = warm_kt is not None

    # ── 모드 결정 ──────────────────────────────────────────────────────────
    # ── 모드 결정 ──────────────────────────────────────────────────────────
    def _plan_cold_channel(self, m: int, topk_ids: torch.Tensor, *,
                           needs_host_ids: bool = True) -> _ColdChannel:
        """cold 호출을 내보낼 통로를 고른다 (graph-safe 여부와 그 귀결).

        needs_host_ids=False는 "이 호출에 expert_ids의 host 소비자가 없다"는 뜻이다
        (순수 GPU prefill: cold가 GPU로 가고 hybrid·warm-CPU도 없다). 그때 blocking
        D2H는 큐에 쌓인 GPU 작업을 비우기만 하므로 건너뛴다 (qwen 레인, 2026-10-30).
        """
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
        if use_cold_stream:
            return _ColdChannel(
                graph_flow=graph_flow, use_cold_stream=True, stream_arg=stream_arg,
                qlen_ptr=self._qlen_pin_for(m).data_ptr(), ids_cpu=None,
            )
        # 순수 eager(스트림 통합 없음): cold가 expert_ids를 host에서 읽으므로 blocking
        # D2H 1회. 그 blocking이 뒤따르는 host 쓰기(qlen_pin)의 사실상 throttle이다
        # (계약 ④ Task8-3). cold_async는 자기 stream에서 async로 내린다.
        if self._cold_async or not needs_host_ids:
            ids_cpu = None
        else:
            with _nvtx("s1.topk_d2h"):
                ids_cpu = topk_ids.to("cpu")
        return _ColdChannel(
            graph_flow=False, use_cold_stream=False,
            stream_arg=stream_arg, qlen_ptr=self._qlen_pin.data_ptr(),
            ids_cpu=ids_cpu,
        )

    def _qlen_pin_for(self, m: int) -> torch.Tensor:
        """graph/stream 경로의 qlen 버퍼 — **bs별로 격리**한다 (Finding A). 캡처가
        baked하는 포인터를 eager의 _qlen_pin과 공유하면 나중의 eager prefill 쓰기에
        노출돼 replay마다 cold가 L토큰짜리로 돌아 perf가 붕괴한다 (실측 30B decode
        graph 328ms/tok vs eager 106ms/tok). bs가 다른 replay끼리도 절대 공유하지 않는다."""
        pin = self._qlen_pins_graph.get(m)
        if pin is None:
            pin = torch.full((1,), m, dtype=torch.int32)
            self._qlen_pins_graph[m] = pin
        return pin

    def _select_cpu_dispatch(self, channel: _ColdChannel, cpu_cold: bool,
                             cpu_warm: bool, hybrid: bool) -> CpuDispatch:
        """남은 CPU 몫을 어느 통로로 보낼지 고른다. 우선순위가 곧 의미다 —
        실험적 ASYNC_FLAG > 곁 스트림 SIDE_STREAM > 기본 HOST_BLOCK."""
        if not (cpu_cold or cpu_warm):
            return CpuDispatch.NONE
        has_side_stream = self._res.cold_stream is not None
        if (self._cold_async and has_side_stream
                and not channel.graph_flow and not hybrid):
            return CpuDispatch.ASYNC_FLAG
        # 곁 스트림은 cold 하나만 남았을 때만 — warm·hybrid는 main에서 staging과
        # 마스크를 공유하므로 함께 옮길 수 없다.
        if (self._cold_split and has_side_stream and channel.use_cold_stream
                and cpu_cold and not cpu_warm and not hybrid):
            return CpuDispatch.SIDE_STREAM
        return CpuDispatch.HOST_BLOCK

    def _plan_flow(self, layer_idx: int, m: int, k: int, topk_ids: torch.Tensor,
                   topk_weights: torch.Tensor, device) -> _LayerFlow:
        """이 호출의 실행 계획을 **1회** 확정한다 — run_layer 본문의 분기는 전부
        여기서 정해진 `_LayerFlow` 필드를 읽기만 한다.

        순서에 의미가 있다: 통로(graph/stream) → 이 호출의 형태(M·마스킹·그룹핑)
        → cold를 GPU가 읽는가 → hybrid로 쪼개는가 → 남은 CPU 몫의 dispatch.
        뒤 단계가 앞 단계를 뒤집는 곳(전량 GPU cold면 CPU는 cold를 내지 않는다)도
        여기 한곳에 모여 있다.
        """
        dims = self._plan.dims
        # D2H가 필요한지는 통로를 열기 **전에** 안다: cold가 GPU로 가고 hybrid·warm-CPU가
        # 없으면 host는 expert_ids를 읽지 않는다. 아래 cold_on_gpu/cpu_warm과 같은 식이다
        # (grouping은 m >= grouped_min_m과 동치).
        has_cold_rows = self._layer_has_cold[layer_idx]
        cold_gpu_pre = bool(
            has_cold_rows and m >= self._grouped_min_m
            and self._cold_gpu_min_m is not None and m >= self._cold_gpu_min_m
            and self._layer_cold_gpu[layer_idx]
        )
        warm_cpu_pre = bool(
            self._layer_warm_kt.get(layer_idx, False)
            and ((self._warm_cpu_max_m is not None and 1 < m < self._warm_cpu_max_m)
                 or (self._warm_cpu_min_m is not None and m >= self._warm_cpu_min_m))
        )
        channel = self._plan_cold_channel(
            m, topk_ids,
            needs_host_ids=((has_cold_rows and not cold_gpu_pre)
                            or self._hybrid_frac is not None or warm_cpu_pre),
        )

        # sparsity (계약 ①): cold만, decode에서만 (prefill-dense).
        masking = self._sparse and m == 1
        # prefill 형태: pair를 expert로 묶어 GPU 티어가 W를 expert당 한 번 읽는다.
        # 레이어당 1회 만들어 phase·티어가 공유한다 (topk가 같다).
        grouping: Optional[Grouping] = (
            build_grouping(topk_ids, dims.num_experts)
            if m >= self._grouped_min_m else None
        )
        # GPU sparse 티어와 rejoin이 같은 fp32 텐서를 쓴다 — 캐스팅이 두 번
        # 일어나면 스텝마다 할당이 하나 더 생긴다.
        topk_w32 = topk_weights if topk_weights.dtype is torch.float32 \
            else topk_weights.to(torch.float32)

        # cold를 GPU가 읽는가 (대배치 prefill). 이 결정 하나가 "cold partial을
        # CPU와 GPU가 둘 다 내는" 이중계산을 막는다 — hybrid만이 예외이고,
        # 거기서는 두 partial의 (m, j) 행이 서로소다.
        cold_on_gpu = bool(
            has_cold_rows and grouping is not None
            and self._cold_gpu_min_m is not None and m >= self._cold_gpu_min_m
            and self._layer_cold_gpu[layer_idx]
        )
        # hybrid: GPU는 마스크된 expert만(제한 그룹핑), CPU는 나머지(kt skip 마스크).
        # 두 partial은 각자 남의 행을 0으로 두므로 합이 곧 전체다.
        hybrid = bool(cold_on_gpu and self._hybrid_frac is not None
                      and not channel.graph_flow and not channel.use_cold_stream
                      and not self._cold_async)   # 마스크 즉시쓰기가 host 선행과 충돌
        cold_grouping_gu = cold_grouping_dn = grouping
        if self._cold is not None and getattr(self._cold, "hybrid_masks", None) is not None:
            if hybrid:
                # 패턴은 여기서 정하고 kt 마스크로의 **쓰기**는 각 phase의 submit
                # 직전이다 (_cpu_partials_begin) — 직전 phase의 kt 작업이 sync로
                # 끝난 뒤여야 하므로. GPU-local 노드만 채우고 원격 노드 마스크는
                # 항상 0이다 (그 shard는 CPU 전부).
                self._hybrid_masks = self._balance_hybrid(
                    channel.ids_cpu, dims.num_experts, device)
                cold_grouping_gu = build_grouping(topk_ids, dims.num_experts,
                                                  expert_mask=self._hybrid_masks["gu"][1])
                cold_grouping_dn = build_grouping(topk_ids, dims.num_experts,
                                                  expert_mask=self._hybrid_masks["dn"][1])
            else:
                for mk in self._cold.hybrid_masks:
                    mk.zero_()
        cpu_cold = has_cold_rows and not (cold_on_gpu and not hybrid)

        # Small prefill uses the warm CPU instance; the upper bound is exclusive.
        # The legacy minimum remains an explicit experimental override.
        cpu_warm = bool(
            self._layer_warm_kt.get(layer_idx, False)
            and ((self._warm_cpu_max_m is not None and 1 < m < self._warm_cpu_max_m)
                 or (self._warm_cpu_min_m is not None and m >= self._warm_cpu_min_m
                     and grouping is not None and not channel.graph_flow
                     and not channel.use_cold_stream))
        )

        probe_on = self._mask_stats is not None and not channel.graph_flow
        return _LayerFlow(
            graph_flow=channel.graph_flow, use_cold_stream=channel.use_cold_stream,
            stream_arg=channel.stream_arg, qlen_ptr=channel.qlen_ptr,
            ids_cpu=channel.ids_cpu,
            m=m, k=k, masking=masking, topk_w32=topk_w32, grouping=grouping,
            cold_on_gpu=cold_on_gpu, hybrid=hybrid,
            cpu_dispatch=self._select_cpu_dispatch(channel, cpu_cold, cpu_warm, hybrid),
            cpu_cold=cpu_cold, cpu_warm=cpu_warm,
            # GPU warm 티어는 CPU가 warm을 내는 호출에서만 빠진다 (이중계산 방지).
            skip_gpu_warm=cpu_warm,
            cold_grouping_gateup=cold_grouping_gu, cold_grouping_down=cold_grouping_dn,
            probe_on=probe_on, stats_on=bool(masking and probe_on),
        )

    # ── 본체 ──────────────────────────────────────────────────────────────
    def run_layer(self, layer_idx: int, hidden: torch.Tensor,
                  topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                  swiglu_limit: Optional[float] = None,
                  activation: Optional["Activation"] = None) -> torch.Tensor:
        """hidden [M, H] bf16 cuda, topk_ids [M, k] int64, topk_weights [M, k].
        activation: 모델의 gate/up 활성화 (activation.py; None = silu). swiglu_limit: 구 API —
        silu의 clamp (DSV4-Flash/GLM 10.0). 둘 다 rejoin#1에서만 적용된다.
        반환 [M, H] bf16 cuda (router 가중 expert 합 완료).

        두 phase가 같은 네 걸음을 밟는다: 관측 → CPU 몫 개시 → GPU 티어 발행 →
        CPU 몫 합류 → rejoin. 실행 모드는 상단의 `_plan_flow()`가 전부 확정한다.
        """
        tiers = self._tiers[layer_idx]
        dims = self._plan.dims
        inter, h = dims.intermediate_size, dims.hidden_size

        _nvtx_push(f"prism.L{layer_idx}")
        flow = self._plan_flow(layer_idx, hidden.shape[0], topk_ids.shape[1],
                               topk_ids, topk_weights, hidden.device)

        # ── Phase 1: gateup ──────────────────────────────────────────────
        self._observe(layer_idx, GATEUP, hidden, topk_ids, flow)
        pending = self._cpu_partials_begin(layer_idx, GATEUP, hidden, topk_ids, flow)
        gu_parts = self._run_gateup(
            tiers, hidden, topk_ids, flow.topk_w32, flow.m, flow.k, inter,
            flow.masking, grouping=flow.grouping, cold_gpu=flow.cold_on_gpu,
            cold_grouping=flow.cold_grouping_gateup, cold_partial=flow.hybrid,
            skip_warm=flow.skip_gpu_warm)
        cold_gu, warm_gu = self._cpu_partials_end(GATEUP, pending, flow, hidden.device)

        # rejoin#1: fp32 합 → act → bf16 (계약 ⑤)
        # 융합 커널 한 launch: Σ partial(fp32) → silu·up → bf16 (rejoin.py). torch
        # 사슬(캐스팅 3 + add 2 + split/silu/mul/cast)은 prefill에서 88 MB 텐서를
        # ~10번 왕복해 층당 2.5 ms였다 (2026-08-27 nsys).
        with _nvtx("rejoin1.acc+act"):
            act = rejoin_gateup(gu_parts + [cold_gu, warm_gu], inter, swiglu_limit,
                                activation=activation)

        # ── Phase 2: down ────────────────────────────────────────────────
        self._observe(layer_idx, DOWN, act, topk_ids, flow)
        pending = self._cpu_partials_begin(layer_idx, DOWN, act, topk_ids, flow)
        down_parts = self._run_down(
            tiers, act, topk_ids, flow.topk_w32, flow.m, flow.k, inter, h,
            flow.masking, grouping=flow.grouping, cold_gpu=flow.cold_on_gpu,
            cold_grouping=flow.cold_grouping_down, cold_partial=flow.hybrid,
            skip_warm=flow.skip_gpu_warm)
        cold_dn, warm_dn = self._cpu_partials_end(DOWN, pending, flow, act.device)

        # rejoin#2: fp32 합 → router 가중 expert합 → bf16
        with _nvtx("rejoin2.acc+wsum"):
            out = rejoin_down(down_parts + [cold_dn, warm_dn], flow.topk_w32)
        _nvtx_pop()                               # prism.L{layer_idx}
        return out

    # ── 관측 ─────────────────────────────────────────────────────────────
    def _observe(self, layer_idx: int, phase: str, x: torch.Tensor,
                 topk_ids: torch.Tensor, flow: _LayerFlow) -> None:
        """effective sparsity 실측 + NaN 진단 프로브 — 커널과 **같은 입력**으로
        마스크를 재계산해 누적한다. eager에서만 돈다 (graph 경로는 파이썬이
        replay되지 않으므로 캡처 중 관측하면 워밍업 토큰이 섞인다)."""
        if not flow.probe_on:
            return
        stats = self._mask_stats
        if phase == GATEUP:
            # NaN 진단은 prefill(m>1)에서도 돈다 — decode 전용 관측이 못 보는 구간이다.
            stats.probe_calls += 1
            stats.probe(layer_idx, "gateup_in", x, flow.m)
            if flow.stats_on:
                stats.observe_gateup(layer_idx, x, topk_ids, flow.topk_w32)
        else:
            stats.probe(layer_idx, "act_after_rejoin1", x, flow.m)
            if flow.stats_on:
                stats.observe_down(layer_idx, x, topk_ids, flow.topk_w32)

    # ── CPU partial 조달 (begin/end 쌍) ────────────────────────────────────
    def _cold_entry(self, port: _PhasePorts, warm: bool = False):
        """cold backend의 이 phase submit 진입점. **호출 시점에** 이름으로 꺾는다 —
        backend를 런타임에 감싸는 테스트·프로파일러가 있으므로 생성 시점에
        메서드를 붙들어두면 안 된다."""
        return getattr(self._cold, port.submit_warm_name if warm else port.submit_cold_name)

    def _cpu_partials_begin(self, layer_idx: int, phase: str, x: torch.Tensor,
                            topk_ids: torch.Tensor, flow: _LayerFlow):
        """이 phase의 CPU(cold/warm) 작업을 **개시**한다 — dispatch별로 enqueue
        방식만 다르다. 반환은 `_cpu_partials_end`에 그대로 넘길 진행 중 상태로,
        ASYNC_FLAG만 여기서 device partial을 내고 나머지는 (None, None)이다.

        호출자는 이 사이에 GPU 티어를 발행한다 — 그 겹침이 2-phase의 전부다.
        """
        dispatch = flow.cpu_dispatch
        if dispatch is CpuDispatch.NONE:
            return None, None
        port = self._ports[phase]
        if flow.hybrid:
            # host 즉시쓰기: 이 phase의 GPU 몫 패턴을 submit 직전에 쓴다 —
            # 직전 phase/층의 kt 작업은 sync로 끝났다 (HOST_BLOCK 전용 경로).
            self._cold.hybrid_masks[self._hybrid_local_node].copy_(
                self._hybrid_masks["gu" if port.is_gateup else "dn"][0])
        if dispatch is CpuDispatch.ASYNC_FLAG:
            return self._cold_phase_async(layer_idx, port, x, topk_ids, flow)
        if dispatch is CpuDispatch.SIDE_STREAM:
            self._cold_side_stream_submit(layer_idx, port, x, topk_ids, flow)
            return None, None
        self._cold_host_block_submit(layer_idx, port, x, topk_ids, flow)
        return None, None

    def _cpu_partials_end(self, phase: str, pending, flow: _LayerFlow, device):
        """개시한 CPU 작업을 **합류**시키고 (cold_partial, warm_partial)을 낸다.
        둘 다 device 텐서이거나 None이며, rejoin이 곧바로 fp32로 누산한다."""
        dispatch = flow.cpu_dispatch
        if dispatch is CpuDispatch.NONE:
            return None, None
        if dispatch is CpuDispatch.ASYNC_FLAG:
            torch.cuda.current_stream().wait_stream(self._res.cold_stream)  # partial H2D 완료
            return pending
        port = self._ports[phase]
        if dispatch is CpuDispatch.SIDE_STREAM:
            return self._cold_side_stream_join(port, flow.m, device), None
        return self._cold_host_block_collect(port, flow, device)

    # ── dispatch: HOST_BLOCK ─────────────────────────────────────────────
    def _cold_host_block_submit(self, layer_idx: int, port: _PhasePorts,
                                x: torch.Tensor, topk_ids: torch.Tensor,
                                flow: _LayerFlow) -> None:
        """[입력 D2H] → [submit] (enqueue-only, 즉시 반환). `cold.{tag}.window`
        nvtx 구간을 열고 나간다 — 닫는 쪽은 collect다."""
        st = self._res.staging
        with _nvtx(f"cold.{port.tag}.{port.fill_label}(D2H-block)"):
            # stream 통합 시 non_blocking: kt host node가 같은 stream에 순서대로
            # 실행되므로 host-측 완료 보장이 불필요하다. expert_ids도 device
            # 텐서에서 pinned로 async D2H한다 (graph·stream 공통).
            port.fill_inputs(x, topk_ids if flow.use_cold_stream else flow.ids_cpu,
                             flow.use_cold_stream)
            if port.is_gateup:
                if flow.use_cold_stream:
                    # qlen은 고정 pinned 상수 버퍼라 여기서 쓰지 않는다.
                    pin = self._qlen_pins_graph[flow.m]
                    assert int(pin[0]) == flow.m, (
                        f"qlen_pin_graph[{flow.m}]={int(pin[0])} != {flow.m} — "
                        f"stream/graph path must never write this buffer"
                    )
                else:
                    self._qlen_pin[0] = flow.m
        if port.is_gateup and flow.masking:
            with _nvtx("cold.fill_topk_w"):
                st.fill_topk_w(flow.topk_w32, non_blocking=flow.use_cold_stream)
        w_ptr = st.topk_w_ptr() if flow.masking else 0
        if flow.cpu_cold:
            with _nvtx(f"cold.{port.tag}.submit"):
                self._cold_entry(port)(
                    layer_idx, flow.qlen_ptr, flow.k, st.expert_ids_ptr(),
                    port.input_ptr(), port.cold_partial_ptr(),
                    cuda_stream=flow.stream_arg, weights_ptr=w_ptr)
        if flow.cpu_warm:
            with _nvtx(f"warm.{port.tag}.submit"):
                self._cold_entry(port, warm=True)(
                    layer_idx, flow.qlen_ptr, flow.k, st.expert_ids_ptr(),
                    port.input_ptr(), port.warm_partial_ptr(),
                    cuda_stream=flow.stream_arg, weights_ptr=0)
        _nvtx_push(f"cold.{port.tag}.window")     # CPU expert 연산 재실 구간

    def _cold_host_block_collect(self, port: _PhasePorts, flow: _LayerFlow, device):
        """[블로킹 sync] → [partial H2D]. sync는 두 인스턴스(cold·warm) 모두를
        기다린다 — 같은 CPUInfer 풀이다."""
        with _nvtx(f"cold.{port.tag}.sync(host-block)"):
            self._cold.sync(cuda_stream=flow.stream_arg)
        _nvtx_pop()                               # cold.{tag}.window
        cold_p = warm_p = None
        with _nvtx(f"cold.{port.tag}.h2d_out"):
            if flow.cpu_cold:
                cold_p = port.cold_out(flow.m).to(
                    device, non_blocking=flow.use_cold_stream)
            if flow.cpu_warm:
                warm_p = port.warm_out(flow.m).to(
                    device, non_blocking=flow.use_cold_stream)
        return cold_p, warm_p

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

    # ── dispatch: SIDE_STREAM ────────────────────────────────────────────
    def _cold_side_stream_submit(self, layer_idx: int, port: _PhasePorts,
                                 x: torch.Tensor, topk_ids: torch.Tensor,
                                 flow: _LayerFlow) -> None:
        """cold 한 phase의 [입력 D2H] → [submit host node]를 **곁 스트림**에 얹는다.

        host node 방식은 그대로다 — 바뀌는 것은 그것이 어느 stream에 붙느냐뿐이다.
        곁 스트림에 두면 뒤따르는 sync가 main의 hot/warm 커널 완료를 기다리지 않는다.
        `use_cold_stream` 경로에서만 고른다 (그 경로에서는 warm·hybrid가 모두 꺼져
        있으므로 cold 하나만 다루면 된다 — `_select_cpu_dispatch` 참조).
        """
        st, cs = self._res.staging, self._res.cold_stream
        cs.wait_stream(torch.cuda.current_stream())   # hidden/act·topk 준비 완료
        with torch.cuda.stream(cs), _nvtx(f"cold.{port.name}.split.submit"):
            w_ptr = 0
            if flow.masking:
                st.fill_topk_w(flow.topk_w32, non_blocking=True)
                w_ptr = st.topk_w_ptr()
            port.fill_inputs(x, topk_ids, True)
            self._cold_entry(port)(
                layer_idx, flow.qlen_ptr, flow.k, st.expert_ids_ptr(),
                port.input_ptr(), port.cold_partial_ptr(),
                cuda_stream=cs.cuda_stream, weights_ptr=w_ptr)

    def _cold_side_stream_join(self, port: _PhasePorts, m: int, device) -> torch.Tensor:
        """곁 스트림의 [sync host node] → [partial H2D]를 마치고 main에 합류시킨다.

        rejoin이 이 partial을 main에서 읽으므로 합류는 없앨 수 없다 — 없앨 수 있는
        것은 sync가 GPU 커널을 기다리던 쪽이었다.
        """
        cs = self._res.cold_stream
        with torch.cuda.stream(cs), _nvtx(f"cold.{port.name}.split.join"):
            self._cold.sync(cuda_stream=cs.cuda_stream)
            out = port.cold_out(m).to(device, non_blocking=True)
        main = torch.cuda.current_stream()
        main.wait_stream(cs)
        # 캡처 밖에서만: 이 블록은 cs에서 할당됐고 main이 읽는다 — 할당자가 cs의
        # 다음 할당에 재사용하지 못하게 막는다. 캡처 중에는 graph 전용 풀이라
        # 재사용 규칙이 다르고 record_stream이 유효하지 않다.
        if not torch.cuda.is_current_stream_capturing():
            out.record_stream(main)
        return out

    # ── dispatch: ASYNC_FLAG ─────────────────────────────────────────────
    def _cold_phase_async(self, layer_idx: int, port: _PhasePorts, x: torch.Tensor,
                          topk_ids: torch.Tensor, flow: _LayerFlow):
        """cold(+warm-kt) 한 phase를 cold stream에 **enqueue만** 하고 partial(device)을
        돌려준다. 순서: [입력 D2H] → [submit host node] → [signal host node] →
        [wait_flag 커널] → [partial H2D]. 호출자는 rejoin 전에 current stream이
        cold stream을 기다리게 한다. host는 어디서도 블록되지 않는다."""
        res, cs = self._res, self._res.cold_stream
        st = res.staging
        cs.wait_stream(torch.cuda.current_stream())  # 입력(hidden/act, topk) 준비 완료
        qlen_ptr = self._qlen_pin_for(flow.m).data_ptr()
        with torch.cuda.stream(cs), _nvtx(f"cold.{port.name}.async"):
            port.fill_inputs(x, topk_ids, True)      # device → pinned async
            if port.is_gateup and flow.masking:
                st.fill_topk_w(flow.topk_w32, non_blocking=True)
            w_ptr = st.topk_w_ptr() if flow.masking else 0
            sarg = cs.cuda_stream
            if flow.cpu_cold:
                self._cold_entry(port)(layer_idx, qlen_ptr, flow.k, st.expert_ids_ptr(),
                                 port.input_ptr(), port.cold_partial_ptr(),
                                 cuda_stream=sarg, weights_ptr=w_ptr)
            if flow.cpu_warm:
                self._cold_entry(port, warm=True)(layer_idx, qlen_ptr, flow.k, st.expert_ids_ptr(),
                                 port.input_ptr(), port.warm_partial_ptr(),
                                 cuda_stream=sarg, weights_ptr=0)
            res.cold_seq += 1
            self._cold.submit_signal(sarg, res.cold_flag.data_ptr(), res.cold_seq)
            from sglang.jit_kernel.prism_grouped import wait_flag
            wait_flag(res.cold_flag, res.cold_seq, cs)
            dev = x.device
            cold_p = port.cold_out(flow.m).to(dev, non_blocking=True) if flow.cpu_cold else None
            warm_p = port.warm_out(flow.m).to(dev, non_blocking=True) if flow.cpu_warm else None
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
