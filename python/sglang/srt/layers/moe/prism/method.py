"""Prism의 sglang 접점 — quant-method wrapper (S7).

차단선 (설계 문서): FusedMoE 위로는 Prism을 모른다. 이 모듈이 유일한
등록 지점이며, 활성화는 env `SGLANG_PRISM_PLAN=<plan.json>` 로만 된다.

kt(KTEPWrapperMethod)와 달리 **상속하지 않는 standalone method**다 —
조사 결과(2026-08-20) kt의 장부는 GPU/CPU "expert 단위" 분할용이고, kt의
CPU weight는 디스크 사전 변환 파일에서 오므로(full 텐서가 프로세스에 없음)
Prism의 "full 텐서 로드 → K-슬라이스 → 주입" 흐름과 겹치는 재사용 면이
없다. 대신 이 method는:

  create_weights: full expert weight를 **CPU 파라미터**로 할당 (sglang
    weight_loader가 그대로 채움 — GPU 메모리 안 씀)
  process_weights_after_loading: 슬라이스(prepare_layer_weights) →
    cold 주입(KtColdBackend) → executor 등록 → CPU 파라미터 해제
    (full 텐서 소멸 — 계약 ③)
  apply: executor.run_layer 위임 한 줄

P0 제약: TP=1. batch size 제약은 없다 — GPU 티어가 pair-native가 되면서
그룹 조성의 host 결정이 사라졌고, eager·캡처·prefill이 모두 같은 경로다
(2026-08-25). executor가 캡처 구간을 자동 감지해 cold를 stream 통합
(kt host node)으로 돌린다.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.moe.prism.plan import Plan, PlanError, Proj, Tier, parse_plan, validate_static
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

logger = logging.getLogger(__name__)

_ENV_PLAN = "SGLANG_PRISM_PLAN"
_ENV_MAX_TOKENS = "SGLANG_PRISM_MAX_TOKENS"
_ENV_CPUINFER = "SGLANG_PRISM_CPUINFER_THREADS"
# 실험 노브: 콤마 구분 NUMA node id 리스트 (예: "1" = node 1 단독).
# 설정 시 subpool을 그 노드들에만 만들고, plan의 shard "node i"는
# i번째 서브풀(= 리스트의 i번째 실제 노드)로 매핑된다. 미설정 = 전 노드.
_ENV_NUMA_MAP = "SGLANG_PRISM_NUMA_MAP"
# eager에서도 cold submit/sync를 stream host node로 보내는 opt-in.
# (graph 경로는 env와 무관하게 항상 stream 통합 — executor 참조.)
_ENV_COLD_STREAM = "SGLANG_PRISM_COLD_STREAM"
# prefill grouped GEMM 임계 M (미설정 = executor 기본값). 벤치·회귀 비교용 노브.
_ENV_GROUPED_MIN_M = "SGLANG_PRISM_GROUPED_MIN_M"
# grouped 경로에서 hot/warm 스트림 분리 (기본 1). "0"으로 끄면 직렬 발행.
_ENV_SPLIT_STREAMS = "SGLANG_PRISM_SPLIT_STREAMS"
# 이 M 이상의 prefill에서 cold를 GPU가 packed slab 제자리 읽기로 계산.
# 미설정 = executor.COLD_GPU_MIN_M(실측 교차점), "0" = 끔 (slab host-register도 안 함
# — 등록은 cold 전량을 pinned로 만들어 프로세스 수명 동안 잠근다).
_ENV_COLD_GPU_MIN_M = "SGLANG_PRISM_COLD_GPU_MIN_M"
# cold hybrid: cold GPU 조건에서 expert의 이 비율만 GPU, 나머지 CPU 동시 계산 (미설정 = 끔).
_ENV_COLD_HYBRID_FRAC = "SGLANG_PRISM_COLD_HYBRID_FRAC"
# warm을 kt 포맷 slab(pinned) 한 벌로 (row-major pinned 대신). GPU는 packed GEMV /
# cold-layout grouped로 읽는다. cold GPU view가 필요하다 (COLD_GPU_MIN_M=0이면 불가).
_ENV_WARM_KT = "SGLANG_PRISM_WARM_KT"
# warm-kt 모드에서 이 M 이상의 prefill은 warm을 CPU(warm-kt 인스턴스)가 계산 (미설정 = GPU).
_ENV_WARM_CPU_MIN_M = "SGLANG_PRISM_WARM_CPU_MIN_M"
# cold를 전용 stream + 플래그 wait로 (블로킹 콜백 없음). eager 전용. hybrid와 동시 불가.
_ENV_COLD_ASYNC = "SGLANG_PRISM_COLD_ASYNC"
# cold submit/sync host node를 곁 스트림에 얹는다 (P2). graph decode에서
# sync가 hot/warm 커널 완료를 기다리던 가짜 의존을 끊는다 — 실측 down phase −31%.
_ENV_COLD_SPLIT = "SGLANG_PRISM_COLD_SPLIT"
# 로딩 훅(process_weights_after_loading)의 torch intra-op 스레드 수. model_runner가
# load_model 진입부에서 전역 1로 고정하는데(평범한 로딩의 memcpy 스레드 경합 방지),
# prism의 훅은 memcpy가 아니라 층당 3.4 GB를 훑는 K-슬라이스 repack이라 그 값이
# 그대로 병목이 된다 (실측 1스레드 4.0 s vs 16스레드 0.87 s = 4.7×).
# 미설정 = 물리 코어 수(HT 제외). cold_load 시점의 kt 워커들은 50 ms 뒤 condvar에서
# 잠들므로(worker_pool.cpp) prepare 구간에 코어를 다 써도 경합하지 않는다.
_ENV_LOAD_THREADS = "SGLANG_PRISM_LOAD_THREADS"


def _load_threads() -> int:
    default = max(1, (os.cpu_count() or 4) // 2)
    return max(1, int(os.environ.get(_ENV_LOAD_THREADS, str(default))))


def _hybrid_local_node(device) -> int:
    from sglang.srt.layers.moe.prism.numa import gpu_numa_node

    phys = gpu_numa_node(device)
    numa_map = os.environ.get(_ENV_NUMA_MAP)
    if numa_map:
        nodes = [int(x) for x in numa_map.split(",")]
        return nodes.index(phys) if phys in nodes else 0
    return phys


def _cold_gpu_min_m():
    from sglang.srt.layers.moe.prism.executor import PrismExecutor

    raw = os.environ.get(_ENV_COLD_GPU_MIN_M)
    if raw is None:
        # 기본값은 스토어 포맷이 정한다 (bf16: 실측 교차점, mxfp4/fp8: grouped 경계 = CPU prefill 없음).
        gmin = os.environ.get(_ENV_GROUPED_MIN_M)
        return _get_runtime().fmt.default_cold_gpu_min_m(
            PrismExecutor.COLD_GPU_MIN_M,
            PrismExecutor.GROUPED_MIN_M if gmin is None else int(gmin))
    v = int(raw)
    return None if v <= 0 else v


def _sglang_capture_mode() -> bool:
    """sglang CudaGraphRunner.capture() 구간(캡처 전 워밍업 2회 포함) 감지.

    executor의 graph-safe 경로 선택 입력으로 주입된다 — 워밍업 run도 graph
    경로를 타야 gather jit compile 같은 lazy init이 캡처 밖에서 끝난다.
    외부 시스템(runner) 접촉은 조립 지점인 이 모듈에 둔다; import 실패
    (runner 미탑재 경량 환경)는 False = "캡처 구간 아님"으로 처리한다.
    """
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
    except Exception:
        return False
    return get_is_capture_mode()


class _PrismRuntime:
    """프로세스 전역 1벌: plan + resources + cold backend + executor.

    (kt의 SharedStagingBuffer/SharedFullContext와 같은 배치 — apply는
    per-layer 호출이라 cross-layer 상태를 전역이 들어야 한다)
    """

    def __init__(self, plan: Plan, calib=None):
        from sglang.srt.layers.moe.prism.kernels import gpu_store_format

        self.plan = plan
        # 스토어 포맷 — plan의 GPU 커널 키가 함의한다 (계약 ①). 파라미터 등록·full 텐서
        # 인출·gather·커널 진입점이 전부 이 객체를 통한다 (formats.py).
        self.fmt = gpu_store_format(plan.kernels.gpu_warm)
        # 모델의 SwiGLU clamp (DSV4-Flash swiglu_limit=10). create_moe_runner가 layer의
        # MoeRunnerConfig에서 읽어 채운다 — 전 층 공통 값이라 프로세스 1벌.
        self.swiglu_limit: Optional[float] = None
        # 모델의 routed_scaling_factor (GLM-5.3 2.5 / DSV4-Flash 1.5). 이것도 create_moe_runner에서만
        # 보인다. 일반 경로에서는 **MoE 러너가** 곱하는데(moe_runner/triton.py:109·164·209,
        # triton_kernels.py:205, flashinfer_cutlass.py:151) prism은 apply를 전유해 러너를 안 쓴다.
        # 모델 쪽 사후 곱셈도 prism에는 걸리지 않는다: deepseek_v2.py의 가드는
        # `not _is_cuda ... or isinstance(KTEPWrapperMethod)`이고(upstream:1021-1027,
        # 포크:810-816) prism은 그 목록에 없으며, maybe_fuse_routed_scale_and_shared_add는
        # 비-fused method에 대해 shared만 더하고 스케일 없이 반환한다
        # (mxfp4_flashinfer_trtllm_moe.py:415-417). 그래서 prism이 직접 곱한다.
        # 정답 규약은 체크포인트 레퍼런스가 정한다: DeepSeek-V4-Flash-0731
        # inference/model.py:588 `weights *= self.route_scale` (Gate.forward 안) —
        # 라우터 가중에 접는 것이고, 곱셈이 선형이므로 expert 가중합 결과에 곱하는 것과 같다.
        self.routed_scaling_factor: Optional[float] = None
        # cold(CPU) 행이 plan 어디에도 없으면 kt 백엔드(스레드풀)를 만들지 않는다 — GPU
        # 스트리밍 전용 plan(mxfp4)에서 idle 워커가 CPU를 점유할 이유가 없다.
        self.has_cold = any(
            plan.expert(l, e).proj(p).has_tier(Tier.COLD)
            for (l, e) in plan.experts for p in Proj
        )
        if self.has_cold:
            # 스토어 포맷과 cold 커널의 호환 (mxfp4 ↔ kt_amx_fp4 / bf16 ↔ kt_*_bf16) — startup에서 즉사.
            self.fmt.check_cold_kernel(plan.kernels.cpu_cold)
        # sparsity 점수·threshold 테이블 (dense plan이면 None). 프로세스에 1벌 —
        # 레이어마다 다시 열면 382MB 자산을 40번 읽게 된다.
        self.calib = calib
        self.max_tokens = int(os.environ.get(_ENV_MAX_TOKENS, "4096"))
        self._resources = None
        self._cold = None
        self._executor = None

    def executor(self, device: torch.device):
        if self._executor is None:
            from sglang.srt.layers.moe.prism.executor import PrismExecutor
            from sglang.srt.layers.moe.prism.kernels import resolve_gpu_kernel
            from sglang.srt.layers.moe.prism.resources import (
                ExecutionResources,
                ResourceSpec,
            )

            resolve_gpu_kernel(self.plan.kernels.gpu_warm)  # 이름 검증
            # 이 포맷의 GEMV/grouped 커널 lazy JIT을 startup으로 앞당긴다 — 첫 호출이 캡처
            # 워밍업이면 컴파일이 캡처 순서에 얽힌다.
            self.fmt.warmup()
            # rejoin Triton 커널도 캡처 워밍업 전에 컴파일한다.
            from sglang.srt.layers.moe.prism.rejoin import warmup as warmup_rejoin

            d = self.plan.dims
            warmup_rejoin(device, d.intermediate_size, d.hidden_size, d.top_k, self.swiglu_limit)
            spec = ResourceSpec.from_plan(
                self.plan, max_tokens=self.max_tokens, device=device)
            self._resources = ExecutionResources(spec)
            # 조립 지점: env·runner 같은 외부 입력은 전부 여기서 읽어 명시
            # 인자로 주입한다 (executor는 hidden input 없음).
            gmin = os.environ.get(_ENV_GROUPED_MIN_M)
            self._executor = PrismExecutor(
                self.plan, self._resources, self.cold() if self.has_cold else None,
                cold_stream=os.environ.get(_ENV_COLD_STREAM) == "1",
                capture_mode_fn=_sglang_capture_mode,
                grouped_min_m=None if gmin is None else int(gmin),
                split_streams=os.environ.get(_ENV_SPLIT_STREAMS, "1") != "0",
                cold_gpu_min_m=_cold_gpu_min_m(),
                cold_hybrid_frac=(tuple(float(v) for v in os.environ[_ENV_COLD_HYBRID_FRAC].split(","))
                                  if os.environ.get(_ENV_COLD_HYBRID_FRAC) else None),
                # plan shard 인덱스 중 GPU-local NUMA 노드: NUMA_MAP이 있으면 그 리스트에서 찾고,
                # 없으면 shard 인덱스 = 물리 노드.
                hybrid_local_node=_hybrid_local_node(device),
                warm_cpu_min_m=(int(os.environ[_ENV_WARM_CPU_MIN_M])
                                if os.environ.get(_ENV_WARM_CPU_MIN_M) else None),
                cold_async=os.environ.get(_ENV_COLD_ASYNC) == "1",
                cold_split=os.environ.get(_ENV_COLD_SPLIT) == "1",
            )
        return self._executor

    def cold(self):
        if self._cold is None:
            from sglang.srt.layers.moe.prism.cold_backend import KtColdBackend
            from sglang.srt.layers.moe.prism.numa import numa_node_count

            # 기본값 = 물리 코어 − 2 (HT 제외, 메인/tokenizer 여유).
            # 과다구독은 submit/sync 고정비를 폭증시킨다 (실측 2026-08-20:
            # 물리 16코어에 60스레드 → sync 회당 1.85ms, 14스레드 → 0.05ms).
            default_threads = max(2, (os.cpu_count() or 4) // 2 - 2)
            threads = int(os.environ.get(_ENV_CPUINFER, str(default_threads)))
            numa_map = os.environ.get(_ENV_NUMA_MAP)
            # cold GPU 읽기가 켜져 있으면 slab을 이 device에 host-register한다.
            gpu_view_device = (
                torch.device(torch.cuda.current_device())
                if _cold_gpu_min_m() is not None else None
            )
            if numa_map:
                from kt_kernel import kt_kernel_ext

                nodes = [int(x) for x in numa_map.split(",")]
                cfg = kt_kernel_ext.WorkerPoolConfig()
                cfg.subpool_count = len(nodes)
                cfg.subpool_numa_map = nodes
                cfg.subpool_thread_count = [
                    threads // len(nodes) + (1 if i < threads % len(nodes) else 0)
                    for i in range(len(nodes))
                ]
                self._cold = KtColdBackend(
                    self.plan,
                    max_tokens=self.max_tokens,
                    num_numa_nodes=len(nodes),
                    cpuinfer=kt_kernel_ext.CPUInfer(cfg),
                    gpu_view_device=gpu_view_device,
                    hybrid_mask=bool(os.environ.get(_ENV_COLD_HYBRID_FRAC)),
                )
            else:
                self._cold = KtColdBackend(
                    self.plan,
                    max_tokens=self.max_tokens,
                    num_numa_nodes=numa_node_count(),
                    cpuinfer_threads=threads,
                    gpu_view_device=gpu_view_device,
                    hybrid_mask=bool(os.environ.get(_ENV_COLD_HYBRID_FRAC)),
                )
        return self._cold


_RUNTIME: Optional[_PrismRuntime] = None


def _get_runtime() -> _PrismRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        plan_path = os.environ[_ENV_PLAN]
        plan = parse_plan(plan_path)
        # sparsity plan이면 자산을 먼저 열어 shape까지 검증한다 (계약 ①):
        # 다른 모델의 calib을 적용하는 것은 dims 불일치와 같은 급의 silent
        # failure이므로 startup에서 죽는 편이 낫다.
        calib = None
        if plan.sparsity is not None:
            from sglang.srt.layers.moe.prism.calib import CalibTables

            calib = CalibTables.load(plan.sparsity)
        validate_static(
            plan, calib_probe=None if calib is None else calib.probe()
        )
        logger.info(
            "[prism] plan loaded: %s (model_id=%s, sparsity=%s)",
            plan_path, plan.model_id,
            "none" if plan.sparsity is None else plan.sparsity.score,
        )
        _RUNTIME = _PrismRuntime(plan, calib)
    return _RUNTIME


class PrismMoEMethod(FusedMoEMethodBase):
    """FusedMoE의 quant_method 자리에 들어가는 Prism 실행기."""

    # loader.device_loading_context에게 "내 파라미터는 CPU가 제자리다"를 알린다 —
    # 없으면 로더가 CPU 파라미터를 offload로 오해해 층당 full weight를 GPU로 올렸다
    # 내리고 버린다 (계약 ③: full expert weight는 host 상주). 클래스 속성이라
    # __getattr__(gpu_method 위임)을 타지 않는다.
    keeps_params_on_host = True

    def __init__(self, gpu_method, layer_id: int):
        self.gpu_method = gpu_method  # create_moe_runner 등 미지 속성의 위임처
        self.layer_id = layer_id
        self._registered = False

    def __getattr__(self, name):
        # __init__ 이전/자기 속성 미존재 시 재귀 방지 (kt의 알려진 함정)
        if name in ("gpu_method", "layer_id", "_registered"):
            raise AttributeError(name)
        return getattr(self.gpu_method, name)

    def create_moe_runner(self, layer, moe_runner_config):
        # base 클래스에 실체(raise NotImplementedError)가 있어 __getattr__이
        # 안 잡는다 — 명시적 위임 (runner는 gpu_method 것이 생성되지만
        # apply를 우리가 전유하므로 사용되지 않음). 모델의 SwiGLU clamp는 여기서만
        # 보이므로(MoeRunnerConfig.swiglu_limit) runtime에 적어 rejoin이 쓴다.
        limit = getattr(moe_runner_config, "swiglu_limit", None)
        runtime = _get_runtime()
        if limit is not None:
            if runtime.swiglu_limit is not None and runtime.swiglu_limit != limit:
                raise PlanError(f"swiglu_limit differs across layers ({runtime.swiglu_limit} vs {limit})")
            runtime.swiglu_limit = float(limit)
        rsf = getattr(moe_runner_config, "routed_scaling_factor", None)
        if rsf is not None:
            if runtime.routed_scaling_factor is not None and runtime.routed_scaling_factor != rsf:
                raise PlanError(
                    f"routed_scaling_factor differs across layers "
                    f"({runtime.routed_scaling_factor} vs {rsf})")
            if runtime.routed_scaling_factor is None:
                logger.info("[prism] routed_scaling_factor = %s (출력에 적용)", float(rsf))
            runtime.routed_scaling_factor = float(rsf)
        try:
            return self.gpu_method.create_moe_runner(layer, moe_runner_config)
        except Exception as exc:  # 내부 method의 runner는 쓰이지 않는다 — 실패해도 진행
            logger.warning("[prism] inner quant method create_moe_runner failed (ignored): %s", exc)
            return None

    # ── Stage 2 훅들 ─────────────────────────────────────────────────────
    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **extra_weight_attrs):
        from sglang.srt.utils import set_weight_attrs

        runtime = _get_runtime()
        dims = runtime.plan.dims
        inter_full = intermediate_size_per_partition * getattr(layer, "moe_tp_size", 1)
        if getattr(layer, "moe_tp_size", 1) != 1:
            raise NotImplementedError("Prism P0 supports TP=1 only")
        if (num_experts, hidden_size, inter_full) != (
            dims.num_experts, dims.hidden_size, dims.intermediate_size
        ):
            raise PlanError(
                f"plan dims {(dims.num_experts, dims.hidden_size, dims.intermediate_size)} "
                f"!= model dims {(num_experts, hidden_size, inter_full)} — "
                f"plan이 다른 모델에 적용되고 있음"
            )
        # full weight를 CPU에 — GPU에는 hot 스토어만 간다 (계약 ③). 파라미터의 이름/shape/
        # dtype/attrs는 포맷이 정한다 (bf16: sglang 기본 w13/w2; mxfp4: DeepSeekMxfp4MoEMethod와
        # 동일한 int8 nibble + fp32 BLOCK 배율 — 그 이름으로 로더가 채운다).
        # trap 방어 (weights.py docstring): gate-first w13 순서 가정 검증
        if getattr(self.gpu_method, "load_up_proj_weight_first", False):
            raise NotImplementedError("Prism assumes gate-first w13 ordering")
        runtime.fmt.create_params(layer, num_experts, hidden_size, inter_full, params_dtype,
                                  extra_weight_attrs)

    def process_weights_after_loading(self, layer) -> None:
        runtime = _get_runtime()
        # 파라미터는 CPU에 그대로 있다: `keeps_params_on_host`가 loader.py의
        # device_loading_context에게 왕복을 건너뛰게 한다. cold 주입은 C++가 host
        # memcpy로 읽으므로 CUDA 텐서의 data_ptr()를 넘기면 device 주소를
        # memcpy하다 segfault한다 (2026-08-20 스모크에서 실제 발생) — 그 왕복이
        # 있던 시절 take_full의 `_host()`가 방어선이었고, 지금은 왕복 자체가 없다.
        import time as _time
        # 로딩 훅만 torch 스레드를 되돌린다 (model_runner.load_model이 전역 1로 고정).
        # 원복은 finally에서 — 추론 경로는 다시 1스레드여야 한다 (kt가 자기 풀을 쓴다).
        _threads_want = _load_threads()
        _threads_saved = torch.get_num_threads()
        if _threads_want != _threads_saved:
            torch.set_num_threads(_threads_want)
        try:
            self._register_layer(layer, runtime, _time)
        finally:
            if _threads_want != _threads_saved:
                torch.set_num_threads(_threads_saved)

    def _register_layer(self, layer, runtime, _time) -> None:
        from sglang.srt.layers.moe.prism.kernels import cold_pack_tile_rows
        from sglang.srt.layers.moe.prism.numa import gpu_numa_node
        from sglang.srt.layers.moe.prism.plan import Proj, Tier
        from sglang.srt.layers.moe.prism.weights import prepare_layer_weights

        _t0 = _time.perf_counter()
        full = runtime.fmt.take_full(layer)
        _t1 = _time.perf_counter()
        # hot 밴드는 이 device에, warm pinned store는 그 GPU의 PCIe root와 같은
        # NUMA 노드에 상주한다 — 둘 다 로더의 입력이고 여기가 결정 지점이다
        # (계약 ③). 원격 소켓 warm은 UVA 읽기에 소켓 간 홉을 추가해 warm의
        # 존립 근거를 무너뜨리는데, 결과는 정확하고 느리기만 해서 어떤 테스트도
        # 잡지 못한다 — 그래서 로더가 배치 후 실제 노드를 검증하고 즉사한다.
        device = torch.device(torch.cuda.current_device())
        warm_kt = os.environ.get(_ENV_WARM_KT) == "1"
        prepared = prepare_layer_weights(
            self.layer_id, full.w13, full.w2, runtime.plan,
            calib=runtime.calib, device=device,
            warm_node=gpu_numa_node(device),
            # cold 스토어의 타일 올림 단위는 커널 키가 함의한다 (계약 ①).
            cold_tile_rows=cold_pack_tile_rows(runtime.plan.kernels.cpu_cold),
            warm_kt=warm_kt,
            fmt=runtime.fmt, w13_scale=full.w13_scale, w2_scale=full.w2_scale,
        )
        del full
        _t2 = _time.perf_counter()
        ep = runtime.plan.expert(self.layer_id, 0)
        if any(ep.proj(p).has_tier(Tier.COLD) for p in Proj):
            runtime.cold().load_layer(self.layer_id, prepared.cold, prepared.thr)
            prepared.cold = None  # 주입 완료 — 소유권은 C++ (계약 ③)
        _t3 = _time.perf_counter()
        if prepared.warm_kt is not None:
            runtime.cold().load_warm_layer(self.layer_id, prepared.warm_kt, prepared.thr,
                                           local_node=_hybrid_local_node(device))

        runtime.executor(device).register_layer(self.layer_id, prepared)

        # full 텐서 소멸 (계약 ③) — host RAM 회수
        runtime.fmt.release(layer)
        self._registered = True
        logger.info("[prism] layer %d registered (hot=%s cold=%s thr=%d) take_full %.1fs prepare %.1fs cold_load %.1fs register %.1fs",
                    self.layer_id, 
                    any(ep.proj(p).has_tier(Tier.HOT) for p in Proj),
                    any(ep.proj(p).has_tier(Tier.COLD) for p in Proj),
                    torch.get_num_threads(),
                    _t1 - _t0, _t2 - _t1, _t3 - _t2, _time.perf_counter() - _t3)

    # ── step-time ────────────────────────────────────────────────────────
    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        runtime = _get_runtime()
        out = runtime.executor(x.device).run_layer(
            self.layer_id, x, topk.topk_ids, topk.topk_weights,
            swiglu_limit=runtime.swiglu_limit,
        )
        if runtime.swiglu_limit is not None:
            # DSV4 2604B 경로 체커: 모델이 "SwiGLU clamp를 적용하는 MoE 경로가 정확히 1회 돌았다"를
            # 층마다 단언한다 (deepseek_v4.py). prism은 clamp를 rejoin에서 적용하므로 같은 신호를 올린다.
            from sglang.srt.debug_utils.deepseek_v4_debug_utils import deepseek_v4_moe_code_path_checker

            deepseek_v4_moe_code_path_checker.observed += 1
        rsf = runtime.routed_scaling_factor
        if rsf is not None and rsf != 1.0:
            # **출력에** 곱한다 — 라우터 가중(topk_weights)에 곱하면 안 된다. 그 배열은 cold
            # 커널의 sparsity 정책이 `s = clip(p − λ(g_e − ḡ), 0, pmax)`로 읽는 입력이고,
            # calib은 `router_weight_norm: sum1`(합 1) 규약으로 만들어졌다. w32를 rsf배 하면
            # λ 항이 rsf배 세져 마스크가 조용히 달라진다. 수학적으로 동일한 두 위치가
            # sparsity 때문에 갈리는 지점이다.
            out = out * rsf
        return StandardCombineInput(hidden_states=out.to(x.dtype))


# ---------------------------------------------------------------------------
# registry 등록 — env가 켜졌을 때만 predicate가 매치된다.
# priority 30: kt(20)보다 바깥 — 단 P0에서는 kt와 동시 사용을 상정하지 않음.
# ---------------------------------------------------------------------------

def _prism_predicate(layer, server_args):
    if not os.environ.get(_ENV_PLAN):
        return None
    return {"layer_id": layer.layer_id}


def _prism_factory(layer, gpu_method, ctx):
    return PrismMoEMethod(gpu_method, ctx["layer_id"])


from sglang.srt.layers.moe.quant_method_registry import register_moe_quant_wrapper

register_moe_quant_wrapper("prism", _prism_predicate, _prism_factory, priority=30)
