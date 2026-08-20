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

P0 제약: TP=1, eager(--disable-cuda-graph) 전용.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.moe.prism.plan import Plan, PlanError, parse_plan, validate_static
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

logger = logging.getLogger(__name__)

_ENV_PLAN = "SGLANG_PRISM_PLAN"
_ENV_MAX_TOKENS = "SGLANG_PRISM_MAX_TOKENS"
_ENV_CPUINFER = "SGLANG_PRISM_CPUINFER_THREADS"
# 실험 노브: 콤마 구분 NUMA node id 리스트 (예: "1" = node 1 단독).
# 설정 시 subpool을 그 노드들에만 만들고, plan의 shard "node i"는
# i번째 서브풀(= 리스트의 i번째 실제 노드)로 매핑된다. 미설정 = 전 노드.
_ENV_NUMA_MAP = "SGLANG_PRISM_NUMA_MAP"


class _PrismRuntime:
    """프로세스 전역 1벌: plan + resources + cold backend + executor.

    (kt의 SharedStagingBuffer/SharedFullContext와 같은 배치 — apply는
    per-layer 호출이라 cross-layer 상태를 전역이 들어야 한다)
    """

    def __init__(self, plan: Plan):
        self.plan = plan
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

            spec = ResourceSpec.from_plan(self.plan, max_tokens=self.max_tokens, device=device)
            self._resources = ExecutionResources(spec)
            self._executor = PrismExecutor(
                self.plan, self._resources, self.cold(), resolve_gpu_kernel(self.plan.kernels.gpu_warm)
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
                )
            else:
                self._cold = KtColdBackend(
                    self.plan,
                    max_tokens=self.max_tokens,
                    num_numa_nodes=numa_node_count(),
                    cpuinfer_threads=threads,
                )
        return self._cold


_RUNTIME: Optional[_PrismRuntime] = None


def _get_runtime() -> _PrismRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        plan_path = os.environ[_ENV_PLAN]
        plan = parse_plan(plan_path)
        validate_static(plan)
        logger.info("[prism] plan loaded: %s (model_id=%s)", plan_path, plan.model_id)
        _RUNTIME = _PrismRuntime(plan)
    return _RUNTIME


class PrismMoEMethod(FusedMoEMethodBase):
    """FusedMoE의 quant_method 자리에 들어가는 Prism 실행기."""

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
        # apply를 우리가 전유하므로 사용되지 않음)
        return self.gpu_method.create_moe_runner(layer, moe_runner_config)

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
        if params_dtype != torch.bfloat16:
            raise NotImplementedError(f"Prism P0 supports bf16 only, got {params_dtype}")

        # full weight를 CPU에 — GPU에는 warm arena만 간다 (계약 ③).
        # trap 방어 (weights.py docstring): gate-first w13 순서 가정 검증
        if getattr(self.gpu_method, "load_up_proj_weight_first", False):
            raise NotImplementedError("Prism assumes gate-first w13 ordering")
        w13 = torch.nn.Parameter(
            torch.empty(num_experts, 2 * inter_full, hidden_size,
                        dtype=params_dtype, device="cpu"),
            requires_grad=False,
        )
        w2 = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, inter_full,
                        dtype=params_dtype, device="cpu"),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13)
        layer.register_parameter("w2_weight", w2)
        set_weight_attrs(w13, extra_weight_attrs)
        set_weight_attrs(w2, extra_weight_attrs)

    def process_weights_after_loading(self, layer) -> None:
        from sglang.srt.layers.moe.prism.plan import Proj, Tier
        from sglang.srt.layers.moe.prism.weights import prepare_layer_weights

        runtime = _get_runtime()
        # 주의: loader.py의 device_loading_context가 이 훅 직전에 파라미터를
        # CUDA로 옮겨둔다 (훅 종료 후 원복). cold 주입은 C++가 host memcpy로
        # 읽으므로 반드시 CPU 사본에서 슬라이스해야 한다 — CUDA 텐서의
        # data_ptr()를 넘기면 device 주소를 memcpy하다 segfault (2026-08-20
        # 스모크에서 실제 발생). TODO: 컨텍스트 왕복(H2D+D2H ~2.4GB/층) 회피.
        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data
        if w13.is_cuda:
            w13 = w13.cpu()
        if w2.is_cuda:
            w2 = w2.cpu()
        prepared = prepare_layer_weights(self.layer_id, w13, w2, runtime.plan)
        ep = runtime.plan.expert(self.layer_id, 0)
        if any(ep.proj(p).has_tier(Tier.COLD) for p in Proj):
            runtime.cold().load_layer(self.layer_id, prepared.cold)
            prepared.cold = None  # 주입 완료 — 소유권은 C++ (계약 ③)

        device = torch.device(torch.cuda.current_device())
        runtime.executor(device).register_layer(self.layer_id, prepared)

        # full 텐서 소멸 (계약 ③) — host RAM 회수
        layer.w13_weight.data = torch.empty(0, dtype=layer.w13_weight.dtype)
        layer.w2_weight.data = torch.empty(0, dtype=layer.w2_weight.dtype)
        self._registered = True
        logger.info("[prism] layer %d registered (cold=%s)", self.layer_id,
                    any(ep.proj(p).has_tier(Tier.COLD) for p in Proj))

    # ── step-time ────────────────────────────────────────────────────────
    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        runtime = _get_runtime()
        out = runtime.executor(x.device).run_layer(
            self.layer_id, x, topk.topk_ids, topk.topk_weights
        )
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
