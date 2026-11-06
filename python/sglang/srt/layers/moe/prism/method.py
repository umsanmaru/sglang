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

TP: MoE-TP owner(rank 0)만 expert를 갖고 나머지 rank는 0을 낸다 (is_tp_owner 참조,
2026-09-06 — K3를 GPU 2장 dense + CPU expert로 돌리기 위해). EP(a2a)는 미지원. batch size 제약은 없다 — GPU 티어가 pair-native가 되면서
그룹 조성의 host 결정이 사라졌고, eager·캡처·prefill이 모두 같은 경로다
(2026-08-25). executor가 캡처 구간을 자동 감지해 cold를 stream 통합
(kt host node)으로 돌린다.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.moe.prism.activation import Activation
from sglang.srt.layers.moe.prism.plan import Plan, PlanError, Proj, Tier, parse_plan, validate_static
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

from sglang.srt.layers.moe.prism.runtime.config import (
    PrismEnvConfig,
    _ENV_PLAN as _ENV_PLAN,
    _ENV_MAX_TOKENS as _ENV_MAX_TOKENS,
    _ENV_CPUINFER as _ENV_CPUINFER,
    _ENV_NUMA_MAP as _ENV_NUMA_MAP,
    _ENV_COLD_STREAM as _ENV_COLD_STREAM,
    _ENV_GROUPED_MIN_M as _ENV_GROUPED_MIN_M,
    _ENV_SPLIT_STREAMS as _ENV_SPLIT_STREAMS,
    _ENV_COLD_GPU_MIN_M as _ENV_COLD_GPU_MIN_M,
    _ENV_COLD_HYBRID_FRAC as _ENV_COLD_HYBRID_FRAC,
    _ENV_WARM_KT as _ENV_WARM_KT,
    _ENV_WARM_CPU_MIN_M as _ENV_WARM_CPU_MIN_M,
    _ENV_WARM_CPU_MAX_M as _ENV_WARM_CPU_MAX_M,
    _ENV_COLD_ASYNC as _ENV_COLD_ASYNC,
    _ENV_COLD_SPLIT as _ENV_COLD_SPLIT,
    _ENV_LOAD_THREADS as _ENV_LOAD_THREADS,
    _ENV_MASK_STATS as _ENV_MASK_STATS,
    _ENV_MASK_STATS_EVERY as _ENV_MASK_STATS_EVERY,
    _ENV_MASK_STATS_NANPROBE as _ENV_MASK_STATS_NANPROBE,
)

from sglang.srt.layers.moe.prism.runtime.context import PrismRuntime, RuntimeBindings

# Keep legacy environment-name imports available to profilers and tests.
logger = logging.getLogger(__name__)
_CONFIG = PrismEnvConfig()


def _load_threads() -> int:
    return _CONFIG.load_threads()


def _hybrid_local_node(device) -> int:
    from sglang.srt.layers.moe.prism.numa import gpu_numa_node

    return _CONFIG.hybrid_local_node(gpu_numa_node(device))


def _cold_gpu_min_m():
    from sglang.srt.layers.moe.prism.executor import PrismExecutor

    def default(grouped_min_m):
        return _get_runtime().fmt.default_cold_gpu_min_m(
            PrismExecutor.COLD_GPU_MIN_M,
            PrismExecutor.GROUPED_MIN_M if grouped_min_m is None
            else _CONFIG.parse_grouped_min_m(grouped_min_m))

    return _CONFIG.cold_gpu_min_m(default)


def _warm_kt_enabled(fmt, *, has_cold: bool = True, all_proj_warm: bool = True):
    return _CONFIG.warm_kt_enabled(
        fmt.name, _cold_gpu_min_m, has_cold=has_cold, all_proj_warm=all_proj_warm)


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


class _PrismRuntime(PrismRuntime):
    """Legacy constructor; singleton and live integration bindings stay here."""

    def __init__(self, plan: Plan, calib=None):
        super().__init__(plan, calib, bindings=RuntimeBindings(
            config=lambda: _CONFIG,
            cold_gpu_min_m=lambda: _cold_gpu_min_m(),
            warm_kt_enabled=lambda fmt: _warm_kt_enabled(fmt),
            hybrid_local_node=lambda device: _hybrid_local_node(device),
            capture_mode_fn=lambda: _sglang_capture_mode,
        ))


_RUNTIME: Optional[_PrismRuntime] = None


def moe_tp_coords() -> tuple[int, int, int]:
    """(moe_tp_rank, moe_tp_size, moe_ep_size). 분산 초기화 전(단위 테스트·프로파일러)에는 (0, 1, 1)."""
    try:
        from sglang.srt.runtime_context import get_parallel

        p = get_parallel()
        return int(p.moe_tp_rank), int(p.moe_tp_size), int(p.moe_ep_size)
    except Exception:
        return 0, 1, 1


def is_tp_owner() -> bool:
    """MoE-TP에서 routed expert를 실제로 가진 rank인가 (rank 0). TP=1이면 항상 True.

    TP 설계 (kt KTEPWrapperMethod와 같은 배치): expert 가중치·CPU 풀·GPU 티어는 **rank 0 하나**가
    전부 갖고(K-split은 TP shard가 아니라 whole row로), 나머지 rank는 0을 낸다 — 모델의 post-experts
    reduce(all-reduce / reduce-scatter)는 합이므로 결과가 같다. attention·shared expert·latent proj는
    평범한 TP로 GPU에 갈라진다. 이게 "GPU 2장에 dense, CPU에 expert"의 최소 형태다."""
    rank, size, _ = moe_tp_coords()
    return size == 1 or rank == 0


def _get_runtime() -> _PrismRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        plan_path = _CONFIG.plan_path
        from sglang.srt.layers.moe.prism.loading_cache import load_plan_cached

        import time

        metadata_start = time.perf_counter()
        plan = load_plan_cached(plan_path)
        plan_ready = time.perf_counter()
        # sparsity plan이면 자산을 먼저 열어 shape까지 검증한다 (계약 ①):
        # 다른 모델의 calib을 적용하는 것은 dims 불일치와 같은 급의 silent
        # failure이므로 startup에서 죽는 편이 낫다. 비-owner rank는 자산을 열지 않는다
        # (executor를 만들지 않으므로 필요 없고, 382 MB를 rank마다 읽을 이유가 없다).
        calib = None
        if plan.sparsity is not None and is_tp_owner():
            from sglang.srt.layers.moe.prism.calib import CalibTables

            calib = CalibTables.load(plan.sparsity)
        calib_ready = time.perf_counter()
        validate_static(
            plan, calib_probe=None if calib is None else calib.probe()
        )
        logger.info("[prism] metadata loading: plan %.2fs calibration %.2fs validation %.2fs",
                    plan_ready - metadata_start, calib_ready - plan_ready,
                    time.perf_counter() - calib_ready)
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
        rank, size, ep = moe_tp_coords()
        if ep != 1:
            raise NotImplementedError(
                f"Prism does not support expert parallelism (moe_ep_size={ep}); use plain TP")
        self.tp_rank, self.tp_size = rank, size
        self.is_owner = is_tp_owner()
        # FusedMoE.weight_loader(fused_moe_triton/layer.py)의 prism hook이 읽는 모드:
        #   None   TP=1 — 평범한 로딩
        #   "full" owner — TP shard를 자르지 않고 whole row를 받는다
        #   "skip" 비-owner — 이 층의 routed expert 텐서를 버린다
        self.prism_tp_mode = None if size == 1 else ("full" if self.is_owner else "skip")

    def __getattr__(self, name):
        # __init__ 이전/자기 속성 미존재 시 재귀 방지 (kt의 알려진 함정)
        if name in ("gpu_method", "layer_id", "_registered", "tp_rank", "tp_size", "is_owner",
                    "prism_tp_mode"):
            raise AttributeError(name)
        return getattr(self.gpu_method, name)

    def create_moe_runner(self, layer, moe_runner_config):
        # base 클래스에 실체(raise NotImplementedError)가 있어 __getattr__이
        # 안 잡는다 — 명시적 위임 (runner는 gpu_method 것이 생성되지만
        # apply를 우리가 전유하므로 사용되지 않음). 모델의 활성화(activation / swiglu_limit /
        # gemm1_alpha / gemm1_clamp_limit)는 여기서만 보이므로(MoeRunnerConfig) runtime에 적어
        # rejoin이 쓴다. 모르는 활성화는 from_runner_config가 즉사시킨다 — 조용히 silu로 계산하면
        # "돌아가는데 다른 모델"이 된다.
        act = Activation.from_runner_config(moe_runner_config)
        runtime = _get_runtime()
        if runtime.activation is None:
            logger.info("[prism] MoE activation = %s (rejoin#1에서 적용)", act)
            runtime.activation = act
        elif runtime.activation != act:
            raise PlanError(f"MoE activation differs across layers ({runtime.activation} vs {act})")
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
        # FusedMoE는 intermediate를 moe_tp_size로 갈라 넘긴다. prism의 expert는 TP shard가 아니라
        # whole row(inter_full)이고 owner rank에만 있다 (is_tp_owner 참조).
        inter_full = intermediate_size_per_partition * getattr(layer, "moe_tp_size", 1)
        if not self.is_owner:
            # 파라미터를 하나도 만들지 않는다: 모델 load_weights는 `name not in params_dict`로
            # expert 텐서를 건너뛰고(kimi_k3.py), 로더 hook은 prism_tp_mode="skip"으로 버리며,
            # apply는 0을 낸다. dummy 로더도 만질 파라미터가 없다.
            logger.info("[prism] layer %d: MoE-TP rank %d/%d holds no experts (owner = rank 0)",
                        self.layer_id, self.tp_rank, self.tp_size)
            return
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
        # mxfp4 파라미터 명명은 감싸는 GPU method가 정한다 (formats.Mxfp4Format 참조):
        # DeepSeekMxfp4MoEMethod → *_weight_scale_inv fp32 / Mxfp4MoEMethod(K3, compressed-tensors)
        # → *_weight_scale u8. 로더는 이름으로 채우므로 여기서 어긋나면 배율이 조용히 1.0으로 남는다.
        from sglang.srt.layers.moe.prism.formats import Mxfp4Format

        layer._prism_mxfp4_naming = Mxfp4Format.naming_for_method(self.gpu_method)
        runtime.fmt.create_params(layer, num_experts, hidden_size, inter_full, params_dtype,
                                  extra_weight_attrs)

    def process_weights_after_loading(self, layer) -> None:
        if not self.is_owner:
            self._registered = True   # 가진 것이 없다 — 등록할 것도 없다
            return
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
        # A single packed host copy serves small-prefill CPU and buffered GPU.
        # 자동 켜짐은 cold가 있고 세 proj 모두 어딘가에 warm 행이 있을 때만이다 — warm-kt
        # 인스턴스는 gate/up/down을 함께 요구하므로(_warm_kt_tensors) 한쪽만 있으면 즉사한다.
        warm_kt = _warm_kt_enabled(
            runtime.fmt, has_cold=runtime.has_cold,
            all_proj_warm=all(any(runtime.plan.expert(self.layer_id, e).proj(p).has_tier(Tier.WARM)
                                  for e in range(runtime.plan.dims.num_experts)) for p in Proj))
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
        if not self.is_owner:
            # 비-owner rank의 몫은 0 — 모델의 post-experts reduce(합)가 owner의 결과를 전 rank에 퍼뜨린다.
            # (K3 `_forward_unfused`: experts → +shared → all_reduce; DSV2/GLM: 같은 자리의 all-reduce /
            # reduce-scatter. EP a2a 경로는 reduce를 생략하므로 지원하지 않는다 — __init__에서 즉사.)
            return StandardCombineInput(hidden_states=torch.zeros_like(x))
        runtime = _get_runtime()
        out = runtime.executor(x.device).run_layer(
            self.layer_id, x, topk.topk_ids, topk.topk_weights,
            activation=runtime.activation,
        )
        if runtime.activation is not None and runtime.activation.kind == "silu" \
                and runtime.activation.limit is not None:
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
    if not _CONFIG.enabled:
        return None
    return {"layer_id": layer.layer_id}


def _prism_factory(layer, gpu_method, ctx):
    return PrismMoEMethod(gpu_method, ctx["layer_id"])


from sglang.srt.layers.moe.quant_method_registry import register_moe_quant_wrapper

register_moe_quant_wrapper("prism", _prism_predicate, _prism_factory, priority=30)
