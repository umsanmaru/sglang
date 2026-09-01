"""Prism dense의 sglang 접점 — linear quant-method wrapper.

차단선: `LinearBase` 위로는 Prism을 모른다. 이 모듈이 유일한 등록 지점이며,
활성화는 env `SGLANG_PRISM_LINEAR_PLAN=<plan.json>` 로만 된다 (MoE의
`SGLANG_PRISM_PLAN`과 **독립**이다 — 둘을 따로 켜고 끌 수 있어야 벤치가 성립한다).

MoE `moe/prism/method.py`와 같은 자리, 같은 세 훅:

    create_weights                 full weight를 **CPU 파라미터**로 할당
                                   (sglang weight_loader가 그대로 채운다)
    process_weights_after_loading  절단 → executor 등록 → CPU 파라미터 해제
                                   (full 텐서 소멸 — 계약 ③)
    apply                          executor.run 위임 한 줄

MoE와 갈리는 것:

  * **좌표가 prefix에서 온다.** `LinearBase`에는 `layer_id`가 없다 — registry의
    predicate가 `split_prefix`로 `(layer, name)`을 뽑아 ctx로 넘긴다.
  * **`apply`가 텐서를 그대로 돌려준다.** `dispatch_output`/`CombineInput` 래핑이
    없고, bias는 여기서 더한다.
  * **TP 방어가 `create_weights`에 있다.** predicate 시점에는 `tp_size`가 아직
    없다 (서브클래스가 `super().__init__()` 반환 후 대입한다 —
    `layers/linear_method_registry.py` 참조).

P0 제약: TP=1, bf16 스토어, cold 미배선. 셋 다 조용히 넘어가지 않고 즉사한다.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch

from sglang.srt.layers.prism.geometry import PlanError, Tier
from sglang.srt.layers.prism.kernels import gpu_store_format_tag

logger = logging.getLogger(__name__)

_ENV_PLAN = "SGLANG_PRISM_LINEAR_PLAN"
_ENV_MAX_TOKENS = "SGLANG_PRISM_LINEAR_MAX_TOKENS"
# cold CPU 스레드 수. 벤치는 28이 최적이었지만(물리 16코어 = 8×2소켓; 14→28은 7%
# 개선, 56은 9배 악화) 실서버는 GPU 스트림·스케줄러와 경합하므로 재측정 대상이다.
_ENV_COLD_THREADS = "SGLANG_PRISM_LINEAR_COLD_THREADS"
# MoE와 공유하는 이름. dense 전용 값이 없으면 이쪽을 본다 — 실행
# 스크립트가 이미 이 이름을 내보내고 있어서, 안 맞추면 지정한 스레드
# 수가 조용히 무시되고 기본값 28로 돈다.
_ENV_CPUINFER = "SGLANG_PRISM_CPUINFER_THREADS"

# 래핑이 **로더 선택을 바꾸는** 메서드들. `ColumnParallelLinear.__init__`이
# `self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED`로 v1/v2
# weight_loader를 고르는데(`linear.py:367`), 우리가 감싸면 그 이름이 바뀐다 —
# v2를 쓰던 메서드는 조용히 v1으로 떨어져 shard 로딩이 어긋난다. 이름으로 하는
# 판정이라 래퍼가 흉내낼 수 없으므로, 해당 메서드는 지금 감싸지 않는다.
# (bf16 `UnquantizedLinearMethod`는 애초에 v1이라 영향이 없다.)
_V2_LOADER_UNSUPPORTED = frozenset({
    "CompressedTensorsLinearMethod", "AWQMarlinLinearMethod", "AWQLinearMethod",
    "AWQLinearAscendMethod", "GPTQMarlinLinearMethod", "Fp8LinearMethod",
    "BlockInt8LinearMethod", "MarlinLinearMethod", "QQQLinearMethod",
    "GPTQMarlin24LinearMethod", "TPUInt8LinearMethod", "GPTQLinearMethod",
})


class _LinearRuntime:
    """프로세스 전역 1벌: plan + executor.

    `apply`가 per-layer 호출이라 cross-layer 상태를 전역이 들어야 한다 (MoE의
    `_PrismRuntime`과 같은 배치).
    """

    def __init__(self, plan):
        self.plan = plan
        self.max_tokens = int(os.environ.get(_ENV_MAX_TOKENS, "4096"))
        self._executor = None
        self._checked = False
        self._calib = None

    @property
    def calib(self):
        """calib 자산 — plan에 sparsity가 없으면 None이고 열지도 않는다.

        `check_plan`을 여기서 1회 돌린다: plan이 마스킹하겠다고 한 모든
        (층, proj, 조각)이 자산에 실제로 있는지, 그리고 전부 0이 아닌지를 첫 층
        prepare **전에** 확인해야 한다 — 42층을 다 채운 뒤 죽으면 5분을 버린다.
        """
        if self.plan.sparsity is None:
            return None
        if self._calib is None:
            from sglang.srt.layers.prism.linear.calib import LinearCalibTables

            self._calib = LinearCalibTables.load(self.plan.sparsity)
            self._calib.check_plan(self.plan)
            logger.info("[prism-linear] calib loaded: %s (score=%s)",
                        self.plan.sparsity.calib.path, self.plan.sparsity.score)
        return self._calib

    def executor(self, device: torch.device):
        if self._executor is None:
            from sglang.srt.layers.prism.linear.executor import LinearExecutor
            from sglang.srt.layers.prism.linear.formats import FORMATS
            from sglang.srt.layers.prism.linear.rejoin import warmup as warmup_rejoin

            # **캡처 안에서 처음 일어나면 안 되는 일**을 전부 startup으로 앞당긴다:
            # JIT 컴파일과 지연 할당. 지금 쓰는 `breakable` prefill 백엔드는 캡처 전에
            # forward를 2회 돌려주지만(`breakable_cuda_graph_backend.py:117`) 그건 그
            # 백엔드의 성질이지 계약이 아니다 — `tc_piecewise`로 바꾸거나 decode graph만
            # 켜면 컴파일이 캡처 순서에 얽힌다. MoE `method.py:180`과 같은 이유·같은 자리.
            warmup_rejoin(device)
            for tag in {pp.kernels.gpu_warm for pp in self.plan.projs.values()}:
                FORMATS[gpu_store_format_tag(tag)].warmup()
            cold, resources = self._cold_backend()
            self._executor = LinearExecutor(max_tokens=self.max_tokens, device=device,
                                            cold=cold, resources=resources)
            self._executor.warmup(device)
            if resources is not None:
                resources.warmup()
        return self._executor

    def _cold_backend(self):
        """plan에 COLD 행이 있을 때만 만든다. 없으면 kt를 import조차 하지 않는다."""
        if not any(pp.has_tier(Tier.COLD) for pp in self.plan.projs.values()):
            return None, None
        from sglang.srt.layers.prism.linear.cold_backend import KtLinearColdBackend
        from sglang.srt.layers.prism.linear.resources import LinearColdResources
        from sglang.srt.layers.prism.numa import numa_node_count

        nodes = numa_node_count()
        threads = int(os.environ.get(_ENV_COLD_THREADS,
                             os.environ.get(_ENV_CPUINFER, "28")))
        logger.info("[prism-linear] cold backend: %d NUMA nodes, %d CPUInfer threads",
                    nodes, threads)
        return (
            KtLinearColdBackend(self.plan, max_tokens=self.max_tokens,
                                num_numa_nodes=nodes, cpuinfer_threads=threads),
            LinearColdResources(max_tokens=self.max_tokens),
        )

    def finalize(self) -> None:
        """cold 그룹을 굳히고 pack한다 (첫 step 1회).

        로딩 중에 못 하는 이유는 그룹의 expert 수다 — `check_coverage`가 좌표가
        아니라 이름으로 세는 것과 같은 이유로, 어느 layer가 어떤 projection을
        갖는지는 plan만으로 알 수 없다.
        """
        if self._executor is not None:
            self._executor.finalize()

    def check_coverage(self) -> None:
        """plan의 proj 이름이 **하나라도** 걸렸는가 (첫 step에서 1회).

        아무 layer에도 안 걸린 이름은 오타다 — 조용히 stock 경로로 돌아 "켰는데
        안 켜진" 상태가 되는데, 성능만 달라져 어떤 테스트도 안 잡는다.

        **좌표 단위가 아니라 이름 단위**로 보는 이유: plan의 `projs`는 전 layer로
        전개되지만 모델은 layer마다 projection 집합이 다를 수 있다 (Qwen3.8-27B는
        `self_attn.qkv_proj`가 64층 중 full_attention 16층에만 있고 나머지는
        `linear_attn.*`이다). 좌표를 전부 요구하면 정상 모델에서 오탐이 난다.
        대신 이름별 매칭 수를 로그로 남겨 배분이 의도대로인지 볼 수 있게 한다.
        """
        if self._checked or self._executor is None:
            return
        self._checked = True
        got = self._executor.registered()
        counts = {name: 0 for name in self.plan.names()}
        for _, name in got:
            counts[name] = counts.get(name, 0) + 1
        never = sorted(n for n, c in counts.items() if c == 0)
        if never:
            raise PlanError(
                f"prism dense: these planned projections never matched any linear "
                f"layer: {never}. plan의 proj 이름이 이 모델의 prefix와 다르다 "
                f"(예: 'mlp.gate_up_proj' vs 'mlp.gate_proj')"
            )
        logger.info(
            "[prism-linear] coverage: %s (총 %d개 projection 등록)",
            ", ".join(f"{n}×{c}" for n, c in sorted(counts.items())), len(got),
        )


_RUNTIME: Optional[_LinearRuntime] = None


def _runtime() -> _LinearRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        from sglang.srt.layers.prism.linear.plan import parse_plan, validate_static

        path = os.environ[_ENV_PLAN]
        plan = parse_plan(path)
        validate_static(plan)
        logger.info(
            "[prism-linear] plan loaded: %s (model_id=%s, projs=%s, sparsity=%s)",
            path, plan.model_id, sorted(plan.names()),
            "none" if plan.sparsity is None else plan.sparsity.score,
        )
        _RUNTIME = _LinearRuntime(plan)
    return _RUNTIME


class PrismLinearMethod:
    """`LinearBase.quant_method` 자리에 들어가는 Prism 실행기."""

    # loader.py의 `device_loading_context` 왕복 opt-out (MoE `PrismMoEMethod`와 같은 계약).
    # 그 컨텍스트는 CPU offload용이고 판정 기준이 `device.type == "cpu"` 하나뿐이라,
    # **의도적으로** host에 파라미터를 만드는 우리를 offload로 오해한다. 그대로 두면
    # layer마다 full weight를 GPU로 올렸다 내린 뒤 그 사본을 버린다 — 순수 낭비인데
    # **에러 없이 느려지기만** 해서 어떤 테스트도 안 잡는다.
    keeps_params_on_host = True

    def __init__(self, inner, layer_idx: int, name: str):
        self.inner = inner          # 미지 속성의 위임처
        self.layer_idx = layer_idx
        self.name = name

    def __getattr__(self, attr):
        # __init__ 이전/자기 속성 미존재 시 재귀 방지 (kt·MoE prism의 알려진 함정)
        if attr in ("inner", "layer_idx", "name"):
            raise AttributeError(attr)
        return getattr(self.inner, attr)

    # ── Stage 1: 파라미터 ────────────────────────────────────────────────
    def create_weights(self, layer, input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype, **extra_weight_attrs):
        from sglang.srt.layers.prism.linear.plan import check_dims, check_partition
        from sglang.srt.utils import set_weight_attrs

        rt = _runtime()
        where = f"layer {self.layer_idx} proj '{self.name}'"
        if input_size_per_partition != input_size:
            raise NotImplementedError(
                f"{where}: Prism dense supports TP=1 only (K is sharded: "
                f"{input_size_per_partition} of {input_size}) — RowParallelLinear의 "
                f"TP는 prism과 같은 K축을 쪼갠다"
            )
        n = sum(output_partition_sizes)
        if n != output_size:
            raise NotImplementedError(
                f"{where}: Prism dense supports TP=1 only (N is sharded: {n} of {output_size})"
            )
        pp = rt.plan.get(self.layer_idx, self.name)
        if pp is None:                       # predicate가 걸렀어야 한다
            raise PlanError(f"{where}: wrapped but not in the plan")
        check_dims(pp, input_size, n, where)
        check_partition(pp, output_partition_sizes, where)
        if params_dtype != torch.bfloat16:
            raise NotImplementedError(
                f"{where}: Prism dense v1 supports bf16 params only, got {params_dtype}"
            )

        # full weight를 **CPU**에 — GPU에는 hot 스토어만 간다 (계약 ③). cold 주입은
        # C++가 host memcpy로 읽으므로 device 텐서를 넘기면 segfault다.
        weight = torch.nn.Parameter(
            torch.empty(n, input_size, dtype=params_dtype, device="cpu"),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    # ── Stage 2: 절단·등록 ───────────────────────────────────────────────
    def process_weights_after_loading(self, layer) -> None:
        import time

        from sglang.srt.layers.prism.linear.weights import prepare_linear_weights
        from sglang.srt.layers.prism.numa import gpu_numa_node

        rt = _runtime()
        where = f"layer {self.layer_idx} proj '{self.name}'"
        t0 = time.perf_counter()
        # 로더의 device_loading_context가 파라미터를 CUDA로 옮겨뒀을 수 있다
        # (훅 종료 후 원복). cold 슬라이스는 host 메모리에서 해야 한다.
        w = layer.weight.data
        w = w.cpu() if w.is_cuda else w

        device = torch.device(torch.cuda.current_device())
        prepared = prepare_linear_weights(
            self.layer_idx, self.name, w, rt.plan,
            device=device, warm_node=gpu_numa_node(device), pin_memory=True,
            calib=rt.calib,
        )
        del w
        rt.executor(device).register(self.layer_idx, self.name, prepared,
                                     sparsity=rt.plan.sparsity)

        # full 텐서 소멸 (계약 ③) — host RAM 회수
        layer.weight.data = torch.empty(0, dtype=layer.weight.dtype)
        logger.debug(
            "[prism-linear] %s registered (hot=%d warm=%d cold=%d) %.2fs",
            where, prepared.rows(Tier.HOT), prepared.rows(Tier.WARM),
            prepared.rows(Tier.COLD), time.perf_counter() - t0,
        )

    # ── step-time ────────────────────────────────────────────────────────
    def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
              ) -> torch.Tensor:
        rt = _runtime()
        if not isinstance(x, torch.Tensor):
            # DeepseekV2MLP의 gemm_output_zero_allocator 경로 (uint8 weight 전용).
            # 조용히 무시하면 사전할당 버퍼가 버려지고 모양이 어긋난다.
            raise NotImplementedError(
                f"layer {self.layer_idx} proj '{self.name}': prism dense does not "
                f"support the pre-allocated-output call form"
            )
        rt.finalize()
        rt.check_coverage()
        shape = x.shape
        x2d = x if x.dim() == 2 else x.view(-1, shape[-1])
        # sparsity는 decode에만 (계약 ①: prefill은 dense).
        masking = rt.plan.sparsity is not None and x2d.shape[0] == 1
        out = rt.executor(x.device).run(self.layer_idx, self.name, x2d, masking=masking)
        if bias is not None:
            out = out + bias
        return out if x.dim() == 2 else out.view(*shape[:-1], out.shape[-1])


# ---------------------------------------------------------------------------
# registry 등록 — env가 켜졌을 때만 predicate가 매치된다.
# ---------------------------------------------------------------------------


def _predicate(layer, prefix: str, server_args):
    from sglang.srt.layers.prism.linear.plan import split_prefix

    if not os.environ.get(_ENV_PLAN):
        return None
    coord = split_prefix(prefix)
    if coord is None:                    # lm_head, embedding 이웃, vision tower …
        return None
    layer_idx, name = coord
    plan = _runtime().plan
    if plan.get(layer_idx, name) is None:
        return None
    return {"layer_idx": layer_idx, "name": name}


def _factory(layer, inner, ctx):
    cls = inner.__class__.__name__
    if cls in _V2_LOADER_UNSUPPORTED:
        raise NotImplementedError(
            f"layer {ctx['layer_idx']} proj '{ctx['name']}': cannot wrap {cls} — "
            f"그 메서드는 v2 weight_loader를 쓰는데 래핑하면 클래스 이름이 바뀌어 "
            f"sglang이 조용히 v1으로 떨어진다 (linear.py의 WEIGHT_LOADER_V2_SUPPORTED). "
            f"bf16(UnquantizedLinearMethod) plan만 지원한다"
        )
    return PrismLinearMethod(inner, ctx["layer_idx"], ctx["name"])


from sglang.srt.layers.linear_method_registry import register_linear_quant_wrapper

register_linear_quant_wrapper("prism_linear", _predicate, _factory, priority=30)
