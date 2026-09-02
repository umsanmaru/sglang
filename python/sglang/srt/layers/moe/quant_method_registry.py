"""Registry for MoE quant-method wrappers.

Plugin slot used by FusedMoE.__init__ to optionally wrap a base GPU
quant_method (Fp8MoEMethod, ModelOptNvFp4FusedMoEMethod, …) with a
model-specific wrapper (e.g. KTEPWrapperMethod for DeepSeek V4 Flash CPU/GPU
expert split).

Plugins register themselves at import time. The DSV4 plugin is pulled in
when sglang.srt.models.deepseek_v4 is auto-discovered by ModelRegistry —
no base file imports the wrapper directly.
"""

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
    from sglang.srt.server_args import ServerArgs


_Predicate = Callable[[Any, "ServerArgs"], Optional[Any]]
_Factory = Callable[[Any, "FusedMoEMethodBase", Any], "FusedMoEMethodBase"]

# Each entry: (priority, wrapper_id, predicate, factory). LOWER priority runs
# FIRST (i.e. wraps the innermost method), matching the original PR #38 layout
# where Phase 2 (mxfp4) wrapped before Phase 3 (kt_ep). Iteration is sorted by
# priority on each `maybe_wrap_moe_quant_method` call so registration order
# from import sequence does not silently change wrap order.
_QUANT_WRAPPERS: List[Tuple[int, str, _Predicate, _Factory]] = []


def register_moe_quant_wrapper(
    wrapper_id: str,
    predicate: _Predicate,
    factory: _Factory,
    priority: int = 100,
) -> None:
    """Register a wrapper.

    Args:
      wrapper_id: stable id used by `is_wrapped_method` for isinstance-style
        checks without importing the wrapper class.
      predicate: (layer, server_args) -> Optional[ctx]. Return None to skip,
        otherwise return an opaque ctx that's passed to factory.
      factory: (layer, gpu_method, ctx) -> wrapped quant_method.
      priority: lower numbers wrap first (i.e. innermost). For DSV4: mxfp4
        registers at priority 10 (Phase 2), kt_ep at priority 20 (Phase 3).
    """
    for _, existing_id, _, _ in _QUANT_WRAPPERS:
        if existing_id == wrapper_id:
            return
    _QUANT_WRAPPERS.append((priority, wrapper_id, predicate, factory))


def _maybe_import_prism() -> None:
    """Prism(K-split tiered MoE offload)은 모델 파일이 아니라 env로 활성화되므로
    여기서 지연 import한다 (import 시 self-register). env가 없으면 no-op."""
    import os

    if os.environ.get("SGLANG_PRISM_PLAN"):
        import sglang.srt.layers.moe.prism.method  # noqa: F401


def maybe_wrap_moe_quant_method(
    layer: Any, gpu_method: "FusedMoEMethodBase", server_args: "ServerArgs"
) -> "FusedMoEMethodBase":
    _maybe_import_prism()
    """Iterate predicates in priority order (lower first); chain-wrap with each
    that matches. For DSV4, the final method is
    KTEPWrapperMethod(DeepSeekMxfp4MoEMethod(gpu_method)) because mxfp4
    is registered at priority 10 and kt_ep at priority 20."""
    method = gpu_method
    for _priority, wrapper_id, predicate, factory in sorted(_QUANT_WRAPPERS):
        ctx = predicate(layer, server_args)
        if ctx is not None:
            method = factory(layer, method, ctx)
            if getattr(method, "_quant_wrapper_id", None) is None:
                method._quant_wrapper_id = wrapper_id
    if method is not gpu_method:
        _detach_stale_module_entry(layer, "quant_method", gpu_method)
    return method


def _detach_stale_module_entry(layer, attr: str, inner) -> None:
    """호출자의 `layer.<attr> = wrapper` 대입이 통과하게 낡은 등록을 떼어낸다.

    내부 method 중 일부는 자신이 nn.Module이다 — `UnquantizedFusedMoEMethod`는
    `BaseFusedOp(nn.Module)`를 상속하지만 `Fp8MoEMethod`와 mxfp4 경로는 아니다.
    Module인 경우 호출자가 원래 했던 `self.quant_method = <inner>`가 그것을
    `layer._modules`에 등록해버렸고, 그 뒤 `nn.Module.__setattr__`은 같은 이름에
    non-Module 대입을 **거부한다**:

        TypeError: cannot assign 'PrismMoEMethod' as child module
                   'quant_method' (torch.nn.Module or None expected)

    래퍼를 Module로 만들지 않는 것은 의도다 — 모듈 트리에 들어가면 Prism이 host에
    둔 텐서가 `.to(device)` 순회에 노출되고, 그것은 티어 배치가 절대 잃어선 안 되는
    성질이다. 그래서 래퍼를 바꾸는 대신 낡은 등록을 지운다. 내부 method는 래퍼의
    속성으로 계속 살아 있고 파라미터·버퍼를 들고 있지 않으므로 state dict에서
    빠지는 것도 없다.
    """
    modules = layer.__dict__.get("_modules")
    if modules is not None and modules.get(attr, None) is inner:
        del modules[attr]


def is_wrapped_method(method: Any, wrapper_id: str) -> bool:
    """isinstance replacement that doesn't require importing the wrapper class."""
    return getattr(method, "_quant_wrapper_id", None) == wrapper_id
