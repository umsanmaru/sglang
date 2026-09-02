"""Registry for linear (non-MoE) quant-method wrappers.

Mirror of `layers/moe/quant_method_registry.py`, for `LinearBase` instead of
`FusedMoE`. The MoE side has had a wrapper slot since Phase 2/3 (mxfp4_deepseek,
kt_ep, prism); the dense side had none — `LinearBase.__init__` picked its
`quant_method` and no plugin could get between that choice and the subclass's
`create_weights()` call.

That gap is the whole point of this module. `create_weights` is where a wrapper
decides *where the parameters get allocated* (Prism allocates the full weight on
CPU, slices it into K-tiers, and frees the original), so a wrapper attached after
the fact is already too late.

Three things differ from the MoE registry, all forced by what `LinearBase` is:

  * **`prefix` is part of the predicate signature.** `FusedMoE` carries
    `layer.layer_id`; `LinearBase` carries nothing that says which projection of
    which decoder layer it is. The dotted state-dict prefix
    (`model.layers.7.self_attn.wq_b`) is the only coordinate available, so it is
    passed explicitly rather than fished off the layer.

  * **Every linear in the model passes through here** — lm_head, MoE routers,
    LoRA shims, vision towers. A predicate that does not recognise a prefix MUST
    return None. Default state is an empty registry, and the fast path (nothing
    registered, no lazy-import env set) is one `os.environ.get` and a truth test.

  * **`tp_rank`/`tp_size` are NOT readable from a predicate.** Subclasses assign
    them *after* `super().__init__()` returns (see `ColumnParallelLinear.__init__`
    and `RowParallelLinear.__init__`), so the hook fires before they exist. A
    wrapper that cares about TP must check in `create_weights`, which does run
    after the assignment.

Plugins register themselves at import time, exactly as on the MoE side.
"""

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
    from sglang.srt.server_args import ServerArgs


# (layer, prefix, server_args) -> Optional[ctx]; None means "not mine".
# server_args is Optional: unlike FusedMoE (which needs the global anyway), linear
# layers are routinely built with no server running — unit tests, weight-conversion
# tools, `python -c` probes. Predicates must tolerate None there.
_Predicate = Callable[[Any, str, Optional["ServerArgs"]], Optional[Any]]
# (layer, inner_method, ctx) -> wrapping method
_Factory = Callable[[Any, "QuantizeMethodBase", Any], "QuantizeMethodBase"]

# Each entry: (priority, wrapper_id, predicate, factory). LOWER priority runs
# FIRST (i.e. wraps the innermost method), matching the MoE registry so the two
# sides read the same way. Sorted on every call, so registration order from
# import sequence cannot silently change wrap order.
_LINEAR_WRAPPERS: List[Tuple[int, str, _Predicate, _Factory]] = []


def register_linear_quant_wrapper(
    wrapper_id: str,
    predicate: _Predicate,
    factory: _Factory,
    priority: int = 100,
) -> None:
    """Register a wrapper. Re-registering the same `wrapper_id` is a no-op.

    Args:
      wrapper_id: stable id used by `is_wrapped_linear_method` for
        isinstance-style checks without importing the wrapper class.
      predicate: (layer, prefix, server_args) -> Optional[ctx]. Return None to
        skip — the common case, since every linear in the model is offered here.
        Otherwise return an opaque ctx that's passed to factory.
      factory: (layer, inner_method, ctx) -> wrapped quant_method.
      priority: lower numbers wrap first (i.e. innermost).
    """
    for _, existing_id, _, _ in _LINEAR_WRAPPERS:
        if existing_id == wrapper_id:
            return
    _LINEAR_WRAPPERS.append((priority, wrapper_id, predicate, factory))


def unregister_linear_quant_wrapper(wrapper_id: str) -> bool:
    """Remove a wrapper by id. Returns whether one was removed.

    Exists for tests: registration is an import side effect, so a test that
    registers a probe has no other way to put the process back the way it found
    it, and a leaked probe would silently wrap every later test's layers.
    """
    for i, (_, existing_id, _, _) in enumerate(_LINEAR_WRAPPERS):
        if existing_id == wrapper_id:
            del _LINEAR_WRAPPERS[i]
            return True
    return False


def _maybe_import_prism_linear() -> None:
    """Prism dense(K-split linear)는 모델 파일이 아니라 env로 활성화되므로 여기서
    지연 import한다 (import 시 self-register). env가 없으면 no-op.

    MoE 쪽 `_maybe_import_prism`과 같은 이유·같은 모양이다. env가 켜졌는데 모듈이
    없으면 ImportError로 즉사하는 것이 의도다 — plan을 줬는데 조용히 stock 경로로
    도는 것은 "켰는데 안 켜진" 상태이고, 성능만 달라져 어떤 테스트도 잡지 못한다.
    """
    import os

    if os.environ.get("SGLANG_PRISM_LINEAR_PLAN"):
        import sglang.srt.layers.prism.linear.method  # noqa: F401


def maybe_wrap_linear_quant_method(
    layer: Any,
    linear_method: Optional["QuantizeMethodBase"],
    prefix: str,
    server_args: Optional["ServerArgs"] = None,
) -> Optional["QuantizeMethodBase"]:
    """Offer `layer` to every registered wrapper, innermost (lowest) priority first.

    Called as the last statement of `LinearBase.__init__`. Returns
    `linear_method` unchanged when nothing matches, which is the overwhelmingly
    common case — keep it cheap.

    `server_args` is fetched lazily rather than imported at module scope: this
    module is imported from `linear.py`, and `server_args` pulls in a large slice
    of the runtime. It stays None when no server is running — linear layers are
    routinely built standalone (unit tests, weight-conversion tools), and the
    hook must not turn that into a crash the moment some plugin registers itself.
    """
    _maybe_import_prism_linear()
    if not _LINEAR_WRAPPERS:
        return linear_method

    if server_args is None:
        from sglang.srt.server_args import get_global_server_args

        try:
            server_args = get_global_server_args()
        except ValueError:
            server_args = None  # no server in this process; predicates handle it

    method = linear_method
    for _priority, wrapper_id, predicate, factory in sorted(_LINEAR_WRAPPERS):
        ctx = predicate(layer, prefix, server_args)
        if ctx is not None:
            method = factory(layer, method, ctx)
            if getattr(method, "_quant_wrapper_id", None) is None:
                method._quant_wrapper_id = wrapper_id
    if method is not linear_method:
        _detach_stale_module_entry(layer, "quant_method", linear_method)
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


def is_wrapped_linear_method(method: Any, wrapper_id: str) -> bool:
    """isinstance replacement that doesn't require importing the wrapper class."""
    return getattr(method, "_quant_wrapper_id", None) == wrapper_id
