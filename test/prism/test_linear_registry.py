"""dense linear의 quant-method 래퍼 훅 테스트 (GPU 불필요).

MoE 쪽 `maybe_wrap_moe_quant_method`에 대응하는 `LinearBase`용 슬롯이 실제로
동작하는지 확인한다. 여기서 지키는 불변식 셋:

  1. **아무도 등록하지 않으면 아무 일도 안 일어난다.** 이 훅은 sglang의 모든
     linear가 지나가는 자리라, 기본 상태에서 관찰 가능한 차이가 있으면 안 된다.
  2. **create_weights보다 먼저 붙는다.** 이것이 훅의 존재 이유다 — Prism dense는
     파라미터를 CPU에 할당하고 K-슬라이스한 뒤 원본을 놓는데, create_weights가
     이미 돈 뒤에 붙는 래퍼는 그 결정을 못 한다. 붙는 시점에 `weight`가 아직
     없다는 것을 단언한다.
  3. **tp_size는 predicate에서 안 보이고 create_weights에서는 보인다.**
     서브클래스가 `super().__init__()` 반환 *후에* 대입하기 때문이다. 이 순서를
     모르고 predicate에서 TP를 보려 하면 조용히 AttributeError가 아니라 조용히
     "TP=1인 척"이 된다 — 그래서 양쪽을 다 못박는다.

standalone 실행(서버 없음)이라 `get_global_server_args()`가 죽는 환경이기도
하다. 그 경로도 여기서 같이 커버된다 (predicate가 server_args=None을 받는다).
"""

import pytest
import torch

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.linear_method_registry import (
    is_wrapped_linear_method,
    maybe_wrap_linear_quant_method,
    register_linear_quant_wrapper,
    unregister_linear_quant_wrapper,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

H, N = 32, 64
TP = {"tp_rank": 0, "tp_size": 1}  # dist 초기화 회피


class _Probe:
    """최소 래퍼 — prism의 `PrismMoEMethod`와 같은 모양(위임 + 관찰)."""

    def __init__(self, inner, ctx):
        self.inner = inner
        self.ctx = ctx
        self.calls = []

    def __getattr__(self, name):
        # 재귀 방지: 자기 속성은 여기 오기 전에 잡힌다
        if name in ("inner", "ctx", "calls"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def create_weights(self, layer, *args, **kwargs):
        self.calls.append(
            {
                # 훅 시점이 create_weights **직전**인가 (불변식 2)
                "weight_exists": hasattr(layer, "weight"),
                # 이때는 서브클래스가 tp를 이미 대입했는가 (불변식 3)
                "tp_size": getattr(layer, "tp_size", None),
            }
        )
        return self.inner.create_weights(layer, *args, **kwargs)


@pytest.fixture
def registry():
    """등록은 import 부수효과라, 누수되면 이후 모든 테스트의 layer를 감싼다."""
    ids = []

    def _reg(wrapper_id, predicate, factory, priority=100):
        register_linear_quant_wrapper(wrapper_id, predicate, factory, priority)
        ids.append(wrapper_id)

    yield _reg
    for wid in ids:
        unregister_linear_quant_wrapper(wid)


def _seen_predicate(seen, match=None):
    """prefix를 기록하고, match가 주어지면 그 prefix에만 붙는 predicate."""

    def predicate(layer, prefix, server_args):
        seen.append((prefix, server_args))
        if match is not None and prefix != match:
            return None
        return {"prefix": prefix}

    return predicate


# ── 1. 기본 상태 ────────────────────────────────────────────────────────────


def test_empty_registry_is_noop():
    lin = ColumnParallelLinear(H, N, bias=False, prefix="model.layers.0.q", **TP)
    assert isinstance(lin.quant_method, UnquantizedLinearMethod)
    assert not is_wrapped_linear_method(lin.quant_method, "anything")


def test_passthrough_returns_same_object():
    """등록이 없으면 들어온 method 객체 그대로 (동일성까지)."""
    inner = UnquantizedLinearMethod()
    assert maybe_wrap_linear_quant_method(object(), inner, "p") is inner


# ── 2. 붙는다 / 안 붙는다 ───────────────────────────────────────────────────


def test_wraps_matching_prefix(registry):
    seen = []
    probes = []

    def factory(layer, inner, ctx):
        p = _Probe(inner, ctx)
        probes.append(p)
        return p

    registry(
        "probe", _seen_predicate(seen, match="model.layers.3.mlp.down_proj"), factory
    )

    hit = RowParallelLinear(
        N, H, bias=False, prefix="model.layers.3.mlp.down_proj", **TP
    )
    miss = RowParallelLinear(
        N, H, bias=False, prefix="model.layers.3.mlp.up_proj", **TP
    )

    assert len(probes) == 1
    assert hit.quant_method is probes[0]
    assert isinstance(miss.quant_method, UnquantizedLinearMethod)
    assert is_wrapped_linear_method(hit.quant_method, "probe")
    # predicate는 양쪽 모두에게 물어봤다
    assert [p for p, _ in seen] == [
        "model.layers.3.mlp.down_proj",
        "model.layers.3.mlp.up_proj",
    ]


def test_predicate_receives_none_server_args_standalone(registry):
    """서버 없이 만들어도 훅이 죽지 않는다 (get_global_server_args가 raise하는 환경)."""
    seen = []
    registry("probe", _seen_predicate(seen), lambda l, i, c: _Probe(i, c))

    ColumnParallelLinear(H, N, bias=False, prefix="model.layers.0.q", **TP)

    assert len(seen) == 1
    assert seen[0][1] is None


def test_all_linearbase_subclasses_pass_through_hook(registry):
    """훅 자리가 LinearBase.__init__이므로 서브클래스 전부가 걸려야 한다."""
    seen = []
    registry("probe", _seen_predicate(seen), lambda l, i, c: _Probe(i, c))

    ReplicatedLinear(H, N, bias=False, prefix="repl")
    ColumnParallelLinear(H, N, bias=False, prefix="col", **TP)
    MergedColumnParallelLinear(H, [N, N], bias=False, prefix="merged", **TP)
    RowParallelLinear(N, H, bias=False, prefix="row", **TP)

    assert [p for p, _ in seen] == ["repl", "col", "merged", "row"]


# ── 3. 타이밍 (훅의 존재 이유) ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: ColumnParallelLinear(H, N, bias=False, prefix="col", **TP),
            id="column",
        ),
        pytest.param(
            lambda: RowParallelLinear(N, H, bias=False, prefix="row", **TP), id="row"
        ),
        pytest.param(
            lambda: MergedColumnParallelLinear(H, [N, N], bias=False, prefix="m", **TP),
            id="merged",
        ),
    ],
)
def test_wrapper_owns_create_weights(registry, build):
    """래퍼의 create_weights가 실제로 불리고, 그 시점에 weight는 아직 없고
    tp_size는 이미 있다."""
    probes = []

    def factory(layer, inner, ctx):
        p = _Probe(inner, ctx)
        probes.append(p)
        return p

    registry("probe", lambda l, p, sa: {"prefix": p}, factory)

    lin = build()

    assert len(probes) == 1 and probes[0].calls, "래퍼의 create_weights가 안 불렸다"
    call = probes[0].calls[0]
    assert call["weight_exists"] is False, "훅이 create_weights 뒤에 붙었다"
    assert call["tp_size"] == 1, "create_weights 시점에 tp_size가 아직 없다"
    # 위임이 실제로 통과해서 파라미터가 만들어졌다
    assert isinstance(lin.weight, torch.nn.Parameter)


def test_wrapper_controls_parameter_placement(registry):
    """훅의 목적: 래퍼가 파라미터를 어디에 놓을지 정할 수 있다 (Prism은 CPU full 텐서).

    inner에 위임하지 않고 직접 등록하는 래퍼로 확인한다.
    """

    class _CpuAlloc:
        def create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra,
        ):
            w = torch.nn.Parameter(
                torch.zeros(
                    sum(output_partition_sizes),
                    input_size_per_partition,
                    dtype=params_dtype,
                    device="cpu",
                ),
                requires_grad=False,
            )
            w.prism_owned = True
            layer.register_parameter("weight", w)

    registry("cpu-alloc", lambda l, p, sa: {}, lambda l, i, c: _CpuAlloc())

    lin = ColumnParallelLinear(H, N, bias=False, prefix="col", **TP)

    assert getattr(lin.weight, "prism_owned", False) is True
    assert lin.weight.device.type == "cpu"


# ── 4. 체이닝 순서 ──────────────────────────────────────────────────────────


def test_lower_priority_wraps_innermost(registry):
    """MoE registry와 같은 규약: 낮은 priority가 먼저 = 안쪽."""
    order = []

    def make(name, priority):
        def factory(layer, inner, ctx):
            order.append(name)
            probe = _Probe(inner, ctx)
            probe.name = name
            return probe

        registry(name, lambda l, p, sa: {}, factory, priority)

    make("outer", 30)
    make("inner", 10)  # 등록 순서를 일부러 뒤집는다 — 정렬이 이기는지

    lin = ColumnParallelLinear(H, N, bias=False, prefix="col", **TP)

    assert order == ["inner", "outer"]
    assert lin.quant_method.name == "outer"
    assert lin.quant_method.inner.name == "inner"
    # wrapper_id 태그는 가장 바깥이 아니라 **처음 태깅된 것**을 유지한다
    # (MoE registry와 동일 — 안쪽 래퍼가 자기 id를 이미 박았다)
    assert is_wrapped_linear_method(lin.quant_method, "inner")


def test_duplicate_registration_is_noop(registry):
    calls = []
    registry(
        "dup", lambda l, p, sa: calls.append(p) or {}, lambda l, i, c: _Probe(i, c)
    )
    # 같은 id 재등록은 무시된다 (import가 두 번 돌아도 이중 래핑 없음)
    register_linear_quant_wrapper(
        "dup", lambda l, p, sa: {}, lambda l, i, c: _Probe(i, c)
    )

    ColumnParallelLinear(H, N, bias=False, prefix="col", **TP)

    assert len(calls) == 1


def test_unregister_removes(registry):
    registry("gone", lambda l, p, sa: {}, lambda l, i, c: _Probe(i, c))
    assert unregister_linear_quant_wrapper("gone") is True
    assert unregister_linear_quant_wrapper("gone") is False

    lin = ColumnParallelLinear(H, N, bias=False, prefix="col", **TP)
    assert isinstance(lin.quant_method, UnquantizedLinearMethod)


# ── 5. 위임된 method가 정상 동작한다 ────────────────────────────────────────


def test_wrapped_layer_still_computes(registry):
    """래퍼가 apply를 안 건드리면 forward가 그대로 돌아야 한다."""
    registry("probe", lambda l, p, sa: {}, lambda l, i, c: _Probe(i, c))

    lin = ColumnParallelLinear(H, N, bias=False, prefix="col", **TP)
    with torch.no_grad():
        lin.weight.normal_()
    x = torch.randn(4, H)

    out, _ = lin(x)
    torch.testing.assert_close(out, x @ lin.weight.t())
