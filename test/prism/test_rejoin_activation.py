"""rejoin#1 활성화 변형 — silu(+clamp) / situ(Kimi K3) 가 torch 참조와 맞는가.

참조는 activation.py `Activation.reference` (= sglang SiluAndMul / SituAndMul.forward_native 식,
fp32). 커널은 partial들을 fp32로 더한 뒤 같은 식을 걸고 bf16 한 번 라운딩하므로, 참조도 같은
순서(fp32 합 → 활성화 → bf16)로 만들면 tolerance는 fp32 결합 순서 차이만이다.
"""
import pytest
import torch

from sglang.srt.layers.moe.prism.activation import Activation
from sglang.srt.layers.moe.prism.plan import PlanError
from sglang.srt.layers.moe.prism.rejoin import rejoin_gateup

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

K3_SITU = Activation.situ(alpha=4.0, limit=25.0)


def _parts(n, m=3, k=4, inter=384, scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [(torch.randn(m, k, 2 * inter, generator=g, device="cuda") * scale).to(torch.bfloat16)
            for _ in range(n)]


def _ref(parts, act: Activation):
    acc = sum(p.float() for p in parts)
    return act.reference_gateup(acc).to(torch.bfloat16)


@cuda_required
@pytest.mark.parametrize("n_parts", [1, 2, 5])
@pytest.mark.parametrize("act", [
    Activation.silu(), Activation.silu(10.0), K3_SITU, Activation.situ(4.0, None), Activation.situ(1.0, 25.0),
], ids=str)
def test_rejoin_matches_reference(n_parts, act):
    inter = 384
    # 큰 스케일로 clamp/saturation 영역까지 덮는다 (|x| 최대 ~40)
    parts = _parts(n_parts, inter=inter, scale=8.0)
    out = rejoin_gateup(parts, inter, activation=act)
    ref = _ref(parts, act)
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), rtol=1.6e-2, atol=2e-2)


@cuda_required
def test_situ_differs_from_silu_and_limit_matters():
    inter = 256
    parts = _parts(2, inter=inter, scale=8.0)
    a = rejoin_gateup(parts, inter, activation=Activation.silu())
    b = rejoin_gateup(parts, inter, activation=K3_SITU)
    c = rejoin_gateup(parts, inter, activation=Activation.situ(4.0, None))
    assert not torch.equal(a, b)
    assert not torch.equal(b, c)  # linear_beta soft clip은 |u|>~10에서 눈에 보인다


@cuda_required
def test_situ_small_inputs_are_accurate():
    """tanh 안정형 근사의 0 근방 정확도 — 실제 activation 크기(|x|≲1)에서 bf16 해상도 아래여야."""
    inter = 256
    parts = _parts(1, inter=inter, scale=0.05, seed=3)
    out = rejoin_gateup(parts, inter, activation=K3_SITU)
    ref = _ref(parts, K3_SITU)
    torch.testing.assert_close(out.float(), ref.float(), rtol=8e-3, atol=1e-6)


@cuda_required
def test_legacy_swiglu_limit_api_equals_activation():
    inter = 256
    parts = _parts(2, inter=inter, scale=8.0)
    a = rejoin_gateup(parts, inter, 10.0)
    b = rejoin_gateup(parts, inter, activation=Activation.silu(10.0))
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        rejoin_gateup(parts, inter, 10.0, activation=K3_SITU)


class _Cfg:
    def __init__(self, **kw):
        self.activation = "silu"
        self.swiglu_limit = None
        self.gemm1_alpha = None
        self.gemm1_clamp_limit = None
        self.__dict__.update(kw)


def test_from_runner_config():
    assert Activation.from_runner_config(_Cfg()) == Activation.silu()
    assert Activation.from_runner_config(_Cfg(swiglu_limit=10.0)) == Activation.silu(10.0)
    # Kimi K3: FusedMoE(activation="situ", gemm1_alpha=4.0, gemm1_clamp_limit=25.0)
    k3 = Activation.from_runner_config(_Cfg(activation="situ", gemm1_alpha=4.0, gemm1_clamp_limit=25.0))
    assert k3 == K3_SITU
    with pytest.raises(PlanError):
        Activation.from_runner_config(_Cfg(activation="gelu"))
    with pytest.raises(PlanError):  # swigluoai 류 alpha가 silu에 붙은 경우는 미지원 — 즉사
        Activation.from_runner_config(_Cfg(activation="silu", gemm1_alpha=1.702))
    with pytest.raises(PlanError):
        Activation("situ", 25.0, None)


def test_reference_matches_sglang_situ_native():
    """activation.py 참조식 == sglang SituAndMul.forward_native (K3가 dense 경로에서 쓰는 것)."""
    from sglang.srt.layers.activation import SituAndMul

    x = torch.randn(5, 2 * 64) * 8
    ref = SituAndMul(beta=4.0, linear_beta=25.0).forward_native(x)
    ours = K3_SITU.reference_gateup(x).to(x.dtype)
    torch.testing.assert_close(ours, ref)
