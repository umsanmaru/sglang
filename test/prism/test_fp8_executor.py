"""FP8 스토어의 executor 경로 — 같은 가중치를 bf16(dequant)로 푼 plan과의 등가성.

e4m3 값은 가수 3비트라 bf16 정확표현이고, 이 테스트의 배율은 전부 2의 거듭제곱이라
dequant도 정확하다. 따라서 fp8 스토어(hot+warm)로 계산한 층 출력과 그 dequant를 bf16
스토어로 넣은 층 출력은 **같은 W8A16 계산**이다: 정확표현 입력(작은 정수 x·코드)에서는
비트일치, 랜덤 입력에서는 fp32 누산 순서 차이만큼의 tolerance.

decode(M=1, worklist)를 덮고, 정렬 위반(128-블록을 쪼개는 인덱스)은 로더가 즉사하는지 본다.
prefill(grouped)·cold 3-tier는 `test_fp8_prefill.py`.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from fp8_ref import dequant_ckpt, random_expert_ckpt  # noqa: E402

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DIMS = {"hidden_size": 256, "intermediate_size": 256, "num_layers": 1,
        "num_experts": 8, "top_k": 2, "dtype": "bfloat16"}
E, H, I, TOPK = 8, 256, 256, 2
MAX_TOKENS = 64
BLK = 128


def _plan(gpu_kernel, gu_bands, dn_bands):
    from sglang.srt.layers.moe.prism.plan import parse_plan, validate_static

    raw = {
        "schema_version": 1, "model_id": "test/tiny-fp8", "dims": dict(DIMS),
        "kernels": {"gpu_warm": gpu_kernel, "cpu_cold": "kt_amx_bf16"},
        "default": {"gate": {"bands": gu_bands, "cold_shards": []},
                    "up": {"bands": list(gu_bands), "cold_shards": []},
                    "down": {"bands": dn_bands, "cold_shards": []}},
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def _fp8_weights(seed, exact):
    """체크포인트 형태: w13 u8 [E, 2I, H], w2 u8 [E, H, I], 배율 fp32 [E, N/128, K/128]."""
    g = torch.Generator().manual_seed(seed)
    c13, s13, c2, s2 = [], [], [], []
    for _ in range(E):
        c, s = random_expert_ckpt(2 * I, H, g, exact=exact); c13.append(c); s13.append(s)
        c, s = random_expert_ckpt(H, I, g, exact=exact); c2.append(c); s2.append(s)
    return (torch.stack(c13), torch.stack(c2), torch.stack(s13), torch.stack(s2))


def _dequant(w13, w2, s13, s2):
    d13 = torch.stack([dequant_ckpt(w13[e], s13[e]) for e in range(E)])
    d2 = torch.stack([dequant_ckpt(w2[e], s2[e]) for e in range(E)])
    return d13.to(torch.bfloat16), d2.to(torch.bfloat16)


def _executor(plan, prepared_kwargs):
    from sglang.srt.layers.moe.prism.executor import PrismExecutor
    from sglang.srt.layers.moe.prism.resources import ExecutionResources, ResourceSpec
    from sglang.srt.layers.moe.prism.weights import prepare_layer_weights

    prepared = prepare_layer_weights(0, plan=plan, device=torch.device("cuda"), **prepared_kwargs)
    spec = ResourceSpec.from_plan(plan, max_tokens=MAX_TOKENS, device=torch.device("cuda"))
    ex = PrismExecutor(plan, ExecutionResources(spec), None)
    ex.register_layer(0, prepared)
    return ex


def _inputs(m, seed, exact):
    g = torch.Generator().manual_seed(seed)
    if exact:
        x = torch.randint(-1, 2, (m, H), generator=g).to(torch.bfloat16)
        w = torch.randint(0, 3, (m, TOPK), generator=g).to(torch.float32)
    else:
        x = (torch.randn(m, H, generator=g) / 4).to(torch.bfloat16)
        w = torch.rand(m, TOPK, generator=g)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(m)])
    return x.cuda(), ids.cuda(), w.cuda()


GU = [[0, 128, "hot"], [128, 256, "warm"]]
DN = [[0, 128, "hot"], [128, 256, "warm"]]


@cuda_required
@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("limit", [None, 1.5])
def test_fp8_layer_matches_bf16_dequant(m, exact, limit):
    w13, w2, s13, s2 = _fp8_weights(seed=1, exact=exact)
    d13, d2 = _dequant(w13, w2, s13, s2)
    ex8 = _executor(_plan("gemv_worklist_fp8", GU, DN),
                    dict(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2))
    ex16 = _executor(_plan("gemv_worklist", GU, DN), dict(w13=d13, w2=d2))
    x, ids, w = _inputs(m, seed=2 + m, exact=exact)
    out8 = ex8.run_layer(0, x, ids, w, swiglu_limit=limit)
    out16 = ex16.run_layer(0, x, ids, w, swiglu_limit=limit)
    torch.cuda.synchronize()
    if exact:
        assert torch.equal(out8, out16)
    else:
        torch.testing.assert_close(out8.float(), out16.float(), rtol=3e-2, atol=3e-2)


@cuda_required
def test_fp8_swiglu_limit_changes_output():
    """limit이 실제로 적용된다 (큰 활성에서 clamp 유무로 출력이 달라져야 한다)."""
    w13, w2, s13, s2 = _fp8_weights(seed=3, exact=False)
    ex8 = _executor(_plan("gemv_worklist_fp8", GU, DN),
                    dict(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2))
    x, ids, w = _inputs(2, seed=4, exact=False)
    x = x * 8
    a = ex8.run_layer(0, x, ids, w, swiglu_limit=None)
    b = ex8.run_layer(0, x, ids, w, swiglu_limit=1.5)
    torch.cuda.synchronize()
    assert not torch.equal(a, b)


@cuda_required
def test_fp8_rejects_unaligned_index_and_cold():
    from sglang.srt.layers.moe.prism.formats import FP8
    from sglang.srt.layers.moe.prism.plan import PlanError

    w13, w2, s13, s2 = _fp8_weights(seed=5, exact=True)
    # 128-블록을 쪼개는 밴드 경계(64)는 페어 정렬은 만족하지만 fp8 정렬 위반
    with pytest.raises(PlanError, match="aligned|scale block"):
        _executor(_plan("gemv_worklist_fp8", [[0, 64, "hot"], [64, 256, "warm"]], DN),
                  dict(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2))
    # fp8 cold 스토어는 fp8 타일 커널만 소비할 수 있다 (startup에서 즉사)
    with pytest.raises(PlanError, match="cannot consume"):
        FP8.check_cold_kernel("kt_amx_bf16")
    FP8.check_cold_kernel("kt_tile_k2_fp8b128")


@cuda_required
def test_fp8_cuda_graph_replay_matches_eager():
    """decode 서버 경로(CUDA graph)에서 fp8 티어가 캡처·재생되고 eager와 비트일치한다."""
    w13, w2, s13, s2 = _fp8_weights(seed=7, exact=True)
    ex = _executor(_plan("gemv_worklist_fp8", GU, DN),
                   dict(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2))
    x, ids, w = _inputs(1, seed=8, exact=True)
    eager = ex.run_layer(0, x, ids, w, swiglu_limit=10.0).clone()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            ex.run_layer(0, x, ids, w, swiglu_limit=10.0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = ex.run_layer(0, x, ids, w, swiglu_limit=10.0)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    x2, ids2, w2_ = _inputs(1, seed=9, exact=True)
    x.copy_(x2); ids.copy_(ids2); w.copy_(w2_)
    g.replay(); torch.cuda.synchronize()
    assert torch.equal(out, ex.run_layer(0, x, ids, w, swiglu_limit=10.0))
