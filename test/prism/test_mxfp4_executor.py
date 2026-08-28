"""MXFP4 스토어의 executor 경로 — 같은 가중치를 bf16(dequant)로 푼 plan과의 등가성.

e2m1 × 2^e 는 bf16 정확표현이라, mxfp4 스토어(hot+warm)로 계산한 층 출력과 그 dequant를
bf16 스토어로 넣은 층 출력은 **같은 W4A16 계산**이다: 정확표현 입력(작은 정수 x·코드, 배율
2^0)에서는 비트일치, 랜덤 입력에서는 fp32 누산 순서 차이만큼의 tolerance. decode(M=1,
worklist)와 prefill(M≥16, grouped)을 모두 덮고, 정렬 위반(32-블록을 쪼개는 인덱스)은
로더가 즉사하는지 본다.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from mxfp4_ref import dequant_ckpt, random_expert_ckpt  # noqa: E402

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DIMS = {"hidden_size": 256, "intermediate_size": 128, "num_layers": 1,
        "num_experts": 8, "top_k": 2, "dtype": "bfloat16"}
E, H, I, TOPK = 8, 256, 128, 2
MAX_TOKENS = 64


def _plan(gpu_kernel, gu_bands, dn_bands):
    from sglang.srt.layers.moe.prism.plan import parse_plan, validate_static

    raw = {
        "schema_version": 1, "model_id": "test/tiny-mxfp4", "dims": dict(DIMS),
        "kernels": {"gpu_warm": gpu_kernel, "cpu_cold": "kt_amx_bf16"},
        "default": {"gate": {"bands": gu_bands, "cold_shards": []},
                    "up": {"bands": list(gu_bands), "cold_shards": []},
                    "down": {"bands": dn_bands, "cold_shards": []}},
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def _fp4_weights(seed, exact):
    """체크포인트 형태: w13 int8 [E, 2I, H/2], w2 int8 [E, H, I/2], 배율 fp32(E8M0 캐스팅)."""
    g = torch.Generator().manual_seed(seed)
    c13, s13, c2, s2 = [], [], [], []
    for _ in range(E):
        c, s = random_expert_ckpt(2 * I, H, g, exact=exact); c13.append(c); s13.append(s)
        c, s = random_expert_ckpt(H, I, g, exact=exact); c2.append(c); s2.append(s)
    w13, w2 = torch.stack(c13), torch.stack(c2)
    s13 = torch.stack(s13); s2 = torch.stack(s2)
    to_f32 = lambda s: torch.ldexp(torch.ones(s.shape), s.int() - 127)  # 로더가 넣는 fp32 배율
    return w13, w2, to_f32(s13), to_f32(s2), s13, s2


def _dequant(w13, w2, s13_u8, s2_u8):
    d13 = torch.stack([dequant_ckpt(w13[e].view(torch.uint8), s13_u8[e]) for e in range(E)])
    d2 = torch.stack([dequant_ckpt(w2[e].view(torch.uint8), s2_u8[e]) for e in range(E)])
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


GU = [[0, 64, "hot"], [64, 256, "warm"]]
DN = [[0, 64, "hot"], [64, 128, "warm"]]


@cuda_required
@pytest.mark.parametrize("m", [1, 4, 40])
@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("limit", [None, 1.5])
def test_mxfp4_layer_matches_bf16_dequant(m, exact, limit):
    w13, w2, s13, s2, s13_u8, s2_u8 = _fp4_weights(seed=1, exact=exact)
    d13, d2 = _dequant(w13, w2, s13_u8, s2_u8)
    ex4 = _executor(_plan("gemv_worklist_mxfp4", GU, DN),
                    dict(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2))
    ex16 = _executor(_plan("gemv_worklist", GU, DN), dict(w13=d13, w2=d2))
    x, ids, w = _inputs(m, seed=2 + m, exact=exact)
    out4 = ex4.run_layer(0, x, ids, w, swiglu_limit=limit)
    out16 = ex16.run_layer(0, x, ids, w, swiglu_limit=limit)
    torch.cuda.synchronize()
    if exact:
        assert torch.equal(out4, out16)
    else:
        torch.testing.assert_close(out4.float(), out16.float(), rtol=3e-2, atol=3e-2)


@cuda_required
def test_mxfp4_swiglu_limit_changes_output():
    """limit이 실제로 적용된다 (큰 활성에서 clamp 유무로 출력이 달라져야 한다)."""
    w13, w2, s13, s2, _, _ = _fp4_weights(seed=3, exact=False)
    ex4 = _executor(_plan("gemv_worklist_mxfp4", GU, DN),
                    dict(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2))
    x, ids, w = _inputs(2, seed=4, exact=False)
    x = x * 8  # gate/up 합이 ±1.5를 넘도록
    a = ex4.run_layer(0, x, ids, w, swiglu_limit=None)
    b = ex4.run_layer(0, x, ids, w, swiglu_limit=1.5)
    torch.cuda.synchronize()
    assert not torch.equal(a, b)


@cuda_required
def test_mxfp4_rejects_unaligned_index_and_cold():
    from sglang.srt.layers.moe.prism.plan import PlanError

    w13, w2, s13, s2, _, _ = _fp4_weights(seed=5, exact=True)
    # 32-블록을 쪼개는 밴드 경계(48)는 페어 정렬은 만족하지만 mxfp4 정렬 위반
    with pytest.raises(PlanError, match="aligned|scale block"):
        _executor(_plan("gemv_worklist_mxfp4", [[0, 48, "hot"], [48, 256, "warm"]], DN),
                  dict(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2))
    # mxfp4 cold 스토어는 fp4 cold 커널만 소비할 수 있다 (bf16 kt 커널과의 조합은 startup에서 즉사)
    from sglang.srt.layers.moe.prism.formats import MXFP4
    with pytest.raises(PlanError, match="cannot consume"):
        MXFP4.check_cold_kernel("kt_amx_bf16")
    MXFP4.check_cold_kernel("kt_amx_fp4")
    # 2의 거듭제곱이 아닌 배율은 MXFP4가 아니다
    with pytest.raises(PlanError, match="powers of two"):
        _executor(_plan("gemv_worklist_mxfp4", GU, DN),
                  dict(w13=w13, w2=w2, w13_scale=s13 * 1.5, w2_scale=s2))


@cuda_required
def test_mxfp4_cuda_graph_replay_matches_eager():
    """decode 서버 경로(CUDA graph)에서 mxfp4 티어가 캡처·재생되고 eager와 비트일치한다."""
    w13, w2, s13, s2, _, _ = _fp4_weights(seed=7, exact=True)
    ex = _executor(_plan("gemv_worklist_mxfp4", GU, DN),
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
    # 입력을 바꿔 재생 → 새 eager와 일치 (주소 고정, 값 갱신)
    x2, ids2, w2_ = _inputs(1, seed=9, exact=True)
    x.copy_(x2); ids.copy_(ids2); w.copy_(w2_)
    g.replay(); torch.cuda.synchronize()
    assert torch.equal(out, ex.run_layer(0, x, ids, w, swiglu_limit=10.0))


# ─── 3-tier (hot/warm GPU + cold CPU kt fp4, cold GPU prefill) ─────────────────────────
pytest.importorskip("kt_kernel", reason="kt_kernel required for the cold tier")


def _plan3(gpu_kernel, cpu_kernel):
    from sglang.srt.layers.moe.prism.numa import numa_node_count
    from sglang.srt.layers.moe.prism.plan import parse_plan, validate_static

    nn = numa_node_count()
    def shards(N):
        half = (N // 2 // 32) * 32
        return [[0, 0, half], [1, half, N]] if nn >= 2 else [[0, 0, N]]
    gu = {"bands": [[0, 64, "hot"], [64, 128, "warm"], [128, 256, "cold"]], "cold_shards": shards(I)}
    dn = {"bands": [[0, 32, "hot"], [32, 64, "warm"], [64, 128, "cold"]], "cold_shards": shards(H)}
    raw = {"schema_version": 1, "model_id": "test/tiny-mxfp4-3tier", "dims": dict(DIMS),
           "kernels": {"gpu_warm": gpu_kernel, "cpu_cold": cpu_kernel},
           "default": {"gate": gu, "up": dict(gu), "down": dn}}
    plan = parse_plan(raw); validate_static(plan)
    return plan


def _executor3(plan, prepared_kwargs, cold_gpu_min_m=None):
    from sglang.srt.layers.moe.prism.cold_backend import KtColdBackend
    from sglang.srt.layers.moe.prism.executor import PrismExecutor
    from sglang.srt.layers.moe.prism.numa import numa_node_count
    from sglang.srt.layers.moe.prism.resources import ExecutionResources, ResourceSpec
    from sglang.srt.layers.moe.prism.weights import prepare_layer_weights

    prepared = prepare_layer_weights(0, plan=plan, device=torch.device("cuda"), **prepared_kwargs)
    cold = KtColdBackend(plan, max_tokens=MAX_TOKENS, num_numa_nodes=numa_node_count(),
                         cpuinfer_threads=4,
                         gpu_view_device=torch.device("cuda") if cold_gpu_min_m else None)
    cold.load_layer(0, prepared.cold, prepared.thr)
    spec = ResourceSpec.from_plan(plan, max_tokens=MAX_TOKENS, device=torch.device("cuda"))
    ex = PrismExecutor(plan, ExecutionResources(spec), cold, cold_gpu_min_m=cold_gpu_min_m)
    ex.register_layer(0, prepared)
    return ex


@cuda_required
@pytest.mark.parametrize("m", [1, 3, 40])
@pytest.mark.parametrize("exact", [True, False])
def test_mxfp4_three_tier_matches_bf16(m, exact):
    """cold(kt AMX_FP4 partial, CPU decode) 포함 3-tier가 bf16 dequant 3-tier(kt_amx_bf16)와 같다."""
    w13, w2, s13, s2, s13_u8, s2_u8 = _fp4_weights(seed=31, exact=exact)
    d13, d2 = _dequant(w13, w2, s13_u8, s2_u8)
    ex4 = _executor3(_plan3("gemv_worklist_mxfp4", "kt_amx_fp4"),
                     dict(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2))
    ex16 = _executor3(_plan3("gemv_worklist", "kt_amx_bf16"), dict(w13=d13, w2=d2))
    x, ids, w = _inputs(m, seed=32 + m, exact=exact)
    out4 = ex4.run_layer(0, x, ids, w, swiglu_limit=10.0)
    out16 = ex16.run_layer(0, x, ids, w, swiglu_limit=10.0)
    torch.cuda.synchronize()
    if exact:
        assert torch.equal(out4, out16)
    else:
        torch.testing.assert_close(out4.float(), out16.float(), rtol=3e-2, atol=3e-2)


@cuda_required
def test_mxfp4_cold_gpu_prefill_matches_cpu_cold():
    """prefill(M≥16)에서 cold를 GPU가 kt fp4 slab 제자리 읽기로 계산 ↔ CPU cold — 정확표현 비트일치."""
    w13, w2, s13, s2, _, _ = _fp4_weights(seed=41, exact=True)
    kw = dict(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2)
    ex_cpu = _executor3(_plan3("gemv_worklist_mxfp4", "kt_amx_fp4"), kw, cold_gpu_min_m=None)
    ex_gpu = _executor3(_plan3("gemv_worklist_mxfp4", "kt_amx_fp4"), kw, cold_gpu_min_m=16)
    x, ids, w = _inputs(48, seed=42, exact=True)
    a = ex_cpu.run_layer(0, x, ids, w, swiglu_limit=10.0)
    b = ex_gpu.run_layer(0, x, ids, w, swiglu_limit=10.0)
    torch.cuda.synchronize()
    assert torch.equal(a, b)


@cuda_required
def test_mxfp4_three_tier_graph_replay_matches_eager():
    """3-tier(cold kt host node 포함) decode를 CUDA graph로 캡처·**2회 이상 재생** — kt의 host-submit
    래퍼가 일회용이면 두 번째 replay가 use-after-free로 죽는다 (2026-08-28 회귀)."""
    w13, w2, s13, s2, _, _ = _fp4_weights(seed=51, exact=True)
    ex = _executor3(_plan3("gemv_worklist_mxfp4", "kt_amx_fp4"),
                    dict(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2))
    ex._force_graph_path = True
    x, ids, w = _inputs(1, seed=52, exact=True)
    eager = ex.run_layer(0, x, ids, w, swiglu_limit=10.0).clone()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            ex.run_layer(0, x, ids, w, swiglu_limit=10.0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = ex.run_layer(0, x, ids, w, swiglu_limit=10.0)
    for _ in range(3):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)
    x2, ids2, w2_ = _inputs(1, seed=53, exact=True)
    x.copy_(x2); ids.copy_(ids2); w.copy_(w2_)
    g.replay(); torch.cuda.synchronize()
    ex._force_graph_path = False
    assert torch.equal(out, ex.run_layer(0, x, ids, w, swiglu_limit=10.0))
