"""FP8 prefill 경로 — grouped GEMM(hot/warm)과 cold(kt fp8 타일)의 executor 등가성.

사용자 결정(2026-08-28): **mxfp4와 fp8의 prefill에는 AMX 구간이 없다** — cold도 GPU가
kt 타일 slab을 제자리 읽는다 (`default_cold_gpu_min_m` = grouped 경계). 여기서 보는 것:

1. prefill(M ≥ GROUPED_MIN_M)의 3-tier 출력이 같은 가중치를 dequant한 bf16 plan과 같다.
2. cold를 GPU가 읽은 결과 == CPU(kt tile decode 경로)가 계산한 결과.
3. fp8 포맷의 cold GPU 기본 임계가 grouped 경계다 (AMX prefill로 떨어지지 않는다).
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from fp8_ref import dequant_ckpt, random_expert_ckpt  # noqa: E402

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

E, H, I, TOPK = 8, 512, 512, 2
DIMS = {"hidden_size": H, "intermediate_size": I, "num_layers": 1,
        "num_experts": E, "top_k": TOPK, "dtype": "bfloat16"}
MAX_TOKENS = 128
BLK = 128


def _fp8_weights(seed, exact):
    g = torch.Generator().manual_seed(seed)
    c13, s13, c2, s2 = [], [], [], []
    for _ in range(E):
        c, s = random_expert_ckpt(2 * I, H, g, exact=exact); c13.append(c); s13.append(s)
        c, s = random_expert_ckpt(H, I, g, exact=exact); c2.append(c); s2.append(s)
    return torch.stack(c13), torch.stack(c2), torch.stack(s13), torch.stack(s2)


def _dequant(w13, w2, s13, s2):
    d13 = torch.stack([dequant_ckpt(w13[e], s13[e]) for e in range(E)])
    d2 = torch.stack([dequant_ckpt(w2[e], s2[e]) for e in range(E)])
    return d13.to(torch.bfloat16), d2.to(torch.bfloat16)


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


def _plan3(gpu_kernel, cpu_kernel, n_align):
    """3-tier plan. cold N shard는 커널이 요구하는 정렬(fp8 타일은 256)을 지켜야 한다."""
    from sglang.srt.layers.moe.prism.numa import numa_node_count
    from sglang.srt.layers.moe.prism.plan import parse_plan, validate_static

    nn = numa_node_count()

    def shards(N):
        half = (N // 2 // n_align) * n_align
        return [[0, 0, half], [1, half, N]] if nn >= 2 and half else [[0, 0, N]]

    gu = {"bands": [[0, 128, "hot"], [128, 256, "warm"], [256, H, "cold"]], "cold_shards": shards(I)}
    dn = {"bands": [[0, 128, "hot"], [128, 256, "warm"], [256, I, "cold"]], "cold_shards": shards(H)}
    raw = {"schema_version": 1, "model_id": "test/tiny-fp8-3tier", "dims": dict(DIMS),
           "kernels": {"gpu_warm": gpu_kernel, "cpu_cold": cpu_kernel},
           "default": {"gate": gu, "up": dict(gu), "down": dn}}
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def _executor3(plan, prepared_kwargs, cold_gpu_min_m=None):
    from sglang.srt.layers.moe.prism.cold_backend import KtColdBackend
    from sglang.srt.layers.moe.prism.executor import PrismExecutor
    from sglang.srt.layers.moe.prism.kernels import cold_pack_tile_rows
    from sglang.srt.layers.moe.prism.numa import numa_node_count
    from sglang.srt.layers.moe.prism.resources import ExecutionResources, ResourceSpec
    from sglang.srt.layers.moe.prism.weights import prepare_layer_weights

    prepared = prepare_layer_weights(
        0, plan=plan, device=torch.device("cuda"),
        cold_tile_rows=cold_pack_tile_rows(plan.kernels.cpu_cold), **prepared_kwargs)
    cold = KtColdBackend(plan, max_tokens=MAX_TOKENS, num_numa_nodes=numa_node_count(),
                         cpuinfer_threads=4,
                         gpu_view_device=torch.device("cuda") if cold_gpu_min_m else None)
    cold.load_layer(0, prepared.cold, prepared.thr)
    spec = ResourceSpec.from_plan(plan, max_tokens=MAX_TOKENS, device=torch.device("cuda"))
    ex = PrismExecutor(plan, ExecutionResources(spec), cold, cold_gpu_min_m=cold_gpu_min_m)
    ex.register_layer(0, prepared)
    return ex


def test_fp8_cold_gpu_default_is_the_grouped_boundary():
    """fp8/mxfp4는 prefill에 AMX를 쓰지 않는다 — cold GPU 임계가 grouped 경계여야 한다."""
    from sglang.srt.layers.moe.prism.formats import BF16, FP8, MXFP4

    assert FP8.default_cold_gpu_min_m(1536, 16) == 16
    assert MXFP4.default_cold_gpu_min_m(1536, 16) == 16
    assert BF16.default_cold_gpu_min_m(1536, 16) == 1536  # bf16만 CPU cold 교차점을 쓴다


pytest.importorskip("kt_kernel", reason="kt_kernel required for the cold tier")


@cuda_required
@pytest.mark.parametrize("m", [1, 40])
@pytest.mark.parametrize("exact", [True, False])
def test_fp8_three_tier_matches_bf16(m, exact):
    """cold(kt TileK2FP8B128 partial) 포함 3-tier가 bf16 dequant 3-tier와 같다.

    m=1은 decode(worklist + kt CPU 커널), m=40은 prefill(grouped + kt CPU 커널)이다."""
    w13, w2, s13, s2 = _fp8_weights(seed=31, exact=exact)
    d13, d2 = _dequant(w13, w2, s13, s2)
    ex8 = _executor3(_plan3("gemv_worklist_fp8", "kt_tile_k2_fp8b128", 256),
                     dict(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2))
    ex16 = _executor3(_plan3("gemv_worklist", "kt_amx_bf16", 32), dict(w13=d13, w2=d2))
    x, ids, w = _inputs(m, seed=32 + m, exact=exact)
    out8 = ex8.run_layer(0, x, ids, w, swiglu_limit=10.0)
    out16 = ex16.run_layer(0, x, ids, w, swiglu_limit=10.0)
    torch.cuda.synchronize()
    torch.testing.assert_close(out8.float(), out16.float(), rtol=3e-2, atol=3e-2)


@cuda_required
def test_fp8_cold_gpu_prefill_matches_cpu_cold():
    """prefill에서 cold를 GPU가 kt 타일 slab 제자리 읽기로 계산 ↔ CPU cold (kt 타일 커널)."""
    w13, w2, s13, s2 = _fp8_weights(seed=41, exact=True)
    kw = dict(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2)
    plan = _plan3("gemv_worklist_fp8", "kt_tile_k2_fp8b128", 256)
    ex_cpu = _executor3(plan, kw, cold_gpu_min_m=None)
    ex_gpu = _executor3(plan, kw, cold_gpu_min_m=16)
    x, ids, w = _inputs(48, seed=42, exact=True)
    a = ex_cpu.run_layer(0, x, ids, w, swiglu_limit=10.0)
    b = ex_gpu.run_layer(0, x, ids, w, swiglu_limit=10.0)
    torch.cuda.synchronize()
    # CPU(AVX-512 fp32 누산 × fp32 배율)와 GPU(bf16 dequant + tensor core)는 같은 W8A16이지만
    # 결합 순서가 다르다 — 배율이 2의 거듭제곱이라 W는 양쪽 다 정확하고, 남는 것은 누산 순서다.
    torch.testing.assert_close(a.float(), b.float(), rtol=2e-2, atol=2e-2)
