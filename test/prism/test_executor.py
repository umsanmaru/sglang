"""Prism vertical slice의 심장 테스트: 3-tier rejoin 정확성 (GPU + kt AMX 필요).

- torch fp32 full-MoE 레퍼런스 대비 tolerance (decode/prefill)
- plan 불변성: "warm+cold 혼합", "전부 cold", "전부 warm", "전부 hot",
  "hot+warm+cold"가 동일 입력에서 일치 — rejoin의 이중계산/누락 검출 (계약 ⑤).
  티어 경계를 어디로 옮겨도 출력이 같다는 것이 K-split의 정의이므로, HOT
  구현의 검증도 여기에 얹는 것이 맞다.
"""

import pytest
import torch

pytest.importorskip("kt_kernel", reason="kt_kernel required")
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from sglang.srt.layers.moe.prism.cold_backend import KtColdBackend
from sglang.srt.layers.moe.prism.executor import PrismExecutor
from sglang.srt.layers.moe.prism.kernels import resolve_gpu_kernel
from sglang.srt.layers.moe.prism.numa import numa_node_count
from sglang.srt.layers.moe.prism.plan import Tier, parse_plan, validate_static
from sglang.srt.layers.moe.prism.resources import ExecutionResources, ResourceSpec
from sglang.srt.layers.moe.prism.stagers import PerSlotCopyStager
from sglang.srt.layers.moe.prism.weights import prepare_layer_weights

DIMS = {
    "hidden_size": 256,
    "intermediate_size": 128,
    "num_layers": 1,
    "num_experts": 8,
    "top_k": 2,
    "dtype": "bfloat16",
}
MAX_TOKENS = 16
NUM_NODES = numa_node_count()


def proj_entry(bands, N):
    has_cold = any(t == "cold" for _, _, t in bands)
    shards = []
    if has_cold:
        half = (N // 2 // 32) * 32
        shards = [[0, 0, half], [1, half, N]] if NUM_NODES >= 2 else [[0, 0, N]]
    return {"bands": bands, "cold_shards": shards}


def make_plan(kind):
    if kind == "mixed":
        gate_up = proj_entry([[0, 64, "warm"], [64, 256, "cold"]], 128)
        down = proj_entry([[0, 64, "warm"], [64, 128, "cold"]], 256)
    elif kind == "all_cold":
        gate_up = proj_entry([[0, 256, "cold"]], 128)
        down = proj_entry([[0, 128, "cold"]], 256)
    elif kind == "all_warm":
        gate_up = proj_entry([[0, 256, "warm"]], 128)
        down = proj_entry([[0, 128, "warm"]], 256)
    elif kind == "all_hot":
        gate_up = proj_entry([[0, 256, "hot"]], 128)
        down = proj_entry([[0, 128, "hot"]], 256)
    elif kind == "three_tier":
        # down의 K=128은 ROW_GROUP=64로 밴드 2개가 한계라 hot+cold로 둔다
        # (proj마다 티어 조합이 달라도 되는 것 자체가 검증 대상).
        gate_up = proj_entry([[0, 64, "hot"], [64, 128, "warm"], [128, 256, "cold"]], 128)
        down = proj_entry([[0, 64, "hot"], [64, 128, "cold"]], 256)
    else:
        raise ValueError(f"unknown plan kind {kind!r}")
    raw = {
        "schema_version": 1,
        "model_id": "test/tiny",
        "dims": dict(DIMS),
        "kernels": {"gpu_warm": "torch_bmm", "cpu_cold": "kt_amx_bf16"},
        "default": {"gate": dict(gate_up), "up": dict(gate_up), "down": down},
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def make_weights(seed=0):
    torch.manual_seed(seed)
    e, h, i = DIMS["num_experts"], DIMS["hidden_size"], DIMS["intermediate_size"]
    w13 = (torch.randn(e, 2 * i, h) / 10.0).to(torch.bfloat16)
    w2 = (torch.randn(e, h, i) / 10.0).to(torch.bfloat16)
    return w13, w2


def build_executor(plan, w13, w2, **executor_kwargs):
    """executor_kwargs는 PrismExecutor 생성자에 그대로 전달된다
    (예: force_graph_path=True, cold_stream=True, capture_mode_fn=...)."""
    from sglang.srt.layers.moe.prism.plan import Proj

    prepared = prepare_layer_weights(0, w13, w2, plan, device=torch.device("cuda"))
    ep = plan.expert(0, 0)
    has_cold = any(ep.proj(p).has_tier(Tier.COLD) for p in Proj)

    cold = None
    if has_cold:
        cold = KtColdBackend(plan, max_tokens=MAX_TOKENS, num_numa_nodes=NUM_NODES)
        cold.load_layer(0, prepared.cold, prepared.thr)
    spec = ResourceSpec.from_plan(plan, max_tokens=MAX_TOKENS, device=torch.device("cuda"))
    res = ExecutionResources(spec)
    ex = PrismExecutor(plan, res, cold, resolve_gpu_kernel(plan.kernels.gpu_warm),
                       stager=PerSlotCopyStager(), **executor_kwargs)
    ex.register_layer(0, prepared)
    return ex


def moe_reference(x, ids, weights, w13, w2):
    """torch fp32 full MoE (silu(gate)*up → down → router 가중 합)."""
    i = DIMS["intermediate_size"]
    gate_w, up_w = w13[:, :i, :].float(), w13[:, i:, :].float()
    down_w = w2.float()
    out = torch.zeros(x.shape[0], DIMS["hidden_size"])
    xf = x.float()
    for m in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[m, j])
            g = xf[m] @ gate_w[e].t()
            u = xf[m] @ up_w[e].t()
            a = torch.nn.functional.silu(g) * u
            out[m] += float(weights[m, j]) * (a @ down_w[e].t())
    return out


def make_inputs(qlen, seed):
    torch.manual_seed(seed)
    x = (torch.randn(qlen, DIMS["hidden_size"]) / 10.0).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(DIMS["num_experts"])[: DIMS["top_k"]] for _ in range(qlen)])
    w = torch.rand(qlen, DIMS["top_k"], dtype=torch.float32)
    return x, ids, w


def run(ex, x, ids, w):
    return ex.run_layer(0, x.cuda(), ids.cuda(), w.cuda()).cpu()


def rel_diff(a, b):
    return (torch.mean(torch.abs(a.float() - b.float())) / (torch.mean(torch.abs(b.float())) + 1e-8)).item()


@cuda_required
@pytest.mark.parametrize("qlen", [1, 16])
@pytest.mark.parametrize("kind", ["mixed", "all_cold", "all_warm", "all_hot", "three_tier"])
def test_layer_matches_reference(kind, qlen):
    plan = make_plan(kind)
    w13, w2 = make_weights()
    ex = build_executor(plan, w13, w2)
    x, ids, w = make_inputs(qlen, seed=10 + qlen)
    out = run(ex, x, ids, w)
    ref = moe_reference(x, ids, w, w13, w2)
    d = rel_diff(out, ref)
    print(f"{kind} qlen={qlen}: rel diff = {d:.6f}")
    assert d < 0.03, f"{kind} qlen={qlen}: diff {d:.6f}"


@cuda_required
@pytest.mark.parametrize("qlen", [1, 16])
def test_plan_invariance(qlen):
    """같은 weight·입력에서 세 plan의 출력이 일치해야 한다 — 티어 경계
    어디에 있든 결과가 같다는 K-split의 핵심 성질 (이중계산/누락 검출)."""
    w13, w2 = make_weights(seed=1)
    x, ids, w = make_inputs(qlen, seed=20 + qlen)
    kinds = ["mixed", "all_cold", "all_warm", "all_hot", "three_tier"]
    outs = {}
    for kind in kinds:
        ex = build_executor(make_plan(kind), w13, w2)
        outs[kind] = run(ex, x, ids, w)
    # 기준을 all_cold 하나로 잡는다: 모든 티어 배치가 같은 값에 수렴해야 하므로
    # 쌍별 비교가 아니라 공통 기준 대비가 결함 위치를 좁혀준다.
    diffs = {kind: rel_diff(outs[kind], outs["all_cold"]) for kind in kinds}
    print(f"invariance qlen={qlen}: " +
          ", ".join(f"{kind}~cold {d:.6f}" for kind, d in diffs.items()))
    for kind, d in diffs.items():
        assert d < 0.02, f"{kind} diverges from all_cold: {d:.6f}"
