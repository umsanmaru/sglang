"""dense COLD 배선 — 티어 배치 불변성과 계약 ②-4 불변식.

**이 파일의 주된 테스트는 `test_tier_placement_is_bitwise`다** (계약 ⑤-5). cold를
붙이면서 생기는 결함은 거의 전부 "값이 조금 틀리는" 형태이고, 그건 정확도 테스트도
서버도 통과한다. 정확히 표현 가능한 입력에서 **티어를 어디에 두든 비트일치**를
요구하는 것이 이중계산·누락·좌표 뒤섞임의 유일한 검출기다.

dense에는 인덱스 자산이 없어 티어 멤버십이 항상 연속 밴드의 합집합이다 — MoE
계약 ⑤-5가 요구하는 "무작위 순열 인덱스" 픽스처는 dense가 인덱스 자산을 갖게 될
때 함께 온다. 지금 잡을 수 있는 것은 **밴드 경계를 옮겼을 때의 비트일치**까지다.

kt와 CUDA가 둘 다 필요하다.
"""
from __future__ import annotations

import pytest
import torch

kt = pytest.importorskip("kt_kernel")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from sglang.srt.layers.prism.geometry import PlanError
from sglang.srt.layers.prism.linear.cold_backend import KtLinearColdBackend
from sglang.srt.layers.prism.linear.executor import LinearExecutor
from sglang.srt.layers.prism.linear.plan import parse_plan, validate_static
from sglang.srt.layers.prism.linear.resources import LinearColdResources
from sglang.srt.layers.prism.linear.weights import prepare_linear_weights

H, I, LAYERS, NODES = 256, 512, 2, 2
TILE = 32          # kt_amx_bf16의 cold_pack_tile_rows


def _shards(n):
    step = (n // NODES) // TILE * TILE
    out, cur = [], 0
    for i in range(NODES):
        end = n if i == NODES - 1 else cur + step
        out.append([i, cur, end])
        cur = end
    return out


def _bands(k, hot, warm):
    """[hot | warm | cold]. 경계는 TILE 배수라 cold 패딩이 없다 (비트일치 요구)."""
    h = int(k * hot) // TILE * TILE
    w = int(k * warm) // TILE * TILE
    out = []
    for start, end, tier in ((0, h, "hot"), (h, h + w, "warm"), (h + w, k, "cold")):
        if end > start:
            out.append([start, end, tier])
    return out


def _plan_dict(hot, warm, *, kernel="kt_amx_bf16"):
    def part(k, n, name=None):
        d = {"bands": _bands(k, hot, warm)}
        if any(b[2] == "cold" for b in d["bands"]):
            d["cold_shards"] = _shards(n)
        return {"name": name, "n": n, **d} if name else d

    return {
        "schema_version": 1,
        "model_id": "test",
        "dims": {"num_layers": LAYERS, "dtype": "bfloat16"},
        "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": kernel},
        "projs": {
            "mlp.gate_up_proj": {"k": H, "n": 2 * I,
                                 "parts": [part(H, I, "gate"), part(H, I, "up")]},
            "mlp.down_proj": {"k": I, "n": H, **part(I, H)},
        },
    }


def _exact_weights(seed=0):
    """정확히 표현 가능한 값 — bf16 라운딩이 무손실이라야 티어 배치가 비트일치한다.

    x는 8개만 ±1이고 W는 [-2, 2] 정수이므로 부분합의 절댓값이 16을 넘지 않는다.
    bf16은 정수 256까지 정확하므로 어느 티어 조합으로 쪼개도 결과가 같다.
    """
    g = torch.Generator().manual_seed(seed)
    w = {}
    for layer in range(LAYERS):
        for name, k, n in (("mlp.gate_up_proj", H, 2 * I), ("mlp.down_proj", I, H)):
            w[(layer, name)] = torch.randint(-2, 3, (n, k), generator=g).to(torch.bfloat16)
    xs = {}
    for name, k in (("mlp.gate_up_proj", H), ("mlp.down_proj", I)):
        x = torch.zeros(1, k)
        idx = torch.randperm(k, generator=g)[:8]
        x[0, idx] = torch.randint(0, 2, (8,), generator=g).float() * 2 - 1
        xs[name] = x.to(torch.bfloat16).cuda()
    return w, xs


def _run_all(plan_dict, weights, xs):
    """한 plan으로 전 좌표를 돌려 `{(layer, name): out}`을 낸다."""
    plan = parse_plan(plan_dict)
    validate_static(plan)
    dev = torch.device("cuda")
    has_cold = any(pp.has_tier_cold() if hasattr(pp, "has_tier_cold") else
                   any(p.bands and any(b.tier.value == "cold" for b in p.bands)
                       for p in pp.parts)
                   for pp in plan.projs.values())
    cold = res = None
    if has_cold:
        cold = KtLinearColdBackend(plan, max_tokens=8, num_numa_nodes=NODES,
                                   cpuinfer_threads=8)
        res = LinearColdResources(max_tokens=8)
    ex = LinearExecutor(max_tokens=8, device=dev, cold=cold, resources=res)
    ex.warmup(dev)
    for (layer, name), w in weights.items():
        ex.register(layer, name,
                    prepare_linear_weights(layer, name, w, plan, device=dev,
                                           warm_node=None, pin_memory=True))
    ex.finalize()
    out = {}
    for (layer, name) in weights:
        out[(layer, name)] = ex.run(layer, name, xs[name]).float().cpu()
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("hot,warm", [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0),
                                      (0.25, 0.25), (0.5, 0.25)])
def test_tier_placement_is_bitwise(hot, warm):
    """티어를 어디에 두든 **비트일치** (계약 ⑤-5).

    안 잡으면: 한 티어가 자기 K행을 빠뜨리거나 두 티어가 같은 행을 더해도 값이
    "그럴듯하게" 나온다. 정확도 테스트는 tolerance라 통과하고, 서버도 통과하고,
    벤치 결론만 틀린다.
    """
    weights, xs = _exact_weights()
    ref = _run_all(_plan_dict(1.0, 0.0), weights, xs)     # all-hot 기준
    got = _run_all(_plan_dict(hot, warm), weights, xs)
    for key in ref:
        assert torch.equal(got[key], ref[key]), (
            f"{key}: hot={hot} warm={warm} 배치에서 출력이 달라졌다 "
            f"(max diff {(got[key] - ref[key]).abs().max().item()})"
        )


def test_reference_matches_dense_gemm():
    """혼합 plan이 단일 GEMM 레퍼런스와 일치 (계약 ⑤-4)."""
    weights, xs = _exact_weights(seed=3)
    got = _run_all(_plan_dict(0.25, 0.25), weights, xs)
    for (layer, name), w in weights.items():
        ref = (xs[name].float().cpu() @ w.float().T)
        assert torch.equal(got[(layer, name)], ref), f"{(layer, name)} != dense GEMM"


def test_layers_merge_into_one_instance():
    """같은 형상의 슬롯은 layer를 넘어 한 인스턴스로 접힌다 (계약 ②-1).

    안 잡으면: 슬롯마다 인스턴스를 만드는 퇴화 구성으로 되돌아가도 결과는 맞아서
    **메모리와 기동 시간만** 나빠진다 (실 형상에서 1.6 GB → 87 GB).
    """
    plan = parse_plan(_plan_dict(0.0, 0.0))
    validate_static(plan)
    cold = KtLinearColdBackend(plan, max_tokens=8, num_numa_nodes=NODES,
                               cpuinfer_threads=8)
    torch.manual_seed(0)
    for layer in range(LAYERS):
        for name, k, n in (("mlp.gate_up_proj", H, 2 * I), ("mlp.down_proj", I, H)):
            w = torch.randint(-2, 3, (n, k)).to(torch.bfloat16)
            cold.register(layer, name,
                          prepare_linear_weights(layer, name, w, plan, pin_memory=False))
    cold.finalize()
    groups = {g.key.label: g.num_experts for g in cold.groups()}
    assert len(groups) == 2, f"형상 그룹이 2개여야 한다 (gateup, down): {groups}"
    assert all(e == LAYERS for e in groups.values()), (
        f"각 그룹의 expert 수는 layer 수여야 한다: {groups}")
    # gate|up이 한 unit으로 묶여 out이 [gate 열 | up 열]이다
    gu = next(g for g in cold.groups() if g.key.entry == "gateup")
    assert gu.out_cols == 2 * I


def test_mismatched_shards_split_into_two_groups():
    """그룹 안에서 노드 테이블이 같은 것은 검사가 아니라 **구성**이다 (계약 ②-4).

    `GroupKey`가 shard 테이블을 담으므로 다른 테이블은 다른 인스턴스가 된다. 이
    테스트가 지키는 것은 그 설계 자체다 — shard를 키에서 빼면 두 layer가 한
    인스턴스에 들어가고, 노드 테이블은 config 스칼라라 kt가 **죽지 않고 엉뚱한
    열에 쓴다**.
    """
    base = _plan_dict(0.0, 0.0)
    skew = {**base, "projs": {**base["projs"],
                              "mlp.down_proj": {**base["projs"]["mlp.down_proj"],
                                                "cold_shards": [[0, 0, 64], [1, 64, H]]}}}
    plan_a, plan_b = parse_plan(base), parse_plan(skew)
    validate_static(plan_a)
    validate_static(plan_b)
    cold = KtLinearColdBackend(plan_a, max_tokens=8, num_numa_nodes=NODES,
                               cpuinfer_threads=8)
    w = torch.zeros(H, I, dtype=torch.bfloat16)
    cold.register(0, "mlp.down_proj",
                  prepare_linear_weights(0, "mlp.down_proj", w, plan_a, pin_memory=False))
    cold._plan = plan_b
    cold.register(1, "mlp.down_proj",
                  prepare_linear_weights(1, "mlp.down_proj", w, plan_b, pin_memory=False))
    cold.finalize()
    groups = [g for g in cold.groups() if g.key.entry == "down"]
    assert len(groups) == 2, f"shard가 다르면 그룹이 갈려야 한다: {[g.key for g in groups]}"
    assert all(g.num_experts == 1 for g in groups)


def test_sparse_plan_is_rejected():
    """sparse plan은 아직 배선되지 않았다 — 조용히 dense로 돌면 안 된다."""
    d = _plan_dict(0.0, 0.0)
    d["sparsity"] = {"score": "k2wl2", "calib": {"path": "/x", "sha256": "0" * 64},
                     "pmax": 0.9, "grid": 0.005, "ng": 201, "renorm_it": 3}
    for proj in d["projs"].values():
        for part in proj.get("parts", [proj]):
            part.update({"calib": "g", "p": 0.5, "lambda": 0.0})
    plan = parse_plan(d)
    with pytest.raises(NotImplementedError, match="sparse"):
        KtLinearColdBackend(plan, max_tokens=8, num_numa_nodes=NODES)
