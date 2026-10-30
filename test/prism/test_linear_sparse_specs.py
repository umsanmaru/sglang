"""dense warm 티어의 `LinearSparseSpec` 조립 (GPU 불필요 — device=cpu로 본다).

여기서 잡는 것 셋:

  1. **HOT에는 spec이 붙지 않는다** (계약 ①, `tiers.SPARSE_TIERS`). 붙으면 hot이
     조용히 마스킹되고, 그러면 세 티어 마스크의 합집합이 full-K가 아니게 되어
     "같은 행을 hot↔warm으로 옮기면 출력이 달라진다"가 더 심한 형태로 나타난다.
  2. **score가 a/c의 유무를 정한다.** k1에서 a/c를 채우면 커널이 페어 에너지로
     판정해 plan이 말한 마스크와 달라진다 (`per_k` 플래그가 진입점을 고른다).
  3. **조각마다 spec이 따로다.** gate와 up은 thr 곡선도 p도 다르므로 proj 하나에
     spec 하나를 달면 한쪽 마스크가 다른 쪽에 적용된다.
"""

from __future__ import annotations

import copy

import pytest
import torch

from sglang.srt.layers.prism.geometry import Tier
from sglang.srt.layers.prism.linear.calib import LinearCalibTables
from sglang.srt.layers.prism.linear.plan import parse_plan
from sglang.srt.layers.prism.linear.tiers import build_part_specs, build_part_tiers
from sglang.srt.layers.prism.linear.weights import prepare_linear_weights

K, N, NG = 128, 64, 201
CPU = torch.device("cpu")

PLAN = {
    "schema_version": 1,
    "model_id": "test/dense",
    "dims": {"num_layers": 1, "dtype": "bfloat16"},
    "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_tile_k2_bf16"},
    "sparsity": {"score": "k2wl2", "calib": {"path": "/x", "sha256": "0" * 64},
                 "pmax": 0.9, "grid": 0.005, "ng": NG, "renorm_it": 3},
    "projs": {
        "mlp.gate_up_proj": {
            "k": K, "n": 2 * N,
            "parts": [
                {"name": "gate", "n": N, "bands": [[0, 32, "hot"], [32, K, "warm"]],
                 "calib": "g", "p": 0.5, "lambda": 0.0},
                {"name": "up", "n": N, "bands": [[0, 32, "hot"], [32, K, "warm"]],
                 "calib": "u", "p": 0.3, "lambda": 0.0},
            ],
        },
    },
}


def _tables():
    torch.manual_seed(0)
    t = {}
    for key in ("g", "u"):
        t[f"wn_{key}"] = torch.rand(1, 1, K) + 0.5
        t[f"c{key}"] = torch.randn(1, 1, K // 2)
        t[f"t{key}2l"] = torch.rand(1, 1, NG)
        t[f"t{key}"] = torch.rand(1, 1, NG)
    return t


def _prepared(score="k2wl2"):
    raw = copy.deepcopy(PLAN)
    raw["sparsity"]["score"] = score
    if score in ("k1", "k1w1"):
        raw["kernels"]["cpu_cold"] = "kt_tile_k1_bf16"
    plan = parse_plan(raw)
    torch.manual_seed(1)
    w = torch.randn(2 * N, K, dtype=torch.bfloat16)
    cal = LinearCalibTables(_tables(), score, NG)
    return prepare_linear_weights(0, "mlp.gate_up_proj", w, plan, device=CPU,
                                  pin_memory=False, calib=cal), plan


def test_only_warm_gets_a_spec():
    prepared, plan = _prepared()
    for part in prepared.parts:
        specs = build_part_specs(part, plan.sparsity, CPU)
        assert set(specs) == {Tier.WARM}
        tiers = build_part_tiers(part, specs)
        assert tiers[Tier.HOT].spec is None       # hot은 절대 마스킹하지 않는다
        assert tiers[Tier.WARM].spec is not None


def test_k2wl2_carries_pair_material():
    prepared, plan = _prepared()
    part = prepared.part("gate")
    sp = build_part_specs(part, plan.sparsity, CPU)[Tier.WARM]
    assert not sp.per_k
    assert sp.a is not None and sp.c is not None
    assert sp.a.numel() == part.warm.k_rows
    assert sp.c.numel() == part.warm.k_rows // 2
    # a = wn² (GPU와 kt가 같은 정의를 봐야 마스크가 갈리지 않는다)
    rows = part.warm.k_index.to(torch.int64)
    torch.testing.assert_close(sp.a, (_tables()["wn_g"][0, 0][rows] ** 2))


def test_k1_carries_no_pair_material():
    prepared, plan = _prepared(score="k1")
    sp = build_part_specs(prepared.part("gate"), plan.sparsity, CPU)[Tier.WARM]
    assert sp.per_k and sp.a is None and sp.c is None
    torch.testing.assert_close(sp.thr, _tables()["tg"][0].contiguous())


def test_k1w1_carries_wn_but_not_pair_dot():
    """score k1w1: per-k 마스크 + a(wn²). 커널은 `x²·a >= thr²`로 판정한다."""
    prepared, plan = _prepared(score="k1w1")
    part = prepared.part("gate")
    sp = build_part_specs(part, plan.sparsity, CPU)[Tier.WARM]
    assert sp.per_k and sp.c is None
    assert sp.a is not None and sp.a.numel() == part.warm.k_rows
    rows = part.warm.k_index.to(torch.int64)
    torch.testing.assert_close(sp.a, (_tables()["wn_g"][0, 0][rows] ** 2))
    torch.testing.assert_close(sp.thr, _tables()["tg"][0].contiguous())   # k1과 같은 곡선


def test_specs_are_per_part():
    """gate와 up의 예산·곡선이 섞이지 않는다."""
    prepared, plan = _prepared()
    g = build_part_specs(prepared.part("gate"), plan.sparsity, CPU)[Tier.WARM]
    u = build_part_specs(prepared.part("up"), plan.sparsity, CPU)[Tier.WARM]
    assert (g.p, u.p) == (0.5, 0.3)
    assert not torch.equal(g.thr, u.thr)


def test_no_material_means_no_spec():
    """`"sparse": false` 또는 warm 행이 없으면 dense로 돈다 (spec 없음)."""
    raw = copy.deepcopy(PLAN)
    for part in raw["projs"]["mlp.gate_up_proj"]["parts"]:
        part["bands"] = [[0, K, "hot"]]
    plan = parse_plan(raw)
    torch.manual_seed(1)
    w = torch.randn(2 * N, K, dtype=torch.bfloat16)
    prepared = prepare_linear_weights(0, "mlp.gate_up_proj", w, plan, device=CPU,
                                      pin_memory=False,
                                      calib=LinearCalibTables(_tables(), "k2wl2", NG))
    for part in prepared.parts:
        assert build_part_specs(part, plan.sparsity, CPU) == {}
