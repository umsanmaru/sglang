"""Prism Plan 스키마·검증 불변식 테스트 (GPU 불필요).

CONTRACTS.md ①의 불변식들이 실제로 강제되는지 확인한다. 특히 커버리지/
disjoint 위반은 런타임에서 조용한 이중계산/누락이 되므로 여기서 전부 잡는다.
"""

import copy

import pytest

from sglang.srt.layers.moe.prism.plan import (
    COL_GROUP,
    KNOWN_SPARSITY_SCORES,
    PAIR_GROUP,
    ROW_GROUP,
    SPARSITY_SCHEMA_VERSION,
    PlanError,
    Proj,
    Tier,
    parse_plan,
    validate_static,
)

# 작은 dims로 (layer, expert) 그리드를 감당 가능하게 유지
DIMS = {
    "hidden_size": 256,
    "intermediate_size": 128,
    "num_layers": 2,
    "num_experts": 4,
    "top_k": 2,
    "dtype": "bfloat16",
}


def make_raw_plan():
    gate_up = {
        "bands": [[0, 64, "warm"], [64, 256, "cold"]],
        "cold_shards": [[0, 0, 64], [1, 64, 128]],
    }
    down = {
        "bands": [[0, 64, "warm"], [64, 128, "cold"]],
        "cold_shards": [[0, 0, 128], [1, 128, 256]],
    }
    return {
        "schema_version": 1,
        "model_id": "test/tiny",
        "dims": dict(DIMS),
        "kernels": {"gpu_warm": "torch_bmm", "cpu_cold": "kt_amx_bf16"},
        "default": {
            "gate": copy.deepcopy(gate_up),
            "up": copy.deepcopy(gate_up),
            "down": down,
        },
    }


def check(raw, **kwargs):
    validate_static(parse_plan(raw), **kwargs)


def test_valid_plan_passes():
    check(make_raw_plan())


def test_all_cold_and_all_warm_plans_pass():
    raw = make_raw_plan()
    for proj, K, N in (("gate", 256, 128), ("up", 256, 128), ("down", 128, 256)):
        raw["default"][proj] = {
            "bands": [[0, K, "cold"]],
            "cold_shards": [[0, 0, N]],
        }
    check(raw)

    raw = make_raw_plan()
    for proj, K in (("gate", 256), ("up", 256), ("down", 128)):
        raw["default"][proj] = {"bands": [[0, K, "warm"]], "cold_shards": []}
    check(raw)


def test_gate_up_may_differ():
    # 스키마 수준에서 gate/up 분할 독립 (P0의 동일성 요구는 cold 로드 시점)
    raw = make_raw_plan()
    raw["default"]["up"]["bands"] = [[0, 128, "warm"], [128, 256, "cold"]]
    check(raw)


def test_band_gap_rejected():
    raw = make_raw_plan()
    raw["default"]["gate"]["bands"] = [[0, 64, "warm"], [128, 256, "cold"]]
    with pytest.raises(PlanError, match="gap"):
        check(raw)


def test_band_overlap_rejected():
    raw = make_raw_plan()
    raw["default"]["gate"]["bands"] = [[0, 128, "warm"], [64, 256, "cold"]]
    with pytest.raises(PlanError, match="overlap"):
        check(raw)


def test_incomplete_coverage_rejected():
    raw = make_raw_plan()
    raw["default"]["gate"]["bands"] = [[0, 192, "cold"]]  # K=256인데 192까지만
    with pytest.raises(PlanError, match="cover"):
        check(raw)


def test_misaligned_band_rejected():
    """정렬 요구는 **페어**뿐이다 (2026-08-25). 커널 타일까지의 올림은 로더가
    하고 cold 인스턴스 안에서 끝나므로 plan은 그 값을 모른다."""
    assert ROW_GROUP == 2
    raw = make_raw_plan()
    raw["default"]["gate"]["bands"] = [[0, 97, "warm"], [97, 256, "cold"]]
    with pytest.raises(PlanError, match="aligned"):
        check(raw)


def test_pair_aligned_band_accepted():
    """타일 배수가 아니어도 페어 배수면 유효하다 — planner 해상도가 목적이다."""
    raw = make_raw_plan()
    raw["default"]["gate"]["bands"] = [[0, 98, "warm"], [98, 256, "cold"]]
    check(raw)


def test_missing_cold_shards_rejected():
    raw = make_raw_plan()
    raw["default"]["gate"]["cold_shards"] = []
    with pytest.raises(PlanError, match="cold_shards is empty"):
        check(raw)


def test_shards_without_cold_band_rejected():
    raw = make_raw_plan()
    raw["default"]["gate"]["bands"] = [[0, 256, "warm"]]  # cold 없음
    with pytest.raises(PlanError, match="no COLD band"):
        check(raw)


def test_shard_gap_rejected():
    raw = make_raw_plan()
    raw["default"]["gate"]["cold_shards"] = [[0, 0, 32], [1, 64, 128]]
    with pytest.raises(PlanError, match="gap"):
        check(raw)


def test_misaligned_shard_rejected():
    assert COL_GROUP == 32
    raw = make_raw_plan()
    raw["default"]["gate"]["cold_shards"] = [[0, 0, 48], [1, 48, 128]]
    with pytest.raises(PlanError, match="aligned"):
        check(raw)


def test_unaligned_dims_rejected():
    raw = make_raw_plan()
    raw["dims"]["hidden_size"] = 200  # ROW_GROUP=64로 나눠지지 않음
    with pytest.raises(PlanError, match="divisible"):
        check(raw)


def test_unknown_kernel_rejected_when_registry_given():
    raw = make_raw_plan()
    with pytest.raises(PlanError, match="unknown gpu_warm"):
        check(raw, known_gpu_kernels=["other_kernel"])
    check(raw, known_gpu_kernels=["torch_bmm"], known_cpu_kernels=["kt_amx_bf16"])


def test_missing_default_and_incomplete_overrides_rejected():
    raw = make_raw_plan()
    default = raw.pop("default")
    raw["overrides"] = [{"layer": 0, "expert": 0, **default}]
    with pytest.raises(PlanError, match="every \\(layer, expert\\)"):
        check(raw)


def test_override_replaces_default():
    raw = make_raw_plan()
    ov = copy.deepcopy(raw["default"])
    ov["gate"]["bands"] = [[0, 256, "cold"]]
    ov["gate"]["cold_shards"] = [[0, 0, 128]]
    raw["overrides"] = [{"layer": 1, "expert": 2, **ov}]
    plan = parse_plan(raw)
    validate_static(plan)
    assert plan.expert(1, 2).gate.rows(Tier.COLD) == 256
    assert plan.expert(0, 0).gate.rows(Tier.WARM) == 64


def test_bad_schema_version_rejected():
    raw = make_raw_plan()
    raw["schema_version"] = 99
    with pytest.raises(PlanError, match="unsupported schema_version"):
        parse_plan(raw)


def test_file_roundtrip(tmp_path):
    import json

    path = tmp_path / "plan.json"
    path.write_text(json.dumps(make_raw_plan()))
    plan = parse_plan(path)
    validate_static(plan)
    assert plan.dims.k_of(Proj.GATE) == 256
    assert plan.dims.k_of(Proj.DOWN) == 128
    assert plan.dims.n_of(Proj.DOWN) == 256


# ---------------------------------------------------------------------------
# sparsity (schema_version 2) — k2wl2 입력기반 마스킹의 예산·자산 검증
# ---------------------------------------------------------------------------

SHA = "a" * 64


def make_raw_sparse_plan():
    """v1 plan에 model-global sparsity 블록 + 전 proj 예산을 얹은 것."""
    raw = make_raw_plan()
    raw["schema_version"] = SPARSITY_SCHEMA_VERSION
    raw["sparsity"] = {
        "score": "k2wl2",
        "calib": {"path": "assets/tiny/gatedyn_calib.pt", "sha256": SHA},
        "pmax": 0.9,
        "grid": 0.005,
        "ng": 201,
        "renorm_it": 3,
    }
    for proj in ("gate", "up", "down"):
        raw["default"][proj]["p"] = 0.5
        raw["default"][proj]["lambda"] = 4.305
    return raw


def probe_of(shapes):
    """calib_probe 대역 — CalibRef를 무시하고 주어진 shape 표를 돌려준다."""
    return lambda ref: shapes


def expected_shapes():
    K = {"gate": DIMS["hidden_size"], "up": DIMS["hidden_size"],
         "down": DIMS["intermediate_size"]}
    L, E, NG = DIMS["num_layers"], DIMS["num_experts"], 201
    out = {}
    for proj, k in K.items():
        out[f"thr_{proj}"] = (L, E, NG)
        out[f"wn_{proj}"] = (L, E, k)
        out[f"pair_dot_{proj}"] = (L, E, k // PAIR_GROUP)
    return out


def test_pair_group_divides_row_group():
    """밴드 경계가 마스킹 페어를 쪼개지 않는다는 불변식 (plan.py import assert의 명시)."""
    assert PAIR_GROUP == 2 and ROW_GROUP % PAIR_GROUP == 0


def test_valid_sparse_plan_passes():
    check(make_raw_sparse_plan())


def test_sparse_plan_roundtrip():
    plan = parse_plan(make_raw_sparse_plan())
    validate_static(plan)
    assert plan.sparsity is not None
    assert plan.sparsity.score == "k2wl2"
    assert plan.sparsity.calib.sha256 == SHA
    pp = plan.expert(0, 0).gate
    assert pp.sparsity_p == 0.5 and pp.sparsity_lambda == 4.305


def test_v1_plan_has_no_sparsity():
    plan = parse_plan(make_raw_plan())
    validate_static(plan)
    assert plan.sparsity is None
    assert plan.expert(0, 0).gate.sparsity_p is None


def test_sparsity_requires_schema_version_2():
    raw = make_raw_sparse_plan()
    raw["schema_version"] = 1
    with pytest.raises(PlanError, match="requires schema_version"):
        parse_plan(raw)


def test_budget_without_block_rejected():
    raw = make_raw_plan()  # sparsity 블록 없음
    raw["default"]["gate"]["p"] = 0.5
    raw["default"]["gate"]["lambda"] = 1.0
    with pytest.raises(PlanError, match="no model-global"):
        check(raw)


def test_block_without_budget_rejected():
    raw = make_raw_sparse_plan()
    del raw["default"]["down"]["p"]
    del raw["default"]["down"]["lambda"]
    with pytest.raises(PlanError, match="no \\(p, lambda\\)"):
        check(raw)


def test_half_budget_rejected():
    """p만 있고 lambda가 없으면 파서에서 즉사 (쌍으로만 유효)."""
    raw = make_raw_sparse_plan()
    del raw["default"]["up"]["lambda"]
    with pytest.raises(PlanError, match="malformed proj entry"):
        parse_plan(raw)


def test_p_above_pmax_rejected():
    raw = make_raw_sparse_plan()
    raw["default"]["gate"]["p"] = 0.95  # pmax=0.9
    with pytest.raises(PlanError, match="not in \\[0, pmax"):
        check(raw)


def test_negative_lambda_rejected():
    raw = make_raw_sparse_plan()
    raw["default"]["up"]["lambda"] = -1.0
    with pytest.raises(PlanError, match="must be >= 0"):
        check(raw)


def test_unknown_score_rejected():
    assert "k2wa" not in KNOWN_SPARSITY_SCORES
    raw = make_raw_sparse_plan()
    raw["sparsity"]["score"] = "k2wa"
    with pytest.raises(PlanError, match="unknown sparsity.score"):
        check(raw)


def test_bad_sha256_rejected():
    raw = make_raw_sparse_plan()
    raw["sparsity"]["calib"]["sha256"] = "deadbeef"
    with pytest.raises(PlanError, match="64 hex"):
        check(raw)


def test_grid_not_spanning_pmax_rejected():
    """격자가 pmax에 못 닿으면 idx가 clamp되어 threshold가 조용히 포화한다."""
    raw = make_raw_sparse_plan()
    raw["sparsity"]["ng"] = 21  # (21-1)*0.005 = 0.1 < pmax 0.9
    with pytest.raises(PlanError, match="must reach pmax"):
        check(raw)


def test_bad_pmax_rejected():
    raw = make_raw_sparse_plan()
    raw["sparsity"]["pmax"] = 1.5
    with pytest.raises(PlanError, match="pmax must be in"):
        check(raw)


def test_malformed_sparsity_block_rejected():
    raw = make_raw_sparse_plan()
    del raw["sparsity"]["calib"]["path"]
    with pytest.raises(PlanError, match="malformed sparsity block"):
        parse_plan(raw)


def test_calib_probe_accepts_matching_shapes():
    check(make_raw_sparse_plan(), calib_probe=probe_of(expected_shapes()))


def test_calib_probe_rejects_wrong_layer_count():
    shapes = expected_shapes()
    shapes["wn_gate"] = (DIMS["num_layers"] + 1,) + shapes["wn_gate"][1:]
    with pytest.raises(PlanError, match="calib table 'wn_gate' shape"):
        check(make_raw_sparse_plan(), calib_probe=probe_of(shapes))


def test_calib_probe_rejects_wrong_pair_length():
    """pair_dot은 K/PAIR_GROUP이어야 한다 — K를 그대로 쓴 자산은 즉사."""
    shapes = expected_shapes()
    shapes["pair_dot_down"] = (DIMS["num_layers"], DIMS["num_experts"],
                               DIMS["intermediate_size"])
    with pytest.raises(PlanError, match="calib table 'pair_dot_down' shape"):
        check(make_raw_sparse_plan(), calib_probe=probe_of(shapes))


def test_calib_probe_rejects_missing_table():
    shapes = expected_shapes()
    del shapes["thr_up"]
    with pytest.raises(PlanError, match="missing tables"):
        check(make_raw_sparse_plan(), calib_probe=probe_of(shapes))


def test_expected_calib_shapes_axes():
    """thr은 격자 축(ng), wn은 K 축, pair_dot은 K/2 축."""
    plan = parse_plan(make_raw_sparse_plan())
    shapes = plan.sparsity.expected_calib_shapes(plan.dims)
    assert shapes["thr_gate"][2] == 201
    assert shapes["wn_gate"][2] == DIMS["hidden_size"]
    assert shapes["wn_down"][2] == DIMS["intermediate_size"]
    assert shapes["pair_dot_gate"][2] == DIMS["hidden_size"] // 2
    assert shapes["pair_dot_down"][2] == DIMS["intermediate_size"] // 2
