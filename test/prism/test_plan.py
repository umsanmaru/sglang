"""Prism Plan 스키마·검증 불변식 테스트 (GPU 불필요).

CONTRACTS.md ①의 불변식들이 실제로 강제되는지 확인한다. 특히 커버리지/
disjoint 위반은 런타임에서 조용한 이중계산/누락이 되므로 여기서 전부 잡는다.
"""

import copy

import pytest

from sglang.srt.layers.moe.prism.plan import (
    COL_GROUP,
    ROW_GROUP,
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
    assert ROW_GROUP == 64
    raw = make_raw_plan()
    raw["default"]["gate"]["bands"] = [[0, 96, "warm"], [96, 256, "cold"]]
    with pytest.raises(PlanError, match="aligned"):
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
