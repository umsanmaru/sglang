"""티어 인덱스 테스트 — 표현·밴드 등가·검증.

이 파일이 지키는 성질은 둘이다:
1. `from_bands`가 밴드를 **손실 없이** 인덱스로 옮긴다 (전환기의 등가 다리).
2. `validate_layer`가 순열 위반·페어 쪼갬을 잡는다 — 밴드 검증이 사라진 자리를
   메우는 두 방어선 중 하나 (다른 하나는 셔플 인덱스에서의 정수 비트일치).
"""

import pytest
import torch

from sglang.srt.layers.moe.prism.index import (
    MAX_K,
    LayerIndex,
    TierIndex,
    from_bands,
    validate_layer,
)
from sglang.srt.layers.moe.prism.plan import (
    ModelDims,
    PlanError,
    Proj,
    Tier,
    parse_plan,
    validate_static,
)

DIMS = {
    "hidden_size": 256,
    "intermediate_size": 128,
    "num_layers": 2,
    "num_experts": 4,
    "top_k": 2,
    "dtype": "bfloat16",
}
TINY = ModelDims(**DIMS)


def make_plan(gate=None, up=None, down=None, overrides=None):
    def proj_entry(bands, N):
        has_cold = any(t == "cold" for _, _, t in bands)
        return {
            "bands": bands,
            "cold_shards": [[0, 0, N // 2], [1, N // 2, N]] if has_cold else [],
        }

    raw = {
        "schema_version": 1,
        "model_id": "test/tiny",
        "dims": dict(DIMS),
        "kernels": {"gpu_warm": "torch_bmm", "cpu_cold": "kt_amx_bf16"},
        "default": {
            "gate": proj_entry(gate or [[0, 64, "warm"], [64, 256, "cold"]], 128),
            "up": proj_entry(up or [[0, 64, "warm"], [64, 256, "cold"]], 128),
            "down": proj_entry(down or [[0, 64, "warm"], [64, 128, "cold"]], 256),
        },
        "overrides": overrides or [],
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


# ── 밴드 → 인덱스 등가 ──────────────────────────────────────────────────────


def test_from_bands_is_lossless_and_contiguous():
    li = from_bands(make_plan(), 0)
    warm = li.get(Proj.GATE, Tier.WARM)
    cold = li.get(Proj.GATE, Tier.COLD)
    assert warm.contiguous and cold.contiguous
    for e in range(TINY.num_experts):
        assert warm.for_expert(e).to(torch.int64).tolist() == list(range(0, 64))
        assert cold.for_expert(e).to(torch.int64).tolist() == list(range(64, 256))
    assert warm.total_rows == 64 * TINY.num_experts
    assert warm.start_of(2) == 0 and cold.start_of(2) == 64


def test_absent_tier_is_none_not_empty():
    """부재는 길이 0이 아니라 None — 스토어(HotStore.gate = None)와 어휘를 맞춘다."""
    li = from_bands(make_plan(), 0)
    assert li.get(Proj.GATE, Tier.HOT) is None
    assert li.get(Proj.DOWN, Tier.HOT) is None


def test_multiple_bands_per_tier_concatenate():
    """티어당 단일 밴드 제약은 인덱스 표현에 없다."""
    li = from_bands(
        make_plan(gate=[[0, 64, "warm"], [64, 128, "cold"], [128, 192, "warm"],
                        [192, 256, "cold"]]),
        0,
    )
    warm = li.get(Proj.GATE, Tier.WARM)
    assert warm.for_expert(0).to(torch.int64).tolist() == (
        list(range(0, 64)) + list(range(128, 192))
    )
    assert not warm.contiguous  # 두 구간이라 단위 stride가 아니다


def test_per_expert_variable_length():
    """expert마다 다른 밴드 → 가변 길이가 자연히 나온다."""
    ov = [{
        "layer": 0, "expert": 1,
        "gate": {"bands": [[0, 128, "warm"], [128, 256, "cold"]],
                 "cold_shards": [[0, 0, 64], [1, 64, 128]]},
        "up": {"bands": [[0, 64, "warm"], [64, 256, "cold"]],
               "cold_shards": [[0, 0, 64], [1, 64, 128]]},
        "down": {"bands": [[0, 64, "warm"], [64, 128, "cold"]],
                 "cold_shards": [[0, 0, 128], [1, 128, 256]]},
    }]
    li = from_bands(make_plan(overrides=ov), 0)
    warm = li.get(Proj.GATE, Tier.WARM)
    assert warm.k_rows(0) == 64 and warm.k_rows(1) == 128 and warm.k_rows(2) == 64
    validate_layer(li, TINY, 0)


@pytest.mark.parametrize("layer", [0, 1])
def test_valid_plan_validates(layer):
    validate_layer(from_bands(make_plan(), layer), TINY, layer)


# ── 검증 ───────────────────────────────────────────────────────────────────


def _layer_index(gate_hot, gate_warm, gate_cold, E=4):
    """gate만 채운 LayerIndex (up/down은 full cold로 유효하게)."""
    tiers = {
        (Proj.GATE, Tier.HOT): TierIndex.from_rows([gate_hot] * E),
        (Proj.GATE, Tier.WARM): TierIndex.from_rows([gate_warm] * E),
        (Proj.GATE, Tier.COLD): TierIndex.from_rows([gate_cold] * E),
    }
    for proj, K in ((Proj.UP, 256), (Proj.DOWN, 128)):
        tiers[(proj, Tier.COLD)] = TierIndex.from_rows([list(range(K))] * E)
    return LayerIndex(tiers)


def test_shuffled_index_is_valid_and_not_contiguous():
    """셔플은 정당하다 — 순열이고 페어가 붙어 있으면 된다."""
    pairs = [[2 * p, 2 * p + 1] for p in range(128)]
    torch.manual_seed(0)
    order = torch.randperm(128).tolist()
    hot = [v for p in order[:16] for v in pairs[p]]
    cold = [v for p in order[16:] for v in pairs[p]]
    li = _layer_index(hot, [], cold)
    validate_layer(li, TINY, 0)
    assert not li.get(Proj.GATE, Tier.HOT).contiguous


def test_pair_order_within_a_pair_is_free():
    """페어 안 순서는 자유 — wn이 같은 인덱스로 동행해 점수식이 대칭."""
    rows = [v for p in range(128) for v in (2 * p + 1, 2 * p)]
    validate_layer(_layer_index([], [], rows), TINY, 0)


def test_duplicate_row_is_caught():
    """개수는 맞는데 0,1이 두 번이고 254,255가 빠진 구성 — 커버리지 검사를
    통과하고 순열 검사에서만 걸린다 (이중계산 + 누락이 상쇄된 최악의 형태)."""
    rows = list(range(254)) + [0, 1]
    with pytest.raises(PlanError, match="순열이 아니다"):
        validate_layer(_layer_index([], [], rows), TINY, 0)


def test_missing_row_is_caught():
    with pytest.raises(PlanError, match="커버리지 위반"):
        validate_layer(_layer_index([], [], list(range(254))), TINY, 0)


def test_split_pair_is_caught():
    rows = [0, 2, 1, 3] + list(range(4, 256))
    with pytest.raises(PlanError, match="pair split"):
        validate_layer(_layer_index([], [], rows), TINY, 0)


def test_odd_length_is_caught():
    with pytest.raises(PlanError, match="PAIR_GROUP"):
        validate_layer(_layer_index([0, 1, 2], [], list(range(3, 256))), TINY, 0)


def test_row_off_length_mismatch_is_caught():
    ti = TierIndex.from_rows([[0, 1], [2, 3]])
    bad = TierIndex(row_off=ti.row_off[:-1], idx=ti.idx, contiguous=ti.contiguous)
    with pytest.raises(PlanError, match="expects|experts"):
        validate_layer(LayerIndex({(Proj.GATE, Tier.COLD): bad}), TINY, 0)


def test_index_beyond_uint16_is_rejected():
    with pytest.raises(PlanError, match="exceeds"):
        TierIndex.from_rows([[0, MAX_K + 1]])


# ── 잡다 ───────────────────────────────────────────────────────────────────


def test_to_device_preserves_values():
    ti = TierIndex.from_rows([[0, 1], [2, 3, 4, 5]])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    moved = ti.to(dev)
    assert moved.for_expert(1).cpu().to(torch.int64).tolist() == [2, 3, 4, 5]
    assert moved.contiguous == ti.contiguous
    assert moved.row_off.device.type == dev


def test_start_of_rejects_non_contiguous():
    ti = TierIndex.from_rows([[4, 5, 0, 1]])
    with pytest.raises(ValueError, match="contiguous"):
        ti.start_of(0)
