"""dense plan의 **인덱스 티어** — 컴파일러 plan(`tile_list`)이 오는 형태 (GPU 불필요).

밴드형은 "연속 구간의 합집합"이라 커버·무중첩이 구조로 보장됐다. 인덱스형은 그
보장이 사라지므로 검증이 전부다. 여기서 잡는 것:

  1. **커버·중복·범위** — 빠진 행은 계산에서 사라지고 출력이 "그럴듯하게 나쁜" 값이
     된다. 중복은 그 행을 두 번 더한다. 둘 다 정확도 테스트로는 안 잡힌다.
  2. **페어 무결성** (계약 ① `PAIR_GROUP`) — 반쪽만 가진 티어는 k2wl2 점수를
     재구성할 수 없고 cold 커널의 skip 단위(VNNI 페어)도 깨진다.
  3. **순서 보존** — 인덱스 순서가 스토어 행 순서이고 그것이 곧 점수 재료·마스크
     비트의 순서다. 어디선가 정렬하면 마스크가 다른 행을 가리킨다.
  4. **절단이 실제로 그 행을 가져오는가** — `weights.prepare`가 순열 인덱스로
     gather한 결과가 원본의 그 행들과 같아야 한다.
"""

from __future__ import annotations

import copy

import pytest
import torch

from sglang.srt.layers.prism.geometry import PlanError, Tier
from sglang.srt.layers.prism.linear.plan import parse_plan, validate_static
from sglang.srt.layers.prism.linear.weights import prepare_linear_weights, tier_rows

K, N, NAME = 64, 32, "self_attn.o_proj"
CPU = torch.device("cpu")

# 순열: hot이 뒤쪽 페어를, cold가 앞쪽을 가져간다 (컴파일러가 중요도 순으로 뽑은 모습).
HOT = [62, 63, 30, 31, 4, 5, 40, 41]
COLD = [r for r in range(K) if r not in HOT]


def _plan(indices=None, **over):
    d = {
        "schema_version": 1,
        "model_id": "test/idx",
        "dims": {"num_layers": 1, "dtype": "bfloat16"},
        "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_tile_k2_bf16"},
        "projs": {
            NAME: {
                "k": K, "n": N,
                "indices": {"hot": list(HOT), "cold": list(COLD)}
                if indices is None else indices,
                "cold_shards": [[0, 0, N]],
            },
        },
    }
    d["projs"][NAME].update(over)
    if indices is not None and not any(t == "cold" for t in indices):
        d["projs"][NAME].pop("cold_shards")
    return d


@pytest.fixture
def weight():
    torch.manual_seed(0)
    return torch.randn(N, K, dtype=torch.bfloat16)


# ── 스키마 ─────────────────────────────────────────────────────────────────


def test_indices_plan_parses_and_keeps_order():
    plan = parse_plan(_plan())
    validate_static(plan)
    part = plan.proj(0, NAME).sole if hasattr(plan.proj(0, NAME), "sole") else plan.proj(0, NAME).parts[0]
    assert part.rows(Tier.HOT) == len(HOT) and part.rows(Tier.COLD) == len(COLD)
    assert part.has_tier(Tier.HOT) and not part.has_tier(Tier.WARM)
    assert tier_rows(part, Tier.HOT) == HOT          # 정렬하지 않는다
    assert tier_rows(part, Tier.WARM) == []


def test_bands_and_indices_together_die():
    d = _plan()
    d["projs"][NAME]["bands"] = [[0, K, "cold"]]
    with pytest.raises(PlanError, match="exactly one of 'bands' / 'indices'"):
        parse_plan(d)


def test_neither_bands_nor_indices_dies():
    d = _plan()
    d["projs"][NAME].pop("indices")
    with pytest.raises(PlanError, match="exactly one of 'bands' / 'indices'"):
        parse_plan(d)


# ── 커버·중복·범위 ─────────────────────────────────────────────────────────


def test_missing_row_dies():
    d = _plan({"hot": list(HOT), "cold": COLD[:-2]})
    with pytest.raises(PlanError, match="rows are in no tier"):
        validate_static(parse_plan(d))


def test_duplicate_row_dies():
    dup = list(COLD)
    dup[:2] = HOT[:2]                       # hot과 겹친다
    d = _plan({"hot": list(HOT), "cold": dup})
    with pytest.raises(PlanError, match="more than one tier position"):
        validate_static(parse_plan(d))


def test_out_of_range_row_dies():
    d = _plan({"hot": [K, K + 1] + HOT[2:], "cold": list(COLD)})
    with pytest.raises(PlanError, match=r"out of \[0, 64\)"):
        validate_static(parse_plan(d))


def test_odd_row_count_dies():
    d = _plan({"hot": HOT[:3], "cold": list(COLD) + HOT[3:]})
    with pytest.raises(PlanError, match="not a multiple of ROW_GROUP"):
        validate_static(parse_plan(d))


def test_split_pair_dies():
    """페어의 반쪽만 가진 티어 — k2wl2 점수를 재구성할 수 없다."""
    d = _plan({"hot": [62, 30, 63, 31, 4, 5, 40, 41], "cold": list(COLD)})
    with pytest.raises(PlanError, match="splits a pair"):
        validate_static(parse_plan(d))


def test_empty_tiers_die():
    d = _plan({"hot": [], "cold": []})
    with pytest.raises(PlanError, match="no rows in any tier"):
        validate_static(parse_plan(d))


# ── 절단이 그 행을 가져오는가 ──────────────────────────────────────────────


def test_gather_follows_the_permutation(weight):
    plan = parse_plan(_plan())
    validate_static(plan)
    prepared = prepare_linear_weights(0, NAME, weight, plan, device=CPU,
                                      pin_memory=False)
    part = prepared.parts[0]
    # hot 스토어는 [k_tier, N] K-major — plan 순서의 행이 그 순서로 들어간다.
    torch.testing.assert_close(part.hot.w_flat,
                               weight.t()[torch.tensor(HOT)].contiguous())
    assert [int(v) for v in part.hot.k_index] == HOT
    assert not part.hot.contiguous                 # 순열이라 포인터 오프셋 불가
    # cold는 [N, k_pad] ckpt 방향, 타일(32) 배수까지 0 패딩.
    assert part.cold.real_rows == len(COLD)
    torch.testing.assert_close(part.cold.w_flat[:, : len(COLD)],
                               weight[:, torch.tensor(COLD)].contiguous())


def test_all_rows_are_computed_exactly_once(weight):
    """세 티어의 인덱스 합집합이 [0, k)의 순열이다 — 절단 뒤에도 그런가."""
    plan = parse_plan(_plan())
    prepared = prepare_linear_weights(0, NAME, weight, plan, device=CPU,
                                      pin_memory=False)
    part = prepared.parts[0]
    got = [int(v) for v in part.hot.k_index]
    got += [int(v) for v in part.cold.k_index[: part.cold.real_rows]]
    assert sorted(got) == list(range(K))
