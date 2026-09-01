"""Prism dense Plan 스키마·검증 불변식 테스트 (GPU 불필요).

MoE 쪽 `test_plan.py`의 dense 대응물. 커버리지/정렬/shard 위반은 런타임에서
조용한 이중계산·누락·엉뚱한 행 슬라이스가 되므로 전부 여기서 잡는다.
"""

import copy

import pytest

from sglang.srt.layers.prism.geometry import COL_GROUP, PlanError, Tier
from sglang.srt.layers.prism.linear.plan import (
    check_dims,
    parse_plan,
    split_prefix,
    validate_static,
)

K_Q, N_Q = 1536, 8192  # self_attn.wq_b
K_D, N_D = 4096, 7168  # mlp.down_proj

PLAN = {
    "schema_version": 1,
    "model_id": "test/dsv4",
    "dims": {"num_layers": 4, "dtype": "bfloat16"},
    "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_amx_bf16"},
    "projs": {
        "self_attn.wq_b": {
            "k": K_Q,
            "n": N_Q,
            "bands": [[0, 192, "warm"], [192, K_Q, "cold"]],
            "cold_shards": [[0, 0, N_Q // 2], [1, N_Q // 2, N_Q]],
        },
        "mlp.down_proj": {
            "k": K_D,
            "n": N_D,
            "bands": [[0, K_D, "hot"]],
        },
    },
}


def _plan(**mutate):
    """PLAN 사본에 mutate를 적용해 파싱."""
    raw = copy.deepcopy(PLAN)
    for path, value in mutate.items():
        node = raw
        keys = path.split("/")
        for key in keys[:-1]:
            node = node[int(key)] if isinstance(node, list) else node[key]
        last = keys[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    return parse_plan(raw)


# ── prefix → 좌표 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("model.layers.7.self_attn.wq_b", (7, "self_attn.wq_b")),
        ("model.layers.0.mlp.down_proj", (0, "mlp.down_proj")),
        ("model.layers.61.mlp.gate_up_proj", (61, "mlp.gate_up_proj")),
        # 멀티모달 래퍼는 텍스트 스택을 model.language_model.layers.N. 으로 짓는다
        ("model.language_model.layers.7.self_attn.qkv_proj", (7, "self_attn.qkv_proj")),
        ("model.language_model.layers.0.linear_attn.in_proj_qkv",
         (0, "linear_attn.in_proj_qkv")),
        # 디코더 레이어 밖 — 훅을 지나는 정상적인 non-target들
        ("lm_head", None),
        ("model.embed_tokens", None),
        ("model.layers.3", None),
        ("visual.blocks.0.attn.qkv", None),
        ("", None),
    ],
)
def test_split_prefix(prefix, expected):
    assert split_prefix(prefix) == expected


# ── 전개 (default → 완전 명시형) ────────────────────────────────────────────


def test_expands_to_every_layer():
    plan = parse_plan(PLAN)
    assert len(plan.projs) == 4 * 2
    assert plan.names() == {"self_attn.wq_b", "mlp.down_proj"}
    for layer in range(4):
        assert plan.proj(layer, "self_attn.wq_b").k == K_Q
        assert plan.proj(layer, "mlp.down_proj").n == N_D


def test_get_returns_none_for_unplanned():
    plan = parse_plan(PLAN)
    assert plan.get(0, "self_attn.wo_b") is None
    assert plan.get(99, "mlp.down_proj") is None


def test_override_replaces_bands_and_inherits_geometry():
    raw = copy.deepcopy(PLAN)
    raw["overrides"] = [{"layer": 2, "self_attn.wq_b": {"bands": [[0, K_Q, "hot"]]}}]
    plan = parse_plan(raw)

    o = plan.proj(2, "self_attn.wq_b")
    assert [b.tier for b in o.sole.bands] == [Tier.HOT]
    assert (o.k, o.n) == (K_Q, N_Q)  # 상속
    assert o.sole.cold_shards == ()
    # 다른 layer는 그대로
    assert [b.tier for b in plan.proj(1, "self_attn.wq_b").sole.bands] == [
        Tier.WARM,
        Tier.COLD,
    ]


def test_override_unknown_proj_dies():
    raw = copy.deepcopy(PLAN)
    raw["overrides"] = [{"layer": 0, "mlp.never_heard_of": {"bands": [[0, 8, "hot"]]}}]
    with pytest.raises(PlanError, match="not declared"):
        parse_plan(raw)


def test_override_layer_out_of_range_dies():
    raw = copy.deepcopy(PLAN)
    raw["overrides"] = [{"layer": 9, "mlp.down_proj": {"bands": [[0, K_D, "hot"]]}}]
    with pytest.raises(PlanError, match="out of range"):
        parse_plan(raw)


# ── 헤더 ────────────────────────────────────────────────────────────────────


def test_unsupported_schema_version_dies():
    with pytest.raises(PlanError, match="schema_version"):
        _plan(schema_version=99)


def test_empty_projs_dies():
    with pytest.raises(PlanError, match="non-empty"):
        _plan(projs={})


def test_missing_k_dies():
    raw = copy.deepcopy(PLAN)
    del raw["projs"]["mlp.down_proj"]["k"]
    with pytest.raises(PlanError, match="k or n"):
        parse_plan(raw)


def test_unknown_kernel_dies():
    plan = _plan(**{"kernels/gpu_warm": "no_such_kernel"})
    with pytest.raises(Exception, match="unknown gpu_warm"):
        validate_static(plan)


def test_unknown_tier_dies():
    raw = copy.deepcopy(PLAN)
    raw["projs"]["mlp.down_proj"]["bands"] = [[0, K_D, "lukewarm"]]
    with pytest.raises(PlanError, match="malformed bands"):
        parse_plan(raw)


# ── 밴드 기하 (K축) ─────────────────────────────────────────────────────────


def test_valid_plan_passes():
    validate_static(parse_plan(PLAN))


def _bad_bands(bands):
    raw = copy.deepcopy(PLAN)
    raw["projs"]["mlp.down_proj"]["bands"] = bands
    return parse_plan(raw)


def test_band_gap_dies():
    with pytest.raises(PlanError, match="gap"):
        validate_static(_bad_bands([[0, 1024, "hot"], [2048, K_D, "cold"]]))


def test_band_overlap_dies():
    with pytest.raises(PlanError, match="overlap"):
        validate_static(_bad_bands([[0, 2048, "hot"], [1024, K_D, "cold"]]))


def test_band_undercover_dies():
    with pytest.raises(PlanError, match=r"cover \[0, 2048\) but k=4096"):
        validate_static(_bad_bands([[0, 2048, "hot"]]))


def test_band_pair_misalignment_dies():
    with pytest.raises(PlanError, match="not aligned"):
        validate_static(_bad_bands([[0, 1023, "hot"], [1023, K_D, "cold"]]))


def test_empty_band_dies():
    with pytest.raises(PlanError, match="empty/negative band"):
        validate_static(_bad_bands([[0, 0, "hot"], [0, K_D, "cold"]]))


def test_no_bands_dies():
    with pytest.raises(PlanError, match="no bands"):
        validate_static(_bad_bands([]))


# ── cold shard 기하 (N축) ───────────────────────────────────────────────────


def _bad_shards(shards, bands=None):
    raw = copy.deepcopy(PLAN)
    if bands is not None:
        raw["projs"]["self_attn.wq_b"]["bands"] = bands
    raw["projs"]["self_attn.wq_b"]["cold_shards"] = shards
    return parse_plan(raw)


def test_shards_without_cold_band_dies():
    with pytest.raises(PlanError, match="no COLD band"):
        validate_static(
            _bad_shards([[0, 0, N_Q]], bands=[[0, K_Q, "hot"]])
        )


def test_cold_band_without_shards_dies():
    with pytest.raises(PlanError, match="cold_shards is empty"):
        validate_static(_bad_shards([]))


def test_shard_gap_dies():
    with pytest.raises(PlanError, match="shard gap"):
        validate_static(_bad_shards([[0, 0, 1024], [1, 2048, N_Q]]))


def test_shard_undercover_dies():
    with pytest.raises(PlanError, match=r"shards cover \[0, 1024\) but n=8192"):
        validate_static(_bad_shards([[0, 0, 1024]]))


def test_shard_misalignment_dies():
    bad = COL_GROUP + 1
    with pytest.raises(PlanError, match="not aligned"):
        validate_static(_bad_shards([[0, 0, bad], [1, bad, N_Q]]))


def test_negative_node_dies():
    with pytest.raises(PlanError, match="negative numa node"):
        validate_static(_bad_shards([[-1, 0, N_Q]]))


# ── 로드 타임 치수 대조 ─────────────────────────────────────────────────────


def test_check_dims_accepts_match():
    pp = parse_plan(PLAN).proj(0, "self_attn.wq_b")
    check_dims(pp, K_Q, N_Q, "where")


@pytest.mark.parametrize("k,n", [(K_Q + 2, N_Q), (K_Q, N_Q * 2), (7168, 2048)])
def test_check_dims_rejects_mismatch(k, n):
    pp = parse_plan(PLAN).proj(0, "self_attn.wq_b")
    with pytest.raises(PlanError, match="다른 모델"):
        check_dims(pp, k, n, "layer 0 proj 'self_attn.wq_b'")


# ── 티어 조회 ───────────────────────────────────────────────────────────────


def test_rows_and_has_tier():
    proj = parse_plan(PLAN).proj(0, "self_attn.wq_b")
    pp = proj.sole
    assert pp.rows(Tier.WARM) == 192
    assert pp.rows(Tier.COLD) == K_Q - 192
    assert pp.rows(Tier.HOT) == 0
    assert pp.has_tier(Tier.WARM) and pp.has_tier(Tier.COLD)
    assert not pp.has_tier(Tier.HOT)
    # 세 티어 합 = K (계약 ①의 커버리지가 곧 이 등식이다)
    assert sum(pp.rows(t) for t in Tier) == proj.k  # K는 proj의 것


# ── proj별 커널 (혼합 포맷) ────────────────────────────────────────────────


def test_kernels_default_to_top_level():
    plan = parse_plan(PLAN)
    for name in ("self_attn.wq_b", "mlp.down_proj"):
        assert plan.proj(0, name).kernels == plan.kernels


def test_proj_kernels_override():
    """한 모델 안에서 proj마다 형식이 갈린다 (DSV4 wo_a는 나머지가 fp8이어도 bf16)."""
    raw = copy.deepcopy(PLAN)
    raw["kernels"] = {"gpu_warm": "gemv_worklist_fp8", "cpu_cold": "kt_tile_k2_fp8b128"}
    raw["projs"]["mlp.down_proj"]["kernels"] = {
        "gpu_warm": "gemv_worklist", "cpu_cold": "kt_amx_bf16"
    }
    plan = parse_plan(raw)
    validate_static(plan)

    assert plan.proj(0, "self_attn.wq_b").kernels.gpu_warm == "gemv_worklist_fp8"
    assert plan.proj(1, "mlp.down_proj").kernels.gpu_warm == "gemv_worklist"
    assert plan.kernels.gpu_warm == "gemv_worklist_fp8"  # top-level은 그대로


def test_proj_kernels_partial_override():
    """한쪽만 덮으면 나머지는 top-level에서 상속된다."""
    raw = copy.deepcopy(PLAN)
    raw["projs"]["mlp.down_proj"]["kernels"] = {"cpu_cold": "kt_tile_k2_bf16"}
    pp = parse_plan(raw).proj(0, "mlp.down_proj")
    assert pp.kernels.gpu_warm == "gemv_worklist"       # 상속
    assert pp.kernels.cpu_cold == "kt_tile_k2_bf16"     # 덮어씀


def test_proj_kernels_inherited_by_override_entry():
    """overrides는 밴딩만 바꾼다 — kernels/k/n은 모델 기하라 layer마다 같다."""
    raw = copy.deepcopy(PLAN)
    raw["projs"]["mlp.down_proj"]["kernels"] = {"gpu_warm": "gemv_worklist_fp8",
                                                "cpu_cold": "kt_tile_k2_fp8b128"}
    raw["overrides"] = [{"layer": 1, "mlp.down_proj": {"bands": [[0, K_D, "hot"]]}}]
    plan = parse_plan(raw)
    assert plan.proj(1, "mlp.down_proj").kernels.gpu_warm == "gemv_worklist_fp8"


def test_unknown_proj_kernel_dies():
    raw = copy.deepcopy(PLAN)
    raw["projs"]["mlp.down_proj"]["kernels"] = {"gpu_warm": "no_such_kernel"}
    with pytest.raises(Exception, match="unknown gpu_warm"):
        validate_static(parse_plan(raw))


# ── halves (N축 분할) ──────────────────────────────────────────────────────
#
# `mlp.gate_up_proj`는 weight가 `[2I, K]` 하나지만 sparsity가 gate/up을 따로
# 캘리브한다 (자산의 wn_g ≠ wn_u, tg2l ≠ tu2l). 마스크가 K축인데 두 절반이 다른
# 마스크를 요구하므로 한 번의 GEMV로 N=2I를 훑을 수 없다 → N축 분할.

I_MLP, H_MLP = 256, 128

GATEUP = {
    "schema_version": 1,
    "model_id": "test/split",
    "dims": {"num_layers": 2, "dtype": "bfloat16"},
    "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_tile_k2_bf16"},
    "projs": {
        "mlp.gate_up_proj": {
            "k": H_MLP,
            "n": 2 * I_MLP,
            "parts": [
                {"name": "gate", "n": I_MLP,
                 "bands": [[0, 64, "hot"], [64, H_MLP, "cold"]],
                 "cold_shards": [[0, 0, I_MLP]]},
                # up은 다른 밴딩 — 분할의 존재 이유
                {"name": "up", "n": I_MLP, "bands": [[0, H_MLP, "cold"]],
                 "cold_shards": [[0, 0, I_MLP // 2], [1, I_MLP // 2, I_MLP]]},
            ],
        }
    },
}


def test_halves_split_n_in_order():
    pp = parse_plan(GATEUP).proj(0, "mlp.gate_up_proj")
    validate_static(parse_plan(GATEUP))
    assert pp.split and len(pp.parts) == 2
    # 순서는 HALF_ORDER가 정한다 (JSON dict 순서가 아니라)
    assert [p.name for p in pp.parts] == ["gate", "up"]
    assert [(p.n_start, p.n_end) for p in pp.parts] == [(0, I_MLP), (I_MLP, 2 * I_MLP)]


def test_halves_carry_independent_bands():
    pp = parse_plan(GATEUP).proj(0, "mlp.gate_up_proj")
    assert pp.part("gate").rows(Tier.HOT) == 64
    assert pp.part("up").rows(Tier.HOT) == 0
    assert pp.part("up").rows(Tier.COLD) == H_MLP
    # cold shard도 조각마다 (좌표는 조각 로컬 = [0, I))
    assert len(pp.part("gate").cold_shards) == 1
    assert len(pp.part("up").cold_shards) == 2


def test_parts_order_is_the_list_order():
    """리스트에서는 **순서가 곧 의미**다 — 뒤집으면 뒤집힌 대로 잘린다.

    dict였을 때는 JSON 순서에 정확성을 걸 수 없어 이름 집합만 보고 순서를 코드가
    정했다. 리스트는 그 모호함이 없다: 잘못 적으면 gate 자리에 up이 들어가고,
    그건 `check_partition`이 실제 layer의 output_partition_sizes와 대조해 잡는다.
    """
    raw = copy.deepcopy(GATEUP)
    raw["projs"]["mlp.gate_up_proj"]["parts"].reverse()
    pp = parse_plan(raw).proj(0, "mlp.gate_up_proj")
    assert [p.name for p in pp.parts] == ["up", "gate"]
    assert pp.part("up").n_start == 0        # 적은 대로 앞에 온다


def test_unequal_parts():
    """`qkv_proj`는 [12288, 1024, 1024]로 불균등하다 (q에 attention 게이트가 실린다)."""
    raw = copy.deepcopy(GATEUP)
    raw["projs"]["mlp.gate_up_proj"] = {
        "k": H_MLP, "n": 12288 + 2 * 1024,
        "parts": [{"name": n, "n": sz, "bands": [[0, H_MLP, "hot"]]}
                  for n, sz in (("q", 12288), ("k", 1024), ("v", 1024))]}
    validate_static(parse_plan(raw))
    pp = parse_plan(raw).proj(0, "mlp.gate_up_proj")
    assert [(p.name, p.n_start, p.n_end) for p in pp.parts] == [
        ("q", 0, 12288), ("k", 12288, 13312), ("v", 13312, 14336)]


@pytest.mark.parametrize(
    "parts,msg",
    [
        ([{"name": "gate", "n": I_MLP, "bands": [[0, H_MLP, "hot"]]}], "list of 2 or more"),
        ([{"n": I_MLP, "bands": [[0, H_MLP, "hot"]]},
          {"name": "up", "n": I_MLP, "bands": [[0, H_MLP, "hot"]]}], "needs 'name' and 'n'"),
        ([{"name": "gate", "n": I_MLP, "bands": [[0, H_MLP, "hot"]]},
          {"name": "gate", "n": I_MLP, "bands": [[0, H_MLP, "hot"]]}], "duplicate part name"),
        ([{"name": "gate", "n": 8, "bands": [[0, H_MLP, "hot"]]},
          {"name": "up", "n": 8, "bands": [[0, H_MLP, "hot"]]}], "parts sum to 16"),
    ],
)
def test_bad_parts_die(parts, msg):
    raw = copy.deepcopy(GATEUP)
    raw["projs"]["mlp.gate_up_proj"]["parts"] = parts
    with pytest.raises(PlanError, match=msg):
        parse_plan(raw)


def test_check_partition():
    """요구는 "일치"가 아니라 **포함**이다 — plan 경계가 layer 분할 경계의 부분집합.

    통짜 plan이 4분할 layer를 덮는 경우를 받아야 한다: calib이 안 덮는 projection은
    마스킹을 안 하니 쪼갤 이유가 없고, 통짜가 스토어·launch가 적어 낫다
    (`linear_attn.in_proj_qkvz`가 실제로 그 경우다).
    """
    from sglang.srt.layers.prism.linear.plan import check_partition

    pp = parse_plan(GATEUP).proj(0, "mlp.gate_up_proj")
    check_partition(pp, [I_MLP, I_MLP], "where")          # 정확 일치
    with pytest.raises(PlanError, match="sum to"):
        check_partition(pp, [2 * I_MLP + 8], "where")     # 합이 다르다

    # 통짜 plan이 여러 분할 layer를 덮는 것은 정상이다
    raw = copy.deepcopy(GATEUP)
    raw["projs"]["mlp.gate_up_proj"] = {
        "k": H_MLP, "n": 16384, "bands": [[0, H_MLP, "hot"]]}
    solo = parse_plan(raw).proj(0, "mlp.gate_up_proj")
    check_partition(solo, [2048, 2048, 6144, 6144], "where")

    # 경계가 layer 분할을 가로지르면 조각이 남의 행을 섞는다
    raw["projs"]["mlp.gate_up_proj"] = {
        "k": H_MLP, "n": 16384,
        "parts": [{"name": "a", "n": 8192, "bands": [[0, H_MLP, "hot"]]},
                  {"name": "b", "n": 8192, "bands": [[0, H_MLP, "hot"]]}]}
    straddle = parse_plan(raw).proj(0, "mlp.gate_up_proj")
    with pytest.raises(PlanError, match="가로지른다"):
        check_partition(straddle, [2048, 2048, 6144, 6144], "where")

    # 크기가 뒤집히면 경계가 안 맞아 잡힌다 (gate/up 뒤바뀜 방어)
    raw["projs"]["mlp.gate_up_proj"] = {
        "k": H_MLP, "n": 2 * I_MLP,
        "parts": [{"name": "gate", "n": 8, "bands": [[0, H_MLP, "hot"]]},
                  {"name": "up", "n": 2 * I_MLP - 8, "bands": [[0, H_MLP, "hot"]]}]}
    with pytest.raises(PlanError, match="가로지른다"):
        check_partition(parse_plan(raw).proj(0, "mlp.gate_up_proj"),
                        [I_MLP, I_MLP], "where")


# ── sparsity 블록 ──────────────────────────────────────────────────────────


def _find(raw, name):
    """parts 리스트에서 이름으로 조각 dict를 찾는다 (테스트 편의)."""
    for d in raw["projs"]["mlp.gate_up_proj"]["parts"]:
        if d["name"] == name:
            return d
    raise KeyError(name)


def _sparse_plan(**part_over):
    raw = copy.deepcopy(GATEUP)
    raw["sparsity"] = {
        "score": "k2wl2",
        "calib": {"path": "assets/x.pt", "sha256": "0" * 64},
        "pmax": 0.9, "grid": 0.005, "ng": 201, "renorm_it": 3,
    }
    for half, tab in (("gate", "g"), ("up", "u")):
        _find(raw, half).update(
            {"calib": tab, "p": 0.5, "lambda": 0.0})
    for half, over in part_over.items():
        _find(raw, half).update(over)
    return raw


def test_sparsity_block_parses():
    plan = parse_plan(_sparse_plan())
    validate_static(plan)
    assert plan.sparsity.score == "k2wl2" and plan.sparsity.ng == 201
    assert plan.sparsity.calib.sha256 == "0" * 64
    for half, tab in (("gate", "g"), ("up", "u")):
        part = plan.proj(0, "mlp.gate_up_proj").part(half)
        assert part.sparsity_p == 0.5 and part.sparsity_lambda == 0.0
        assert part.calib == tab and part.sparse is True


def test_budget_without_sparsity_dies():
    """예산만 있고 sparsity 블록이 없으면 값이 조용히 버려진다."""
    raw = copy.deepcopy(GATEUP)
    _find(raw, "gate").update({"p": 0.5, "lambda": 0.0})
    with pytest.raises(PlanError, match="조용히 버려진다"):
        validate_static(parse_plan(raw))


def test_sparsity_without_budget_dies():
    """sparsity 블록만 있고 예산이 없으면 마스킹이 조용히 사라진다."""
    raw = _sparse_plan()
    del _find(raw, "up")["p"]
    del _find(raw, "up")["lambda"]
    with pytest.raises(PlanError, match=r"\['p', 'lambda'\] missing"):
        validate_static(parse_plan(raw))


def test_half_budget_dies():
    raw = _sparse_plan()
    del _find(raw, "gate")["lambda"]
    with pytest.raises(PlanError, match="p and lambda must both"):
        parse_plan(raw)


def test_p_out_of_range_dies():
    with pytest.raises(PlanError, match=r"p=1.5 out of \[0, 1\]"):
        validate_static(parse_plan(_sparse_plan(gate={"p": 1.5, "lambda": 0.0})))


def test_unknown_score_dies():
    raw = _sparse_plan()
    raw["sparsity"]["score"] = "l1"
    with pytest.raises(PlanError, match="unknown sparsity score"):
        validate_static(parse_plan(raw))


def test_coordinates_lists_every_layer_proj():
    plan = parse_plan(GATEUP)
    assert plan.coordinates() == {(0, "mlp.gate_up_proj"), (1, "mlp.gate_up_proj")}


# ── calib 키 + sparse opt-out ──────────────────────────────────────────────
#
# 빠뜨림과 의도적 제외를 **구분**하는 것이 목적이다. calib이 안 덮는 projection이
# 실재하므로(Muse-Glimmer의 `output_gate_proj`) 끄는 길은 있어야 하지만, 그냥 안
# 적으면 마스킹이 조용히 사라지므로 명시를 요구한다.


def _sp(**over):
    raw = copy.deepcopy(GATEUP)
    raw["sparsity"] = {"score": "k2wl2",
                       "calib": {"path": "assets/x.pt", "sha256": "0" * 64},
                       "pmax": 0.9, "grid": 0.005, "ng": 201, "renorm_it": 3}
    for half, tab in (("gate", "g"), ("up", "u")):
        _find(raw, half).update(
            {"calib": tab, "p": 0.5, "lambda": 0.0})
    for half, o in over.items():
        _find(raw, half).update(o)
    return raw


def test_calib_key_parses():
    plan = parse_plan(_sp())
    validate_static(plan)
    pp = plan.proj(0, "mlp.gate_up_proj")
    assert pp.part("gate").calib == "g" and pp.part("up").calib == "u"
    assert pp.part("gate").sparse is True


def test_sparse_false_opts_out():
    """calib이 안 덮는 projection은 명시적으로 뺀다."""
    raw = _sp(up={"sparse": False, "calib": None, "p": None, "lambda": None})
    h = _find(raw, "up")
    for k in ("calib", "p", "lambda"):
        h.pop(k)
    h["sparse"] = False
    plan = parse_plan(raw)
    validate_static(plan)
    pp = plan.proj(0, "mlp.gate_up_proj")
    assert pp.part("up").sparse is False and pp.part("up").calib is None
    assert pp.part("gate").sparse is True


def test_missing_calib_dies_not_silently_opts_out():
    """빠뜨리면 죽어야 한다 — 안 죽으면 마스킹이 조용히 사라진다."""
    raw = _sp()
    del _find(raw, "up")["calib"]
    with pytest.raises(PlanError, match=r"\['calib'\] missing"):
        validate_static(parse_plan(raw))


def test_sparse_false_with_budget_dies():
    raw = _sp(gate={"sparse": False})
    with pytest.raises(PlanError, match="제외하려면"):
        validate_static(parse_plan(raw))


def test_calib_without_sparsity_block_dies():
    raw = copy.deepcopy(GATEUP)
    _find(raw, "gate")["calib"] = "g"
    with pytest.raises(PlanError, match="조용히 버려진다"):
        validate_static(parse_plan(raw))


def test_non_string_calib_dies():
    raw = _sp(gate={"calib": 3})
    with pytest.raises(PlanError, match="must be a string table key"):
        parse_plan(raw)


def test_non_bool_sparse_dies():
    raw = _sp(gate={"sparse": "no"})
    with pytest.raises(PlanError, match="must be a boolean"):
        parse_plan(raw)
