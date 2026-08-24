"""calib 자산 어댑터 테스트 (GPU 불필요).

핵심 성질은 test_weights.py와 같은 계열이다: **K축 밴드로 나눈 점수 재료를
다시 이어 붙이면 원본과 비트 단위로 같아야 한다.** wn은 K축, pair_dot은 K/2
축이라 절단 인덱스가 다르므로, 여기가 틀리면 두 티어가 서로 다른 채널의
threshold를 쓰면서도 아무 에러가 나지 않는다.
"""

import hashlib

import pytest
import torch

from sglang.srt.layers.moe.prism.calib import CalibTables
from sglang.srt.layers.moe.prism.plan import (
    PAIR_GROUP,
    CalibRef,
    ModelDims,
    PlanError,
    Proj,
    SparsitySpec,
)

L, E, H, I, NG = 2, 4, 256, 128, 201

DIMS = ModelDims(
    hidden_size=H,
    intermediate_size=I,
    num_layers=L,
    num_experts=E,
    top_k=2,
    dtype="bfloat16",
)


def make_asset(tmp_path, **overrides):
    """합성 gatedyn_calib.pt. 값은 인덱스로 만들어 슬라이스 대조가 가능하게 한다."""
    torch.manual_seed(0)
    blob = {
        "tg2l": torch.rand(L, E, NG),
        "tu2l": torch.rand(L, E, NG),
        "td2l": torch.rand(L, E, NG),
        "wn_g": torch.rand(L, E, H),
        "wn_u": torch.rand(L, E, H),
        "wn_d": torch.rand(L, E, I),
        "cg": torch.rand(L, E, H // PAIR_GROUP),
        "cu": torch.rand(L, E, H // PAIR_GROUP),
        "cd": torch.rand(L, E, I // PAIR_GROUP),
        # 자산에는 다른 score 계열 테이블과 스칼라도 함께 들어 있다 — 무시돼야 한다.
        "tg2w": torch.rand(L, E, NG),
        "lam0": 4.305,
        "PMAX": 0.9,
    }
    blob.update(overrides)
    for key in [k for k, v in blob.items() if v is None]:
        del blob[key]
    path = tmp_path / "gatedyn_calib.pt"
    torch.save(blob, path)
    return path, blob


def make_spec(path, sha256=None, score="k2wl2"):
    digest = sha256 or hashlib.sha256(path.read_bytes()).hexdigest()
    return SparsitySpec(
        score=score,
        calib=CalibRef(path=str(path), sha256=digest),
        pmax=0.9,
        grid=0.005,
        ng=NG,
        renorm_it=3,
    )


def load(tmp_path, **overrides):
    path, blob = make_asset(tmp_path, **overrides)
    return CalibTables.load(make_spec(path)), blob


# ── 로딩 ────────────────────────────────────────────────────────────────


def test_load_exposes_logical_names(tmp_path):
    tables, _ = load(tmp_path)
    shapes = tables.shapes()
    assert shapes["thr_gate"] == (L, E, NG)
    assert shapes["wn_gate"] == (L, E, H)
    assert shapes["wn_down"] == (L, E, I)
    assert shapes["pair_dot_gate"] == (L, E, H // PAIR_GROUP)
    assert shapes["pair_dot_down"] == (L, E, I // PAIR_GROUP)
    assert set(shapes) == {
        f"{stem}_{proj.value}"
        for stem in ("thr", "wn", "pair_dot")
        for proj in Proj
    }


def test_digest_mismatch_rejected(tmp_path):
    path, _ = make_asset(tmp_path)
    with pytest.raises(PlanError, match="digest mismatch"):
        CalibTables.load(make_spec(path, sha256="b" * 64))


def test_digest_check_can_be_skipped(tmp_path):
    path, _ = make_asset(tmp_path)
    CalibTables.load(make_spec(path, sha256="b" * 64), verify_digest=False)


def test_missing_table_rejected(tmp_path):
    path, _ = make_asset(tmp_path, cd=None)
    with pytest.raises(PlanError, match="has no 'cd'"):
        CalibTables.load(make_spec(path))


def test_non_tensor_table_rejected(tmp_path):
    path, _ = make_asset(tmp_path, wn_d=1.0)
    with pytest.raises(PlanError, match="must be a 3-D tensor"):
        CalibTables.load(make_spec(path))


def test_missing_file_rejected(tmp_path):
    spec = make_spec(tmp_path / "nope.pt", sha256="c" * 64)
    with pytest.raises(PlanError, match="cannot read calib asset"):
        CalibTables.load(spec)


def test_unsupported_score_rejected(tmp_path):
    path, _ = make_asset(tmp_path)
    with pytest.raises(NotImplementedError, match="k2wl2"):
        CalibTables.load(make_spec(path, score="k2wa"))


def test_tables_are_fp32_contiguous(tmp_path):
    path, _ = make_asset(tmp_path, wn_g=torch.rand(L, E, H).to(torch.bfloat16))
    tables = CalibTables.load(make_spec(path))
    band = tables.slice_band(0, Proj.GATE, 0, H, "t")
    assert band.wn.dtype is torch.float32 and band.wn.is_contiguous()


# ── 조회·절단 ───────────────────────────────────────────────────────────


def test_thr_is_per_layer_expert_curve(tmp_path):
    tables, blob = load(tmp_path)
    for layer in range(L):
        assert torch.equal(tables.thr(layer, Proj.UP), blob["tu2l"][layer])


@pytest.mark.parametrize(
    "proj,k,split",
    [(Proj.GATE, H, 64), (Proj.UP, H, 192), (Proj.DOWN, I, 64)],
)
def test_band_slices_reassemble_to_original(tmp_path, proj, k, split):
    """warm [0, split) + cold [split, K)를 이어 붙이면 원본과 비트 동일."""
    tables, blob = load(tmp_path)
    layer = 1
    warm = tables.slice_band(layer, proj, 0, split, "warm")
    cold = tables.slice_band(layer, proj, split, k, "cold")

    wn_key = {Proj.GATE: "wn_g", Proj.UP: "wn_u", Proj.DOWN: "wn_d"}[proj]
    dot_key = {Proj.GATE: "cg", Proj.UP: "cu", Proj.DOWN: "cd"}[proj]
    assert torch.equal(
        torch.cat([warm.wn, cold.wn], dim=1), blob[wn_key][layer]
    )
    assert torch.equal(
        torch.cat([warm.pair_dot, cold.pair_dot], dim=1), blob[dot_key][layer]
    )
    # 페어 축 길이가 K축 길이의 절반이어야 한다 (여기가 어긋나면 조용한 오답)
    assert warm.pair_dot.shape[1] == warm.k_rows // PAIR_GROUP
    assert cold.pair_dot.shape[1] == cold.k_rows // PAIR_GROUP


def test_slice_band_pair_offset_is_halved(tmp_path):
    """cold 밴드의 pair_dot은 start/2에서 시작해야 한다 (start가 아니라)."""
    tables, blob = load(tmp_path)
    cold = tables.slice_band(0, Proj.GATE, 64, H, "cold")
    assert torch.equal(cold.pair_dot, blob["cg"][0][:, 32:])


def test_odd_band_bound_is_assertion(tmp_path):
    tables, _ = load(tmp_path)
    with pytest.raises(AssertionError, match="splits a masking pair"):
        tables.slice_band(0, Proj.GATE, 0, 65, "bad")


# ── validate_static 연동 ────────────────────────────────────────────────


def test_probe_matches_expected_shapes(tmp_path):
    tables, _ = load(tmp_path)
    spec = make_spec(tmp_path / "gatedyn_calib.pt")
    expected = spec.expected_calib_shapes(DIMS)
    assert tables.probe()(spec.calib) == {k: tuple(v) for k, v in expected.items()}
    tables.check_dims(DIMS, spec)


def test_check_dims_rejects_wrong_model(tmp_path):
    tables, _ = load(tmp_path)
    spec = make_spec(tmp_path / "gatedyn_calib.pt")
    other = ModelDims(
        hidden_size=H,
        intermediate_size=I,
        num_layers=L,
        num_experts=E + 1,  # expert 수가 다른 모델
        top_k=2,
        dtype="bfloat16",
    )
    with pytest.raises(PlanError, match="calib table"):
        tables.check_dims(other, spec)
