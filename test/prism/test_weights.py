"""Stage 2 로더 테스트: 절단·변환이 정보를 보존하는지 (재조립 = 원본).

핵심 성질: 어떤 plan이든 warm 슬라이스(방향 복원) + cold 슬라이스를 K축으로
이어 붙이면 원본과 비트 단위로 같아야 한다 — 이게 깨지면 이후 모든 수치
테스트가 의미를 잃는다.
"""

import copy

import pytest
import torch

from sglang.srt.layers.moe.prism.plan import (
    PlanError,
    Proj,
    Tier,
    parse_plan,
    validate_static,
)
from sglang.srt.layers.moe.prism.calib import CalibTables
from sglang.srt.layers.moe.prism.plan import PAIR_GROUP, CalibRef, SparsitySpec
from sglang.srt.layers.moe.prism.weights import prepare_layer_weights

DIMS = {
    "hidden_size": 256,
    "intermediate_size": 128,
    "num_layers": 2,
    "num_experts": 4,
    "top_k": 2,
    "dtype": "bfloat16",
}

PIN = torch.cuda.is_available()  # CUDA 없으면 pinned 없이 로직만 검증


def make_plan(gate_bands=None, up_bands=None, down_bands=None):
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
            "gate": proj_entry(gate_bands or [[0, 64, "warm"], [64, 256, "cold"]], 128),
            "up": proj_entry(up_bands or [[0, 64, "warm"], [64, 256, "cold"]], 128),
            "down": proj_entry(down_bands or [[0, 64, "warm"], [64, 128, "cold"]], 256),
        },
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def make_weights():
    torch.manual_seed(0)
    e, h, i = DIMS["num_experts"], DIMS["hidden_size"], DIMS["intermediate_size"]
    w13 = torch.randn(e, 2 * i, h, dtype=torch.bfloat16)
    w2 = torch.randn(e, h, i, dtype=torch.bfloat16)
    return w13, w2


def sources(w13, w2):
    i = DIMS["intermediate_size"]
    return {Proj.GATE: w13[:, :i, :], Proj.UP: w13[:, i:, :], Proj.DOWN: w2}


def reassemble(prepared, proj, K):
    """warm(방향 복원) + cold를 K축 순서대로 이어 붙인 [E, N, K]."""
    pieces = []
    warm = prepared.warm.band(proj)
    if warm is not None:
        pieces.append((warm.k_offset, warm.weights.transpose(1, 2)))
    cold = prepared.cold.band(proj)
    if cold is not None:
        pieces.append((cold.k_offset, cold.weights))
    pieces.sort(key=lambda p: p[0])
    assert pieces and pieces[0][0] == 0
    out = torch.cat([t for _, t in pieces], dim=2)
    assert out.shape[2] == K
    return out


def test_roundtrip_reassembly_bitexact():
    plan = make_plan()
    w13, w2 = make_weights()
    src = sources(w13, w2)
    prepared = prepare_layer_weights(0, w13, w2, plan, pin_memory=PIN)
    for proj in Proj:
        K = plan.dims.k_of(proj)
        assert torch.equal(reassemble(prepared, proj, K), src[proj])


def test_warm_store_layout_and_offsets():
    plan = make_plan()
    w13, w2 = make_weights()
    prepared = prepare_layer_weights(0, w13, w2, plan, pin_memory=PIN)
    for proj, N in ((Proj.GATE, 128), (Proj.UP, 128), (Proj.DOWN, 256)):
        band = prepared.warm.band(proj)
        assert band.k_offset == 0 and band.k_rows == 64
        assert band.weights.shape == (4, 64, N)  # GEMM-ready [E, k, N]
        assert band.weights.dtype == torch.bfloat16
        assert band.weights.is_contiguous()
        if PIN:
            assert band.weights.is_pinned()
        cold = prepared.cold.band(proj)
        assert cold.k_offset == 64
        assert cold.weights.shape[2] == plan.dims.k_of(proj) - 64  # [E, N, k_cold]
        assert not cold.weights.is_cuda


def test_gate_up_bands_may_differ():
    plan = make_plan(up_bands=[[0, 128, "warm"], [128, 256, "cold"]])
    w13, w2 = make_weights()
    src = sources(w13, w2)
    prepared = prepare_layer_weights(0, w13, w2, plan, pin_memory=PIN)
    assert prepared.warm.band(Proj.GATE).k_rows == 64
    assert prepared.warm.band(Proj.UP).k_rows == 128
    for proj in Proj:
        assert torch.equal(
            reassemble(prepared, proj, plan.dims.k_of(proj)), src[proj]
        )


def test_all_cold_and_all_warm():
    plan = make_plan(
        gate_bands=[[0, 256, "cold"]],
        up_bands=[[0, 256, "warm"]],
        down_bands=[[0, 128, "cold"]],
    )
    w13, w2 = make_weights()
    src = sources(w13, w2)
    prepared = prepare_layer_weights(0, w13, w2, plan, pin_memory=PIN)
    assert prepared.warm.band(Proj.GATE) is None
    assert prepared.cold.band(Proj.UP) is None
    for proj in Proj:
        assert torch.equal(
            reassemble(prepared, proj, plan.dims.k_of(proj)), src[proj]
        )


def test_hot_band_not_implemented():
    plan = make_plan(gate_bands=[[0, 64, "hot"], [64, 256, "cold"]])
    w13, w2 = make_weights()
    with pytest.raises(NotImplementedError, match="HOT"):
        prepare_layer_weights(0, w13, w2, plan, pin_memory=PIN)


def test_multi_band_not_implemented():
    plan = make_plan(
        gate_bands=[[0, 64, "warm"], [64, 128, "cold"], [128, 192, "warm"], [192, 256, "cold"]]
    )
    w13, w2 = make_weights()
    with pytest.raises(NotImplementedError, match="one warm band"):
        prepare_layer_weights(0, w13, w2, plan, pin_memory=PIN)


def test_nonuniform_experts_not_implemented():
    raw_plan = make_plan()
    raw = {
        "schema_version": 1,
        "model_id": "test/tiny",
        "dims": dict(DIMS),
        "kernels": {"gpu_warm": "torch_bmm", "cpu_cold": "kt_amx_bf16"},
        "default": {
            "gate": {"bands": [[0, 64, "warm"], [64, 256, "cold"]],
                      "cold_shards": [[0, 0, 128]]},
            "up": {"bands": [[0, 64, "warm"], [64, 256, "cold"]],
                    "cold_shards": [[0, 0, 128]]},
            "down": {"bands": [[0, 64, "warm"], [64, 128, "cold"]],
                      "cold_shards": [[0, 0, 256]]},
        },
        "overrides": [{
            "layer": 0, "expert": 1,
            "gate": {"bands": [[0, 128, "warm"], [128, 256, "cold"]],
                      "cold_shards": [[0, 0, 128]]},
            "up": {"bands": [[0, 64, "warm"], [64, 256, "cold"]],
                    "cold_shards": [[0, 0, 128]]},
            "down": {"bands": [[0, 64, "warm"], [64, 128, "cold"]],
                      "cold_shards": [[0, 0, 256]]},
        }],
    }
    plan = parse_plan(raw)
    validate_static(plan)
    w13, w2 = make_weights()
    with pytest.raises(NotImplementedError, match="uniform geometry"):
        prepare_layer_weights(0, w13, w2, plan, pin_memory=PIN)
    del raw_plan


def test_shape_mismatch_rejected():
    plan = make_plan()
    w13, w2 = make_weights()
    with pytest.raises(PlanError, match="shape mismatch"):
        prepare_layer_weights(0, w13[:, :, :128], w2, plan, pin_memory=PIN)


# ---------------------------------------------------------------------------
# sparsity: wn/pair_dot이 weight와 같은 밴드 절단을 받는지
# ---------------------------------------------------------------------------

NG = 201


def make_sparse_plan(gate_bands=None, up_bands=None, down_bands=None,
                     p=0.5, lam=4.305):
    """make_plan과 같은 기하 + schema_version 2 sparsity."""
    def proj_entry(bands, N):
        has_cold = any(t == "cold" for _, _, t in bands)
        return {
            "bands": bands,
            "cold_shards": [[0, 0, N // 2], [1, N // 2, N]] if has_cold else [],
            "p": p,
            "lambda": lam,
        }

    raw = {
        "schema_version": 2,
        "model_id": "test/tiny",
        "dims": dict(DIMS),
        "kernels": {"gpu_warm": "torch_bmm", "cpu_cold": "kt_amx_bf16"},
        "sparsity": {
            "score": "k2wl2",
            "calib": {"path": "unused", "sha256": "a" * 64},
            "pmax": 0.9, "grid": 0.005, "ng": NG, "renorm_it": 3,
        },
        "default": {
            "gate": proj_entry(gate_bands or [[0, 64, "warm"], [64, 256, "cold"]], 128),
            "up": proj_entry(up_bands or [[0, 64, "warm"], [64, 256, "cold"]], 128),
            "down": proj_entry(down_bands or [[0, 64, "warm"], [64, 128, "cold"]], 256),
        },
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def make_calib(tmp_path):
    """합성 자산 + CalibTables (digest 검사는 건너뛴다 — 여기 관심사가 아니다)."""
    L, E = DIMS["num_layers"], DIMS["num_experts"]
    h, i = DIMS["hidden_size"], DIMS["intermediate_size"]
    torch.manual_seed(1)
    blob = {
        "tg2l": torch.rand(L, E, NG), "tu2l": torch.rand(L, E, NG),
        "td2l": torch.rand(L, E, NG),
        "wn_g": torch.rand(L, E, h), "wn_u": torch.rand(L, E, h),
        "wn_d": torch.rand(L, E, i),
        "cg": torch.rand(L, E, h // PAIR_GROUP),
        "cu": torch.rand(L, E, h // PAIR_GROUP),
        "cd": torch.rand(L, E, i // PAIR_GROUP),
    }
    path = tmp_path / "calib.pt"
    torch.save(blob, path)
    spec = SparsitySpec(
        score="k2wl2", calib=CalibRef(path=str(path), sha256="a" * 64),
        pmax=0.9, grid=0.005, ng=NG, renorm_it=3,
    )
    return CalibTables.load(spec, verify_digest=False), blob


def test_dense_plan_has_no_calib():
    w13, w2 = make_weights()
    prepared = prepare_layer_weights(0, w13, w2, make_plan(), pin_memory=PIN)
    assert prepared.thr is None
    for proj in Proj:
        assert prepared.warm.band(proj).calib is None
        assert prepared.cold.band(proj).calib is None


def test_calib_without_sparsity_rejected(tmp_path):
    w13, w2 = make_weights()
    calib, _ = make_calib(tmp_path)
    with pytest.raises(PlanError, match="both be present"):
        prepare_layer_weights(0, w13, w2, make_plan(), calib=calib, pin_memory=PIN)


def test_sparsity_without_calib_rejected():
    w13, w2 = make_weights()
    with pytest.raises(PlanError, match="both be present"):
        prepare_layer_weights(0, w13, w2, make_sparse_plan(), pin_memory=PIN)


def test_thr_curves_attached(tmp_path):
    w13, w2 = make_weights()
    calib, blob = make_calib(tmp_path)
    prepared = prepare_layer_weights(
        1, w13, w2, make_sparse_plan(), calib=calib, pin_memory=PIN
    )
    assert set(prepared.thr) == set(Proj)
    key = {Proj.GATE: "tg2l", Proj.UP: "tu2l", Proj.DOWN: "td2l"}
    for proj in Proj:
        assert prepared.thr[proj].shape == (DIMS["num_experts"], NG)
        assert torch.equal(prepared.thr[proj], blob[key[proj]][1])


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_calib_bands_reassemble_to_original(tmp_path, layer_idx):
    """warm.calib + cold.calib을 K축으로 이으면 그 레이어의 원본과 비트 동일.

    weight 재조립 테스트와 같은 성질 — 여기가 어긋나면 두 티어가 서로 다른
    채널의 노름/내적을 쓰면서도 아무 에러가 나지 않는다.
    """
    w13, w2 = make_weights()
    calib, blob = make_calib(tmp_path)
    prepared = prepare_layer_weights(
        layer_idx, w13, w2, make_sparse_plan(), calib=calib, pin_memory=PIN
    )
    wn_key = {Proj.GATE: "wn_g", Proj.UP: "wn_u", Proj.DOWN: "wn_d"}
    dot_key = {Proj.GATE: "cg", Proj.UP: "cu", Proj.DOWN: "cd"}
    for proj in Proj:
        warm = prepared.warm.band(proj).calib
        cold = prepared.cold.band(proj).calib
        assert torch.equal(
            torch.cat([warm.wn, cold.wn], dim=1), blob[wn_key[proj]][layer_idx]
        )
        assert torch.equal(
            torch.cat([warm.pair_dot, cold.pair_dot], dim=1),
            blob[dot_key[proj]][layer_idx],
        )


def test_calib_band_rows_track_weight_rows(tmp_path):
    """calib 밴드의 K 길이가 weight 밴드의 k_rows와 정확히 같아야 한다."""
    w13, w2 = make_weights()
    calib, _ = make_calib(tmp_path)
    plan = make_sparse_plan(
        gate_bands=[[0, 192, "warm"], [192, 256, "cold"]],
        up_bands=[[0, 192, "warm"], [192, 256, "cold"]],
        down_bands=[[0, 64, "warm"], [64, 128, "cold"]],
    )
    prepared = prepare_layer_weights(0, w13, w2, plan, calib=calib, pin_memory=PIN)
    for proj in Proj:
        for band in (prepared.warm.band(proj), prepared.cold.band(proj)):
            assert band.calib.k_rows == band.k_rows
            assert band.calib.pair_dot.shape[1] == band.k_rows // PAIR_GROUP


def test_all_warm_plan_has_no_cold_calib(tmp_path):
    """cold 밴드가 없으면 cold.calib도 없다 (밴드와 동행한다는 성질)."""
    w13, w2 = make_weights()
    calib, blob = make_calib(tmp_path)
    plan = make_sparse_plan(
        gate_bands=[[0, 256, "warm"]],
        up_bands=[[0, 256, "warm"]],
        down_bands=[[0, 128, "warm"]],
    )
    prepared = prepare_layer_weights(0, w13, w2, plan, calib=calib, pin_memory=PIN)
    for proj in Proj:
        assert prepared.cold.band(proj) is None
        assert torch.equal(
            prepared.warm.band(proj).calib.wn,
            blob[{Proj.GATE: "wn_g", Proj.UP: "wn_u", Proj.DOWN: "wn_d"}[proj]][0],
        )
