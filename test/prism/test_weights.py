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
