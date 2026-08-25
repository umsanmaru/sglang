"""ExecutionResources 테스트: storage identity 불변, in-place 계약, 크기 산정."""

import pytest
import torch

from sglang.srt.layers.moe.prism.plan import Proj, parse_plan, validate_static
from sglang.srt.layers.moe.prism.resources import (
    ColdStaging,
    DeviceArena,
    ExecutionResources,
    ResourceSpec,
)

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DIMS = {
    "hidden_size": 256,
    "intermediate_size": 128,
    "num_layers": 2,
    "num_experts": 4,
    "top_k": 2,
    "dtype": "bfloat16",
}


def make_plan():
    raw = {
        "schema_version": 1,
        "model_id": "test/tiny",
        "dims": dict(DIMS),
        "kernels": {"gpu_warm": "torch_bmm", "cpu_cold": "kt_amx_bf16"},
        "default": {
            "gate": {"bands": [[0, 64, "warm"], [64, 256, "cold"]],
                      "cold_shards": [[0, 0, 128]]},
            "up": {"bands": [[0, 128, "warm"], [128, 256, "cold"]],
                    "cold_shards": [[0, 0, 128]]},
            "down": {"bands": [[0, 64, "warm"], [64, 128, "cold"]],
                      "cold_shards": [[0, 0, 256]]},
        },
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def make_spec(device="cuda"):
    return ResourceSpec.from_plan(
        make_plan(), max_tokens=8, device=torch.device(device)
    )


def test_spec_from_plan_takes_max_warm_rows():
    spec = make_spec("cpu")  # spec 산정 자체는 device 무관
    assert spec.k_warm_gate == 64
    assert spec.k_warm_up == 128
    assert spec.k_warm_down == 64
    assert spec.n_slots == DIMS["top_k"]
    assert spec.n_of(Proj.GATE) == 128 and spec.n_of(Proj.DOWN) == 256
    # hot 밴드가 없는 plan은 k_hot 전부 0
    assert (spec.k_hot_gate, spec.k_hot_up, spec.k_hot_down) == (0, 0, 0)


def make_hot_plan():
    raw = {
        "schema_version": 1,
        "model_id": "test/tiny-hot",
        "dims": dict(DIMS),
        "kernels": {"gpu_warm": "torch_bmm", "cpu_cold": "kt_amx_bf16"},
        "default": {
            "gate": {"bands": [[0, 64, "hot"], [64, 128, "warm"], [128, 256, "cold"]],
                      "cold_shards": [[0, 0, 128]]},
            "up": {"bands": [[0, 64, "hot"], [64, 128, "warm"], [128, 256, "cold"]],
                    "cold_shards": [[0, 0, 128]]},
            "down": {"bands": [[0, 64, "hot"], [64, 128, "cold"]],
                      "cold_shards": [[0, 0, 256]]},
        },
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def test_spec_from_plan_takes_max_hot_rows():
    spec = ResourceSpec.from_plan(
        make_hot_plan(), max_tokens=8, device=torch.device("cpu")
    )
    assert (spec.k_hot_gate, spec.k_hot_up, spec.k_hot_down) == (64, 64, 64)
    assert spec.k_hot_of(Proj.GATE) == 64


@cuda_required
def test_hot_arena_views_and_sel():
    spec = ResourceSpec.from_plan(
        make_hot_plan(), max_tokens=8, device=torch.device("cuda")
    )
    res = ExecutionResources(spec)
    gate = res.hot_view(Proj.GATE)
    assert gate.shape == (spec.n_slots, 64, 128)
    assert gate.dtype == torch.bfloat16 and gate.is_cuda
    assert res.hot_view(Proj.DOWN).shape == (spec.n_slots, 64, 256)
    # storage identity 불변 (계약 ④)
    assert res.hot_view(Proj.GATE).data_ptr() == gate.data_ptr()
    sel = res.hot_sel_device()
    assert sel.dtype == torch.int32 and sel.is_cuda and sel.shape == (spec.n_slots,)
    # warm sel과 물리적으로 분리 (스트림 간 공유 금지)
    assert sel.data_ptr() != res.sel_device().data_ptr()


@cuda_required
def test_hot_arena_absent_without_hot_bands():
    res = ExecutionResources(make_spec())
    with pytest.raises(KeyError):
        res.hot_view(Proj.GATE)


@cuda_required
def test_arena_views_shapes_and_aliasing():
    spec = make_spec()
    arena = DeviceArena(spec)
    gate = arena.view(Proj.GATE)
    up = arena.view(Proj.UP)
    down = arena.view(Proj.DOWN)
    assert gate.shape == (2, 64, 128)
    assert up.shape == (2, 128, 128)
    assert down.shape == (2, 64, 256)
    # gate/up은 서로소 구간
    assert gate.data_ptr() + gate.numel() * 2 <= up.data_ptr()
    # down은 flat 선두 재사용 (gate와 같은 주소에서 시작)
    assert down.data_ptr() == gate.data_ptr()
    # storage identity 불변: view는 항상 같은 객체 주소를 돌려준다
    assert arena.view(Proj.GATE).data_ptr() == gate.data_ptr()


@cuda_required
def test_arena_down_reuse_sizing():
    # down이 gate+up 합보다 큰 경우 flat이 down 기준으로 잡히는지
    spec = ResourceSpec(
        max_tokens=4, top_k=2, hidden_size=1024, intermediate_size=64,
        k_warm_gate=32, k_warm_up=32, k_warm_down=64, n_slots=2,
        device=torch.device("cuda"),
    )
    arena = DeviceArena(spec)
    down_bytes = 2 * 64 * 1024 * 2
    gateup_bytes = 2 * 32 * 64 * 2 * 2
    assert arena.nbytes == max(down_bytes, gateup_bytes) == down_bytes


@cuda_required
def test_staging_inplace_identity_and_capacity():
    spec = make_spec()
    staging = ColdStaging(spec, pin_memory=True)
    x1 = torch.randn(3, 256, dtype=torch.bfloat16)
    v1 = staging.fill_x(x1)
    ptr = v1.data_ptr()
    assert torch.equal(v1.cpu(), x1)
    assert v1.is_pinned()
    # 같은 버퍼에 다시 채워도 주소 불변 (in-place 계약)
    v2 = staging.fill_x(torch.randn(5, 256, dtype=torch.bfloat16))
    assert v2.data_ptr() == ptr
    # 용량 초과는 즉사
    with pytest.raises(ValueError, match="exceed staging capacity"):
        staging.fill_x(torch.randn(9, 256, dtype=torch.bfloat16))


@cuda_required
def test_staging_shapes():
    spec = make_spec()
    staging = ColdStaging(spec, pin_memory=True)
    act = torch.randn(4, 2, 128, dtype=torch.bfloat16)
    assert staging.fill_act(act).shape == (4, 2, 128)
    assert staging.gateup_out(4).shape == (4, 2, 256)
    assert staging.gateup_out(4).dtype == torch.bfloat16  # 계약 ⑤ 개정: wire=bf16
    assert staging.down_out(4).shape == (4, 2, 256)  # hidden=256
    ids = torch.randint(0, 4, (4, 2), dtype=torch.int64)
    assert torch.equal(staging.fill_expert_ids(ids).cpu(), ids)


@cuda_required
def test_execution_resources_bundle():
    res = ExecutionResources(make_spec(), pin_memory=True)
    assert isinstance(res.warm_stream, torch.cuda.Stream)
    assert res.arena.view(Proj.GATE).is_cuda
    res.evt_staged.record(res.warm_stream)
    res.evt_staged.synchronize()
