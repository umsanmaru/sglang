"""ExecutionResources 테스트: storage identity 불변, in-place 계약, 크기 산정."""

import pytest
import torch

from sglang.srt.layers.moe.prism.plan import Proj, parse_plan, validate_static
from sglang.srt.layers.moe.prism.resources import (
    ColdStaging,
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


def test_spec_from_plan_carries_dims():
    """티어별 K 치수와 n_slots가 사라졌다 — 스토어가 flat + offset이 되면서
    크기를 로더가 알고, arena가 없어지면서 slot 개념 자체가 없다."""
    spec = ResourceSpec.from_plan(
        make_plan(), max_tokens=32, device=torch.device("cpu"))
    assert spec.max_tokens == 32
    assert spec.top_k == DIMS["top_k"]
    assert spec.hidden_size == DIMS["hidden_size"]
    assert spec.intermediate_size == DIMS["intermediate_size"]
    assert spec.n_of(Proj.DOWN) == DIMS["hidden_size"]
    assert spec.n_of(Proj.GATE) == DIMS["intermediate_size"]
    assert not hasattr(spec, "n_slots")


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
    """남은 영구 storage는 staging 하나다 — arena·stager 스크래치·sel 버퍼는
    소비자(bmm의 연속 배치 축)와 함께 사라졌다. warm_stream은 storage가 아니라
    스트림 핸들이고, prefill grouped 경로의 hot∥warm 겹침용으로 되살아났다
    (2026-08-26)."""
    res = ExecutionResources(make_spec(), pin_memory=True)
    assert isinstance(res.staging, ColdStaging)
    assert not hasattr(res, "arena")
    assert isinstance(res.warm_stream, torch.cuda.Stream)
