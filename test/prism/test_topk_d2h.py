"""Host IDs are required by CPU/hybrid consumers, but not pure GPU prefill."""
import pytest
import torch

pytest.importorskip("kt_kernel")
from test_executor import build_executor, make_inputs, make_plan, make_weights, run


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["gpu", "cpu_fallback", "hybrid", "warm_cpu"])
def test_host_ids_and_output_match_legacy(mode, monkeypatch):
    kwargs = dict(cold_gpu_view=True, grouped_min_m=16,
                  cold_gpu_min_m=64 if mode == "cpu_fallback" else 16)
    if mode == "hybrid":
        kwargs["cold_hybrid_frac"] = (0.9, 0.9)
    if mode == "warm_cpu":
        kwargs.update(warm_kt=True, warm_cpu_min_m=16)
    ex = build_executor(make_plan("mixed"), *make_weights(), **kwargs)
    inputs = make_inputs(16, seed=42)
    original = ex._plan_flow
    host_ids = []

    def observe(*args, **kwargs):
        flow = original(*args, **kwargs)
        host_ids.append(flow.ids_cpu is not None)
        return flow

    monkeypatch.setattr(ex, "_plan_flow", observe)
    actual = run(ex, *inputs)
    assert host_ids == [mode != "gpu"]

    def legacy(*args, **kwargs):
        kwargs["needs_host_ids"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(ex, "_plan_flow", legacy)
    expected = run(ex, *inputs)
    assert torch.equal(actual, expected)
