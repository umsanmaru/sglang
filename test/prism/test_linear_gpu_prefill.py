"""Staged prefill must preserve all columns and leave decode routing intact."""
import pytest
import torch

from sglang.srt.layers.prism.linear.executor import LinearExecutor
from sglang.srt.layers.prism.linear.plan import parse_plan
from sglang.srt.layers.prism.linear.weights import prepare_linear_weights
from sglang.srt.layers.prism.linear.prefill import DenseGpuPrefill

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_prefill_retains_weights_and_decode_uses_existing_tiers(monkeypatch):
    # Numerical test is independent of the runner's NUMA affinity. Real placement
    # is checked by the server benchmark launched on the GPU-local socket.
    from sglang.srt.layers.prism import numa
    monkeypatch.setattr(numa, "gpu_numa_node", lambda device: None)
    device = torch.device("cuda:0")
    k, n = 128, 320
    weight = torch.randint(-2, 3, (n, k)).to(torch.bfloat16)
    original = weight.clone()
    plan = parse_plan({"schema_version": 1, "model_id": "test",
                       "dims": {"num_layers": 1, "dtype": "bfloat16"},
                       "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_tile_k2_bf16"},
                       "projs": {"proj": {"k": k, "n": n,
                                  "bands": [[0, 64, "hot"], [64, k, "warm"]]}}})
    executor = LinearExecutor(device=device, prefill_gpu_bytes=32 * 1024)
    prepared = prepare_linear_weights(0, "proj", weight, plan, device=device, pin_memory=True)
    executor.register(0, "proj", prepared, prefill_weight=weight)
    executor.finalize()
    weight.zero_()
    for m in (17, 1):
        x = torch.randint(-2, 3, (m, k), device=device).to(torch.bfloat16)
        actual = executor.run(0, "proj", x)
        expected = (x.float() @ original.to(device).float().t()).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shared_buffer_orders_cross_stream_copies():
    engine = DenseGpuPrefill(torch.device("cuda:0"), 8192)
    weights = [torch.randint(-2, 3, (n, 128)).to(torch.bfloat16).pin_memory() for n in (96, 160)]
    xs = [torch.randint(-2, 3, (17, 128), device="cuda").to(torch.bfloat16) for _ in weights]
    outputs = []
    for x, weight in zip(xs, weights):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            outputs.append(engine.run(x, weight))
    torch.cuda.synchronize()
    for x, weight, output in zip(xs, weights, outputs):
        expected = (x.float() @ weight.cuda().float().t()).bfloat16()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
