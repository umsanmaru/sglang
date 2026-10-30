"""Bounded packed staging versus independent indexed GEMM, including slot reuse."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _packed_store(rows, n=64, seed=1):
    from sglang.srt.layers.moe.prism.cold_gpu import ColdSlab
    from sglang.srt.layers.moe.prism.plan import Proj

    generator = torch.Generator().manual_seed(seed)
    weights, packed, offsets = [], [], [0]
    for kr in rows:
        w = torch.randint(-2, 3, (kr, n), generator=generator).to(torch.bfloat16)
        weights.append(w)
        # Independent AMX tile packing: N blocks, K blocks, N steps, K steps.
        chunks = []
        for nb in range(0, n, 64):
            for kb in range(0, kr, 64):
                for ns in range(nb, min(nb + 64, n), 32):
                    for ks in range(kb, min(kb + 64, kr), 32):
                        tile = w[ks:ks + 32, ns:ns + 32]
                        chunks.append(tile.reshape(16, 2, 2, 16).permute(2, 0, 3, 1).reshape(-1))
        data = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.bfloat16)
        packed.append(data)
        offsets.append(offsets[-1] + data.numel())
    row_off = torch.tensor([0] + list(torch.tensor(rows).cumsum(0).tolist()),
                           dtype=torch.int32, device="cuda")
    kidx = torch.cat([torch.randperm(max(rows), generator=generator)[:r]
                     for r in rows]).to(dtype=torch.uint16, device="cuda")
    slab = ColdSlab(torch.cat(packed).pin_memory(),
                    torch.tensor(offsets[:-1], dtype=torch.int64, device="cuda"),
                    row_off, kidx, n, 0, 64, 64, 0, Proj.GATE,
                    expert_offsets=tuple(offsets), k_max=max(rows))
    return slab, torch.cat(weights).cuda()


@pytest.mark.parametrize("m", [1536, 2048])
@pytest.mark.parametrize("pair", [False, True])
def test_buffered_packed_reuses_topk_slots(m, pair, monkeypatch):
    from sglang.jit_kernel import prism_buffered
    from sglang.jit_kernel.prism_grouped import grouped_gemm_cold_gateup, grouped_gemm_indexed
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    # E is not divisible by top-k; zero-row and unselected experts are included.
    rows, topk = [32, 96, 0, 64, 32, 96, 64], 3
    gate, wg = _packed_store(rows)
    up, wu = _packed_store(rows[::-1], seed=2)
    ids = torch.randint(0, len(rows), (m, topk), device="cuda")
    ids[:, 0] = 1
    ids[ids == 5] = 6
    grouping = build_grouping(ids, len(rows))
    x = torch.randint(-2, 3, (m * topk if pair else m, 96), device="cuda").bfloat16()
    out = torch.full((m, topk, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
    ref = torch.empty_like(out)
    stream = torch.cuda.current_stream()
    for slab, w, col in [(gate, wg, 0), (up, wu, 64)]:
        grouped_gemm_indexed(x, grouping, w, slab.row_off, slab.k_index,
                             ref, col, pair, stream)
    allocations = []
    original_empty = torch.empty

    def tracked_empty(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        if tensor.numel():
            allocations.append(tensor.numel())
        return tensor

    with monkeypatch.context() as patch:
        patch.setattr(prism_buffered.torch, "empty", tracked_empty)
        grouped_gemm_cold_gateup(x, grouping, gate, up, out, 0, 64, pair, stream)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    assert allocations == [topk * max(rows) * 64]


def test_buffered_packed_capture_replay():
    from sglang.jit_kernel.prism_grouped import grouped_gemm_cold
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    slab, _ = _packed_store([32, 64, 96, 32, 64])
    ids = torch.randint(0, 5, (1536, 2), device="cuda")
    grouping = build_grouping(ids, 5)
    x = torch.ones(1536, 96, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1536, 2, 64, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        grouped_gemm_cold(x, grouping, slab, out, 0, False, side)
    torch.cuda.current_stream().wait_stream(side)
    expected = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        grouped_gemm_cold(x, grouping, slab, out, 0, False, side)
    for scale in [2, 3]:
        x.fill_(scale)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected * scale)


def test_buffered_packed_all_zero_rows():
    from sglang.jit_kernel.prism_grouped import grouped_gemm_cold
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    slab, _ = _packed_store([0, 0, 0])
    ids = torch.randint(0, 3, (1536, 1), device="cuda")
    x = torch.ones(1536, 32, dtype=torch.bfloat16, device="cuda")
    out = torch.full((1536, 1, 64), float("nan"), dtype=torch.bfloat16, device="cuda")
    grouped_gemm_cold(x, build_grouping(ids, 3), slab, out, 0, False,
                      torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0
