"""Indexed BF16 prefill correctness and the top-k device weight bound."""
import pytest
import torch

from sglang.jit_kernel.prism_prefill import Projection, grouped_prefill
from sglang.srt.layers.moe.prism.grouping import build_grouping


def pack(weights, nb, kb, nc=False):
    pieces = []
    for w in weights:
        k, n = w.shape
        for n0 in range(0, n, nb):
            for k0 in range(0, k, kb):
                block = w[k0:k0 + kb, n0:n0 + nb]
                ks, ns = block.shape
                tiles = block.reshape(ks // 32, 16, 2, ns // 32, 2, 16)
                order = (0, 3, 4, 1, 5, 2) if nc else (3, 0, 4, 1, 5, 2)
                pieces.append(tiles.permute(order).reshape(-1))
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.bfloat16)


@pytest.mark.parametrize('layout', ['rowmajor', 'kt_bf16', 'kt_bf16_nc'])
@pytest.mark.parametrize('pair_input', [False, True])
def test_indexed_buffered(layout, pair_input):
    torch.manual_seed(8)
    e, topk, m, h, n = 7, 3, 73, 128, 96
    ks = [0, 32, 64, 96, 32, 64, 32]
    weights = [torch.randn(k, n, dtype=torch.bfloat16) * .03 for k in ks]
    source = (torch.cat(weights).flatten() if layout == 'rowmajor'
              else pack(weights, 64, 64, nc=layout.endswith('_nc'))).pin_memory()
    ro = torch.tensor([0] + ks).cumsum(0).int()
    idx = torch.cat([torch.randperm(h)[:k] for k in ks]).to(torch.uint16).cuda()
    p = Projection(source, ro.cuda(), idx, n, layout,
                   (ro[:-1].long() * n).cuda(), 64, 64, tuple(ro.tolist()))
    ids = torch.randint(0, e, (m, topk), device='cuda')
    # Exercise skipped experts with a nonempty pair range (hybrid scheduling).
    mask = torch.ones(e, dtype=torch.bool, device='cuda'); mask[5] = False
    g = build_grouping(ids, e, expert_mask=mask)
    x = torch.randn(m * topk if pair_input else m, h, device='cuda', dtype=torch.bfloat16)
    out = torch.full((m, topk, n + 16), 7., device='cuda', dtype=torch.bfloat16)
    expected = out.clone()
    for expert in range(e):
        if expert == 5:
            continue
        pairs = (ids.flatten() == expert).nonzero().flatten()
        rows = pairs if pair_input else pairs // topk
        cols = idx[int(ro[expert]):int(ro[expert + 1])].long()
        val = x[rows][:, cols] @ weights[expert].cuda()
        expected.view(-1, n + 16)[pairs, 8:8 + n] = val
    stats = {}
    grouped_prefill(x, g, [(p, 8)], out, pair_input, torch.cuda.current_stream(), stats=stats)
    torch.testing.assert_close(out, expected, rtol=.02, atol=.002)
    assert stats['weight_slots'] == topk
    assert stats['weight_elements'] == topk * max(ks) * n


def test_all_zero_projection():
    ids = torch.zeros((5, 1), device='cuda', dtype=torch.long)
    ro = torch.zeros(2, device='cuda', dtype=torch.int32)
    p = Projection(torch.empty(0, dtype=torch.bfloat16, device='cuda'), ro,
                   torch.empty(0, dtype=torch.uint16, device='cuda'), 64,
                   rows_host=(0, 0))
    out = torch.ones(5, 1, 64, device='cuda', dtype=torch.bfloat16)
    grouped_prefill(torch.ones(5, 32, device='cuda', dtype=torch.bfloat16),
                    build_grouping(ids, 1), [(p, 0)], out, False,
                    torch.cuda.current_stream())
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize('overlap', ['0', '1'])
def test_multiple_projections_and_chunks(overlap, monkeypatch):
    monkeypatch.setenv('SGLANG_PRISM_BUFFERED_OVERLAP', overlap)
    torch.manual_seed(12)
    # Reuse both banks several times, tail batch, and >1024 rows in an expert.
    e, m, topk, k, n = 7, 1050, 3, 32, 64
    ro = torch.arange(e + 1, device='cuda', dtype=torch.int32) * k
    idx = torch.arange(k).repeat(e).to(torch.uint16).cuda()
    weights = torch.randn(e, k, n, dtype=torch.bfloat16).pin_memory()
    p = Projection(weights, ro, idx, n, rows_host=tuple(range(0, (e + 1) * k, k)))
    ids = torch.randint(1, e, (m, topk), device='cuda'); ids[:, 0] = 0
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    out = torch.empty(m, topk, n * 2, device='cuda', dtype=torch.bfloat16)
    grouping = build_grouping(ids, e)
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    grouped_prefill(x, grouping, [(p, 0), (p, n)], out, False, stream)
    torch.cuda.current_stream().wait_stream(stream)
    for expert in range(e):
        rows, ranks = (ids == expert).nonzero(as_tuple=True)
        expected = x[rows] @ weights[expert].cuda()
        torch.testing.assert_close(out[rows, ranks, :n], expected)
        torch.testing.assert_close(out[rows, ranks, n:], expected)


@pytest.mark.parametrize("sparse", [False, True])
def test_warm_cpu_boundary_and_nc_runtime(monkeypatch, tmp_path, sparse):
    from test_executor import build_executor, make_weights, make_inputs, run, moe_reference, rel_diff
    from test_nonuniform_geometry import E, H, I, DIMS, _entry, _contig
    from sglang.srt.layers.moe.prism.plan import parse_plan, validate_static
    overrides = []
    for e in range(E):
        overrides.append({'layer': 0, 'expert': e,
                          'gate': _entry(_contig(H, 64, 0 if e == 3 else 32), I),
                          'up': _entry(_contig(H, 64, 32), I),
                          'down': _entry(_contig(I, 32, 32), H)})
    plan = parse_plan({'schema_version': 1, 'model_id': 'test/tiny', 'dims': dict(DIMS),
                       'kernels': {'gpu_warm': 'gemv_worklist', 'cpu_cold': 'kt_tile_k2_bf16'},
                       'overrides': overrides})
    validate_static(plan)
    monkeypatch.setenv('SGLANG_PRISM_BUFFERED_MIN_M', '32')
    monkeypatch.setenv('SGLANG_PRISM_BUFFERED_PREFILL', '1')
    import test_executor
    if sparse:
        from dataclasses import replace
        from test_nonuniform_sparse import make_calib, make_plan as sparse_plan
        calib, _, path = make_calib(tmp_path, 0.0)
        plan = replace(plan, schema_version=2, sparsity=sparse_plan('uniform', path).sparsity,
                       experts={key: replace(ep,
                           gate=replace(ep.gate, sparsity_p=.5, sparsity_lambda=4.),
                           up=replace(ep.up, sparsity_p=.5, sparsity_lambda=4.),
                           down=replace(ep.down, sparsity_p=.5, sparsity_lambda=4.))
                                for key, ep in plan.experts.items()})
        prepare = test_executor.prepare_layer_weights
        monkeypatch.setattr(test_executor, 'prepare_layer_weights',
                            lambda *args, **kwargs: prepare(*args, calib=calib, **kwargs))
    backend = test_executor.KtColdBackend
    monkeypatch.setattr(test_executor, 'KtColdBackend',
                        lambda *args, **kwargs: backend(*args, cpuinfer_threads=14, **kwargs))
    weights = make_weights(seed=17)
    ex = build_executor(plan, *weights, warm_kt=True, cold_gpu_view=True,
                        grouped_min_m=16, cold_gpu_min_m=32, warm_cpu_max_m=32)
    calls = []
    original = ex._cold.submit_warm_gateup
    def submit(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(ex._cold, 'submit_warm_gateup', submit)
    for m in (1, 16, 32):
        calls.clear()
        x, ids, router = make_inputs(m, seed=18 + m)
        ids[0, 0], ids[0, 1] = 3, 4  # valid top-k: no duplicate expert IDs
        actual = run(ex, x, ids, router)
        expected = moe_reference(x, ids, router, *weights)
        assert rel_diff(actual, expected) < .03
        assert bool(calls) == (m == 16)


def test_dispatch_keeps_small_expert_gemms_on_wres(monkeypatch):
    from sglang.jit_kernel.prism_prefill import enabled
    monkeypatch.setenv('SGLANG_PRISM_BUFFERED_PREFILL', 'auto')
    monkeypatch.setenv('SGLANG_PRISM_BUFFERED_MIN_M', '1536')
    out = torch.empty(1536, 10, 1, device='cuda', dtype=torch.bfloat16)
    balanced = torch.arange(15360, device='cuda').view(1536, 10) % 512
    g = build_grouping(balanced, 512)
    assert not enabled(out, g)
    concentrated = build_grouping(balanced % 32, 512)
    assert enabled(out, concentrated)
    monkeypatch.setenv('SGLANG_PRISM_BUFFERED_PREFILL', '1')
    assert enabled(out, g)
    monkeypatch.setenv('SGLANG_PRISM_BUFFERED_PREFILL', '0')
    assert not enabled(out, concentrated)
