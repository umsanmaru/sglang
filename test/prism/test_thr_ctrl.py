"""Online threshold controller (thr_ctrl.py): GPU mask counters, the window update, live tables.

Needs CUDA (counters and the update kernel live on the device); kt is not needed. The kt side
of the counters is tested in kt-kernel test/prism/test_partial_moe.py.
"""

from __future__ import annotations

import math

import pytest
import torch

from sglang.srt.layers.moe.prism.plan import Proj, Tier, parse_plan, validate_static
from sglang.srt.layers.moe.prism.thr_ctrl import PROJ_ORDER, ThresholdController

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

E, K, TOPK, NG, GRID = 4, 64, 2, 21, 0.05
P = 0.5


def make_plan(*, window=4096, mu=1.0, alpha_min=0.25, alpha_max=4.0, calib_path="unused"):
    """One layer, every K row warm, so full-K sparsity equals masked sparsity."""
    proj = {"bands": [[0, K, "warm"]], "p": P, "lambda": 0.0}
    raw = {
        "schema_version": 2, "model_id": "t",
        "dims": {"hidden_size": K, "intermediate_size": K, "num_layers": 1,
                 "num_experts": E, "top_k": TOPK, "dtype": "bfloat16"},
        "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_amx_bf16"},
        "default": {"gate": dict(proj), "up": dict(proj), "down": dict(proj)},
        "sparsity": {"score": "k2wl2", "calib": {"path": calib_path, "sha256": "a" * 64},
                     "pmax": 0.9, "grid": GRID, "ng": NG, "renorm_it": 0,
                     "controller": {"window": window, "mu": mu,
                                    "alpha_min": alpha_min, "alpha_max": alpha_max}},
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def _alpha(alpha, d, mu, lo=0.25, hi=4.0):
    return min(max(alpha * (1.0 - mu * (d - P)), lo), hi)


@cuda_required
def test_step_moves_alpha_by_full_k_window_deltas():
    """Every `window` steps alpha moves by that window's realized full-K sparsity, and both live
    tables become calib * alpha; between boundaries nothing changes.

    The denominator is window * top_k * K/2 pairs, so unmasked (hot) rows count as kept; the
    second window must use counter deltas, not running totals; alpha clamps; a proj with no
    masked rows keeps its alpha.
    """
    W, MU = 4, 4.0
    ctrl = ThresholdController(make_plan(window=W, mu=MU))
    calib = {p: torch.rand(E, NG) + 0.5 for p in PROJ_ORDER}
    s = ctrl.register_layer(0, calib, torch.device("cuda"))
    full = W * TOPK * K // 2

    def run_window(gate, up):
        """gate counts go to the GPU counter, up counts to kt's pinned counter; [kept, total]."""
        s.gpu_ctr[0] += torch.tensor(gate, device="cuda")
        s.cpu_ctr[1] += torch.tensor(up)
        before = s.live_dev.clone()
        for _ in range(W - 1):
            ctrl.step(0)
        torch.cuda.synchronize()
        assert torch.equal(s.live_dev, before), "tables changed before the window boundary"
        ctrl.step(0)
        torch.cuda.synchronize()

    alpha = [1.0, 1.0, 1.0]
    # Window 1: gate cuts 3/4 of full K, up cuts 1/8, down has no masked rows.
    run_window(gate=[full // 4, full], up=[7 * full // 8, full])
    alpha = [_alpha(alpha[0], 0.75, MU), _alpha(alpha[1], 0.125, MU), 1.0]
    alpha_log = [alpha[0]]
    assert s.alpha.tolist() == pytest.approx(alpha, rel=1e-6)
    # Window 2: gate masks only half of K and keeps half of that, so d = 1/4 (not 1/2);
    # up cuts 7/8 and clamps at alpha_min.
    run_window(gate=[full // 4, full // 2], up=[full // 8, full])
    alpha = [_alpha(alpha[0], 0.25, MU), _alpha(alpha[1], 0.875, MU), 1.0]
    alpha_log.append(alpha[0])
    assert s.alpha.tolist() == pytest.approx(alpha, rel=1e-6)
    assert alpha[1] == 0.25

    expect = s.calib_dev * s.alpha.view(3, 1, 1)
    assert torch.equal(s.live_dev, expect)
    assert torch.equal(s.live_host, expect.cpu())
    report = ctrl.report()[0]
    assert report["windows_done"] == 2
    windows = report["windows"]
    assert [w["gate"]["alpha"] for w in windows] == pytest.approx(alpha_log)
    assert [w["gate"]["sparsity"] for w in windows] == pytest.approx([0.75, 0.25])
    assert [w["up"]["sparsity"] for w in windows] == pytest.approx([0.125, 0.875])
    assert all(math.isnan(w["down"]["sparsity"]) for w in windows)


@cuda_required
def test_step_runs_under_a_bf16_default_dtype():
    """The server loads layers with torch's default dtype set to the model's; buffers stay fp32."""
    ctrl = ThresholdController(make_plan(window=1))
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        s = ctrl.register_layer(0, {p: torch.rand(E, NG) for p in PROJ_ORDER}, torch.device("cuda"))
        ctrl.step(0)
        torch.cuda.synchronize()
    finally:
        torch.set_default_dtype(prev)
    assert int(s.windows_host[0]) == 1


def _store(kr, N, seed):
    """Indexed warm store with small-integer x/a/c, so every pair energy is an exact integer."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-3, 4, (1, K), generator=g).to(torch.bfloat16)
    w = (torch.randn(E * kr, N, generator=g) * 0.1).to(torch.bfloat16)
    kidx = torch.cat([torch.randperm(K, generator=g)[:kr] for _ in range(E)]).to(torch.uint16)
    row_off = torch.arange(E + 1, dtype=torch.int32) * kr
    a = torch.randint(0, 4, (E * kr,), generator=g).float()
    c = torch.randint(-2, 3, (E * kr // 2,), generator=g).float()
    return x, w, kidx, row_off, a, c


def _expected_counts(x, ids, kidx, row_off, a, c, thr2):
    kept = total = 0
    for e in ids.flatten().tolist():
        lo, hi = int(row_off[e]), int(row_off[e + 1])
        xs = x[0, kidx[lo:hi].to(torch.int64)].float()
        x0, x1 = xs[0::2], xs[1::2]
        en = a[lo:hi:2] * x0 * x0 + a[lo + 1:hi:2] * x1 * x1 + 2 * c[lo // 2:hi // 2] * x0 * x1
        kept += int((en.clamp(min=0) >= thr2).sum())
        total += (hi - lo) // 2
    return [kept, total]


@cuda_required
@pytest.mark.parametrize("fused", [False, True])
def test_gemv_counts_each_mask_once(fused):
    """The sparse GEMV adds exactly the kept/total pairs of its masks to the spec's counter.

    N spans several column tiles that each recompute the mask, so counting in more than one
    tile multiplies the totals; the fused gate+up launch must count each slot into its own spec.
    """
    from sglang.jit_kernel import prism_gemv as k
    from sglang.srt.layers.moe.prism.tiers import SparseSpec

    kr, N, T2 = 48, 512, 4.5  # energies are integers, so T2 sits strictly between two of them
    assert N > 64, "must span several bf16 column tiles (kNCol = 64)"
    ids = torch.randint(0, E, (1, TOPK), generator=torch.Generator().manual_seed(7))
    topk_w = torch.rand(1, TOPK, device="cuda")
    stores = [_store(kr, N, seed) for seed in ((11, 12) if fused else (11,))]
    x = stores[0][0]

    def spec(st):
        _, _, _, _, a, c = st
        return SparseSpec(a=a.cuda(), c=c.cuda(), thr=torch.full((E, NG), math.sqrt(T2), device="cuda"),
                          p=P, lam=0.0, pmax=0.9, grid=GRID, ng=NG, renorm_it=0,
                          ctr=torch.zeros(2, dtype=torch.int64, device="cuda"))

    def dev(st):
        return st[1].cuda(), st[3].cuda(), st[2].cuda()  # w, row_off, kidx

    specs = [spec(st) for st in stores]
    out = torch.zeros(1, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.current_stream()
    if fused:
        (wg, rg, kg), (wu, ru, ku) = dev(stores[0]), dev(stores[1])
        k.gemv_worklist_indexed_sparse_gateup(x.cuda(), ids.int().cuda(), topk_w, wg, rg, kg,
                                              wu, ru, ku, out, specs[0], specs[1], 0, N, False, s)
    else:
        w, ro, ki = dev(stores[0])
        k.gemv_worklist_indexed_sparse(x.cuda(), ids.int().cuda(), topk_w, w, ro, ki, out,
                                       specs[0], 0, False, s)
    torch.cuda.synchronize()
    for st, sp in zip(stores, specs):
        _, _, kidx, row_off, a, c = st
        expect = _expected_counts(x, ids, kidx, row_off, a, c, T2)
        assert 0 < expect[0] < expect[1], "mask is trivial"
        assert sp.ctr.tolist() == expect


@cuda_required
@pytest.mark.parametrize("fmt", ["fp8", "mxfp4"])
def test_quantized_gemv_counts_each_mask_once(fmt):
    """The fp8 and mxfp4 worklist kernels count their masks like the bf16 one (pairs, one
    column tile), although each stages and walks K differently."""
    import importlib
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from sglang.jit_kernel import prism_gemv_fp8, prism_gemv_mxfp4
    from sglang.srt.layers.moe.prism.tiers import SparseSpec

    ref = importlib.import_module(f"{fmt}_ref")
    k2, kr, N, T2 = 256, 128, 512, 4.5
    g = torch.Generator().manual_seed(31)
    codes, scales, kidx = [], [], []
    for _ in range(E):
        c_ck, s_ck = ref.random_expert_ckpt(N, k2, g, exact=True)
        rows = ref.aligned_index(k2, kr, g)
        store = ref.row_store if fmt == "fp8" else ref.pairrow_store
        c, s = store(c_ck, s_ck, rows)
        codes.append(c)
        scales.append(s)
        kidx.append(rows)
    kidx = torch.cat(kidx).to(torch.uint16)
    row_off = torch.arange(E + 1, dtype=torch.int32) * kr
    x = torch.randint(-3, 4, (1, k2), generator=g).to(torch.bfloat16)
    a = torch.randint(0, 4, (E * kr,), generator=g).float()
    c = torch.randint(-2, 3, (E * kr // 2,), generator=g).float()
    ids = torch.randint(0, E, (1, TOPK), generator=g)
    sp = SparseSpec(a=a.cuda(), c=c.cuda(), thr=torch.full((E, NG), math.sqrt(T2), device="cuda"),
                    p=P, lam=0.0, pmax=0.9, grid=GRID, ng=NG, renorm_it=0,
                    ctr=torch.zeros(2, dtype=torch.int64, device="cuda"))
    gemv = (prism_gemv_fp8.gemv_fp8_indexed_sparse if fmt == "fp8"
            else prism_gemv_mxfp4.gemv_mxfp4_indexed_sparse)
    out = torch.zeros(1, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv(x.cuda(), ids.int().cuda(), torch.rand(1, TOPK, device="cuda"), torch.cat(codes).cuda(),
         torch.cat(scales).cuda(), row_off.cuda(), kidx.cuda(), out, sp, 0, False,
         torch.cuda.current_stream())
    torch.cuda.synchronize()
    expect = _expected_counts(x, ids, kidx, row_off, a, c, T2)
    assert 0 < expect[0] < expect[1], "mask is trivial"
    assert sp.ctr.tolist() == expect


@cuda_required
def test_sparse_tiers_read_the_live_tables_and_counters(tmp_path):
    """With a controller, every sparse GPU tier reads the controller's live device table by
    pointer and counts into its counter; a private copy of the curve would silently decouple
    the masks from the controller."""
    from sglang.srt.layers.moe.prism.calib import CalibTables
    from sglang.srt.layers.moe.prism.tiers import build_layer_tiers
    from sglang.srt.layers.moe.prism.weights import prepare_layer_weights

    g = torch.Generator().manual_seed(5)
    blob = {key: torch.rand(1, E, K, generator=g) + 0.5 for key in ("wn_g", "wn_u", "wn_d")}
    blob.update({key: torch.rand(1, E, K // 2, generator=g) - 0.5 for key in ("cg", "cu", "cd")})
    blob.update({key: torch.rand(1, E, NG, generator=g) for key in ("tg2l", "tu2l", "td2l")})
    torch.save(blob, tmp_path / "calib.pt")
    plan = make_plan(calib_path=str(tmp_path / "calib.pt"))
    calib = CalibTables.load(plan.sparsity, verify_digest=False)
    w13 = torch.randn(E, 2 * K, K, generator=g).to(torch.bfloat16)
    w2 = torch.randn(E, K, K, generator=g).to(torch.bfloat16)
    device = torch.device("cuda", torch.cuda.current_device())
    prepared = prepare_layer_weights(0, w13, w2, plan, calib=calib, device=device)

    state = ThresholdController(plan).register_layer(0, prepared.thr, device)
    tiers = build_layer_tiers(prepared, plan, 0, thr_ctrl=state)
    gateup, down = tiers.gateup[Tier.WARM], tiers.down[Tier.WARM]
    for proj, spec in ((Proj.GATE, gateup.gate_spec), (Proj.UP, gateup.up_spec),
                       (Proj.DOWN, down.spec)):
        assert spec.thr.data_ptr() == state.device_table(proj).data_ptr(), proj
        assert spec.ctr.data_ptr() == state.gpu_counter(proj).data_ptr(), proj


@cuda_required
def test_controller_holds_budget_under_activation_drift_in_cuda_graph():
    """Shrinking activations push a fixed threshold's sparsity far above the budget; replaying
    mask kernel + controller step from one CUDA graph pulls it back to the budget.

    The graph bakes pointers, not table values, so this fails if the step does not rewrite the
    table the kernel reads in place, or the kernel does not count into the controller.
    """
    from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed_sparse
    from sglang.srt.layers.moe.prism.tiers import SparseSpec

    W, STEPS = 8, 640
    g = torch.Generator().manual_seed(3)
    a = (torch.rand(E * K, generator=g) + 0.5).cuda()
    # Calibrate the fixed threshold at the budget for unit-scale activations, over all experts.
    x0 = torch.randn(1024, 1, K // 2, 2, generator=g)
    en = (x0 ** 2 * a.cpu().view(1, E, K // 2, 2)).sum(-1)
    t0 = math.sqrt(float(en.flatten().quantile(P)))
    final_scale = 0.35
    d_fixed = float((en * final_scale ** 2 < t0 ** 2).float().mean())
    assert d_fixed > P + 0.25, "test setup must actually drift"

    ctrl = ThresholdController(make_plan(window=W, mu=1.0))
    state = ctrl.register_layer(0, {p: torch.full((E, NG), t0) for p in PROJ_ORDER},
                                torch.device("cuda"))
    spec = SparseSpec(a=a, c=torch.zeros(E * K // 2, device="cuda"),
                      thr=state.device_table(Proj.GATE), p=P, lam=0.0, pmax=0.9, grid=GRID,
                      ng=NG, renorm_it=0, ctr=state.gpu_counter(Proj.GATE))
    w = (torch.randn(E * K, 64, generator=g) * 0.1).to(torch.bfloat16).cuda()
    row_off = (torch.arange(E + 1, dtype=torch.int32) * K).cuda()
    kidx = torch.arange(K, dtype=torch.int32).repeat(E).to(torch.uint16).cuda()
    x = torch.zeros(1, K, dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros(1, TOPK, dtype=torch.int32, device="cuda")
    topk_w = torch.full((1, TOPK), 0.5, device="cuda")
    out = torch.zeros(1, TOPK, 64, dtype=torch.bfloat16, device="cuda")

    def step():
        gemv_worklist_indexed_sparse(x, ids, topk_w, w, row_off, kidx, out, spec, 0, False,
                                     torch.cuda.current_stream())
        ctrl.step(0)

    step()  # compile outside the capture, then forget the warmup step
    torch.cuda.synchronize()
    for t in (state.gpu_ctr, state.snap, state.tokens):
        t.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    for t in range(STEPS):
        scale = 1.0 if t < 80 else max(final_scale, 1.0 - (1.0 - final_scale) * (t - 80) / 240)
        x.copy_((torch.randn(1, K, generator=g) * scale).to(torch.bfloat16))
        ids.copy_(torch.randperm(E, generator=g)[:TOPK].view(1, TOPK))
        graph.replay()
    torch.cuda.synchronize()
    report = ctrl.report()[0]
    assert report["windows_done"] == STEPS // W
    tail = [wdw["gate"]["sparsity"] for wdw in report["windows"][-12:]]
    assert sum(tail) / len(tail) == pytest.approx(P, abs=0.05)
    assert state.alpha[0].item() == pytest.approx(final_scale, rel=0.25)
