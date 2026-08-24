"""입력기반 sparsity 테스트 — threshold 조회(GPU)와 cold 마스킹(kt).

warm/hot은 dense로 계산한다 (2026-08-24 결정). 그래서 이 파일이 검증하는
것은 두 가지다:

1. 예산 → threshold 변환 (`slot_thr`) — 루프 레퍼런스와 대조.
2. cold 밴드만 마스킹된 결과 — full-K가 아니라 **cold 구간만** 마스킹한
   레퍼런스와 대조한다. warm이 dense라는 것이 여기서 검증된다.
"""

import math

import pytest
import torch

from sglang.srt.layers.moe.prism.plan import PAIR_GROUP, Proj
from sglang.srt.layers.moe.prism.sparsity import (
    LayerSparsity,
    normalized_router_weights,
)

NG, GRID, PMAX, RENORM = 201, 0.005, 0.9, 3


# ===========================================================================
# Part A — threshold 변환 (CPU)
# ===========================================================================


def test_slot_thr_matches_loop_reference():
    """s_mat + 격자 조회를 요소별 루프로 재계산해 대조."""
    E, k, M, p, lam = 6, 4, 3, 0.4, 3.0
    torch.manual_seed(7)
    table = torch.rand(E, NG)
    sp = LayerSparsity(
        {pr: table for pr in Proj},
        {pr: (p, lam) for pr in Proj},
        pmax=PMAX, grid=GRID, ng=NG, renorm_it=RENORM,
    )
    ids = torch.stack([torch.randperm(E)[:k] for _ in range(M)])
    twn = normalized_router_weights(torch.rand(M, k))
    got = sp.slot_thr(Proj.GATE, ids, twn)

    ref = torch.empty(M, k)
    for m in range(M):
        row = [float(v) for v in twn[m]]
        gbar = sum(row) / k
        s = [min(max(p - lam * (v - gbar), 0.0), PMAX) for v in row]
        for _ in range(RENORM):
            mean = max(sum(s) / k, 1e-6)
            s = [min(max(v * (p / mean), 0.0), PMAX) for v in s]
        for j in range(k):
            idx = min(max(int(round(s[j] / GRID)), 0), NG - 1)
            ref[m, j] = table[int(ids[m, j]), idx]
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_slot_thr_zero_budget_hits_grid_origin():
    """p=0, lam=0이면 s=0 → 격자 0번 — calib이 그 자리에 0을 둔다."""
    E, k = 4, 2
    table = torch.rand(E, NG)
    table[:, 0] = 0.0
    sp = LayerSparsity(
        {pr: table for pr in Proj}, {pr: (0.0, 0.0) for pr in Proj},
        pmax=PMAX, grid=GRID, ng=NG, renorm_it=RENORM,
    )
    ids = torch.tensor([[0, 1]])
    twn = normalized_router_weights(torch.rand(1, k))
    assert torch.equal(sp.slot_thr(Proj.DOWN, ids, twn), torch.zeros(1, k))


def test_normalized_router_weights_is_idempotent():
    w = torch.rand(4, 8)
    once = normalized_router_weights(w)
    assert torch.allclose(once.sum(-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(normalized_router_weights(once), once, atol=1e-7)


# ===========================================================================
# Part B — cold 마스킹 end-to-end (CUDA; mixed plan은 kt 필요)
# ===========================================================================

from sglang.srt.layers.moe.prism.calib import CalibTables  # noqa: E402
from sglang.srt.layers.moe.prism.executor import PrismExecutor  # noqa: E402
from sglang.srt.layers.moe.prism.kernels import resolve_gpu_kernel  # noqa: E402
from sglang.srt.layers.moe.prism.plan import (  # noqa: E402
    CalibRef,
    SparsitySpec,
    Tier,
    parse_plan,
    validate_static,
)
from sglang.srt.layers.moe.prism.resources import (  # noqa: E402
    ExecutionResources,
    ResourceSpec,
)
from sglang.srt.layers.moe.prism.stagers import PerSlotCopyStager  # noqa: E402
from sglang.srt.layers.moe.prism.weights import prepare_layer_weights  # noqa: E402

try:
    import kt_kernel  # noqa: F401

    HAS_KT = True
except Exception:  # pragma: no cover - 환경 의존
    HAS_KT = False

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)
kt_required = pytest.mark.skipif(not HAS_KT, reason="kt_kernel required")

E, H, I, K = 8, 256, 128, 2
EDIMS = {
    "hidden_size": H, "intermediate_size": I, "num_layers": 1,
    "num_experts": E, "top_k": K, "dtype": "bfloat16",
}
MAX_TOKENS = 16
P, LAM = 0.5, 4.0
WARM_ROWS = 64  # mixed plan의 warm 밴드 크기 (gate/up·down 공통)


def _proj_entry(bands, N, sparse):
    has_cold = any(t == "cold" for _, _, t in bands)
    shards = []
    if has_cold:
        half = (N // 2 // 32) * 32
        shards = [[0, 0, half], [1, half, N]]
    entry = {"bands": bands, "cold_shards": shards}
    if sparse:
        entry["p"], entry["lambda"] = P, LAM
    return entry


def make_exec_plan(kind, *, sparse=True):
    if kind == "mixed":
        gate_up = _proj_entry([[0, WARM_ROWS, "warm"], [WARM_ROWS, H, "cold"]], I, sparse)
        down = _proj_entry([[0, WARM_ROWS, "warm"], [WARM_ROWS, I, "cold"]], H, sparse)
    else:  # all_warm — cold가 없으므로 threshold가 아무 영향을 주지 못해야 한다
        gate_up = _proj_entry([[0, H, "warm"]], I, sparse)
        down = _proj_entry([[0, I, "warm"]], H, sparse)
    raw = {
        "schema_version": 2 if sparse else 1,
        "model_id": "test/tiny",
        "dims": dict(EDIMS),
        "kernels": {"gpu_warm": "torch_bmm", "cpu_cold": "kt_amx_bf16"},
        "default": {"gate": dict(gate_up), "up": dict(gate_up), "down": down},
    }
    if sparse:
        raw["sparsity"] = {
            "score": "k2wl2",
            "calib": {"path": "unused", "sha256": "a" * 64},
            "pmax": PMAX, "grid": GRID, "ng": NG, "renorm_it": RENORM,
        }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def make_exec_calib(tmp_path, thr_fill, seed=5):
    torch.manual_seed(seed)
    blob = {
        "wn_g": torch.rand(1, E, H) + 0.5, "wn_u": torch.rand(1, E, H) + 0.5,
        "wn_d": torch.rand(1, E, I) + 0.5,
        "cg": torch.randn(1, E, H // 2) * 0.1,
        "cu": torch.randn(1, E, H // 2) * 0.1,
        "cd": torch.randn(1, E, I // 2) * 0.1,
    }
    for key in ("tg2l", "tu2l", "td2l"):
        blob[key] = torch.full((1, E, NG), float(thr_fill))
    path = tmp_path / f"calib_{thr_fill}.pt"
    torch.save(blob, path)
    spec = SparsitySpec(
        score="k2wl2", calib=CalibRef(path=str(path), sha256="a" * 64),
        pmax=PMAX, grid=GRID, ng=NG, renorm_it=RENORM,
    )
    return CalibTables.load(spec, verify_digest=False), blob


def make_exec_weights(seed=0):
    torch.manual_seed(seed)
    w13 = (torch.randn(E, 2 * I, H) / 10.0).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I) / 10.0).to(torch.bfloat16)
    return w13, w2


def build_exec(plan, w13, w2, calib=None):
    prepared = prepare_layer_weights(0, w13, w2, plan, calib=calib)
    ep = plan.expert(0, 0)
    has_cold = any(ep.proj(p).has_tier(Tier.COLD) for p in Proj)
    cold = None
    if has_cold:
        from sglang.srt.layers.moe.prism.cold_backend import KtColdBackend
        from sglang.srt.layers.moe.prism.numa import numa_node_count

        cold = KtColdBackend(plan, max_tokens=MAX_TOKENS,
                             num_numa_nodes=numa_node_count())
        cold.load_layer(0, prepared.cold)
    spec = ResourceSpec.from_plan(plan, max_tokens=MAX_TOKENS,
                                  device=torch.device("cuda"))
    ex = PrismExecutor(
        plan, ExecutionResources(spec), cold,
        resolve_gpu_kernel(plan.kernels.gpu_warm),
        stager=PerSlotCopyStager(),
    )
    ex.register_layer(0, prepared)
    return ex


def make_exec_inputs(qlen, seed):
    torch.manual_seed(seed)
    x = (torch.randn(qlen, H) / 10.0).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E)[:K] for _ in range(qlen)])
    w = torch.rand(qlen, K, dtype=torch.float32)
    return x, ids, w


def run_exec(ex, x, ids, w):
    return ex.run_layer(0, x.cuda(), ids.cuda(), w.cuda()).cpu()


def rel_diff(a, b):
    return (
        torch.mean(torch.abs(a.float() - b.float()))
        / (torch.mean(torch.abs(b.float())) + 1e-8)
    ).item()


_WN_KEY = {Proj.GATE: "wn_g", Proj.UP: "wn_u", Proj.DOWN: "wn_d"}
_DOT_KEY = {Proj.GATE: "cg", Proj.UP: "cu", Proj.DOWN: "cd"}


def ref_pair_importance(vec, a_sq, c):
    """루프 레퍼런스: imp_j = sqrt(max(a0x0² + a1x1² + 2c·x0x1, 0))."""
    out = torch.empty(vec.shape[0] // PAIR_GROUP)
    for j in range(out.numel()):
        x0, x1 = float(vec[2 * j]), float(vec[2 * j + 1])
        v = (float(a_sq[2 * j]) * x0 * x0 + float(a_sq[2 * j + 1]) * x1 * x1
             + 2.0 * float(c[j]) * x0 * x1)
        out[j] = math.sqrt(max(v, 0.0))
    return out


def _cold_span(ep, proj):
    for b in ep.proj(proj).bands:
        if b.tier is Tier.COLD:
            return b.start, b.end
    return None


def _cold_imp(vec, proj, e, span, blob):
    s, en = span
    a = blob[_WN_KEY[proj]][0, e][s:en] ** 2
    c = blob[_DOT_KEY[proj]][0, e][s // 2 : en // 2]
    return ref_pair_importance(vec[s:en], a, c)


def probe_thresholds(x, ids, w13, w2, plan, blob, q=0.5):
    """**cold 구간**의 실제 imp 분포 분위수 → proj별 threshold.

    무작위 벡터로 잡으면 스케일이 어긋나 전량 통과/차단이 되어 대조가
    0 vs 0의 무의미한 비교가 된다 — nnz assert로도 이중 방어한다.
    """
    gate_w, up_w = w13[:, :I, :].float(), w13[:, I:, :].float()
    ep = plan.expert(0, 0)
    xf = x.float()
    vals = {p: [] for p in Proj}
    for m in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[m, j])
            for proj in (Proj.GATE, Proj.UP):
                span = _cold_span(ep, proj)
                if span:
                    vals[proj].append(_cold_imp(xf[m], proj, e, span, blob))
            act = torch.nn.functional.silu(xf[m] @ gate_w[e].t()) * (
                xf[m] @ up_w[e].t())
            span = _cold_span(ep, Proj.DOWN)
            if span:
                vals[Proj.DOWN].append(_cold_imp(act, Proj.DOWN, e, span, blob))
    return {p: float(torch.cat(v).quantile(q)) for p, v in vals.items() if v}


def cold_masked_reference(x, ids, w, w13, w2, plan, blob, thr):
    """**cold 밴드만** 마스킹한 fp32 레퍼런스 — warm이 dense임을 검증한다.

    warm 구간까지 마스킹하면 이 대조가 깨진다. 즉 이 테스트가 "warm dense"
    결정이 실제 실행에 반영됐는지의 검출기다.
    """
    gate_w, up_w = w13[:, :I, :].float(), w13[:, I:, :].float()
    down_w = w2.float()
    ep = plan.expert(0, 0)

    def apply_mask(vec, proj, e, t):
        span = _cold_span(ep, proj)
        if span is None:
            return vec
        s, en = span
        keep = (_cold_imp(vec, proj, e, span, blob) >= t).repeat_interleave(PAIR_GROUP)
        out = vec.clone()
        out[s:en] = vec[s:en] * keep.to(vec.dtype)
        return out

    out = torch.zeros(x.shape[0], H)
    xf = x.float()
    for m in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[m, j])
            xg = apply_mask(xf[m], Proj.GATE, e, float(thr[Proj.GATE][m, j]))
            xu = apply_mask(xf[m], Proj.UP, e, float(thr[Proj.UP][m, j]))
            act = torch.nn.functional.silu(xg @ gate_w[e].t()) * (xu @ up_w[e].t())
            act = apply_mask(act, Proj.DOWN, e, float(thr[Proj.DOWN][m, j]))
            out[m] += float(w[m, j]) * (act @ down_w[e].t())
    return out


@kt_required
@cuda_required
def test_cold_backend_passes_wn_squared(tmp_path):
    """kt에 넘어간 테이블이 wn²(a)인지 — wn을 그대로 넘기면 마스크가 조용히
    갈린다. 두 구현이 같은 정의를 쓰는지 직접 확인하는 유일한 지점이다."""
    from sglang.srt.layers.moe.prism.cold_backend import KtColdBackend
    from sglang.srt.layers.moe.prism.numa import numa_node_count

    w13, w2 = make_exec_weights()
    calib, blob = make_exec_calib(tmp_path, 0.0)
    plan = make_exec_plan("mixed")
    prepared = prepare_layer_weights(0, w13, w2, plan, calib=calib)
    cold = KtColdBackend(plan, max_tokens=MAX_TOKENS,
                         num_numa_nodes=numa_node_count())
    cold.load_layer(0, prepared.cold)

    tables = cold._wrappers[0]._sparsity_tables
    assert set(tables) == {
        "gate_wn_sq", "gate_pair_dot", "up_wn_sq", "up_pair_dot",
        "down_wn_sq", "down_pair_dot",
    }
    wn_cold = blob["wn_g"][0][:, WARM_ROWS:H]
    assert torch.allclose(tables["gate_wn_sq"], wn_cold * wn_cold, atol=0)
    assert torch.equal(tables["down_pair_dot"],
                       blob["cd"][0][:, WARM_ROWS // 2 :])


@kt_required
@cuda_required
def test_zero_threshold_matches_dense_path(tmp_path):
    """thr=0이면 cold도 전량 통과 — dense plan과 사실상 동일해야 한다."""
    w13, w2 = make_exec_weights()
    x, ids, w = make_exec_inputs(1, seed=31)
    dense = run_exec(build_exec(make_exec_plan("mixed", sparse=False), w13, w2),
                     x, ids, w)
    calib, _ = make_exec_calib(tmp_path, 0.0)
    masked = run_exec(build_exec(make_exec_plan("mixed"), w13, w2, calib), x, ids, w)
    d = rel_diff(masked, dense)
    print(f"thr=0 vs dense: rel diff = {d:.3e}")
    assert d < 1e-5


@cuda_required
def test_all_warm_plan_ignores_threshold(tmp_path):
    """cold 밴드가 없으면 threshold가 아무 영향을 주지 못한다 (warm은 dense).

    warm 마스킹이 남아 있으면 thr=∞에서 출력이 0이 되어 이 테스트가 깨진다.
    """
    w13, w2 = make_exec_weights()
    x, ids, w = make_exec_inputs(1, seed=32)
    dense = run_exec(build_exec(make_exec_plan("all_warm", sparse=False), w13, w2),
                     x, ids, w)
    calib, _ = make_exec_calib(tmp_path, 1e9)
    masked = run_exec(build_exec(make_exec_plan("all_warm"), w13, w2, calib),
                      x, ids, w)
    assert torch.equal(masked, dense)


@kt_required
@cuda_required
def test_prefill_stays_dense(tmp_path):
    """M>1은 마스킹하지 않는다 (prefill-dense/decode-sparse) — thr=∞여도 dense."""
    w13, w2 = make_exec_weights()
    x, ids, w = make_exec_inputs(16, seed=33)
    dense = run_exec(build_exec(make_exec_plan("mixed", sparse=False), w13, w2),
                     x, ids, w)
    calib, _ = make_exec_calib(tmp_path, 1e9)
    masked = run_exec(build_exec(make_exec_plan("mixed"), w13, w2, calib), x, ids, w)
    assert rel_diff(masked, dense) < 1e-6


@kt_required
@cuda_required
def test_cold_masked_matches_reference(tmp_path):
    """중간 threshold에서 **cold 구간만** 마스킹한 레퍼런스와 일치.

    warm까지 마스킹되면 이 대조가 깨진다 — "warm dense"의 검출기다.
    """
    w13, w2 = make_exec_weights(seed=2)
    x, ids, w = make_exec_inputs(1, seed=34)
    calib, blob = make_exec_calib(tmp_path, 0.0)
    plan = make_exec_plan("mixed")
    ex = build_exec(plan, w13, w2, calib)

    picked = probe_thresholds(x, ids, w13, w2, plan, blob, q=0.5)
    sp = ex._sparsity[0]
    for proj, value in picked.items():
        sp._thr[proj] = torch.full_like(sp._thr[proj], value)
    twn = normalized_router_weights(w.cuda())
    thr = {p: sp.slot_thr(p, ids.cuda(), twn).cpu() for p in Proj}

    out = run_exec(ex, x, ids, w)
    ref = cold_masked_reference(x, ids, w, w13, w2, plan, blob, thr)
    d = rel_diff(out, ref)
    print(f"cold-only masked: rel diff = {d:.6f}")
    assert d < 0.03
