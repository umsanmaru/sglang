"""sparse plan + expert별 비균일 K 기하 (컴파일 plan NaN 조사, 2단계).

1단계(test_nonuniform_geometry.py)에서 **dense** 비균일 기하는 torch_bmm/kt_amx도
gemv_worklist/kt_tile_k2도 레퍼런스와 일치했다. 남은 면이 마스킹 경로다.

레퍼런스는 mask_stats와 같은 식(계약 ①)을 full-K에서 계산한다:
    s      = slot_sparsity(topk_w, p, lam, pmax, renorm_it)[j]
    thr    = table[e, lrint(s / grid)]
    e_j    = a[2j]x0² + a[2j+1]x1² + 2c[j]x0x1,  a = wn²
    keep_j = e_j >= thr²           (hot 페어는 항상 keep — SPARSE_TIERS가 warm만)
페어 단위 마스크라 티어가 어디에 있든 keep 판정은 같다 — 달라지는 것은
"마스킹되느냐(warm/cold)"뿐이다.
"""

import pytest
import torch

pytest.importorskip("kt_kernel", reason="kt_kernel required")
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from sglang.srt.layers.moe.prism.calib import CalibTables
from sglang.srt.layers.moe.prism.mask_stats import slot_sparsity
from sglang.srt.layers.moe.prism.plan import (
    PAIR_GROUP,
    CalibRef,
    Proj,
    SparsitySpec,
    Tier,
    parse_plan,
    validate_static,
)

from test_executor import DIMS, MAX_TOKENS, make_inputs, make_weights, rel_diff, run
from test_nonuniform_geometry import _entry, _contig, _interleave, _HOT_GU, _HOT_DN, _PAT, _WARM

H = DIMS["hidden_size"]
I = DIMS["intermediate_size"]
E = DIMS["num_experts"]
NG, GRID, PMAX, RENORM = 201, 0.005, 0.9, 3
P, LAM = 0.5, 4.0


def make_plan(kind, thr_path, *, gpu_warm="gemv_worklist", cpu_cold="kt_tile_k2_bf16"):
    overrides = []
    for e in range(E):
        if kind == "uniform":
            gu, dn = _contig(H, 64, _WARM), _contig(I, 32, _WARM)
        elif kind == "nonuniform":
            gu, dn = _contig(H, _HOT_GU[e], _WARM), _contig(I, _HOT_DN[e], _WARM)
        elif kind == "nonuniform_nowarm":
            gu, dn = _contig(H, _HOT_GU[e], 0), _contig(I, _HOT_DN[e], 0)
        elif kind == "interleave":
            gu, dn = _interleave(H, 32, _PAT[e]), _interleave(I, 32, _PAT[e])
        elif kind == "uniform_nohot":     # hot 없음 = 손 plan의 warm/cold 판
            gu, dn = _contig(H, 0, _WARM), _contig(I, 0, _WARM)
        elif kind == "nonuniform_nohot":  # hot 없이 warm/cold 경계만 expert별로 다름
            gu = _contig(H, 0, 32 * (e % 5) + 32)
            dn = _contig(I, 0, 32 * (e % 3) + 32)
        else:
            raise ValueError(kind)
        ent_gu, ent_dn = _entry(list(gu), I), _entry(list(dn), H)
        for ent in (ent_gu, ent_dn):
            ent["p"], ent["lambda"] = P, LAM
        overrides.append({"layer": 0, "expert": e, "gate": dict(ent_gu),
                          "up": dict(ent_gu), "down": ent_dn})
    raw = {
        "schema_version": 2,
        "model_id": "test/tiny",
        "dims": dict(DIMS),
        "kernels": {"gpu_warm": gpu_warm, "cpu_cold": cpu_cold},
        "overrides": overrides,
        "sparsity": {
            "score": "k2wl2",
            "calib": {"path": str(thr_path), "sha256": "a" * 64},
            "pmax": PMAX, "grid": GRID, "ng": NG, "renorm_it": RENORM,
        },
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def make_calib(tmp_path, thr_fill, seed=5):
    """thr는 grid 전체에 같은 값을 채운다 — s 조회를 상수로 만들어 기하만 본다."""
    torch.manual_seed(seed)
    blob = {
        "wn_g": torch.rand(1, E, H) + 0.5, "wn_u": torch.rand(1, E, H) + 0.5,
        "wn_d": torch.rand(1, E, I) + 0.5,
        "cg": torch.randn(1, E, H // 2) * 0.1,
        "cu": torch.randn(1, E, H // 2) * 0.1,
        "cd": torch.randn(1, E, I // 2) * 0.1,
    }
    fill = {p: thr_fill for p in Proj} if isinstance(thr_fill, (int, float)) else dict(thr_fill)
    for key, proj in (("tg2l", Proj.GATE), ("tu2l", Proj.UP), ("td2l", Proj.DOWN)):
        blob[key] = torch.full((1, E, NG), float(fill[proj]))
    path = tmp_path / "calib.pt"
    torch.save(blob, path)
    spec = SparsitySpec(score="k2wl2", calib=CalibRef(path=str(path), sha256="a" * 64),
                        pmax=PMAX, grid=GRID, ng=NG, renorm_it=RENORM)
    return CalibTables.load(spec, verify_digest=False), blob, path


def build_exec(plan, w13, w2, calib):
    """test_executor.build_executor의 calib 판 (worklist는 PrismExecutor가 plan에서 resolve)."""
    from sglang.srt.layers.moe.prism.cold_backend import KtColdBackend
    from sglang.srt.layers.moe.prism.executor import PrismExecutor
    from sglang.srt.layers.moe.prism.numa import numa_node_count
    from sglang.srt.layers.moe.prism.resources import ExecutionResources, ResourceSpec
    from sglang.srt.layers.moe.prism.weights import prepare_layer_weights

    dev = torch.device("cuda")
    prepared = prepare_layer_weights(0, w13, w2, plan, calib=calib, device=dev)
    has_cold = any(
        plan.expert(0, e).proj(p).has_tier(Tier.COLD)
        for e in range(E) for p in Proj
    )
    cold = None
    if has_cold:
        cold = KtColdBackend(plan, max_tokens=MAX_TOKENS,
                             num_numa_nodes=numa_node_count())
        cold.load_layer(0, prepared.cold, prepared.thr)
    res = ExecutionResources(ResourceSpec.from_plan(plan, max_tokens=MAX_TOKENS, device=dev))
    ex = PrismExecutor(plan, res, cold)
    ex.register_layer(0, prepared)
    return ex


_WN = {Proj.GATE: "wn_g", Proj.UP: "wn_u", Proj.DOWN: "wn_d"}
_DOT = {Proj.GATE: "cg", Proj.UP: "cu", Proj.DOWN: "cd"}


def _pair_energy(vec, proj, e, blob):
    """e_j = a[2j]x0² + a[2j+1]x1² + 2c[j]x0x1 (full-K, fp32)."""
    a = (blob[_WN[proj]][0, e].float() ** 2)
    c = blob[_DOT[proj]][0, e].float()
    x = vec.float()
    x0, x1 = x[0::2], x[1::2]
    return a[0::2] * x0 * x0 + a[1::2] * x1 * x1 + 2.0 * c * x0 * x1


def _masked_pairs(plan, proj, e):
    """마스킹되는 페어(warm/cold)의 bool [K/2] — hot 페어는 False(dense)."""
    K = DIMS["hidden_size"] if proj is not Proj.DOWN else DIMS["intermediate_size"]
    m = torch.zeros(K // PAIR_GROUP, dtype=torch.bool)
    for b in plan.expert(0, e).proj(proj).bands:
        if b.tier in (Tier.WARM, Tier.COLD):
            m[b.start // PAIR_GROUP : b.end // PAIR_GROUP] = True
    return m


def masked_reference(x, ids, w, w13, w2, plan, blob):
    gate_w, up_w = w13[:, :I, :].float(), w13[:, I:, :].float()
    down_w = w2.float()
    out = torch.zeros(x.shape[0], H)
    xf = x.float()
    for m in range(x.shape[0]):
        s = slot_sparsity(w[m], P, LAM, PMAX, RENORM)
        for j in range(ids.shape[1]):
            e = int(ids[m, j])
            gi = int(round(float(s[j]) / GRID))
            gi = max(0, min(NG - 1, gi))

            def apply(vec, proj):
                keep = _pair_energy(vec, proj, e, blob) >= float(
                    blob[{"gate": "tg2l", "up": "tu2l", "down": "td2l"}[proj.value]][0, e, gi]
                ) ** 2
                keep = keep | ~_masked_pairs(plan, proj, e)
                return vec.float() * keep.repeat_interleave(PAIR_GROUP).float()

            g = apply(xf[m], Proj.GATE) @ gate_w[e].t()
            u = apply(xf[m], Proj.UP) @ up_w[e].t()
            act = torch.nn.functional.silu(g) * u
            out[m] += float(w[m, j]) * (apply(act, Proj.DOWN) @ down_w[e].t())
    return out


def _pick_thr(vals, q=0.5, window=0.15):
    """분포에서 **가장 넓은 간극**을 골라 그 중점을 thr²로 쓴다.

    임계값이 실제 에너지 값 근처에 놓이면 그 페어가 사실상 동점이 되어 커널과
    레퍼런스의 fp 차이(bf16 eps ≈ 0.4%) 하나로 keep이 뒤집히고, 페어 하나가
    뒤집히면 이 소형 shape에서 출력이 10%씩 흔들린다 — 결함이 아니라 계측기의
    문제다. q 근방 window 안에서 상대 간극이 최대인 자리를 골라 여유를 만든다.
    """
    srt = torch.cat(vals).clamp_min(0).sort().values
    n = srt.numel()
    lo = max(int((q - window) * (n - 1)), 0)
    hi = min(int((q + window) * (n - 1)), n - 2)
    seg = srt[lo : hi + 2]
    gaps = (seg[1:] - seg[:-1]) / seg[:-1].clamp_min(1e-20)
    i = lo + int(torch.argmax(gaps))
    return float(0.5 * (srt[i] + srt[i + 1])) ** 0.5, float(gaps.max())


def _probe_thr(x, ids, w13, w2, blob, plan, q=0.5):
    """proj별 threshold. down은 **마스킹된 gate/up으로 만든 act**의 분포에서 고른다
    (커널이 보는 것과 같은 act여야 간극 여유가 실제로 여유가 된다)."""
    gate_w, up_w = w13[:, :I, :].float(), w13[:, I:, :].float()
    xf = x.float()
    vals = {p: [] for p in Proj}
    for m in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[m, j])
            for proj in (Proj.GATE, Proj.UP):
                vals[proj].append(_pair_energy(xf[m], proj, e, blob))
    thr = {}
    margins = {}
    for proj in (Proj.GATE, Proj.UP):
        thr[proj], margins[proj] = _pick_thr(vals[proj], q)

    def apply(vec, proj, e):
        keep = _pair_energy(vec, proj, e, blob) >= thr[proj] ** 2
        keep = keep | ~_masked_pairs(plan, proj, e)
        return vec.float() * keep.repeat_interleave(PAIR_GROUP).float()

    for m in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[m, j])
            g = apply(xf[m], Proj.GATE, e) @ gate_w[e].t()
            u = apply(xf[m], Proj.UP, e) @ up_w[e].t()
            act = torch.nn.functional.silu(g) * u
            vals[Proj.DOWN].append(_pair_energy(act, Proj.DOWN, e, blob))
    thr[Proj.DOWN], margins[Proj.DOWN] = _pick_thr(vals[Proj.DOWN], q)
    return thr, margins


@cuda_required
@pytest.mark.parametrize("qlen", [1])
@pytest.mark.parametrize("kind", [
    "uniform", "uniform_nohot", "nonuniform_nohot",
    "nonuniform", "nonuniform_nowarm", "interleave",
])
def test_sparse_nonuniform_matches_reference(kind, qlen, tmp_path):
    w13, w2 = make_weights()
    x, ids, w = make_inputs(qlen, seed=10 + qlen)
    _, blob0, path0 = make_calib(tmp_path, 0.0)
    plan0 = make_plan(kind, path0)
    thr, margins = _probe_thr(x, ids, w13, w2, blob0, plan0)
    # 간극이 너무 좁으면 이 대조는 커널이 아니라 반올림을 재게 된다 — 그때는
    # 조용히 통과/실패하지 말고 계측기 문제임을 알린다.
    thin = {p.value: round(g, 4) for p, g in margins.items() if g < 0.02}
    if thin:
        pytest.skip(f"threshold 간극이 bf16 eps 수준이라 대조 불가: {thin}")
    calib, blob, path = make_calib(tmp_path, thr)
    plan = make_plan(kind, path)
    ex = build_exec(plan, w13, w2, calib)
    out = run(ex, x, ids, w)
    ref = masked_reference(x, ids, w, w13, w2, plan, blob)
    d = rel_diff(out, ref)
    finite = bool(torch.isfinite(out).all())
    print(f"[sparse] {kind} qlen={qlen}: rel diff = {d:.6f} finite={finite}")
    assert finite, f"{kind}: non-finite output"
    assert d < 0.05, f"{kind}: diff {d:.6f}"
