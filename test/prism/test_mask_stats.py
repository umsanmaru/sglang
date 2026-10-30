"""mask_stats.MaskStats — decode 마스크 재계산기의 CPU 단위 테스트 (kt/CUDA 불필요).

레퍼런스는 이 파일의 스칼라 루프다: `moe_base.hpp slot_sparsity/thr_of`와
`pair_mask.hpp`의 식을 한 슬롯·한 페어씩 그대로 옮겼다. 벡터화된 MaskStats가 같은
keep 수를 (layer, proj, tier)별로 내야 한다.
"""
from __future__ import annotations

import math

import torch

from sglang.srt.layers.moe.prism.calib import CalibTables
from sglang.srt.layers.moe.prism.mask_stats import MaskStats, slot_sparsity
from sglang.srt.layers.moe.prism.plan import (
    CalibRef, Proj, SparsitySpec, Tier, parse_plan, validate_static,
)

E, H, I, TOPK, L = 6, 64, 32, 3, 2
NG, GRID, PMAX, RENORM = 201, 0.005, 0.9, 3
P, LAM = 0.5, 1.2


def make_plan():
    def entry(bands, N):
        return {"bands": bands, "cold_shards": [[0, 0, N]], "p": P, "lambda": LAM}
    layer = {
        "gate": entry([[0, 8, "hot"], [8, 24, "warm"], [24, H, "cold"]], I),
        "up":   entry([[0, 8, "hot"], [8, 24, "warm"], [24, H, "cold"]], I),
        "down": entry([[0, 8, "warm"], [8, I, "cold"]], H),
    }
    raw = {
        "schema_version": 2, "model_id": "t",
        "dims": {"hidden_size": H, "intermediate_size": I, "num_layers": L,
                 "num_experts": E, "top_k": TOPK, "dtype": "bfloat16"},
        "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_amx_bf16"},
        "default": layer,
        "sparsity": {"score": "k2wl2", "calib": {"path": "unused", "sha256": "a" * 64},
                     "pmax": PMAX, "grid": GRID, "ng": NG, "renorm_it": RENORM},
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def make_calib(tmp_path, seed=3):
    g = torch.Generator().manual_seed(seed)
    blob = {
        "wn_g": torch.rand(L, E, H, generator=g) + 0.5,
        "wn_u": torch.rand(L, E, H, generator=g) + 0.5,
        "wn_d": torch.rand(L, E, I, generator=g) + 0.5,
        "cg": (torch.rand(L, E, H // 2, generator=g) - 0.5) * 0.2,
        "cu": (torch.rand(L, E, H // 2, generator=g) - 0.5) * 0.2,
        "cd": (torch.rand(L, E, I // 2, generator=g) - 0.5) * 0.2,
    }
    # 단조 곡선: |x|~N(0,1)·wn~1 이면 imp 스케일 ~1 → 0..3 사이로 격자
    ramp = torch.linspace(0.0, 3.0, NG)
    for key in ("tg2l", "tu2l", "td2l"):
        blob[key] = ramp.expand(L, E, NG).clone() * (torch.rand(L, E, 1, generator=g) + 0.5)
    path = tmp_path / "calib.pt"
    torch.save(blob, path)
    spec = SparsitySpec(score="k2wl2", calib=CalibRef(path=str(path), sha256="a" * 64),
                        pmax=PMAX, grid=GRID, ng=NG, renorm_it=RENORM)
    return CalibTables.load(spec, verify_digest=False), blob


# ── 스칼라 레퍼런스 (C++ 그대로) ────────────────────────────────────────────
def ref_slot_sparsity(w, p, lam, pmax, renorm_it):
    k = len(w)
    tot = sum(w)
    inv = 1.0 / (tot if tot > 1e-9 else 1e-9)
    gbar = sum(x * inv for x in w) / k
    clip = lambda v: 0.0 if v < 0 else (pmax if v > pmax else v)
    s = [clip(p - lam * (w[i] * inv - gbar)) for i in range(k)]
    for _ in range(renorm_it):
        mean = max(sum(s) / k, 1e-6)
        s = [clip(v * (p / mean)) for v in s]
    return s


def ref_keep(vec, a_sq, c, thr):
    out = []
    for j in range(len(vec) // 2):
        x0, x1 = float(vec[2 * j]), float(vec[2 * j + 1])
        e = float(a_sq[2 * j]) * x0 * x0 + float(a_sq[2 * j + 1]) * x1 * x1 + 2.0 * float(c[j]) * x0 * x1
        out.append(e >= thr * thr)
    return out


def tier_of(plan, l, e, proj, row):
    for b in plan.expert(l, e).proj(proj).bands:
        if b.start <= row < b.end:
            return {Tier.HOT: 0, Tier.WARM: 1, Tier.COLD: 2}[b.tier]
    raise AssertionError


def test_slot_sparsity_matches_scalar():
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        w = torch.rand(TOPK, generator=g) * 3
        got = slot_sparsity(w, P, LAM, PMAX, RENORM)
        exp = ref_slot_sparsity(w.tolist(), P, LAM, PMAX, RENORM)
        assert torch.allclose(got, torch.tensor(exp), atol=1e-6), (got, exp)


def test_counts_match_scalar_reference(tmp_path):
    plan = make_plan()
    calib, blob = make_calib(tmp_path)
    st = MaskStats(plan, calib, str(tmp_path / "out.json"), every=1000)
    for l in range(L):
        st.register_layer(l)
    g = torch.Generator().manual_seed(7)
    keep_ref = torch.zeros(L, 3, 3, dtype=torch.int64)
    total_ref = torch.zeros(L, 3, 3, dtype=torch.int64)
    keys = {Proj.GATE: ("wn_g", "cg", "tg2l"), Proj.UP: ("wn_u", "cu", "tu2l"),
            Proj.DOWN: ("wn_d", "cd", "td2l")}
    n_tok = 5
    for _ in range(n_tok):
        hidden = torch.randn(1, H, generator=g).to(torch.bfloat16)
        act = torch.randn(1, TOPK, I, generator=g).to(torch.bfloat16)
        ids = torch.randperm(E, generator=g)[:TOPK].unsqueeze(0)
        w = (torch.rand(1, TOPK, generator=g) + 0.1)
        for l in range(L):
            st.observe_gateup(l, hidden, ids, w)
            st.observe_down(l, act, ids, w)
            s = ref_slot_sparsity(w[0].tolist(), P, LAM, PMAX, RENORM)
            for j in range(TOPK):
                e = int(ids[0, j])
                gi = min(max(int(round(s[j] / GRID)), 0), NG - 1)
                for pi, proj in enumerate(Proj):
                    wn_k, c_k, t_k = keys[proj]
                    vec = hidden[0].float() if proj is not Proj.DOWN else act[0, j].float()
                    thr = float(blob[t_k][l, e, gi])
                    kp = ref_keep(vec, blob[wn_k][l, e] ** 2, blob[c_k][l, e], thr)
                    for pj, k_ in enumerate(kp):
                        t = tier_of(plan, l, e, proj, 2 * pj)
                        total_ref[l, pi, t] += 1
                        keep_ref[l, pi, t] += int(k_)
    assert torch.equal(st.total, total_ref)
    assert torch.equal(st.keep, keep_ref), (st.keep - keep_ref)
    assert st.tokens == n_tok
    rep = st.report()
    # 마스킹 티어(warm+cold) 집계가 표에서 재구성된 값과 일치
    m = rep["total"]["all"]["masked"]
    kp = int(keep_ref[:, :, 1:].sum()); tot = int(total_ref[:, :, 1:].sum())
    assert m["rows"] == tot * 2 and math.isclose(m["keep_frac"], kp / tot)
    assert 0.0 < m["keep_frac"] < 1.0, "레퍼런스가 전량 통과/차단이면 테스트가 무의미"
    st.dump()
    assert (tmp_path / "out.json").exists()


def test_report_reflects_thresholds(tmp_path):
    """thr=0 → 전량 keep, thr=∞ → 전량 drop (마스킹 티어 기준)."""
    plan = make_plan()
    for fill, want in ((0.0, 1.0), (1e9, 0.0)):
        blob = {"wn_g": torch.ones(L, E, H), "wn_u": torch.ones(L, E, H), "wn_d": torch.ones(L, E, I),
                "cg": torch.zeros(L, E, H // 2), "cu": torch.zeros(L, E, H // 2), "cd": torch.zeros(L, E, I // 2)}
        for key in ("tg2l", "tu2l", "td2l"):
            blob[key] = torch.full((L, E, NG), fill)
        path = tmp_path / f"c{fill}.pt"
        torch.save(blob, path)
        spec = SparsitySpec(score="k2wl2", calib=CalibRef(path=str(path), sha256="a" * 64),
                            pmax=PMAX, grid=GRID, ng=NG, renorm_it=RENORM)
        calib = CalibTables.load(spec, verify_digest=False)
        st = MaskStats(plan, calib, str(tmp_path / "o.json"), every=1000)
        st.register_layer(0)
        st.observe_gateup(0, torch.randn(1, H).bfloat16(), torch.arange(TOPK)[None], torch.ones(1, TOPK))
        st.observe_down(0, torch.randn(1, TOPK, I).bfloat16(), torch.arange(TOPK)[None], torch.ones(1, TOPK))
        assert st.report()["total"]["all"]["masked"]["keep_frac"] == want


def test_non_finite_router_weight_uses_thr0_like_kt(tmp_path):
    """w에 NaN이 오면 kt thr_of처럼 idx 0(thr=0 → 전량 keep)으로 처리하고 죽지 않는다."""
    plan = make_plan()
    calib, _ = make_calib(tmp_path)
    st = MaskStats(plan, calib, str(tmp_path / "o.json"), every=1000)
    st.register_layer(0)
    ids = torch.arange(TOPK)[None]
    w = torch.tensor([[float("nan"), 1.0, float("inf")]])
    st.observe_gateup(0, torch.randn(1, H).bfloat16(), ids, w)
    st.observe_down(0, torch.randn(1, TOPK, I).bfloat16(), ids, w)
    assert not st.disabled
    assert st.nan_slots == 3 * TOPK   # 세 proj 모두, 슬롯 전부 (inf 하나가 sum을 오염)
    rep = st.report()
    assert rep["nan_slots"] == 3 * TOPK
    assert rep["total"]["all"]["masked"]["keep_frac"] == 1.0   # thr[...,0] = 0 → 전량 keep
