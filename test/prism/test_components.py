"""구성 요소 합성(`closure.compose_exec`)·handshake 추출·그리드 채점의 계약 — 하드웨어 없이.

측정 자체(components.ComponentProfiler)는 CUDA 게이트 테스트 하나가 작은 점에서 돌려
모양만 본다. 수치의 정합은 bench_components가 본다.
"""

from importlib import import_module

import pytest

from sglang.srt.layers.moe.prism.profile.common import Shape

_P = "sglang.srt.layers.moe.prism.profile."
cl = import_module(_P + "closure")
comp = import_module(_P + "components")

BASE = {"hot": 6.0, "warm": 30.0, "gpu_seq": 37.0, "handshake_gpu": 20.0, "cold_eff": 40.0,
        "d2h_ids": 2.0, "d2h_x": 2.5, "d2h_topk_w": 1.5, "d2h_act": 3.0,
        "h2d_partial": 3.5, "rejoin": 4.0, "cold_host": 30.0, "cold_host_pcie": 36.0}


# ─── compose_exec ─────────────────────────────────────────────────────────
def test_gpu_critical_layer():
    r = cl.compose_exec(BASE, group="down", target="layer")
    # gpu_path = 20 + 36 = 56 > C = 40 → 2 + 56 + 3.5 + 4
    assert r["us"] == pytest.approx(65.5)
    assert r["overlap_detail"]["critical"] == "gpu"
    assert set(r["terms"]) == {"d2h_ids", "overlap", "h2d_partial", "rejoin"}


def test_cpu_critical_combined():
    c = dict(BASE, cold_eff=90.0)
    r = cl.compose_exec(c, group="gateup", target="combined")
    assert r["us"] == pytest.approx(2.0 + 90.0)
    assert r["overlap_detail"]["critical"] == "cpu"
    assert "rejoin" not in r["terms"]


def test_runtime_inputs_per_phase():
    gu = cl.compose_exec(BASE, group="gateup", target="runtime")
    assert set(gu["terms"]) == {"d2h_x", "d2h_ids", "d2h_topk_w", "overlap", "h2d_partial", "rejoin"}
    gu_nomask = cl.compose_exec(BASE, group="gateup", target="runtime", masking=False)
    assert "d2h_topk_w" not in gu_nomask["terms"]
    dn = cl.compose_exec(BASE, group="down", target="runtime")
    # down은 expert_ids를 다시 내리지 않는다 (executor fill_down)
    assert set(dn["terms"]) == {"d2h_act", "overlap", "h2d_partial", "rejoin"}
    assert dn["us"] == pytest.approx(3.0 + 56.0 + 3.5 + 4.0)


def test_contention_slows_only_warm_window():
    c = dict(BASE, cold_eff=60.0)
    r = cl.compose_exec(c, group="down", target="combined", contention=True)
    assert r["overlap_detail"]["contention_scale"] == pytest.approx(1.2)
    # 진행: [0,6) 6, [6,36) 30/1.2 = 25, 남은 29 → 65
    assert r["overlap_detail"]["cold_used"] == pytest.approx(65.0)
    assert r["us"] == pytest.approx(2.0 + 65.0)
    nowarm = dict(c, warm=0.0)
    r2 = cl.compose_exec(nowarm, group="down", target="combined", contention=True)
    assert r2["overlap_detail"]["contention_scale"] == 1.0


def test_cold_under_traffic_cases():
    f = cl.cold_under_traffic
    assert f(50.0, 1.0, start=5, length=20) == 50.0            # 경쟁 없음
    assert f(4.0, 2.0, start=5, length=20) == 4.0              # 구간 전에 끝남
    assert f(10.0, 2.0, start=5, length=20) == pytest.approx(15.0)   # 구간 안에서 끝남
    assert f(30.0, 2.0, start=5, length=20) == pytest.approx(40.0)   # 5 + 20 + (30-5-10)


def test_tail_replaces_h2d_rejoin():
    c = dict(BASE, join_tail=17.0)
    r = cl.compose_exec(c, group="down", target="layer", tail=True)
    assert "join_tail" in r["terms"] and "rejoin" not in r["terms"]
    assert r["us"] == pytest.approx(2.0 + 56.0 + 17.0)


def test_component_values_join_tail():
    comp_entry = {"components": {"cold_graph": {"us": 50.0}, "cold_graph_tail": {"us": 66.5}},
                  "handshake": {"handshake_gpu_us": 40.0, "cold_eff_us": 48.0}}
    v = cl.component_values(comp_entry)
    assert v["join_tail"] == pytest.approx(16.5)
    assert v["handshake_gpu"] == 40.0


def test_gpu_branch_seq():
    r = cl.compose_exec(BASE, group="down", target="combined", gpu_branch="seq")
    assert r["overlap_detail"]["gpu"] == 37.0


def test_terms_sum_to_total():
    for g in ("gateup", "down"):
        for t in cl.EXEC_TARGETS:
            r = cl.compose_exec(BASE, group=g, target=t)
            assert sum(r["terms"].values()) == pytest.approx(r["us"], abs=1e-3)


@pytest.mark.parametrize("kw,msg", [
    ({"group": "up"}, "group"), ({"group": "down", "target": "x"}, "target"),
    ({"group": "down", "gpu_branch": "x"}, "gpu_branch")])
def test_bad_args(kw, msg):
    with pytest.raises(ValueError, match=msg):
        cl.compose_exec(BASE, **kw)


def test_missing_component():
    c = dict(BASE)
    del c["handshake_gpu"]
    with pytest.raises(ValueError, match="handshake_gpu"):
        cl.compose_exec(c, group="down")


def test_exec_variants_and_residuals():
    pred = cl.exec_variants(BASE, "down")
    assert {"combined", "layer", "runtime", "combined+pcie"} <= set(pred)
    assert "layer+tail" not in pred                # join_tail이 없으면 꼬리 변형도 없다
    assert "layer+tail" in cl.exec_variants(dict(BASE, join_tail=10.0), "down")
    res = cl.residuals(pred, {"combined": 58.0, "layer": 65.5})
    assert res["layer"]["err"] == 0.0
    assert "runtime" not in res                  # 기준에 없는 target은 짝짓지 않는다


# ─── handshake_from_sleep ─────────────────────────────────────────────────
def _curve(H, C, d2h, Ls):
    return [{"sleep_us": L, "cold_sleep_us": d2h + max(H + L, C)} for L in Ls]


def test_handshake_recovers_synthetic():
    r = comp.handshake_from_sleep(_curve(18.0, 50.0, 2.0, (25, 50, 75, 100, 150)),
                                  2.0, 52.0)
    assert r["handshake_gpu_us"] == pytest.approx(18.0)
    assert r["cold_eff_us"] == pytest.approx(50.0)
    assert r["source"] == "gpu_bound"
    assert all(abs(p["resid"]) < 1e-6 for p in r["fit"])


def test_handshake_falls_back_to_longest():
    r = comp.handshake_from_sleep(_curve(10.0, 100.0, 2.0, (20, 40)), 2.0, 102.0)
    assert r["source"] == "longest_only"


def test_handshake_needs_points():
    with pytest.raises(ValueError):
        comp.handshake_from_sleep([], 2.0, 50.0)


# ─── score ────────────────────────────────────────────────────────────────
def _pt(dtype, h, w, s, g, us, **kw):
    return {"dtype": dtype, "hot_frac": h, "warm_frac": w, "sparsity": s, "group": g,
            "us": us, **kw}


def test_score_counts_and_heldout():
    ref = [_pt("fp8", 0.25, 0.125, 0.5, "gateup", 100.0, aa_spread=0.02),
           _pt("fp8", 0.25, 0.125, 0.5, "down", 100.0, aa_spread=0.04),
           {**_pt("bf16", 0.5, 0.125, 0.8, "gateup", 200.0), "phase": "gateup"}]
    pred = [_pt("fp8", 0.25, 0.125, 0.5, "gateup", 104.0),
            _pt("fp8", 0.25, 0.125, 0.5, "down", 112.0),
            _pt("bf16", 0.5, 0.125, 0.8, "gateup", 190.0),
            _pt("mxfp4", 0.5, 0.125, 0.8, "gateup", 1.0)]        # 기준 없음
    held = [cl.point_key(pred[2])]
    s = cl.score(pred, ref, held_out=held)
    assert s["unmatched_predictions"] == 1
    assert s["all"]["n"] == 3
    assert s["all"]["max_abs_err"] == pytest.approx(0.12)
    assert s["all"]["within_10"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["held_out"]["n"] == 1 and s["tuned"]["n"] == 2
    assert s["all"]["aa_max"] == pytest.approx(0.04)
    assert s["all"]["target_met"] is False


def test_score_aa_band_and_layer_phase():
    ref = [_pt("fp8", 0.25, 0.125, 0.5, "layer", 100.0, aa_values_us=[95.0, 100.0, 110.0])]
    s = cl.score([_pt("fp8", 0.25, 0.125, 0.5, "layer", 108.0)], ref)
    assert s["points"][0]["in_aa_band"] is True
    assert s["by_phase"]["layer"]["all"]["n"] == 1


def test_point_key_rejects_bad_group():
    with pytest.raises(ValueError):
        cl.point_key(_pt("fp8", 0.25, 0.125, 0.5, "up", 1.0))


# ─── CUDA 게이트: 작은 점에서 모양만 ──────────────────────────────────────
def _cuda_and_kt():
    try:
        import torch
        import kt_kernel  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_and_kt(), reason="needs CUDA + kt_kernel")
def test_components_smoke_cuda():
    shape = Shape(experts=16, topk=4, hidden=2048, inter=768)
    with comp.ComponentProfiler(shape, hot_frac=0.25, warm_frac=0.125, sparsity=0.5,
                                dtype="fp8", device=0) as prof:
        out = comp.measure_components(prof, groups=("down",), reps=8, replays=3, rounds=1)
    d = out["down"]
    for k in ("hot", "warm", "rejoin", "d2h_ids", "d2h_act", "h2d_partial",
              "cold_host", "cold_graph", "empty"):
        assert d["components"][k]["us"] > 0
    pred = cl.compose_exec(cl.component_values(d), group="down", target="layer")
    assert pred["us"] > 0


def test_exec_variants_packed_swaps_warm_only_for_runtime():
    v = dict(BASE, join_tail=10.0, warm_packed=20.0)
    pred = cl.exec_variants(v, "down")
    rp = pred["runtime+packed+tail"]
    assert rp["warm_path"] == "packed"
    assert rp["overlap_detail"]["warm"] == 20.0
    assert pred["runtime+tail"]["overlap_detail"]["warm"] == 30.0
    assert pred["layer+tail"]["overlap_detail"]["warm"] == 30.0     # full-layer 기준은 row-major


def test_score_cli_loaders(tmp_path):
    import importlib.util
    import json
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "score_components", pathlib.Path(__file__).with_name("score_components.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"points": [
        {"dtype": "bf16", "hot_frac": 0.25, "warm_frac": 0.125, "sparsity": 0.5,
         "phase": "down", "median_us": 100.0, "aa_spread": 0.03}]}))
    pred = tmp_path / "p.json"
    pred.write_text(json.dumps({"point": {"dtype": "bf16", "hot_frac": 0.25, "warm_frac": 0.125,
                                          "sparsity": 0.5},
                                "groups": {"down": {"predicted": {
                                    "runtime+pcie+tail": {"us": 90.0},
                                    "runtime+packed+pcie+tail": {"us": 103.0}}}}}))
    preds = mod.load_predictions([str(pred)], "runtime+pcie+tail", True)
    assert preds[0]["variant"] == "runtime+packed+pcie+tail"
    s = cl.score(preds, mod.load_reference(str(ref)))
    assert s["all"]["n"] == 1 and s["all"]["max_abs_err"] == pytest.approx(0.03)


def test_tail_follows_critical_branch():
    c = dict(BASE, join_tail=10.0, join_tail_gpu=25.0)
    gpu_crit = cl.compose_exec(c, group="down", target="layer", tail=True)
    assert gpu_crit["terms"]["join_tail_gpu"] == 25.0 and "join_tail" not in gpu_crit["terms"]
    cpu_crit = cl.compose_exec(dict(c, cold_eff=200.0), group="down", target="layer", tail=True)
    assert cpu_crit["terms"]["join_tail"] == 10.0 and "join_tail_gpu" not in cpu_crit["terms"]


def test_component_values_prefers_paired_tails():
    comp_entry = {"components": {"cold_graph": {"us": 50.0}, "cold_graph_tail": {"us": 66.5}},
                  "handshake": {"handshake_gpu_us": 40.0, "cold_eff_us": 48.0},
                  "tails": {"join_tail": {"us": 14.0}, "join_tail_gpu": {"us": 20.0}}}
    v = cl.component_values(comp_entry)
    assert v["join_tail"] == 14.0 and v["join_tail_gpu"] == 20.0


def test_score_loader_runtime_grid_schema(tmp_path):
    import importlib.util
    import json
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "score_components", pathlib.Path(__file__).with_name("score_components.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ref = tmp_path / "grid.json"
    ref.write_text(json.dumps({"schema": "prism.runtime_layer.reference/v1", "points": [
        {"dtype": "fp8", "hot_frac": 0.375, "warm_frac": 0.25, "sparsity": 0.8, "held_out": True,
         "phases": {"layer": {"median_us": 300.0}, "gateup": {"median_us": 180.0, "aa_spread_rel": 0.02},
                    "down": {"median_us": 120.0, "aa_spread_rel": None}}}]}))
    r = mod.load_reference(str(ref))
    assert {x["group"] for x in r} == {"gateup", "down", "layer"}
    assert all(x["held_out"] for x in r)
    assert [x["us"] for x in r] == [180.0, 120.0, 300.0]


def test_compose_layer_adds_chain_per_variant():
    pred = {"gateup": {"runtime+tail": {"us": 150.0}, "layer": {"us": 1.0}},
            "down": {"runtime+tail": {"us": 100.0}}}
    out = cl.compose_layer(pred, -4.0)
    assert set(out) == {"runtime+tail"}
    assert out["runtime+tail"]["us"] == pytest.approx(246.0)
    with pytest.raises(ValueError):
        cl.compose_layer({"gateup": {}}, 0.0)


def test_graph_interference_variant():
    v = dict(BASE, join_tail=10.0, cold_eff=90.0, warm_interference=7.0)
    r = cl.compose_exec(v, group="gateup", target="runtime", contention="graph", tail=True)
    assert r["overlap_detail"]["cold_used"] == pytest.approx(97.0)
    assert "runtime+gint+tail" in cl.exec_variants(v, "gateup")


def test_pooled_handshake_median():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "score_components", pathlib.Path(__file__).with_name("score_components.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    docs = [{"groups": {"gateup": {"values_us": {"handshake_gpu": h1}},
                        "down": {"values_us": {"handshake_gpu": h2}}}}
            for h1, h2 in ((30.0, 50.0), (40.0, 44.0))]
    assert mod.pooled_handshake(docs) == pytest.approx(42.0)
    d = {"groups": {g: {"values_us": dict(BASE, join_tail=10.0)} for g in ("gateup", "down")},
         "chain": {"chain_us": 5.0}}
    mod.repool([d], 10.0)
    assert d["groups"]["down"]["predicted"]["layer"]["overlap_detail"]["handshake_gpu"] == 10.0
    assert "runtime+tail" in d["layer_predicted"]


def test_compose_v3_adds_measured_interference_without_pcie():
    v = dict(BASE, join_tail=10.0, gpu_interference=12.0, cold_eff=90.0)
    r = cl.compose_v3(v, group="gateup")
    base = cl.compose_exec(v, group="gateup", target="runtime", contention=False, tail=True)
    assert r["us"] == pytest.approx(base["us"] + 12.0)
    assert r["overlap_detail"]["contention_scale"] == 1.0
    neg = cl.compose_v3(dict(v, gpu_interference=-3.0), group="gateup")
    assert neg["us"] == pytest.approx(base["us"])
    pk = cl.compose_v3(dict(v, warm_packed=99.0), group="gateup", warm_key="warm_packed")
    assert pk["overlap_detail"]["warm"] == 99.0
