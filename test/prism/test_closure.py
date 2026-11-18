"""closure 검사(`profile/closure.py`)의 계약 — 하드웨어 없이 도는 단위 테스트.

측정 진입점 넷(hot_dense_gemv / WarmColdProfiler / ColdCpuProfiler / FullLayerProfiler)을
가짜로 갈아끼우고 (1) 합성 식이 컴파일러 식과 같은지, (2) 오차 부호와 판정, (3) 빈 티어를
부르지 않는지, (4) 검증이 하드웨어보다 먼저 나는지, (5) 런타임 경로 경고, (6) 티어 쪽과
full-layer의 CPU 풀이 동시에 살지 않는지를 본다. 실측값의 정합은 벤치가 본다.
"""

import json
from importlib import import_module
from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.prism.profile.common import Shape, Timing, split_rows, store_of
from sglang.srt.layers.moe.prism.profile.full_layer import LayerGroupReport, build_splits

_P = "sglang.srt.layers.moe.prism.profile."
cl = import_module(_P + "closure")
hot_mod = import_module(_P + "hot")
wc_mod = import_module(_P + "warm_cold")
cold_mod = import_module(_P + "cold_cpu")
fl_mod = import_module(_P + "full_layer")

SHAPE = Shape(experts=8, topk=2, hidden=2048, inter=768)
AXIS = {"gateup": SHAPE.hidden, "down": SHAPE.inter}
PROJ_AXIS = {"gate": SHAPE.hidden, "up": SHAPE.hidden, "down": SHAPE.inter}


def _t(us: float) -> Timing:
    return Timing(us=us, min_us=us, max_us=us, p90_us=us, replays=1)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """런타임 경로 판정이 바깥 env에 휘둘리지 않게."""
    for k in ("SGLANG_PRISM_WARM_KT", "SGLANG_PRISM_COLD_SPLIT",
              "SGLANG_PRISM_COLD_GPU_MIN_M", "SGLANG_PRISM_GROUPED_MIN_M"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def rig(monkeypatch):
    """측정 진입점 넷을 가짜로. 행 수는 진짜 프로파일러와 같은 식(split_rows/build_splits)으로
    실현하고, 열린 CPU 풀 수(alive)와 순서(events)를 받아 적는다."""
    st = SimpleNamespace(
        tiers={"hot": {"gateup": 10.0, "down": 6.0},
               "warm": {"gateup": 5.0, "down": 3.0},
               "cold": {"gateup": 20.0, "down": 30.0}},
        # full-layer `layer` = handshake 예측 / (1 + target_err[hot_frac]) → err가 정확히 target
        target_err={}, default_err=0.05,
        alive=0, peak=0, events=[], hot_alive=[], warm_k_override=None)

    def opened(name):
        st.alive += 1
        st.peak = max(st.peak, st.alive)
        st.events.append(("open", name))

    def closed(name):
        st.alive -= 1
        st.events.append(("close", name))

    class HotReport:
        """HotGemvReport가 closure에 주는 면만: r[group].k_rows, r.us(group)."""

        def __init__(self, rows):
            self._rows = rows

        def __getitem__(self, g):
            return SimpleNamespace(k_rows=self._rows[g])

        def us(self, g):
            return st.tiers["hot"][g]

    def fake_hot_dense_gemv(shape, *, hot_frac, device, dtype, reps, replays):
        st.events.append(("call", "hot"))
        st.hot_alive.append(st.alive)
        step = store_of(dtype).rows_step()
        return HotReport({g: split_rows(AXIS[g], hot_frac, step=step)
                          for g in ("gateup", "down")})

    class FakeWarm:
        def __init__(self, shape, *, warm_frac, cold_frac, sparsity, numa_split,
                     device, dtype, threads, cpu_kernel):
            step = store_of(dtype).rows_step()
            k = st.warm_k_override
            self.splits = {p: SimpleNamespace(
                k_warm=k if k is not None else split_rows(PROJ_AXIS[p], warm_frac, step=step),
                k_cold=split_rows(PROJ_AXIS[p], cold_frac, step=step)) for p in PROJ_AXIS}
            st.warm_kw = dict(warm_frac=warm_frac, cold_frac=cold_frac, threads=threads,
                              cpu_kernel=cpu_kernel)
            opened("warm")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            closed("warm")

        def measure(self, g, *, reps, replays, only):
            assert only == ("warm_only",)
            return SimpleNamespace(us=lambda name: st.tiers["warm"][g])

    class FakeCold:
        def __init__(self, shape, *, cold_frac, sparsity, proj, numa_split, threads,
                     dtype, cpu_kernel):
            step = store_of(dtype).rows_step()
            self.k_cold = {p: split_rows(PROJ_AXIS[p], cold_frac, step=step)
                           for p in PROJ_AXIS}
            self.proj, self.sparsity = proj, sparsity
            st.cold_kw = dict(threads=threads, cpu_kernel=cpu_kernel)
            opened(f"cold[{proj}]")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            closed(f"cold[{self.proj}]")

        def measure(self, *, iters, replays):
            return SimpleNamespace(us=st.tiers["cold"][self.proj],
                                   keep_frac=1.0 - self.sparsity)

    class FakeFull:
        def __init__(self, shape, *, hot_frac, warm_frac, cold_frac, sparsity,
                     sparsity_spread, dtype, device, numa_split, cpu_kernel, threads, seed):
            assert cold_frac is None
            self.hot_frac = hot_frac
            self.splits = build_splits(shape, hot_frac=hot_frac, warm_frac=warm_frac,
                                       cold_frac=None, dtype=dtype)
            self.params = {"dtype": store_of(dtype).name, "cpuinfer_threads": threads,
                           "sparsity_spread": sparsity_spread}
            opened("full")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            closed("full")

        def measure(self, g, *, reps, replays, flush, rounds):
            t = st.tiers
            pred = cl.compose(t["hot"][g] if self.hot_frac else 0.0,
                              t["warm"][g], t["cold"][g], model="handshake")
            err = st.target_err.get(self.hot_frac, st.default_err)
            layer = pred / (1.0 + err)
            return LayerGroupReport(group=g, timings={
                "combined": _t(layer - 2.0), "layer": _t(layer),
                "combined_split": _t(layer - 3.0), "layer_split": _t(layer - 1.0),
                "hot_only": _t(1.0)}, info={"group": g})

    monkeypatch.setattr(hot_mod, "hot_dense_gemv", fake_hot_dense_gemv)
    monkeypatch.setattr(wc_mod, "WarmColdProfiler", FakeWarm)
    monkeypatch.setattr(cold_mod, "ColdCpuProfiler", FakeCold)
    monkeypatch.setattr(fl_mod, "FullLayerProfiler", FakeFull)
    return st


def run(**kw):
    args = dict(hot_frac=0.25, warm_frac=0.125, sparsity=0.5, dtype="fp8", threads=6)
    args.update(kw)
    return cl.closure(SHAPE, **args)


# ── 합성 식 ─────────────────────────────────────────────────────────
def test_compose_bare_max():
    assert cl.compose(10, 5, 20, model="max") == 20
    assert cl.compose(30, 10, 20, model="max") == 40      # hot + warm는 더한다
    assert cl.compose(0, 0, 20, model="max") == 20


def test_compose_handshake_matches_compiler_defaults():
    """compute_3tier.py:376-382 — max(gpu, cold + 2.5) × contention + 5.25."""
    assert cl.compose(10, 5, 20) == pytest.approx(20 + 2.5 + 5.25)          # CPU 쪽이 길다
    assert cl.compose(30, 10, 20) == pytest.approx(40 + 5.25)               # GPU 쪽이 길다
    assert cl.compose(10, 5, 20, tier_contention=1.2) == pytest.approx(22.5 * 1.2 + 5.25)
    assert cl.compose(10, 5, 20, cold_handshake_us=0, tier_join_us=0) == 20
    # 한쪽 갈래만 있으면 보정 없이 bare max (:383-385)
    assert cl.compose(10, 5, 0) == 15
    assert cl.compose(0, 0, 20) == 20


def test_compose_api_layer():
    """api.py: input_transfer + max(gpu, cold + cold_partial) + merge — handshake 없음."""
    kw = dict(input_transfer_us=1.0, cold_partial_us=2.0, merge_us=3.0)
    assert cl.compose(10, 5, 20, model="api", **kw) == pytest.approx(1 + 22 + 3)
    assert cl.compose(30, 10, 20, model="api", **kw) == pytest.approx(1 + 40 + 3)
    assert cl.compose(10, 5, 0, model="api", **kw) == pytest.approx(15 + 3)   # CPU 없음


def test_compose_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown model"):
        cl.compose(1, 1, 1, model="sum")
    with pytest.raises(ValueError, match="hot_us"):
        cl.compose(-1, 1, 1)
    with pytest.raises(ValueError, match="cold_us"):
        cl.compose(1, 1, float("nan"))


def test_api_overheads_mirror_compiler_metadata():
    ov = cl.api_overheads(SHAPE, "gateup", "fp8")
    assert ov["input_transfer_us"] == pytest.approx(2048 / 5.553e10 * 1e6 + 16.115, abs=1e-4)
    assert ov["cold_partial_us"] == pytest.approx(2 * 768 / 5.553e10 * 1e6 + 16.115, abs=1e-4)
    assert ov["merge_us"] == pytest.approx(4096 * 4 / 1.867e12 * 1e6 + 4.864, abs=1e-4)
    bf = cl.api_overheads(SHAPE, "down", "bf16", meta={"cuda_graph": True,
                                                       "cpu_io_charging": "amortized"})
    assert bf["input_transfer_us"] == pytest.approx(768 * 2 / 5.553e10 * 1e6, abs=1e-4)
    assert bf["merge_us"] == pytest.approx(4096 * 2 * 4 / 1.867e12 * 1e6, abs=1e-4)
    with pytest.raises(ValueError, match="unknown api meta"):
        cl.api_overheads(SHAPE, "down", "bf16", meta={"pcie": 1})


def test_error_sign_convention():
    """양수 = 모델이 비관적(실측보다 크게 예측)."""
    assert cl.closure_error(110.0, 100.0) == pytest.approx(0.10)
    assert cl.closure_error(90.0, 100.0) == pytest.approx(-0.10)
    assert cl.closure_error(1.0, 0.0) is None and cl.closure_error(1.0, None) is None


# ── 리포트 · 판정 ────────────────────────────────────────────────────
def test_report_shape_sign_and_json(rig):
    rep = run()
    json.dumps(rep)
    assert rep["bench"] == "closure" and "positive = model pessimistic" in rep["sign"]
    g = rep["groups"]["gateup"]
    assert g["tiers_us"] == {"hot": 10.0, "warm": 5.0, "cold": 20.0, "gpu_branch": 15.0}
    assert g["predicted_us"]["max"] == 20.0
    assert g["predicted_us"]["handshake"] == pytest.approx(27.75)
    assert g["err"]["handshake"]["layer"] == pytest.approx(0.05, abs=1e-4)
    assert g["err"]["handshake"]["combined"] > g["err"]["handshake"]["layer"]  # combined < layer
    assert set(g["measured_us"]) == {"combined", "layer", "combined_split", "layer_split"}
    assert g["verdict"] == {"model": "handshake", "reference": "layer",
                            "err": g["err"]["handshake"]["layer"],
                            "tolerance": 0.10, "pass": True}
    assert rep["pass"] is True
    # 실현 행 수 (fp8 step 128): hidden 2048 → 512/256/1280, inter 768 → 256/128/384
    assert g["rows"] == {"k_axis": 2048, "hot": 512, "warm": 256, "cold": 1280}
    assert rep["groups"]["down"]["rows"] == {"k_axis": 768, "hot": 256, "warm": 128, "cold": 384}
    assert rep["groups"]["down"]["realized_rows"] == {"hot": 256, "warm": 128, "cold": 384}
    prov = rep["provenance"]
    assert {"env", "source_commit", "kt_kernel_file", "env_flags", "threads",
            "cpu_kernel", "dtype"} <= set(prov)
    assert set(prov["env_flags"]) == set(cl.ENV_FLAGS)
    assert prov["cpu_kernel"] == "kt_tile_k2_fp8b128" and prov["threads"] == 6
    assert rep["notes"] and any("graph" in n and "D2H" in n for n in rep["notes"])
    # 세 CPU 측정이 같은 스레드·커널을 받는다
    assert rig.warm_kw["threads"] == rig.cold_kw["threads"] == 6
    assert rep["full_layer_params"]["cpuinfer_threads"] == 6
    assert rep["full_layer_params"]["sparsity_spread"] == 0.0


def test_tolerance_and_reference_gate(rig):
    rig.default_err = 0.08
    assert run()["pass"] is True
    assert run(tolerance=0.05)["pass"] is False
    rep = run(tolerance=0.05, reference="combined", model="max")
    assert rep["groups"]["gateup"]["verdict"]["reference"] == "combined"
    assert rep["groups"]["gateup"]["verdict"]["model"] == "max"
    rig.default_err = -0.2                                    # 낙관적 모델도 |err|로 걸린다
    rep = run()
    assert rep["pass"] is False and rep["worst_abs_err"] == pytest.approx(0.2, abs=1e-3)
    with pytest.raises(ValueError, match="reference"):
        run(reference="gpu_only")
    with pytest.raises(ValueError, match="tolerance"):
        run(tolerance=0)


def test_sweep_summary_counts_failures_and_errors(rig):
    rig.target_err = {0.25: 0.03, 0.375: -0.15}
    out = cl.closure_sweep(SHAPE, [(0.25, 0.125, 0.5), (0.375, 0.125, 0.9),
                                   (0.7, 0.4, 0.5)], dtype="fp8", threads=6)
    json.dumps(out)
    s = out["summary"]
    assert (s["points"], s["failing"], s["errors"]) == (3, 2, 1)
    assert s["pass"] is False
    assert s["worst_err"] == pytest.approx(-0.15, abs=1e-3)
    assert s["worst_at"]["hot_frac"] == 0.375
    assert "error" in out["reports"][2] and "ValueError" in out["reports"][2]["error"]
    ok = cl.closure_sweep(SHAPE, [{"hot_frac": 0.25, "warm_frac": 0.125, "sparsity": 0.5}],
                          dtype="fp8", threads=6)["summary"]
    assert ok["pass"] is True and ok["failing"] == 0
    with pytest.raises(ValueError, match="unknown model"):
        cl.closure_sweep(SHAPE, [(0.25, 0.125, 0.5)], model="nope")


# ── 빈 티어 ─────────────────────────────────────────────────────────
def test_zero_hot_is_not_called(rig):
    rep = run(hot_frac=0.0, warm_frac=0.25)
    assert ("call", "hot") not in rig.events
    g = rep["groups"]["gateup"]
    assert g["tiers_us"]["hot"] == 0.0 and g["rows"]["hot"] == 0
    assert g["predicted_us"]["handshake"] == pytest.approx(max(5.0, 22.5) + 5.25)


def test_zero_warm_is_not_constructed(rig):
    rep = run(hot_frac=0.25, warm_frac=0.0)
    assert not any(name == "warm" for _, name in rig.events)
    assert rep["groups"]["down"]["tiers_us"]["warm"] == 0.0
    assert rep["tier_params"]["warm_profiler"] is None
    assert rep["runtime_path"]["all_proj_warm"] is False


def test_zero_cold_is_not_constructed(rig):
    rep = run(hot_frac=0.5, warm_frac=0.5)       # fp8 step 128: 두 축 모두 cold 0행
    assert not any(name.startswith("cold") for _, name in rig.events)
    g = rep["groups"]["gateup"]
    assert g["tiers_us"]["cold"] == 0.0 and g["rows"]["cold"] == 0
    assert g["predicted_us"]["handshake"] == g["predicted_us"]["max"] == 15.0


# ── 검증이 하드웨어보다 먼저 ─────────────────────────────────────────
@pytest.mark.parametrize("hot,warm,match", [
    (0.7, 0.4, "> 1"),                       # 합이 1을 넘는다
    (-0.1, 0.2, r"\[0, 1\]"),                # 음수
    (0.25, 1.5, r"\[0, 1\]"),
    (0.97, 0.02, "exceed"),                  # 반올림 + 최소 한 타일로 축을 넘는다
    (0.62, 0.37, "0 cold rows"),             # cold 0.01 요청이 반올림에 먹힌다
])
def test_fraction_validation_before_measuring(rig, hot, warm, match):
    with pytest.raises(ValueError, match=match):
        run(hot_frac=hot, warm_frac=warm)
    assert rig.events == [], "검증 실패인데 측정 진입점이 불렸다"


def test_realized_rows_must_match(rig):
    rig.warm_k_override = 384                 # 프로파일러가 다른 행 수를 만들었다고 치자
    with pytest.raises(ValueError, match="warm rows"):
        run()
    assert rig.alive == 0, "실패해도 열린 프로파일러는 닫혀야 한다"


def test_plan_rows_agrees_with_build_splits():
    r = cl.plan_rows(SHAPE, hot_frac=0.375, warm_frac=0.125, dtype="bf16")
    sp = build_splits(SHAPE, hot_frac=0.375, warm_frac=0.125, dtype="bf16")
    for g, projs in (("gateup", ("gate", "up")), ("down", ("down",))):
        for p in projs:
            for tier in ("hot", "warm", "cold"):
                assert set(sp[p].rows(tier)) == {r["groups"][g][tier]}
        # cold 티어 호출에 넘길 비율이 같은 행을 재현한다
        a = r["groups"][g]
        assert split_rows(a["k_axis"], a["cold_frac"], step=r["step"]) == a["cold"]


def test_warm_profiler_cold_frac_choice():
    # fp8, inter 768에서는 두 축의 cold 비율이 갈려 한 비율로 재현할 수 없다 → 한 타일 (sweep과 같다)
    r = cl.plan_rows(SHAPE, hot_frac=0.25, warm_frac=0.125, dtype="fp8")
    f, src = cl._warm_profiler_cold_frac(r)
    assert src == "one_tile" and f == pytest.approx(128 / 768)
    # 두 축이 같은 비율로 떨어지면 실현값 그대로
    r = cl.plan_rows(Shape(experts=8, topk=2, hidden=2048, inter=1024),
                     hot_frac=0.25, warm_frac=0.25, dtype="fp8")
    f, src = cl._warm_profiler_cold_frac(r)
    assert src == "realized" and f == pytest.approx(0.5)


# ── 런타임 경로 ─────────────────────────────────────────────────────
def test_runtime_warning_on_for_bf16_defaults(rig):
    rep = run(dtype="bf16")
    rt = rep["runtime_path"]
    assert rt["applicable"] and rt["warm_kt"] is True
    assert rt["cold_gpu_min_m"] == 1536          # bf16: executor COLD_GPU_MIN_M
    assert any("warm-kt" in w for w in rep["warnings"])


def test_runtime_warning_off_for_fp8(rig):
    rep = run(dtype="fp8")
    assert rep["runtime_path"]["warm_kt"] is False
    assert rep["runtime_path"]["cold_gpu_min_m"] == 16   # fp8: grouped 경계
    assert not any("warm-kt" in w for w in rep["warnings"])


def test_runtime_warning_off_when_env_disables_it(rig, monkeypatch):
    monkeypatch.setenv("SGLANG_PRISM_WARM_KT", "0")
    rep = run(dtype="bf16")
    assert rep["runtime_path"]["warm_kt"] is False
    assert not any("warm-kt" in w for w in rep["warnings"])
    assert rep["provenance"]["env_flags"]["SGLANG_PRISM_WARM_KT"] == "0"


def test_runtime_path_follows_runtime_config_rules():
    on = cl.runtime_path("bf16", has_cold=True, all_proj_warm=True, environ={})
    assert on["warm_kt"] is True
    # cold GPU 경로를 끄면 자동 warm-kt도 꺼진다 (warm_kt_enabled의 auto 조건)
    assert cl.runtime_path("bf16", has_cold=True, all_proj_warm=True,
                           environ={"SGLANG_PRISM_COLD_GPU_MIN_M": "0"})["warm_kt"] is False
    assert cl.runtime_path("bf16", has_cold=False, all_proj_warm=True, environ={})["warm_kt"] is False
    # 명시적 override는 fp8에서도 켠다
    assert cl.runtime_path("fp8", has_cold=True, all_proj_warm=True,
                           environ={"SGLANG_PRISM_WARM_KT": "1"})["warm_kt"] is True
    assert cl.runtime_path("fp8pt", has_cold=True, all_proj_warm=True,
                           environ={})["applicable"] is False


def test_cold_split_env_warns_about_reference(rig, monkeypatch):
    monkeypatch.setenv("SGLANG_PRISM_COLD_SPLIT", "1")
    rep = run(dtype="fp8")
    assert any("layer_split" in w for w in rep["warnings"])


# ── 동시성 ──────────────────────────────────────────────────────────
def test_tier_and_full_layer_pools_never_overlap(rig):
    run()
    assert rig.peak == 1, f"CPU 풀이 동시에 {rig.peak}개 살았다: {rig.events}"
    assert rig.hot_alive == [0]
    order = [(k, n) for k, n in rig.events]
    assert order == [("call", "hot"), ("open", "warm"), ("close", "warm"),
                     ("open", "cold[gateup]"), ("close", "cold[gateup]"),
                     ("open", "cold[down]"), ("close", "cold[down]"),
                     ("open", "full"), ("close", "full")]
    assert rig.alive == 0
