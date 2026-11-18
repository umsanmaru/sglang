"""runtime-exact 레이어 프로파일러의 회귀망.

CPU 테스트는 합성(plan·calib·행 반올림·리포트 보조)만 본다 — plan이 validate_static을
통과하는지, 티어 행 수가 FullLayerProfiler와 같은 반올림인지, calib 테이블이 요청한 keep을
정확히 실현하는지. CUDA 테스트는 서빙 조립 경로로 작은 구성을 만들어 graph/eager 측정이
도는지와 결정 기록이 채워지는지 본다.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from sglang.srt.layers.moe.prism.plan import Proj, Tier, parse_plan, validate_static  # noqa: E402
from sglang.srt.layers.moe.prism.profile.common import Shape, split_rows  # noqa: E402
from sglang.srt.layers.moe.prism.profile.runtime_layer import (  # noqa: E402
    RuntimePoint,
    aa_summary,
    attach_sparsity,
    build_calib_blob,
    build_plan_raw,
    diff_timing,
    node_shards,
    tier_rows,
    timing_dict,
)

SHAPE = Shape(8, 2, 512, 512)


def _point(**kw):
    base = dict(shape=SHAPE, hot_frac=0.25, warm_frac=0.125, sparsity=0.5, dtype="fp8",
                n_layers=2)
    base.update(kw)
    return RuntimePoint(**base)


@pytest.mark.parametrize("dtype,step", [("bf16", 32), ("fp8", 128), ("mxfp4", 32)])
def test_tier_rows_match_full_layer_rounding(dtype, step):
    p = _point(dtype=dtype)
    for proj in ("gate", "down"):
        k = SHAPE.k_axis(proj)
        hot, warm, cold = tier_rows(k, 0.25, 0.125, p.step)
        assert p.step == step
        assert hot == split_rows(k, 0.25, step=step)
        assert warm == split_rows(k, 0.125, step=step)
        assert hot + warm + cold == k


def test_point_rejects_bad_input():
    with pytest.raises(ValueError):
        _point(dtype="int4")
    with pytest.raises(ValueError):
        _point(hot_frac=0.8, warm_frac=0.5)
    with pytest.raises(ValueError):
        _point(n_layers=0)
    with pytest.raises(ValueError):
        _point(index_mode="random")


def test_node_shards():
    assert node_shards(768, 2, 256) == [[0, 0, 512], [1, 512, 768]]
    assert node_shards(2048, 2, 256) == [[0, 0, 1024], [1, 1024, 2048]]
    assert node_shards(768, 1, 256) == [[0, 0, 768]]
    with pytest.raises(ValueError):
        node_shards(768, 3, 256)


@pytest.mark.parametrize("mode", ["scattered", "contiguous"])
@pytest.mark.parametrize("dtype", ["bf16", "fp8", "mxfp4"])
def test_plan_validates_and_rows_per_expert(mode, dtype, tmp_path):
    p = _point(dtype=dtype, index_mode=mode)
    raw, labels, info = build_plan_raw(p, nodes=2)
    blob, keep = build_calib_blob(p, labels, info["chunk"])
    path = tmp_path / "c.pt"
    torch.save(blob, path)
    attach_sparsity(raw, p, str(path), "0" * 64)
    plan = parse_plan(raw)
    validate_static(plan)
    for layer in range(p.n_layers):
        for e in range(SHAPE.experts):
            for proj, name in ((Proj.GATE, "gate"), (Proj.UP, "up"), (Proj.DOWN, "down")):
                pp = plan.expert(layer, e).proj(proj)
                for tier, want in zip((Tier.HOT, Tier.WARM, Tier.COLD), info["rows"][name]):
                    got = sum(b.end - b.start for b in pp.bands if b.tier == tier)
                    assert got == want
    gate = [labels[(0, e, "gate")] for e in range(SHAPE.experts)]
    up = [labels[(0, e, "up")] for e in range(SHAPE.experts)]
    if mode == "contiguous":
        assert gate == up
    else:
        assert gate != up        # gate/up 인덱스가 서로 독립 (계약: 독립 인덱스)


@pytest.mark.parametrize("sp", [0.5, 0.8])
def test_calib_realizes_keep_exactly(sp):
    p = _point(sparsity=sp, dtype="bf16")
    raw, labels, info = build_plan_raw(p, nodes=2)
    blob, keep = build_calib_blob(p, labels, info["chunk"])
    # 마스크 티어마다 expert별 round(npairs·(1−s)) — 균일 행 수면 비율이 정확히 같다
    for proj in ("gate", "up", "down"):
        for tier in ("warm", "cold"):
            rows = info["rows"][proj][("hot", "warm", "cold").index(tier)]
            want = round(rows // 2 * (1 - sp)) / (rows // 2)
            assert keep[proj][tier] == pytest.approx(want, abs=1e-6)
    # 페어 무결성: 두 채널이 함께 살거나 죽는다
    wn = blob["wn_g"]
    assert torch.equal(wn[..., 0::2] > 0, wn[..., 1::2] > 0)
    assert blob["tg2l"].shape == (p.n_layers, SHAPE.experts, 201)


TK = {"gate": {"warm": 0.75, "cold": 0.25}, "up": {"warm": 0.625, "cold": 0.375},
      "down": {"warm": 1.0, "cold": 0.0}}


def _tier_mask(blob_wn, labels, chunk, proj, tier, layer, e):
    """(layer, e)의 `tier` 밴드 행들에서 wn > 0인 페어의 0/1 벡터."""
    lab = labels[(layer, e, proj)]
    rows = torch.arange(blob_wn.shape[-1]).reshape(-1, chunk)
    sel = [i for i, t in enumerate(lab) if t == tier]
    r = rows[sel].reshape(-1)
    return (blob_wn[layer, e, r] > 0).reshape(-1, 2)


@pytest.mark.parametrize("mode", ["scattered", "contiguous"])
def test_tier_keep_realized_per_tier(mode):
    p = _point(sparsity=0.0, tier_keep=TK, dtype="bf16", index_mode=mode)
    assert p.sparse
    raw, labels, info = build_plan_raw(p, nodes=2)
    assert raw["schema_version"] == 2            # 마스크 plan
    blob, keep = build_calib_blob(p, labels, info["chunk"])
    key = {"gate": "wn_g", "up": "wn_u", "down": "wn_d"}
    for proj in ("gate", "up", "down"):
        for tier in ("warm", "cold"):
            rows = info["rows"][proj][("hot", "warm", "cold").index(tier)]
            want = round(rows // 2 * TK[proj][tier]) / (rows // 2)
            assert keep[proj][tier] == pytest.approx(want, abs=1e-6)
            # expert마다 정확히 그 수 (전 expert 공통 keep)
            for layer in range(p.n_layers):
                for e in range(SHAPE.experts):
                    m = _tier_mask(blob[key[proj]], labels, info["chunk"], proj, tier, layer, e)
                    assert torch.equal(m[:, 0], m[:, 1])          # 페어 무결성
                    assert int(m[:, 0].sum()) == round(rows // 2 * TK[proj][tier])
        # hot은 전부 산다
        m = _tier_mask(blob[key[proj]], labels, info["chunk"], proj, "hot", 0, 0)
        assert bool(m.all())


def test_tier_keep_uniform_equivalent_to_sparsity():
    """모든 티어 keep = 1−s면 균일 경로와 같은 keep 표 (spread 0)."""
    u = _point(sparsity=0.5, dtype="bf16")
    t = _point(sparsity=0.0, dtype="bf16",
               tier_keep={q: {"warm": 0.5, "cold": 0.5} for q in ("gate", "up", "down")})
    ku = build_calib_blob(u, *_labels_chunk(u))[1]
    kt = build_calib_blob(t, *_labels_chunk(t))[1]
    assert ku == kt


def _labels_chunk(p):
    _, labels, info = build_plan_raw(p, nodes=2)
    return labels, info["chunk"]


def test_tier_keep_rejects_bad_input():
    with pytest.raises(ValueError):          # sparsity와 동시 지정
        _point(sparsity=0.5, tier_keep=TK)
    with pytest.raises(ValueError):          # proj 누락
        _point(sparsity=0.0, tier_keep={"gate": TK["gate"], "down": TK["down"]})
    with pytest.raises(ValueError):          # 범위 밖
        _point(sparsity=0.0, tier_keep=dict(TK, down={"warm": 1.2, "cold": 0.1}))
    with pytest.raises(ValueError):          # 티어 키 누락
        _point(sparsity=0.0, tier_keep=dict(TK, up={"warm": 0.5}))
    k = _point(sparsity=0.0, tier_keep=TK).key()
    assert "tk" in k and "sp" not in k.split("/")[-1]


def test_profiler_restores_env_and_cleans_workdir(monkeypatch):
    """구성이 실패해도 close()가 env를 되돌리고 임시 디렉터리를 지운다 (CUDA 불필요)."""
    from sglang.srt.layers.moe.prism.profile import runtime_layer as rl

    monkeypatch.setenv("SGLANG_PRISM_MAX_TOKENS", "77")
    monkeypatch.delenv("SGLANG_PRISM_PLAN", raising=False)
    seen = {}

    def boom(self, point, *, device, workdir, env, verbose):
        import tempfile
        self._tmpdir = tempfile.mkdtemp(prefix="prism_runtime_layer_")
        seen["dir"] = self._tmpdir
        for k, v in (env or {}).items():
            self._set_env(k, v)
        self._set_env("SGLANG_PRISM_PLAN", "/nonexistent/plan.json")
        raise ValueError("synthetic failure")

    monkeypatch.setattr(rl.RuntimeLayerProfiler, "_build", boom)
    with pytest.raises(ValueError):
        rl.RuntimeLayerProfiler(_point(), env={"SGLANG_PRISM_MAX_TOKENS": "1"})
    assert os.environ["SGLANG_PRISM_MAX_TOKENS"] == "77"
    assert "SGLANG_PRISM_PLAN" not in os.environ
    assert not os.path.exists(seen["dir"])


def test_timing_helpers():
    t = timing_dict([3.0, 1.0, 2.0])
    assert t["us"] == 2.0 and t["min_us"] == 1.0
    d = diff_timing([10.0, 11.0, 12.0], [4.0, 4.0, 4.0])
    assert d["us"] == 7.0 and d["flush_us"] == 4.0
    a = aa_summary([100.0, 102.0, 101.0])
    assert a["median"] == 101.0 and a["spread_rel"] == pytest.approx(2 / 101, abs=1e-4)


def test_factorial_conditions():
    from bench_runtime_layer import factorial_conditions

    c = factorial_conditions(4)
    assert c[0]["n_layers"] == 4 and c[0]["phases"] and c[0]["eager"]
    keys = {(x["n_layers"], x["rotate"], x["flush"]) for x in c}
    assert len(keys) == len(c) == 12


# ─── CUDA + kt ─────────────────────────────────────────────────────────────
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda_required
@pytest.mark.parametrize("dtype", ["fp8"])
def test_runtime_profiler_smoke(dtype, tmp_path):
    pytest.importorskip("kt_kernel", reason="kt_kernel required")
    from sglang.srt.layers.moe.prism.profile.runtime_layer import RuntimeLayerProfiler

    p = RuntimePoint(Shape(8, 2, 512, 512), 0.25, 0.25, 0.5, dtype, n_layers=2)
    with RuntimeLayerProfiler(p, workdir=str(tmp_path), verbose=False) as prof:
        rep = prof.report([{"n_layers": 2, "eager": True, "phases": True}], passes=2, replays=3)
    r = rep["results"][0]
    assert r["layer"]["us"] > 0 and r["gateup"]["us"] > 0 and r["down"]["us"] > 0
    assert r["decisions_graph"]["graph_flow"] is True
    assert rep["decisions"]["kt_max_len"] == rep["decisions"]["max_tokens"]
    assert rep["realized_keep"]["masked_all"] == pytest.approx(rep["calib_keep"]["masked_all"], abs=0.02)
