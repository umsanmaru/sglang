"""full-layer 프로파일러와 per-expert primitive의 회귀망.

두 층으로 나뉜다. primitive 테스트는 CUDA도 kt도 필요 없다 (합성만 검증) —
per-expert 구성이 조용히 균일로 되돌아가거나 평균이 어긋나면 여기서 잡힌다.
프로파일러 테스트는 GPU + kt가 있어야 돌고, `check()`로 **마스킹이 실제로
걸렸는지**를 대조한다: 마스킹이 빠져도 성능만 달라지므로 그 대조가 유일한
검출기다.
"""

import pytest
import torch

from sglang.srt.layers.moe.prism.profile import (
    Shape,
    sparse_tables,
    split_rows_varied,
    spread_values,
    store_of,
    tier_indices,
)

E = 32
AXIS = 2048
STEP = 128


# ─── per-expert primitive (CUDA·kt 불필요) ─────────────────────────────────
@pytest.mark.parametrize("experts", [2, 7, 32, 128])
def test_spread_values_mean_is_exact(experts):
    """평균이 정확해야 한다 — 독립 표본이면 E가 작을 때 목표에서 벗어나고,
    그 편차가 모델 대비 오차에 섞인다."""
    vals = spread_values(0.5, experts, spread=0.3, seed=0)
    assert len(vals) == experts
    assert sum(vals) / experts == pytest.approx(0.5, abs=1e-12)
    assert min(vals) >= 0.2 - 1e-12 and max(vals) <= 0.8 + 1e-12


def test_spread_values_permutes_so_two_draws_are_not_anticorrelated():
    """행 수와 sparsity를 각각 뽑았을 때 구조적으로 역상관하면 안 된다.
    (섞지 않으면 짝수 expert가 늘 평균 이하가 되어 keep이 0.5 요청에 0.466이 됐다.)"""
    rows = split_rows_varied(AXIS, 0.125, 128, spread=0.06, step=STEP, seed=10)
    sps = spread_values(0.5, 128, spread=0.3, seed=0)
    _, _, _, keep = sparse_tables(128, rows, sps, seed=0)
    assert keep == pytest.approx(0.5, abs=0.02)


def test_split_rows_varied_respects_step_and_mean():
    rows = split_rows_varied(AXIS, 0.125, E, spread=0.06, step=STEP, seed=1)
    assert len(rows) == E
    assert all(r % STEP == 0 for r in rows)
    assert all(0 < r <= AXIS for r in rows)
    assert sum(rows) / (E * AXIS) == pytest.approx(0.125, abs=0.02)
    assert len(set(rows)) > 1, "spread를 줬는데 전부 같은 값이면 가변이 죽은 것"


def test_split_rows_varied_zero_spread_is_uniform():
    rows = split_rows_varied(AXIS, 0.125, E, spread=0.0, step=STEP, seed=1)
    assert set(rows) == {256}


def test_sparse_tables_accepts_per_expert_rows_and_sparsity():
    rows = split_rows_varied(AXIS, 0.125, E, spread=0.06, step=STEP, seed=1)
    sps = spread_values(0.5, E, spread=0.3, seed=2)
    a, c, thr, keep = sparse_tables(E, rows, sps, seed=0)
    assert a.numel() == sum(rows)
    assert c.numel() == sum(rows) // 2
    assert tuple(thr.shape) == (E, 201)
    # expert별 실현 sparsity가 요청을 따라가야 한다 (균일로 뭉개지면 안 된다)
    off = 0
    realized = []
    for e, k in enumerate(rows):
        realized.append(1.0 - float((a[off:off + k] > 0).float().mean()))
        off += k
    assert realized[0] != pytest.approx(realized[1], abs=1e-6)
    assert sum(realized) / E == pytest.approx(0.5, abs=0.05)
    assert keep == pytest.approx(0.5, abs=0.05)


def test_sparse_tables_uniform_path_unchanged():
    """스칼라 호출은 예전 형태 그대로여야 한다 (하위호환)."""
    a, c, _, keep = sparse_tables(4, 256, 0.9, seed=0)
    assert a.numel() == 4 * 256 and c.numel() == 4 * 128
    assert keep == pytest.approx(0.1, abs=0.02)


def test_tier_indices_three_tiers_are_disjoint_per_expert():
    """같은 seed + 누적 skip이면 hot/warm/cold가 expert마다 서로소여야 한다."""
    hot = split_rows_varied(AXIS, 0.375, E, spread=0.1, step=STEP, seed=11)
    warm = split_rows_varied(AXIS, 0.125, E, spread=0.05, step=STEP, seed=22)
    cold = tuple(AXIS - h - w for h, w in zip(hot, warm))
    skip_w = tuple(hot)
    skip_c = tuple(h + w for h, w in zip(hot, warm))
    hi, ho = tier_indices(AXIS, hot, E, skip=0, seed=0)
    wi, wo = tier_indices(AXIS, warm, E, skip=skip_w, seed=0)
    ci, co = tier_indices(AXIS, cold, E, skip=skip_c, seed=0)
    assert int(ho[-1]) == sum(hot) and int(wo[-1]) == sum(warm) and int(co[-1]) == sum(cold)
    for e in range(E):
        h = set(hi[ho[e]:ho[e + 1]].tolist())
        w = set(wi[wo[e]:wo[e + 1]].tolist())
        c = set(ci[co[e]:co[e + 1]].tolist())
        assert len(h) == hot[e] and len(w) == warm[e] and len(c) == cold[e]
        assert not (h & w) and not (h & c) and not (w & c)
        assert len(h | w | c) == AXIS, "세 티어의 합이 K축 전체여야 한다"


def test_store_accepts_per_expert_rows():
    st = store_of("fp8")
    rows = split_rows_varied(AXIS, 0.125, E, spread=0.06, step=STEP, seed=1)
    parts = st.gpu_store(E, rows, 768, device=None, seed=0)
    assert parts[0].shape[0] == sum(rows)
    assert st.store_bytes(E, rows, 768) > 0
    w, s = st.cold_store(E, 768, rows, seed=0)
    assert w.numel() == 768 * sum(rows)
    assert s.numel() == sum(r // 128 for r in rows) * (768 // 128)


# ─── 프로파일러 (GPU + kt 필요) ────────────────────────────────────────────
def _kt_available() -> bool:
    try:
        import kt_kernel  # noqa: F401
    except Exception:
        return False
    return True


needs_rig = pytest.mark.skipif(
    not torch.cuda.is_available() or not _kt_available(),
    reason="full-layer 프로파일러는 CUDA와 kt_kernel이 둘 다 있어야 한다",
)


@needs_rig
@pytest.mark.parametrize("dtype", ["fp8", "bf16"])
@pytest.mark.parametrize("group", ["gateup", "down"])
def test_check_matches_masked_reference(dtype, group):
    """세 티어 partial의 합이 합성 마스크 레퍼런스와 맞아야 한다.

    마스킹이 빠지면 살아있는 행의 두 배를 더하게 되어 rel err가 1 근처로 뛴다.
    """
    from sglang.srt.layers.moe.prism.profile import FullLayerProfiler

    with FullLayerProfiler(
        Shape(experts=32, topk=8, hidden=2048, inter=768),
        hot_frac=0.375, warm_frac=0.125, sparsity=0.5, sparsity_spread=0.3,
        hot_spread=0.1, warm_spread=0.05, dtype=dtype, device=0, seed=0,
    ) as p:
        rep = p.check(group)
        errs = [v for k, v in rep.items() if k.endswith("_max_rel_err")]
        assert errs, f"대조 항목이 없다: {rep}"
        for e in errs:
            assert e < 0.02, f"{group}/{dtype} rel err {e} — 마스킹이 빠졌을 수 있다"


@needs_rig
def test_measure_reports_rounds_and_overlap():
    """rounds가 실제로 반복되고, 겹침 지표가 기준선과 함께 나와야 한다."""
    from sglang.srt.layers.moe.prism.profile import FullLayerProfiler

    with FullLayerProfiler(
        Shape(experts=32, topk=8, hidden=2048, inter=768),
        hot_frac=0.375, warm_frac=0.125, sparsity=0.5, dtype="fp8", device=0, seed=0,
    ) as p:
        rep = p.measure("down", reps=8, replays=3, rounds=2, flush=False)
        for name in ("cold_graph", "combined", "combined_split", "layer_split"):
            assert name in rep.timings, f"{name} 변형이 없다"
            assert len(rep.info["rounds_us"][name]) == 2
        # 겹침 지표는 같은 조건(flush 없음)의 두 값을 뺀 것이어야 한다
        exposed = rep.info["exposed_us"]
        assert exposed["combined"] == pytest.approx(
            rep.us("combined") - rep.us("cold_graph"), abs=1e-6)
        # cold를 곁 스트림으로 빼면 GPU 노출이 줄어야 한다 (down에서 실측 22 µs → ~0)
        assert exposed["combined_split"] <= exposed["combined"] + 5.0
