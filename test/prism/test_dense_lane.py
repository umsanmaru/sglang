"""dense 레인 래퍼(`profile/dense_lane.py`)의 계약 — 하드웨어 없이 도는 단위 테스트.

밑단(dense_gemv / warm_sparse_gemv / cold_sparse_gemv)을 가짜로 갈아끼우고
(1) 검증이 하드웨어를 건드리기 **전에** 나는지, (2) 밑단에 넘어가는 인자가 dense
레인의 퇴화형(top_k=1, expert=슬롯, proj=down)인지, (3) 결과 필드가 맞는지 본다.
실측값의 정합은 벤치가 보고, 여기서는 배관만 본다.
"""

from importlib import import_module

import pytest

from sglang.srt.layers.moe.prism.profile import (
    DenseShape,
    Shape,
    dense_backends,
    dense_cold,
    dense_hot,
    dense_sweep,
    dense_warm,
)
# 패키지가 같은 이름의 **함수** cold_cpu를 재수출하므로 `from ... import cold_cpu`는
# 모듈이 아니라 함수를 준다 — 모듈을 잡으려면 import_module을 쓴다.
_P = "sglang.srt.layers.moe.prism.profile."
cold_mod = import_module(_P + "cold_cpu")
hot_mod = import_module(_P + "hot")
wc_mod = import_module(_P + "warm_cold")
from sglang.srt.layers.moe.prism.profile.common import SparseGemv, Timing, store_of
from sglang.srt.layers.moe.prism.profile.hot import ProjGemv

SHAPE = DenseShape(k=512, n=1024, slots=4)
T = Timing(us=10.0, min_us=9.0, max_us=11.0, p90_us=10.5, replays=8)


@pytest.fixture
def spy(monkeypatch):
    """밑단 3개를 가짜로 바꾸고 호출 인자를 받아 적는다."""
    calls = {}

    def fake_dense_gemv(k, n, **kw):
        calls["hot"] = dict(k=k, n=n, **kw)
        store = store_of(kw["dtype"])
        return ProjGemv(proj="dense", k_rows=k, n_cols=n, k_axis=k, m=kw["m"],
                        timing=T, dtype=store.name,
                        w_bytes_per_launch=store.store_bytes(kw["m"] * kw["topk"], k, n),
                        w_store_mb=round(store.store_bytes(kw["experts"], k, n) / 1e6, 1))

    def fake_warm(k, n, sparsity, **kw):
        calls["warm"] = dict(k=k, n=n, sparsity=sparsity, **kw)
        store = store_of(kw["dtype"])
        return SparseGemv(where="warm", k_rows=k, n_cols=n, sparsity=sparsity,
                          keep_frac=1.0 - sparsity,
                          dense_bytes=store.store_bytes(kw["m"] * kw["topk"], k, n),
                          timing=T)

    def fake_cold(k, n, sparsity, **kw):
        calls["cold"] = dict(k=k, n=n, sparsity=sparsity, **kw)
        store = store_of(kw["dtype"])
        return SparseGemv(where="cold", k_rows=k, n_cols=n, sparsity=sparsity,
                          keep_frac=1.0 if sparsity is None else 1.0 - sparsity,
                          dense_bytes=store.store_bytes(kw["topk"], k, n),
                          timing=T, tokens=kw["m"], numa_split=kw["numa_split"],
                          node_rows=(512, 512))

    monkeypatch.setattr(hot_mod, "dense_gemv", fake_dense_gemv)
    monkeypatch.setattr(wc_mod, "warm_sparse_gemv", fake_warm)
    monkeypatch.setattr(cold_mod, "cold_sparse_gemv", fake_cold)
    return calls


# ── 밑단에 무엇이 넘어가는가 ────────────────────────────────────────
def test_dense_lane_folds_to_one_weight_per_call(spy):
    """슬롯 = expert 풀, top_k = 1, cold는 proj=down (weight 하나짜리 진입점)."""
    dense_hot(SHAPE, dtype="bf16", device=0, m=8)
    dense_warm(SHAPE, 0.9, dtype="bf16", device=0)
    dense_cold(SHAPE, 0.5, dtype="bf16")

    assert spy["hot"]["experts"] == 4 and spy["hot"]["topk"] == 1 and spy["hot"]["m"] == 8
    assert spy["warm"]["experts"] == 4 and spy["warm"]["topk"] == 1
    assert spy["warm"]["m"] == 1, "warm은 decode 전용이라 토큰을 1로 고정한다"
    assert spy["cold"]["experts"] == 4 and spy["cold"]["topk"] == 1
    assert spy["cold"]["proj"] == "down", "gateup은 GEMV 2개라 dense part 하나와 다르다"
    for tier in ("hot", "warm", "cold"):
        assert spy[tier]["k"] == 512 and spy[tier]["n"] == 1024


def test_cold_dense_path_and_tokens(spy):
    """sparsity=None이 kt dense 경로이고, 그때만 m > 1이 허용된다."""
    r = dense_cold(SHAPE, None, dtype="bf16", m=64)
    assert spy["cold"]["sparsity"] is None and spy["cold"]["m"] == 64
    assert r.masked is False and r.keep_frac == 1.0 and r.tokens == 64
    assert r.us == 10.0 and r.us_per_token == pytest.approx(10.0 / 64)


# ── 검증이 하드웨어보다 먼저 난다 ──────────────────────────────────
def test_validation_fires_before_touching_the_backend(spy):
    with pytest.raises(ValueError, match="배수"):                      # K 정렬
        dense_hot(DenseShape(k=100, n=1024), dtype="fp8")             # fp8 k_align 128
    with pytest.raises(ValueError, match="decode 전용"):               # sparse + m>1
        dense_cold(SHAPE, 0.5, m=8)
    with pytest.raises(ValueError, match="sparsity=0.0"):              # warm에 None
        dense_warm(SHAPE, None)
    with pytest.raises(ValueError, match="배수"):                      # cold N 정렬
        dense_cold(DenseShape(k=512, n=300), 0.5, dtype="fp8")        # N_ALIGN 256
    assert spy == {}, "검증 실패인데 밑단이 불렸다"


def test_shape_rejects_degenerate_values():
    with pytest.raises(ValueError, match="양수"):
        DenseShape(k=0, n=128)
    with pytest.raises(ValueError, match="slots"):
        DenseShape(k=128, n=128, slots=0)


# ── 결과 필드 ──────────────────────────────────────────────────────
def test_weight_bytes_is_one_weight_across_tiers(spy):
    """티어끼리 GB/s를 비교하려면 분모가 같아야 한다 — 셋 다 [k, n] 한 벌."""
    one = store_of("bf16").store_bytes(1, 512, 1024)
    rs = [dense_hot(SHAPE, m=8), dense_warm(SHAPE, 0.75), dense_cold(SHAPE, 0.75)]
    assert [r.weight_bytes for r in rs] == [one] * 3
    # hot의 원본 회계(worklist m×topk)는 버리지 않고 raw에 남는다
    assert rs[0].raw["bytes_per_launch"] == one * 8
    # kept_bytes와 GB/s는 실현 keep 기준
    assert rs[1].keep_frac == pytest.approx(0.25)
    assert rs[1].kept_bytes == int(one * 0.25)
    # gbps 헬퍼가 소수 첫째 자리로 반올림하므로 그만큼만 허용한다
    assert rs[1].gbps == pytest.approx(one * 0.25 / 10.0 / 1000, abs=0.05)


def test_as_dict_is_json_ready(spy):
    import json

    d = dense_cold(SHAPE, 0.9, dtype="bf16", numa_map=[0, 1]).as_dict()
    json.dumps(d)                                    # 직렬화 가능
    assert d["tier"] == "cold" and d["k"] == 512 and d["n"] == 1024 and d["slots"] == 4
    assert d["masked"] is True and d["sparsity"] == 0.9
    assert d["backend"] == "kt_tile_k2_bf16"
    assert set(("us", "min_us", "max_us", "p90_us", "replays")) <= set(d)
    assert d["raw"]["node_rows"] == [512, 512]


# ── 스윕 ───────────────────────────────────────────────────────────
def test_sweep_shape_x_tier_x_sparsity(spy):
    shapes = [SHAPE, DenseShape(k=1024, n=1024, slots=2)]
    p = dense_sweep(shapes, (0.0, 0.9), tiers=("hot", "warm", "cold"), dtype="bf16")
    rows = p["results"]
    # hot은 shape당 1행, warm/cold는 shape × sparsity
    assert len(rows) == 2 * (1 + 2 + 2)
    assert sum(r["tier"] == "hot" for r in rows) == 2
    assert {r["sparsity"] for r in rows if r["tier"] == "warm"} == {0.0, 0.9}
    assert all("error" not in r for r in rows)
    assert p["params"]["dtype"] == "bf16" and len(p["shapes"]) == 2


def test_sweep_records_errors_instead_of_dying(spy):
    """한 칸이 틀려도 나머지는 살아야 한다 — 긴 스윕을 통째로 잃지 않는다."""
    bad = DenseShape(k=100, n=1024)          # fp8 K 정렬 위반
    p = dense_sweep([bad], (0.5,), tiers=("hot", "cold"), dtype="fp8")
    assert len(p["results"]) == 2
    assert all("error" in r and "ValueError" in r["error"] for r in p["results"])


def test_sweep_without_gpu_tiers_does_not_stamp_a_device(spy):
    p = dense_sweep([SHAPE], (0.5,), tiers=("cold",), dtype="bf16")
    assert "gpu" not in p["env"], "cold 전용 스윕이 CUDA를 초기화했다"


# ── 백엔드 선택 ────────────────────────────────────────────────────
def test_kernel_type_is_the_only_backend_axis_and_only_for_fp8(spy):
    """fp8만 변종을 갖는다 — k2/k1은 같은 스토어의 다른 마스크 단위, pt는 스토어 자체가 다르다."""
    shape = DenseShape(k=1024, n=1024, slots=2)          # fp8 k_align 128, N_ALIGN 256
    assert dense_cold(shape, 0.5, dtype="fp8").backend == "kt_tile_k2_fp8b128"
    r1 = dense_cold(shape, 0.5, dtype="fp8", kernel_type="k1")
    assert spy["cold"]["cpu_kernel"] == "kt_tile_k1_fp8b128" and r1.backend == "kt_tile_k1_fp8b128"
    assert r1.store == "fp8" and r1.kernel_type == "k1"
    rp = dense_cold(shape, 0.5, dtype="fp8", kernel_type="pt")
    assert rp.store == "fp8pt", "pt는 배율 기하가 달라 스토어가 갈린다"
    assert rp.backend == "kt_tile_k2_fp8pt" and spy["cold"]["dtype"].name == "fp8pt"


def test_kernel_type_rejected_for_dtypes_without_variants(spy):
    """bf16/mxfp4는 커널이 하나뿐이라 인자 자체를 받지 않는다 (사용자 결정)."""
    for dtype in ("bf16", "mxfp4"):
        with pytest.raises(ValueError, match="kernel_type은 dtype"):
            dense_cold(DenseShape(k=512, n=512), 0.5, dtype=dtype, kernel_type="k2")
    with pytest.raises(ValueError, match="dtype='fp8', kernel_type='pt'"):
        dense_cold(DenseShape(k=512, n=512), 0.5, dtype="fp8pt", kernel_type="pt")
    with pytest.raises(ValueError, match="k1', 'k2', 'pt"):
        dense_cold(DenseShape(k=1024, n=1024), 0.5, dtype="fp8", kernel_type="k3")
    assert spy == {}


def test_pt_relaxes_k_alignment_from_128_to_32(spy):
    """pt는 배율 블록이 없어 K 128 제약이 풀린다 — 같은 shape이 k2에서는 죽는다."""
    shape = DenseShape(k=1056, n=1024)                   # 32의 배수, 128의 배수 아님
    dense_cold(shape, 0.5, dtype="fp8", kernel_type="pt")
    assert spy["cold"]["k"] == 1056
    with pytest.raises(ValueError, match="128의 배수"):
        dense_cold(shape, 0.5, dtype="fp8", kernel_type="k2")


def test_warm_passes_per_k_through_for_k1(spy):
    """k1 warm은 per-k 진입점(*_sparsek1)으로 간다 — 마스크가 x 레벨로 실현된다."""
    shape = DenseShape(k=1024, n=1024, slots=2)
    r = dense_warm(shape, 0.9, dtype="fp8", kernel_type="k1", device=0)
    assert spy["warm"]["per_k"] is True
    assert r.backend == "gemv_fp8_indexed_pinned_sparsek1"
    # 나머지 변종은 페어 마스크 그대로
    for kt, entry in (("k2", "gemv_fp8_indexed_pinned_sparse"),
                      ("pt", "gemv_fp8pt_indexed_pinned_sparse")):
        assert dense_warm(shape, 0.9, dtype="fp8", kernel_type=kt).backend == entry
        assert spy["warm"]["per_k"] is False
    assert dense_warm(DenseShape(k=512, n=512), 0.9, dtype="bf16").backend == \
        "gemv_worklist_indexed_pinned_sparse"
    assert spy["warm"]["per_k"] is False


def test_hot_ignores_the_mask_variant(spy):
    """hot은 마스크를 안 쓰므로 k1과 k2가 같은 커널이다."""
    shape = DenseShape(k=1024, n=1024, slots=2)
    a = dense_hot(shape, dtype="fp8", kernel_type="k1")
    b = dense_hot(shape, dtype="fp8", kernel_type="k2")
    assert a.backend == b.backend == "gemv_fp8_indexed" and a.store == b.store == "fp8"
    assert dense_hot(shape, dtype="fp8", kernel_type="pt").backend == "gemv_fp8pt_indexed"


def test_backends_rows_enumerate_every_selectable_combo():
    """harness가 그대로 순회할 수 있게 (dtype, kernel_type) 행으로 준다."""
    rows = dense_backends()
    assert [(r["dtype"], r["kernel_type"]) for r in rows] == [
        ("bf16", None), ("mxfp4", None), ("fp8", "k2"), ("fp8", "k1"), ("fp8", "pt")]
    by = {(r["dtype"], r["kernel_type"]): r for r in rows}
    assert by[("fp8", "pt")]["store"] == "fp8pt" and by[("fp8", "pt")]["k_align"] == 32
    assert by[("fp8", "k2")]["k_align"] == 128
    assert all(r["warm"] for r in rows), "다섯 조합 모두 warm이 있다"
    assert by[("fp8", "k1")]["warm_gpu"] == "gemv_fp8_indexed_pinned_sparsek1"
    assert by[("bf16", None)]["has_vec"] is True
    for r in rows:
        assert r["hot_gpu"].endswith("_indexed")
        assert r["warm_gpu"].endswith("_pinned_sparse" if r["kernel_type"] != "k1"
                                      else "_pinned_sparsek1")


# ── MoE Shape 과의 변환 ─────────────────────────────────────────────
def test_from_moe_maps_axes_per_proj():
    """gate/up은 [hidden, inter], down은 [inter, hidden] — 축이 뒤집힌다."""
    S = Shape(experts=128, topk=8, hidden=2048, inter=768)
    assert DenseShape.from_moe(S, "gate") == DenseShape(k=2048, n=768, slots=128)
    assert DenseShape.from_moe(S, "up") == DenseShape.from_moe(S, "gate")
    assert DenseShape.from_moe(S, "down") == DenseShape(k=768, n=2048, slots=128)
    assert DenseShape.from_moe(S, "down", slots=88).slots == 88   # 층 수로 갈아끼우기
    with pytest.raises(ValueError, match="gate|up|down"):
        DenseShape.from_moe(S, "gateup")


def test_bare_moe_shape_is_rejected_with_both_conversions(spy):
    """축을 조용히 고르면 harness가 틀린 숫자를 모은다 — 어느 proj인지 물어본다."""
    S = Shape(experts=128, topk=8, hidden=2048, inter=768)
    for call in (lambda: dense_cold(S, 0.9),
                 lambda: dense_warm(S, 0.9),
                 lambda: dense_hot(S)):
        with pytest.raises(TypeError, match="from_moe"):
            call()
    with pytest.raises(TypeError, match="DenseShape이어야"):
        dense_cold((512, 1024), 0.9)
    assert spy == {}


def test_from_moe_shapes_flow_through_the_wrapper(spy):
    """변환한 shape은 평범한 DenseShape이라 그대로 돈다."""
    S = Shape(experts=8, topk=2, hidden=1024, inter=512)
    r = dense_cold(DenseShape.from_moe(S, "down"), 0.5, dtype="bf16")
    assert (spy["cold"]["k"], spy["cold"]["n"], spy["cold"]["experts"]) == (512, 1024, 8)
    assert r.shape.k == 512 and r.shape.n == 1024
