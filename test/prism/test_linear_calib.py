"""dense calib 어댑터 — 자산 로딩·gather·전부-0 게이트 (GPU 불필요).

두 축을 본다:

  1. **gather가 weight와 같은 순서를 만드는가.** 마스크 비트 ↔ packed 타일 대응이
     유지되려면 점수의 행 순서가 스토어의 행 순서와 같아야 한다. 어긋나면 엉뚱한
     채널이 죽는데, 출력이 "그럴듯하게 나쁜" 값이라 눈에 안 띈다.

  2. **캘리브 안 된 (층, projection)을 잡는가.** 자산이 그 자리를 안 덮으면 wn/thr이
     0이고 `imp = 0 >= thr = 0` 이라 **전부 살아남는다** — 출력은 정확하고 sparsity만
     0이 된다. 정확도 테스트도 계약 테스트도 서버도 다 통과하고 벤치 결론만 틀린다.

실물 자산(`assets/*.pt`)이 있으면 그걸로도 돈다. 없으면 합성 자산으로 같은 불변식을
확인한다 — CI가 2.6 GB 자산에 의존하면 안 된다.
"""

import hashlib
from pathlib import Path

import pytest
import torch

from sglang.srt.layers.prism.geometry import PAIR_GROUP, PlanError
from sglang.srt.layers.prism.linear.calib import LinearCalibShard, LinearCalibTables
from sglang.srt.layers.prism.linear.plan import CalibRef, SparsitySpec, parse_plan

L, K, NG = 4, 64, 201
KEYS = ("g", "u", "d", "q", "k", "v", "o")
ASSETS = Path(__file__).resolve().parents[2].parent / "assets"


def _tables(dead_layers=(), keys=KEYS, k=K):
    """합성 자산. dead_layers의 층은 전부 0 — 캘리브 안 된 자리를 흉내낸다."""
    torch.manual_seed(0)
    t = {}
    for key in keys:
        wn = torch.rand(L, 1, k) + 0.5
        c = torch.randn(L, 1, k // PAIR_GROUP)
        thr = torch.rand(L, 1, NG)
        for d in dead_layers:
            wn[d], c[d], thr[d] = 0, 0, 0
        t[f"wn_{key}"], t[f"c{key}"], t[f"t{key}2l"] = wn, c, thr
    return t


@pytest.fixture
def cal():
    return LinearCalibTables(_tables(), "k2wl2", NG)


# ── gather: 스토어와 같은 순서 ─────────────────────────────────────────────


def test_gather_matches_source_rows(cal):
    rows = torch.tensor([4, 5, 10, 11, 20, 21], dtype=torch.int64).to(torch.uint16)
    sh = cal.gather(0, "q", rows)
    src = cal._table("wn_q", "")[0, 0]
    torch.testing.assert_close(sh.wn, src[rows.to(torch.int64)])
    assert sh.pair_dot.numel() == rows.numel() // PAIR_GROUP


def test_pair_dot_follows_the_pairs(cal):
    """페어 내적은 **원본 페어 id**로 모아야 한다 — 순서만 맞추면 다른 페어를 집는다."""
    rows = torch.tensor([8, 9, 2, 3], dtype=torch.int64).to(torch.uint16)
    sh = cal.gather(0, "q", rows)
    src = cal._table("cq", "")[0, 0]
    torch.testing.assert_close(sh.pair_dot, src[torch.tensor([4, 1])])  # 페어 8//2, 2//2


def test_wn_sq_is_the_square(cal):
    sh = cal.gather(0, "g", torch.arange(8, dtype=torch.int64).to(torch.uint16))
    torch.testing.assert_close(sh.wn_sq, sh.wn * sh.wn)


def test_padding_stays_zero(cal):
    """cold는 타일 경계까지 패딩된다 — 그 뒤는 0이어야 kt가 tail을 끌 수 있다."""
    idx = torch.cat([torch.arange(6), torch.zeros(10, dtype=torch.int64)]).to(torch.uint16)
    sh = cal.gather(0, "d", idx, real_rows=6)
    assert sh.wn.numel() == 16
    assert bool((sh.wn[6:] == 0).all()) and bool((sh.pair_dot[3:] == 0).all())
    assert int((sh.wn[:6] != 0).sum()) == 6


def test_odd_row_count_dies(cal):
    with pytest.raises(PlanError, match="PAIR_GROUP"):
        cal.gather(0, "q", torch.arange(5, dtype=torch.int64).to(torch.uint16))


# ── 전부-0 게이트 ──────────────────────────────────────────────────────────


def test_uncalibrated_layer_dies():
    """이 검사가 없으면 마스킹이 조용히 사라지고 성능만 달라진다."""
    cal = LinearCalibTables(_tables(dead_layers=(1, 2)), "k2wl2", NG)
    cal.check(0, "o", K, "살아있는 층")          # 통과
    for dead in (1, 2):
        with pytest.raises(PlanError, match="all zeros at layer"):
            cal.check(dead, "o", K, f"층 {dead}")


def test_zero_masks_everything_in_theory():
    """게이트가 왜 필요한지의 근거 — 0 테이블이면 keep이 100%가 된다."""
    cal = LinearCalibTables(_tables(dead_layers=(1,)), "k2wl2", NG)
    wn = cal._table("wn_o", "")[1, 0]
    thr = cal.thr(1, "o")
    x = torch.randn(K).double()
    a = (wn.double() ** 2)
    imp = (a[0::2] * x[0::2] ** 2 + a[1::2] * x[1::2] ** 2).sqrt()
    keep = imp >= float(thr[100])
    assert float(keep.float().mean()) == 1.0      # 전부 산다 = sparsity 0


# ── 이름·치수 방어 ─────────────────────────────────────────────────────────


def test_missing_table_dies(cal):
    with pytest.raises(PlanError, match="has no table 'wn_zz'"):
        cal.check(0, "zz", K, "w")


def test_k_mismatch_dies(cal):
    with pytest.raises(PlanError, match=r"K=64 but expected 66"):
        cal.check(0, "q", K + 2, "w")


def test_layer_out_of_range_dies(cal):
    with pytest.raises(PlanError, match="out of calib range"):
        cal.check(L, "q", K, "w")


def test_moe_shaped_asset_dies():
    """expert 축이 1이 아니면 dense 자산이 아니다."""
    t = {f"wn_g": torch.rand(L, 8, K), "cg": torch.rand(L, 8, K // 2),
         "tg2l": torch.rand(L, 8, NG)}
    with pytest.raises(PlanError, match=r"expected \[L, 1"):
        LinearCalibTables(t, "k2wl2", NG).check(0, "g", K, "w")


def test_unknown_score_dies():
    with pytest.raises(PlanError, match="unknown sparsity score"):
        LinearCalibTables({}, "l2", NG)


def test_k1_score_uses_unsuffixed_tables():
    """k1은 `tg`(접미사 없음)를 본다. 아직 커널에 배선하지 않았지만 이름은 맞춰 둔다."""
    t = _tables()
    t["tg"] = t.pop("tg2l")
    cal = LinearCalibTables(t, "k1", NG)
    assert cal.thr(0, "g").numel() == NG


# ── 실물 자산 ──────────────────────────────────────────────────────────────


def _load(name):
    p = ASSETS / name
    if not p.is_file():
        pytest.skip(f"{p} 없음")
    spec = SparsitySpec(
        score="k2wl2",
        calib=CalibRef(str(p), hashlib.sha256(p.read_bytes()).hexdigest()),
        pmax=0.9, grid=0.005, ng=201, renorm_it=3,
    )
    return LinearCalibTables.load(spec)


def test_real_muse_glimmer_all_seven_keys():
    """Muse-Glimmer는 52층 전부 표준 attention이라 7개 키가 모든 층에서 살아 있다."""
    cal = _load("muse_glimer.pt")
    H, I, O = 6656, 19968, 4096
    for layer in (0, 25, 51):
        for key, k in (("g", H), ("u", H), ("d", I),
                       ("q", H), ("k", H), ("v", H), ("o", O)):
            cal.check(layer, key, k, f"layer {layer} [{key}]")


def test_real_qwen38_linear_attn_layers_are_uncalibrated():
    """Qwen3.8은 full_attention 16층만 q/k/v/o가 있다 — 나머지 48층은 잡혀야 한다."""
    cal = _load("qwen38_27b.pt")
    H, O = 5120, 6144
    cal.check(3, "o", O, "full_attn 층 3")            # attn_layers = [3, 7, 11, …]
    cal.check(3, "q", H, "full_attn 층 3")
    for dead in (0, 1, 2, 4):
        with pytest.raises(PlanError, match="all zeros at layer"):
            cal.check(dead, "o", O, f"linear_attn 층 {dead}")
    # MLP는 전 층에서 살아 있다
    for layer in (0, 3, 63):
        cal.check(layer, "g", H, f"layer {layer}")


def test_digest_mismatch_dies():
    p = ASSETS / "muse_glimer.pt"
    if not p.is_file():
        pytest.skip("자산 없음")
    spec = SparsitySpec(score="k2wl2", calib=CalibRef(str(p), "0" * 64),
                        pmax=0.9, grid=0.005, ng=201, renorm_it=3)
    with pytest.raises(PlanError, match="digest mismatch"):
        LinearCalibTables.load(spec)

# ── check_plan: plan이 마스킹하겠다고 한 자리를 전부 대조한다 ──────────────


def _plan_with_sparsity(calib_key="g", sparse=True, k=K):
    """조각 이름이 있는 plan — 오류 메시지가 그 이름을 쓴다 (회귀: part.half)."""
    return parse_plan({
        "schema_version": 1, "model_id": "t",
        "dims": {"num_layers": L, "dtype": "bfloat16"},
        "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_amx_bf16"},
        "sparsity": {"score": "k2wl2",
                     "calib": {"path": "/x", "sha256": "0" * 64},
                     "pmax": 0.9, "grid": 0.005, "ng": NG, "renorm_it": 3},
        "projs": {"mlp.gate_up_proj": {
            "k": k, "n": 2 * k,
            "parts": [{"name": "gate", "n": k, "bands": [[0, k, "warm"]],
                       "calib": calib_key, "p": 0.5, "lambda": 0.0,
                       "sparse": sparse},
                      {"name": "up", "n": k, "bands": [[0, k, "warm"]],
                       "calib": "u", "p": 0.5, "lambda": 0.0,
                       "sparse": sparse}]}},
    })


def test_check_plan_passes_on_calibrated_plan(cal):
    cal.check_plan(_plan_with_sparsity())


def test_check_plan_names_the_part_in_the_error():
    """조각 이름이 메시지에 들어가야 한다 — 어느 절반이 문제인지가 진단의 전부다.

    회귀: 이 경로가 `part.half`(존재하지 않는 속성)를 읽어 AttributeError로
    죽었다. 아무도 `check_plan`을 부르지 않아 드러나지 않았다.
    """
    cal = LinearCalibTables(_tables(dead_layers=(2,)), "k2wl2", NG)
    with pytest.raises(PlanError, match=r"\[gate\]"):
        cal.check_plan(_plan_with_sparsity())


def test_check_plan_non_strict_reports_instead_of_dying():
    """plan은 어느 층에 어느 모듈이 실재하는지 모른다 — 전수 대조로 죽이면 안 된다.

    Qwen3.8-27B에서 `make_plan`이 `self_attn.*`를 64층 전부에 선언하지만 그 모듈은
    full_attention 16층에만 있고, calib이 나머지 48층을 0으로 둔 것은 **정확한
    사실**이다. strict로 죽이면 존재하지 않을 좌표가 런을 막는다.
    """
    cal = LinearCalibTables(_tables(dead_layers=(2,)), "k2wl2", NG)
    bad = cal.check_plan(_plan_with_sparsity(), strict=False)
    assert bad and any("[gate]" in b for b in bad), bad
    assert all("layer 2" in b for b in bad), f"죽은 층만 보고해야 한다: {bad}"


def test_check_plan_non_strict_is_empty_when_covered(cal):
    assert cal.check_plan(_plan_with_sparsity(), strict=False) == []


def test_check_plan_skips_parts_that_opt_out():
    """`sparse: false`인 조각은 대조하지 않는다 — calib이 안 덮는 자리를 빼는 길이다."""
    cal = LinearCalibTables(_tables(dead_layers=tuple(range(L))), "k2wl2", NG)
    cal.check_plan(_plan_with_sparsity(sparse=False))
