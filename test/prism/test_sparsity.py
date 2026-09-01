"""입력기반 sparsity 테스트 — cold 마스킹 end-to-end.

warm/hot은 dense로 계산하고, sparsity 수식 전체(예산 → s → 격자 조회 →
페어 판정)는 kt에 있다. 그래서 prism 쪽에서 검증할 수 있는 것은 두 가지다:

1. **cold 밴드만** 마스킹된 결과 — full-K가 아니라 cold 구간만 마스킹한
   레퍼런스와 대조한다. warm이 dense라는 것이 여기서 검증된다.
2. 자산 → kt 주입의 정합성 (wn² 여부, 밴드 절단 offset).

threshold를 특정 값으로 고정하려면 calib 자산의 thr 곡선을 그 값으로 채운다
— 격자 인덱스가 무엇이든 조회 결과가 같아지므로 s를 우회할 수 있다.
"""

import math

import pytest
import torch

from sglang.srt.layers.moe.prism.plan import PAIR_GROUP, Proj

NG, GRID, PMAX, RENORM = 201, 0.005, 0.9, 3


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
from sglang.srt.layers.moe.prism.tiers import SPARSE_TIERS  # noqa: E402
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
    elif kind == "all_hot":
        # HOT만 — SPARSE_TIERS가 hot을 고르지 않으므로 thr이 무효여야 한다.
        gate_up = _proj_entry([[0, H, "hot"]], I, sparse)
        down = _proj_entry([[0, I, "hot"]], H, sparse)
    elif kind == "all_warm":
        # WARM만 — cold가 없어도 threshold가 걸린다 (2026-08-26 계약 개정).
        gate_up = _proj_entry([[0, H, "warm"]], I, sparse)
        down = _proj_entry([[0, I, "warm"]], H, sparse)
    else:
        raise ValueError(f"unknown plan kind {kind!r}")
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
    # thr_fill은 스칼라이거나 {Proj: 값} 매핑이다 — proj마다 점수 스케일이
    # 다르므로(down의 x는 act) 하나로 뭉갤 수 없는 경우가 있다.
    fill = {p: thr_fill for p in Proj} if isinstance(thr_fill, (int, float)) \
        else dict(thr_fill)
    for key, proj in (("tg2l", Proj.GATE), ("tu2l", Proj.UP), ("td2l", Proj.DOWN)):
        blob[key] = torch.full((1, E, NG), float(fill[proj]))
    tag = "_".join(f"{float(fill[p]):.6g}" for p in Proj)
    path = tmp_path / f"calib_{tag}.pt"
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
    prepared = prepare_layer_weights(0, w13, w2, plan, calib=calib,
                                     device=torch.device("cuda"))
    ep = plan.expert(0, 0)
    has_cold = any(ep.proj(p).has_tier(Tier.COLD) for p in Proj)
    cold = None
    if has_cold:
        from sglang.srt.layers.moe.prism.cold_backend import KtColdBackend
        from sglang.srt.layers.moe.prism.numa import numa_node_count

        cold = KtColdBackend(plan, max_tokens=MAX_TOKENS,
                             num_numa_nodes=numa_node_count())
        cold.load_layer(0, prepared.cold, prepared.thr)
    spec = ResourceSpec.from_plan(plan, max_tokens=MAX_TOKENS,
                                  device=torch.device("cuda"))
    ex = PrismExecutor(plan, ExecutionResources(spec), cold)
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


# 마스킹하는 티어 = cold(kt가 항상 마스킹) + GPU 쪽 SPARSE_TIERS. 정책의
# 정의점은 tiers.SPARSE_TIERS 하나이고 여기서 파생한다 — 두 곳에 적으면
# 정책을 바꿀 때 테스트가 조용히 옛 계약을 검사한다.
_MASKED_TIERS = frozenset({Tier.COLD}) | SPARSE_TIERS


def _cold_span(ep, proj):
    for b in ep.proj(proj).bands:
        if b.tier is Tier.COLD:
            return b.start, b.end
    return None


def _masked_span(ep, proj):
    """마스킹되는 티어들이 덮는 K 구간 [start, end).

    2026-08-26부터 WARM도 cold와 **같은 threshold**로 마스킹하므로 대조의
    단위가 "cold 밴드"에서 "마스킹 티어의 합집합"으로 넓어졌다. HOT은 dense라
    (tiers.SPARSE_TIERS) 여기 들어가지 않는다.

    합집합을 구간 하나로 돌려주는 것은 이 테스트의 plan들이 마스킹 티어를
    연속으로 배치하기 때문이다 (warm 앞, cold 뒤). 불연속 배치를 쓰게 되면
    구간 리스트로 넓혀야 한다 — 그때는 여기가 먼저 틀린다.
    """
    spans = [(b.start, b.end) for b in ep.proj(proj).bands
             if b.tier in _MASKED_TIERS]
    if not spans:
        return None
    lo, hi = min(s for s, _ in spans), max(e for _, e in spans)
    covered = sum(e - s for s, e in spans)
    assert covered == hi - lo, (
        f"{proj.value}: 마스킹 티어가 [{lo}, {hi})를 연속으로 덮지 않는다 "
        f"(덮은 행 {covered}) — _masked_span을 구간 리스트로 넓혀야 한다"
    )
    return lo, hi


def _cold_imp(vec, proj, e, span, blob):
    s, en = span
    a = blob[_WN_KEY[proj]][0, e][s:en] ** 2
    c = blob[_DOT_KEY[proj]][0, e][s // 2 : en // 2]
    return ref_pair_importance(vec[s:en], a, c)


def probe_thresholds(x, ids, w13, w2, plan, blob, q=0.5):
    """**마스킹 구간**의 실제 imp 분포 분위수 → proj별 threshold.

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
                span = _masked_span(ep, proj)
                if span:
                    vals[proj].append(_cold_imp(xf[m], proj, e, span, blob))
            act = torch.nn.functional.silu(xf[m] @ gate_w[e].t()) * (
                xf[m] @ up_w[e].t())
            span = _masked_span(ep, Proj.DOWN)
            if span:
                vals[Proj.DOWN].append(_cold_imp(act, Proj.DOWN, e, span, blob))
    return {p: float(torch.cat(v).quantile(q)) for p, v in vals.items() if v}


def masked_reference(x, ids, w, w13, w2, plan, blob, thr):
    """마스킹 티어(warm + cold)를 마스킹한 fp32 레퍼런스.

    2026-08-26 이전에는 cold 밴드만 마스킹했고, 그것이 "warm은 dense"의
    검출기였다. warm이 같은 threshold로 마스킹하게 되면서 대조 구간이
    `_masked_span`으로 넓어졌다 — HOT이 있으면 그 구간은 여전히 dense다.
    """
    gate_w, up_w = w13[:, :I, :].float(), w13[:, I:, :].float()
    down_w = w2.float()
    ep = plan.expert(0, 0)

    def apply_mask(vec, proj, e, t):
        span = _masked_span(ep, proj)
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
    prepared = prepare_layer_weights(0, w13, w2, plan, calib=calib,
                                     device=torch.device("cuda"))
    cold = KtColdBackend(plan, max_tokens=MAX_TOKENS,
                         num_numa_nodes=numa_node_count())
    cold.load_layer(0, prepared.cold, prepared.thr)

    tables = cold._wrappers[0]._sparsity_tables
    assert set(tables) == {
        "gate_wn_sq", "gate_pair_dot", "up_wn_sq", "up_pair_dot",
        "down_wn_sq", "down_pair_dot", "thr_gate", "thr_up", "thr_down",
    }
    # thr 곡선은 밴드와 무관하므로 절단되지 않고 [E, ng]로 그대로 간다.
    assert tables["thr_gate"].shape == (E, NG)
    assert torch.equal(tables["thr_down"], blob["td2l"][0])
    # 점수 재료는 weight와 같은 오프셋의 **flat**이다 (계약 ③) — expert 블록이
    # 이어 붙으므로 [E, rows]를 평탄화한 것과 같아야 한다.
    wn_cold = blob["wn_g"][0][:, WARM_ROWS:H]
    assert torch.allclose(tables["gate_wn_sq"], (wn_cold * wn_cold).reshape(-1), atol=0)
    assert torch.equal(tables["down_pair_dot"],
                       blob["cd"][0][:, WARM_ROWS // 2 :].reshape(-1))


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
def test_warm_masks_with_the_same_threshold(tmp_path):
    """warm도 cold와 **같은 threshold**로 마스킹한다 (2026-08-26 계약 개정).

    thr=∞면 warm 페어가 전부 죽으므로 all-warm plan의 출력은 정확히 0이다.
    이전 계약(warm은 dense)에서는 같은 입력이 dense와 일치했다 — 그 테스트가
    이 구현으로 바뀌면서 발화했고, 그것이 warm 마스킹이 실제로 걸린다는 증거다.
    """
    w13, w2 = make_exec_weights()
    x, ids, w = make_exec_inputs(1, seed=32)
    calib, _ = make_exec_calib(tmp_path, 1e9)
    masked = run_exec(build_exec(make_exec_plan("all_warm"), w13, w2, calib),
                      x, ids, w)
    assert torch.count_nonzero(masked) == 0, (
        "thr=∞인데 살아남은 원소가 있다 — warm 마스크가 걸리지 않았다"
    )


@cuda_required
def test_warm_zero_threshold_is_bit_identical_to_dense(tmp_path):
    """thr=0이면 warm sparse 커널이 dense와 **비트일치**해야 한다.

    이것이 커널에서 압축 리스트를 쓰지 않고 dense 루프 모양(스레드 y가 행
    y, y+NY, …)을 유지한 이유의 검증이다. 생존 페어를 압축하면 누산 순서가
    달라져 전량 keep에서도 마지막 비트가 갈리고, 순서 재현은 계약 ⑤의 요구다.
    tolerance 비교로는 이 회귀가 안 잡힌다 — `torch.equal`이어야 한다.
    """
    w13, w2 = make_exec_weights()
    x, ids, w = make_exec_inputs(1, seed=32)
    dense = run_exec(build_exec(make_exec_plan("all_warm", sparse=False), w13, w2),
                     x, ids, w)
    calib, _ = make_exec_calib(tmp_path, 0.0)
    masked = run_exec(build_exec(make_exec_plan("all_warm"), w13, w2, calib),
                      x, ids, w)
    assert torch.equal(masked, dense)


@cuda_required
def test_hot_stays_dense(tmp_path):
    """HOT은 마스킹하지 않는다 (tiers.SPARSE_TIERS의 정책).

    thr=∞여도 all-hot plan은 dense와 일치한다. 이 정책을 뒤집으면(hot을
    SPARSE_TIERS에 넣으면) 이 테스트가 발화해야 한다 — 정책 변경이 조용히
    일어나지 않게 하는 것이 목적이다.
    """
    assert Tier.HOT not in SPARSE_TIERS
    w13, w2 = make_exec_weights()
    x, ids, w = make_exec_inputs(1, seed=32)
    dense = run_exec(build_exec(make_exec_plan("all_hot", sparse=False), w13, w2),
                     x, ids, w)
    calib, _ = make_exec_calib(tmp_path, 1e9)
    masked = run_exec(build_exec(make_exec_plan("all_hot"), w13, w2, calib),
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

    # thr 곡선을 probe한 값으로 채운 자산으로 다시 만든다 — wn/pair_dot은
    # 같은 seed라 동일하므로 분위수가 그대로 유효하다.
    #
    # **proj별로** 채우는 것이 중요하다. 세 값을 평균 하나로 뭉개면 down이
    # 무너진다: down의 x는 act(=silu(g)·u)라 gate/up의 hidden보다 스케일이
    # 훨씬 작고, gate/up 쪽으로 끌려간 threshold가 down 페어를 전량 죽인다.
    # warm이 dense였을 때는 그 피해가 cold 밴드에 갇혀 있어 드러나지 않았다.
    picked = probe_thresholds(x, ids, w13, w2, plan, blob, q=0.5)
    calib2, blob2 = make_exec_calib(tmp_path, picked)
    assert torch.equal(blob2["wn_g"], blob["wn_g"])  # 자산 재생성이 재현적인지
    ex = build_exec(plan, w13, w2, calib2)
    thr = {p: torch.full((1, K), picked[p]) for p in Proj}

    out = run_exec(ex, x, ids, w)
    ref = masked_reference(x, ids, w, w13, w2, plan, blob, thr)
    d = rel_diff(out, ref)
    print(f"warm+cold masked: rel diff = {d:.6f}")
    # 마스크가 실제로 뭔가 죽였는지 — 전량 통과/차단이면 0 vs 0의 무의미한
    # 대조가 되므로 이중 방어한다.
    assert torch.count_nonzero(out) > 0, "전량 차단 — threshold가 너무 크다"
    assert d < 0.03
