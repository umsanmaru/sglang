"""expert별로 K 기하가 다른 plan에서 rejoin이 맞는가 (컴파일 plan NaN 조사).

컴파일 plan(qwen38_next_exp16gb_block{2,32})은 expert마다 hot/warm/cold 행 수가
다르고, **expert 0에는 hot이 없다**. 손 plan(균일·expert 0에도 hot)은 정상이므로
비균일 기하 자체가 rejoin을 깨는지를 소형 shape에서 격리한다.

레퍼런스는 test_executor.moe_reference와 같은 dense fp32 full-MoE다 — sparsity를
끄고(schema v1) 기하만 본다.
"""

import pytest
import torch

pytest.importorskip("kt_kernel", reason="kt_kernel required")
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from sglang.srt.layers.moe.prism.plan import parse_plan, validate_static

from test_executor import (  # noqa: E402
    DIMS,
    build_executor,
    make_inputs,
    make_weights,
    moe_reference,
    rel_diff,
    run,
)

H = DIMS["hidden_size"]        # 256  (gate/up의 K)
I = DIMS["intermediate_size"]  # 128  (down의 K)
E = DIMS["num_experts"]


def _shards(N):
    half = (N // 2 // 32) * 32
    return [[0, 0, half], [1, half, N]]


def _contig(K, hot, warm):
    """[hot | warm | cold] — hot/warm이 0이면 그 밴드를 안 만든다."""
    out = []
    if hot:
        out.append([0, hot, "hot"])
    if warm:
        out.append([hot, hot + warm, "warm"])
    if hot + warm < K:
        out.append([hot + warm, K, "cold"])
    return out


def _interleave(K, blk, pattern):
    """block32식 인터리브: blk 행 단위로 pattern을 순환시킨다.
    인접한 같은 티어 블록은 하나의 밴드로 합친다 (컴파일 컨버터와 같은 규칙)."""
    out = []
    for i, start in enumerate(range(0, K, blk)):
        tier = pattern[i % len(pattern)]
        end = min(start + blk, K)
        if out and out[-1][2] == tier:
            out[-1][1] = end
        else:
            out.append([start, end, tier])
    return out


def _entry(bands, N):
    e = {"bands": bands}
    if any(b[2] == "cold" for b in bands):
        e["cold_shards"] = _shards(N)
    return e


# expert별 hot 행 수 — expert 0은 0행 (컴파일 plan의 특징)
_HOT_GU = [0, 32, 64, 96, 128, 0, 32, 64]
_HOT_DN = [0, 32, 64, 0, 32, 64, 0, 32]
_WARM = 32

# 인터리브 패턴도 expert마다 다르게 (expert 0에는 hot이 없다)
_PAT = [
    ["cold", "warm"],
    ["cold", "hot", "warm", "cold"],
    ["hot", "cold", "cold", "warm"],
    ["cold", "cold", "warm", "hot"],
    ["warm", "cold", "hot", "cold"],
    ["cold", "warm", "cold", "cold"],
    ["hot", "warm", "cold", "cold"],
    ["cold", "cold", "hot", "warm"],
]


def make_plan(kind, gpu_warm="torch_bmm", cpu_cold="kt_amx_bf16"):
    overrides = []
    for e in range(E):
        if kind == "uniform":            # 대조군: 손 plan과 같은 균일 기하
            gu = _contig(H, 64, _WARM)
            dn = _contig(I, 32, _WARM)
        elif kind == "nonuniform":       # expert마다 다른 hot 크기, expert 0은 hot 없음
            gu = _contig(H, _HOT_GU[e], _WARM)
            dn = _contig(I, _HOT_DN[e], _WARM)
        elif kind == "nonuniform_nowarm":  # (2) warm 분리: 비균일이되 warm 없음
            gu = _contig(H, _HOT_GU[e], 0)
            dn = _contig(I, _HOT_DN[e], 0)
        elif kind == "interleave":       # (3) 밴드 수: block32식 인터리브 + 비균일
            gu = _interleave(H, 32, _PAT[e])
            dn = _interleave(I, 32, _PAT[e])
        else:
            raise ValueError(kind)
        overrides.append(
            {"layer": 0, "expert": e,
             "gate": _entry(list(gu), I), "up": _entry(list(gu), I),
             "down": _entry(list(dn), H)}
        )
    raw = {
        "schema_version": 1,
        "model_id": "test/tiny",
        "dims": dict(DIMS),
        "kernels": {"gpu_warm": gpu_warm, "cpu_cold": cpu_cold},
        "overrides": overrides,
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


@cuda_required
@pytest.mark.parametrize("qlen", [1, 16])
@pytest.mark.parametrize(
    "kind", ["uniform", "nonuniform", "nonuniform_nowarm", "interleave"]
)
def test_nonuniform_matches_reference(kind, qlen):
    plan = make_plan(kind)
    w13, w2 = make_weights()
    ex = build_executor(plan, w13, w2)
    x, ids, w = make_inputs(qlen, seed=10 + qlen)
    out = run(ex, x, ids, w)
    ref = moe_reference(x, ids, w, w13, w2)
    d = rel_diff(out, ref)
    finite = bool(torch.isfinite(out).all())
    print(f"[nonuniform] {kind} qlen={qlen}: rel diff = {d:.6f} finite={finite}")
    assert finite, f"{kind} qlen={qlen}: non-finite output"
    assert d < 0.03, f"{kind} qlen={qlen}: diff {d:.6f}"


# 실전 커널 조합 — 깨지는 서버 plan과 같다 (gemv_worklist + kt_tile_k2_bf16).
@cuda_required
@pytest.mark.parametrize("qlen", [1, 16])
@pytest.mark.parametrize(
    "kind", ["uniform", "nonuniform", "nonuniform_nowarm", "interleave"]
)
def test_nonuniform_real_kernels(kind, qlen):
    plan = make_plan(kind, gpu_warm="gemv_worklist", cpu_cold="kt_tile_k2_bf16")
    w13, w2 = make_weights()
    ex = build_executor(plan, w13, w2)
    x, ids, w = make_inputs(qlen, seed=10 + qlen)
    out = run(ex, x, ids, w)
    ref = moe_reference(x, ids, w, w13, w2)
    d = rel_diff(out, ref)
    finite = bool(torch.isfinite(out).all())
    print(f"[worklist] {kind} qlen={qlen}: rel diff = {d:.6f} finite={finite}")
    assert finite, f"{kind} qlen={qlen}: non-finite output"
    assert d < 0.03, f"{kind} qlen={qlen}: diff {d:.6f}"


# ── expert마다 "빈 티어"가 있는 plan (2026-09-05, 컴파일 plan NaN의 원인) ──────
#
# 컴파일 plan은 expert별로 예산이 갈려 **어떤 티어의 행이 0인 expert**가 나온다
# (전량을 다른 티어가 가진 expert). 균일 plan에서는 절대 나오지 않는 형태라
# 기존 테스트가 전부 통과하는데도 서버는 첫 토큰부터 NaN을 냈다. 두 구멍이었다:
#   1. kt cold: k(e)==0인 expert를 건너뛰지 않고 export가 stale을 남겼다
#      (kt-kernel 528d59e "stage-scoped skip, allow zero cold rows"가 고쳤다 —
#       **바이너리가 그보다 오래되면 여기서 잡힌다**).
#   2. grouped wres 커널(prism_grouped.cuh): out 쓰기가 K 조각 루프 **안**에
#      있어 kr==0이면 통째로 건너뛰고 이전 스텝 값이 남았다 (m >= GROUPED_MIN_M).
# 두 구멍 다 "이전 호출의 값이 남는다"라서, 앞선 호출을 바꿔가며 두 번 본다.


def _empty_tier_plan(victim_bands, proj_key, gpu_warm="gemv_worklist",
                     cpu_cold="kt_tile_k2_bf16", victim=3):
    overrides = []
    for e in range(E):
        gu, dn = _contig(H, 64, _WARM), _contig(I, 32, _WARM)
        if e == victim:
            if proj_key == "down":
                dn = victim_bands
            else:
                gu = victim_bands
        overrides.append({"layer": 0, "expert": e, "gate": _entry(list(gu), I),
                          "up": _entry(list(gu), I), "down": _entry(list(dn), H)})
    raw = {"schema_version": 1, "model_id": "test/tiny", "dims": dict(DIMS),
           "kernels": {"gpu_warm": gpu_warm, "cpu_cold": cpu_cold},
           "overrides": overrides}
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


_EMPTY_TIER_CASES = {
    # (proj, victim 밴드) — victim만 그 proj에서 티어 하나가 0행이다
    "down_no_cold": ("down", [[0, I, "hot"]]),
    "down_no_cold_warm": ("down", [[0, 64, "hot"], [64, I, "warm"]]),
    "down_no_warm": ("down", [[0, 32, "hot"], [32, I, "cold"]]),
    "down_no_hot": ("down", [[0, 32, "warm"], [32, I, "cold"]]),
    "gate_no_cold": ("gate", [[0, H, "hot"]]),
    "gate_no_warm": ("gate", [[0, 64, "hot"], [64, H, "cold"]]),
}


@cuda_required
@pytest.mark.parametrize("m", [2, 16])   # 2 = worklist, 16 = grouped (GROUPED_MIN_M)
@pytest.mark.parametrize("case", sorted(_EMPTY_TIER_CASES))
def test_empty_tier_expert(case, m):
    """티어가 빈 expert의 partial은 0이어야 한다 — 쓰지 않으면 stale이 더해진다.

    victim 슬롯의 기여만 router 가중으로 분리해 본다 (다른 슬롯은 정상 expert).
    앞선 호출을 다르게 두 번 하는 이유: 결함이 "이전 호출의 값이 남는다"라서
    한 번만 보면 우연히 맞을 수 있다."""
    proj_key, bands = _EMPTY_TIER_CASES[case]
    victim = 3
    plan = _empty_tier_plan(bands, proj_key, victim=victim)
    w13, w2 = make_weights()
    ex = build_executor(plan, w13, w2)
    torch.manual_seed(1)
    x = (torch.randn(m, H) / 10.0).to(torch.bfloat16)
    other = (torch.randn(m, H) / 10.0).to(torch.bfloat16)
    ids = torch.tensor([[victim, 5]] * m, dtype=torch.int32)
    ids_other = torch.tensor([[5, 6]] * m, dtype=torch.int32)
    w = torch.zeros(m, DIMS["top_k"], dtype=torch.float32)
    w[:, 0] = 1.0
    ref = moe_reference(x, ids, w, w13, w2)
    for tag, pre in (("cold-start", None), ("after-other", (other, ids_other))):
        if pre is not None:
            run(ex, pre[0], pre[1], torch.ones(m, DIMS["top_k"], dtype=torch.float32))
        out = run(ex, x, ids, w)
        d = rel_diff(out, ref)
        print(f"[empty-tier] {case} m={m} {tag}: rel diff = {d:.6f}")
        assert bool(torch.isfinite(out).all()), f"{case} m={m} {tag}: non-finite"
        assert d < 0.03, f"{case} m={m} {tag}: diff {d:.6f}"
