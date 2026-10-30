"""Prism vertical slice의 심장 테스트: 3-tier rejoin 정확성 (GPU + kt AMX 필요).

- torch fp32 full-MoE 레퍼런스 대비 tolerance (decode/prefill)
- plan 불변성: "warm+cold 혼합", "전부 cold", "전부 warm", "전부 hot",
  "hot+warm+cold"가 동일 입력에서 일치 — rejoin의 이중계산/누락 검출 (계약 ⑤).
  티어 경계를 어디로 옮겨도 출력이 같다는 것이 K-split의 정의이므로, HOT
  구현의 검증도 여기에 얹는 것이 맞다.
"""

import pytest
import torch

pytest.importorskip("kt_kernel", reason="kt_kernel required")
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from sglang.srt.layers.moe.prism.cold_backend import KtColdBackend
from sglang.srt.layers.moe.prism.executor import PrismExecutor
from sglang.srt.layers.moe.prism.numa import numa_node_count
from sglang.srt.layers.moe.prism.plan import Tier, parse_plan, validate_static
from sglang.srt.layers.moe.prism.resources import ExecutionResources, ResourceSpec
from sglang.srt.layers.moe.prism.weights import prepare_layer_weights

DIMS = {
    "hidden_size": 256,
    "intermediate_size": 128,
    "num_layers": 1,
    "num_experts": 8,
    "top_k": 2,
    "dtype": "bfloat16",
}
MAX_TOKENS = 16
NUM_NODES = numa_node_count()


def proj_entry(bands, N):
    has_cold = any(t == "cold" for _, _, t in bands)
    shards = []
    if has_cold:
        half = (N // 2 // 32) * 32
        shards = [[0, 0, half], [1, half, N]] if NUM_NODES >= 2 else [[0, 0, N]]
    return {"bands": bands, "cold_shards": shards}


def make_plan(kind, gpu_warm="torch_bmm"):
    if kind == "mixed":
        gate_up = proj_entry([[0, 64, "warm"], [64, 256, "cold"]], 128)
        down = proj_entry([[0, 64, "warm"], [64, 128, "cold"]], 256)
    elif kind == "all_cold":
        gate_up = proj_entry([[0, 256, "cold"]], 128)
        down = proj_entry([[0, 128, "cold"]], 256)
    elif kind == "all_warm":
        gate_up = proj_entry([[0, 256, "warm"]], 128)
        down = proj_entry([[0, 128, "warm"]], 256)
    elif kind == "all_hot":
        gate_up = proj_entry([[0, 256, "hot"]], 128)
        down = proj_entry([[0, 128, "hot"]], 256)
    elif kind == "three_tier":
        # down의 K=128은 ROW_GROUP=64로 밴드 2개가 한계라 hot+cold로 둔다
        # (proj마다 티어 조합이 달라도 되는 것 자체가 검증 대상).
        gate_up = proj_entry([[0, 64, "hot"], [64, 128, "warm"], [128, 256, "cold"]], 128)
        down = proj_entry([[0, 64, "hot"], [64, 128, "cold"]], 256)
    else:
        raise ValueError(f"unknown plan kind {kind!r}")
    raw = {
        "schema_version": 1,
        "model_id": "test/tiny",
        "dims": dict(DIMS),
        "kernels": {"gpu_warm": gpu_warm, "cpu_cold": "kt_amx_bf16"},
        "default": {"gate": dict(gate_up), "up": dict(gate_up), "down": down},
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def make_weights(seed=0, exact=False):
    """exact=True: bf16-정확표현 작은 정수({-1,0,1}) 가중치 — GEMM 누산 순서와
    무관하게 비트일치가 성립하는 입력 클래스 (worklist vs torch_bmm 동등성
    테스트용). 범위를 {-2..2}가 아니라 {-1,0,1}로 좁힌 이유: down 입력 act는
    silu(gate)*up라 gate/up 스케일에 제곱으로 증폭된다 — {-2..2}였을 때
    down GEMM 누산이 실측으로 fp32 정수 정확표현 한계(2^24)에 근접해
    torch_bmm과 worklist 커널의 리덕션 순서 차이가 실제 반올림 차이로
    드러났다 (동일 seed에서 rel diff ~0.4%, 계약 ⑤가 요구하는 비트일치 위반)."""
    torch.manual_seed(seed)
    e, h, i = DIMS["num_experts"], DIMS["hidden_size"], DIMS["intermediate_size"]
    if exact:
        w13 = torch.randint(-1, 2, (e, 2 * i, h)).to(torch.bfloat16)
        w2 = torch.randint(-1, 2, (e, h, i)).to(torch.bfloat16)
    else:
        w13 = (torch.randn(e, 2 * i, h) / 10.0).to(torch.bfloat16)
        w2 = (torch.randn(e, h, i) / 10.0).to(torch.bfloat16)
    return w13, w2


def build_executor(plan, w13, w2, cold_gpu_view=False, warm_kt=False, **executor_kwargs):
    """executor_kwargs는 PrismExecutor 생성자에 그대로 전달된다
    (예: force_graph_path=True, cold_stream=True, capture_mode_fn=...).
    worklist_max_m은 PrismExecutor로 전달되고, worklist_kernels는 plan의
    gpu_warm 키에서 자동 resolve된다 (torch_bmm이면 None — worklist 비활성)."""
    from sglang.srt.layers.moe.prism.plan import Proj

    prepared = prepare_layer_weights(0, w13, w2, plan, device=torch.device("cuda"),
                                     warm_kt=warm_kt)
    ep = plan.expert(0, 0)
    has_cold = any(ep.proj(p).has_tier(Tier.COLD) for p in Proj)

    cold = None
    if has_cold or prepared.warm_kt is not None:
        cold = KtColdBackend(
            plan, max_tokens=MAX_TOKENS, num_numa_nodes=NUM_NODES,
            gpu_view_device=torch.device("cuda") if (cold_gpu_view or warm_kt) else None,
            hybrid_mask=executor_kwargs.get("cold_hybrid_frac") is not None)
        if has_cold:
            cold.load_layer(0, prepared.cold, prepared.thr)
        if prepared.warm_kt is not None:
            cold.load_warm_layer(0, prepared.warm_kt, prepared.thr, local_node=0)
    spec = ResourceSpec.from_plan(plan, max_tokens=MAX_TOKENS, device=torch.device("cuda"))
    res = ExecutionResources(spec)
    ex = PrismExecutor(plan, res, cold, **executor_kwargs)
    ex.register_layer(0, prepared)
    return ex


def moe_reference(x, ids, weights, w13, w2):
    """torch fp32 full MoE (silu(gate)*up → down → router 가중 합)."""
    i = DIMS["intermediate_size"]
    gate_w, up_w = w13[:, :i, :].float(), w13[:, i:, :].float()
    down_w = w2.float()
    out = torch.zeros(x.shape[0], DIMS["hidden_size"])
    xf = x.float()
    for m in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[m, j])
            g = xf[m] @ gate_w[e].t()
            u = xf[m] @ up_w[e].t()
            a = torch.nn.functional.silu(g) * u
            out[m] += float(weights[m, j]) * (a @ down_w[e].t())
    return out


def make_inputs(m, seed=0, exact=False):
    """exact=True: bf16-정확표현 작은 정수 입력 (make_weights(exact=True)와 짝).
    topk_ids는 이미 정수 permutation이라 항상 정확 — x/topk_weights만 갈린다.
    (매개변수명은 m — 기존 호출부는 위치 인자로 qlen을 넘기므로 무변경 통과.)"""
    torch.manual_seed(seed)
    if exact:
        x = torch.randint(-1, 2, (m, DIMS["hidden_size"])).to(torch.bfloat16)
        w = torch.randint(0, 3, (m, DIMS["top_k"])).to(torch.float32)
    else:
        x = (torch.randn(m, DIMS["hidden_size"]) / 10.0).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(DIMS["num_experts"])[: DIMS["top_k"]] for _ in range(m)])
        w = torch.rand(m, DIMS["top_k"], dtype=torch.float32)
    if exact:
        ids = torch.stack([torch.randperm(DIMS["num_experts"])[: DIMS["top_k"]] for _ in range(m)])
    return x, ids, w


def run(ex, x, ids, w):
    return ex.run_layer(0, x.cuda(), ids.cuda(), w.cuda()).cpu()


def rel_diff(a, b):
    return (torch.mean(torch.abs(a.float() - b.float())) / (torch.mean(torch.abs(b.float())) + 1e-8)).item()


@cuda_required
@pytest.mark.parametrize("qlen", [1, 16])
@pytest.mark.parametrize("kind", ["mixed", "all_cold", "all_warm", "all_hot", "three_tier"])
def test_layer_matches_reference(kind, qlen):
    plan = make_plan(kind)
    w13, w2 = make_weights()
    ex = build_executor(plan, w13, w2)
    x, ids, w = make_inputs(qlen, seed=10 + qlen)
    out = run(ex, x, ids, w)
    ref = moe_reference(x, ids, w, w13, w2)
    d = rel_diff(out, ref)
    print(f"{kind} qlen={qlen}: rel diff = {d:.6f}")
    assert d < 0.03, f"{kind} qlen={qlen}: diff {d:.6f}"


@cuda_required
@pytest.mark.parametrize("qlen", [1, 16])
def test_plan_invariance(qlen):
    """같은 weight·입력에서 세 plan의 출력이 일치해야 한다 — 티어 경계
    어디에 있든 결과가 같다는 K-split의 핵심 성질 (이중계산/누락 검출)."""
    w13, w2 = make_weights(seed=1)
    x, ids, w = make_inputs(qlen, seed=20 + qlen)
    kinds = ["mixed", "all_cold", "all_warm", "all_hot", "three_tier"]
    outs = {}
    for kind in kinds:
        ex = build_executor(make_plan(kind), w13, w2)
        outs[kind] = run(ex, x, ids, w)
    # 기준을 all_cold 하나로 잡는다: 모든 티어 배치가 같은 값에 수렴해야 하므로
    # 쌍별 비교가 아니라 공통 기준 대비가 결함 위치를 좁혀준다.
    diffs = {kind: rel_diff(outs[kind], outs["all_cold"]) for kind in kinds}
    print(f"invariance qlen={qlen}: " +
          ", ".join(f"{kind}~cold {d:.6f}" for kind, d in diffs.items()))
    for kind, d in diffs.items():
        assert d < 0.02, f"{kind} diverges from all_cold: {d:.6f}"




@cuda_required
@pytest.mark.parametrize("kind", ["mixed", "three_tier", "all_cold"])
def test_cold_gpu_matches_cpu_cold(kind):
    """cold를 GPU가 kt packed slab을 제자리 읽어 계산해도(cold_gpu.py) CPU AMX와
    같은 값이어야 한다. 비트일치는 요구하지 않는다 — 두 경로의 fp32 누산 순서가
    다르고 이미 CPU↔GPU 티어 사이엔 tolerance가 계약이다 (plan 불변성과 같은 급).
    grouped 경로 전용이므로 M은 임계값 이상이어야 하고, MAX_TOKENS(16) 안에서
    expert 중복이 필연이 되도록 E=8·k=2·M=16이다."""
    w13, w2 = make_weights(seed=3)
    x, ids, w = make_inputs(16, seed=33)
    plan = make_plan(kind)
    ex_cpu = build_executor(plan, w13, w2, cold_gpu_view=True, grouped_min_m=16)
    ex_gpu = build_executor(plan, w13, w2, cold_gpu_view=True, grouped_min_m=16,
                            cold_gpu_min_m=16, split_streams=True)
    a, b = run(ex_cpu, x, ids, w), run(ex_gpu, x, ids, w)
    d = rel_diff(b, a)
    print(f"cold-gpu {kind}: rel diff vs cpu cold = {d:.6f}")
    assert d < 1e-3, f"{kind}: cold GPU diverges from CPU cold: {d:.6f}"
    # 아래 임계값이면 같은 executor가 CPU cold로 돌아간다 (M=16 < 64) — 경로 선택 검증
    ex_thr = build_executor(plan, w13, w2, cold_gpu_view=True, grouped_min_m=16,
                            cold_gpu_min_m=64)
    assert torch.equal(run(ex_thr, x, ids, w), a)


@cuda_required
@pytest.mark.parametrize("kind", ["mixed", "all_warm"])
@pytest.mark.parametrize("qlen", [1, 16])
def test_warm_kt_matches_rowmajor_warm(kind, qlen):
    """warm을 kt 포맷 slab(pinned)으로 두고 GPU가 packed GEMV(decode)/cold-layout grouped
    (prefill)로 읽어도 row-major pinned warm과 같은 값 — 읽는 원소가 같고 누산 순서도
    같아 정확표현 입력에서 비트일치, 일반 입력은 tolerance."""
    plan = make_plan(kind)
    for exact in (True, False):
        w13, w2 = make_weights(seed=5, exact=exact)
        x, ids, w = make_inputs(qlen, seed=50 + qlen, exact=exact)
        a = run(build_executor(plan, w13, w2, grouped_min_m=16), x, ids, w)
        b = run(build_executor(plan, w13, w2, warm_kt=True, grouped_min_m=16), x, ids, w)
        if exact:
            assert torch.equal(a, b), f"{kind} qlen={qlen}: warm-kt differs on exact ints ({(a != b).sum().item()} elems)"
        else:
            d = rel_diff(b, a)
            assert d < 1e-3, f"{kind} qlen={qlen}: warm-kt rel diff {d}"


@cuda_required
def test_warm_kt_requires_all_projections():
    """warm이 일부 proj에만 있는 plan(three_tier: down은 hot+cold)은 warm-kt 불가 — 즉사."""
    from sglang.srt.layers.moe.prism.plan import PlanError

    w13, w2 = make_weights(seed=6)
    with pytest.raises(PlanError):
        build_executor(make_plan("three_tier"), w13, w2, warm_kt=True)


@cuda_required
@pytest.mark.parametrize("kind", ["mixed", "all_warm"])
def test_warm_kt_cpu_prefill_matches(kind):
    """prefill에서 warm 행을 warm-kt 인스턴스(CPU AMX)가 계산해도 GPU 계산과 같은 값 —
    CPU↔GPU는 fp32 누산 순서가 달라 tolerance (plan 불변성과 같은 급)."""
    plan = make_plan(kind)
    w13, w2 = make_weights(seed=7)
    x, ids, w = make_inputs(16, seed=70)
    a = run(build_executor(plan, w13, w2, warm_kt=True, grouped_min_m=16), x, ids, w)
    b = run(build_executor(plan, w13, w2, warm_kt=True, grouped_min_m=16, warm_cpu_min_m=16), x, ids, w)
    d = rel_diff(b, a)
    print(f"warm-cpu {kind}: rel diff = {d:.6f}")
    assert d < 1e-3, f"{kind}: warm-on-CPU diverges: {d}"


@cuda_required
@pytest.mark.parametrize("kind", ["mixed", "three_tier"])
def test_cold_hybrid_matches_cpu_cold(kind):
    """cold hybrid(expert를 GPU/CPU로 분할, 비용 모델)는 CPU cold와 같은 값 — 두 partial의
    (m, j) 행이 서로소여야 한다. W-resident 커널이 expert 마스크(tile_off)를 무시하면
    GPU가 CPU 몫도 계산해 rejoin에서 이중 합산된다 (2026-08-27 실제 발생) — 이 테스트가
    그 회귀 검출기다. M=16이면 expert당 pair가 적어 비용 모델이 대부분을 CPU에 주므로
    frac을 크게 줘서 양쪽 다 비어 있지 않게 한다."""
    plan = make_plan(kind)
    w13, w2 = make_weights(seed=9)
    x, ids, w = make_inputs(16, seed=90)
    a = run(build_executor(plan, w13, w2, cold_gpu_view=True, grouped_min_m=16), x, ids, w)
    ex = build_executor(plan, w13, w2, cold_gpu_view=True, grouped_min_m=16,
                        cold_gpu_min_m=16, cold_hybrid_frac=(0.9, 0.9), split_streams=True)
    # 분할이 실제로 양쪽에 걸리는지 확인 (한쪽이 비면 이 테스트는 이중 합산을 못 잡는다)
    masks = ex._balance_hybrid(ids, DIMS["num_experts"], torch.device("cuda"))
    gpu_experts = int(masks["gu"][0].sum())
    assert 0 < gpu_experts < DIMS["num_experts"], f"hybrid split degenerate: {gpu_experts}/{DIMS['num_experts']} on GPU"
    b = run(ex, x, ids, w)
    d = rel_diff(b, a)
    print(f"hybrid {kind}: GPU experts {gpu_experts}, rel diff = {d:.6f}")
    assert d < 1e-3, f"{kind}: hybrid diverges from CPU cold: {d}"


@pytest.mark.skip(reason="cold_async는 미완 (partial H2D가 flag wait 앞서 실행되는 순서 버그 "
                         "+ 스핀 커널 stream 오배치로 hang). SGLANG_PRISM_COLD_ASYNC 기본 꺼짐. "
                         "cuStreamWaitValue32로 재구현 예정 — TODO.md 'cold_async' 참조.")
@cuda_required
@pytest.mark.parametrize("kind", ["mixed", "all_cold", "three_tier"])
@pytest.mark.parametrize("qlen", [1, 16])
def test_cold_async_matches_blocking(kind, qlen):
    """cold_async(전용 stream + 플래그 wait, 블로킹 콜백 없음)는 같은 CPU 계산·같은 GPU
    커널이라 기본 경로와 **비트 동일**해야 한다."""
    plan = make_plan(kind)
    w13, w2 = make_weights(seed=11)
    x, ids, w = make_inputs(qlen, seed=110 + qlen)
    a = run(build_executor(plan, w13, w2, grouped_min_m=16), x, ids, w)
    ex = build_executor(plan, w13, w2, grouped_min_m=16, cold_async=True)
    b = run(ex, x, ids, w)
    assert torch.equal(a, b), f"{kind} qlen={qlen}: cold_async differs ({(a != b).sum().item()} elems)"
    # 두 번 연속 호출해도(플래그 seq 증가) 같은 값 — 재사용 버퍼/플래그 오염 검출
    assert torch.equal(run(ex, x, ids, w), a)
