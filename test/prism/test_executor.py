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
from sglang.srt.layers.moe.prism.kernels import resolve_gpu_kernel, resolve_worklist_kernels
from sglang.srt.layers.moe.prism.numa import numa_node_count
from sglang.srt.layers.moe.prism.plan import Tier, parse_plan, validate_static
from sglang.srt.layers.moe.prism.resources import ExecutionResources, ResourceSpec
from sglang.srt.layers.moe.prism.stagers import PerSlotCopyStager
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


def build_executor(plan, w13, w2, *, worklist_max_m=32, **executor_kwargs):
    """executor_kwargs는 PrismExecutor 생성자에 그대로 전달된다
    (예: force_graph_path=True, cold_stream=True, capture_mode_fn=...).
    worklist_max_m은 PrismExecutor로 전달되고, worklist_kernels는 plan의
    gpu_warm 키에서 자동 resolve된다 (torch_bmm이면 None — worklist 비활성)."""
    from sglang.srt.layers.moe.prism.plan import Proj

    prepared = prepare_layer_weights(0, w13, w2, plan, device=torch.device("cuda"))
    ep = plan.expert(0, 0)
    has_cold = any(ep.proj(p).has_tier(Tier.COLD) for p in Proj)

    cold = None
    if has_cold:
        cold = KtColdBackend(plan, max_tokens=MAX_TOKENS, num_numa_nodes=NUM_NODES)
        cold.load_layer(0, prepared.cold, prepared.thr)
    spec = ResourceSpec.from_plan(plan, max_tokens=MAX_TOKENS, device=torch.device("cuda"))
    res = ExecutionResources(spec)
    kwargs = dict(
        worklist_kernels=resolve_worklist_kernels(plan.kernels.gpu_warm),
        worklist_max_m=worklist_max_m,
    )
    kwargs.update(executor_kwargs)
    ex = PrismExecutor(plan, res, cold, resolve_gpu_kernel(plan.kernels.gpu_warm),
                       stager=PerSlotCopyStager(), **kwargs)
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
        w = torch.rand(m, DIMS["top_k"], dtype=torch.float32)
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
@pytest.mark.parametrize("kind", ["mixed", "all_hot", "three_tier"])
@pytest.mark.parametrize("m", [1, 4])
def test_worklist_matches_bmm_path(kind, m):
    """worklist 커널 plan과 torch_bmm plan이 같은 가중치·입력에서 일치.

    gu(gate/up) phase는 두 경로 모두 bf16 재료화(_warm_gemm/worklist 커널
    둘 다 "출력 bf16" 계약)라 비트일치다 — exact=True 입력으로 실측 확인됨
    (hot/warm/cold 세 partial 전부). down phase는 다르다: 기존
    grouping.down_apply(Dedup/SlotOrder 공통, Task 4 이전부터 존재)는 fp32
    누산을 절대 bf16으로 재료화하지 않는 반면, worklist 커널은 계약 ⑤의
    문면대로 down partial을 bf16으로 재료화한다 — 두 경로가 "티어별 partial
    라운딩 횟수"만 다를 뿐 같은 클래스(계약 ⑤ §3: "tier별 partial이 각각
    라운딩된 뒤 fp32 합산 — 같은 클래스, 라운딩 이벤트 수만 증가")이므로,
    §5에 따라 여기서는 bitwise가 아니라 tolerance로 비교한다.

    비트일치 커버리지는 이 테스트가 아니라 (a) 커널 단위 exact-int 테스트
    (test_gemv_worklist)와 (b) worklist 경로 자기 자신의 graph-vs-eager
    테스트가 담당한다 — 이 테스트는 "두 실행 경로가 수치적으로 같은 것을
    계산한다"는 executor 레벨의 근사 등가성만 검증한다.

    tolerance 유도: down partial에 bf16 라운딩이 정확히 1회 추가된다
    (상대오차 ≤ 2^-8), 라우터 가중합(가중치 ~O(1), k=2)을 거치고 최종
    출력이 다시 bf16으로 재료화되며 1회 더 라운딩되어(상대오차 ≤ 2^-8),
    합쳐서 상대오차 ≤ ~2^-7 수준. rtol=2^-6은 여기에 2배 여유를 둔 값이다.

    atol은 고정 2^-6이 아니라 `2^-6 * out_ref.abs().max()`로 출력 스케일에
    맞춰 잡는다 — silu(gate)*up의 이차 증폭 때문에 exact=True 입력에서도
    down 단계 값이 O(1)에 머물지 않고 실측 O(10^3~10^4)까지 커지고
    (아래 실측: |out_ref| max ~6800), 이 스케일에서 고정 atol=2^-6은
    사실상 0이라 rtol만 남는데, k=2 expert의 가중합이 서로 상쇄되는 극소수
    좌표(라우터 가중 합의 값 자체가 0 근처)에서는 상대오차가 분모→0으로
    발산해 어떤 rtol로도 통과하지 못한다 (실측: 고정 atol=2^-6에서
    5/6 파라미터화가 실패, 상대오차 100%·절대오차는 겨우 12~32 — bf16
    반올림 노이즈 그 자체이지 별도 결함이 아님). atol을 출력 스케일에
    proportional하게 앵커링하면 이 근접-영점 좌표에서도 "절대오차가
    출력 스케일 대비 2^-6 이내"라는 같은 유도가 그대로 성립한다 (실측
    max_diff는 32, 스케일-aware atol은 96~107 — 3배 이상 여유).
    exact=True 입력은 여전히 유지한다 — gu phase와 cold 경로를 결정적으로
    만들어 tolerance 비교의 신호 대 잡음비를 높여준다(정확표현이 아니면
    gu/cold 자체도 흔들려 down의 신호를 가린다). m=4는 bs>1 decode
    시나리오(worklist의 존재 이유)."""
    w13, w2 = make_weights(exact=True)          # 기존 헬퍼의 exact 입력 경로 사용
    x, ids, tw = make_inputs(m=m, exact=True)   # 헬퍼에 m 인자가 없으면 추가
    ex_ref = build_executor(make_plan(kind), w13, w2)
    ex_wl = build_executor(make_plan(kind, gpu_warm="gemv_worklist"), w13, w2)
    out_ref = ex_ref.run_layer(0, x.cuda(), ids.cuda(), tw.cuda()).cpu().float()
    out_wl = ex_wl.run_layer(0, x.cuda(), ids.cuda(), tw.cuda()).cpu().float()
    scale = max(out_ref.abs().max().item(), 1.0)  # O(1) 하한 — 전부-0 출력 방어
    torch.testing.assert_close(out_wl, out_ref, rtol=2**-6, atol=(2**-6) * scale)


@cuda_required
def test_worklist_above_threshold_falls_back_to_dedup():
    """M > worklist_max_m(=여기선 2로 강제)이면 기존 Dedup 경로 — 결과 동일.

    이 비교는 bitwise다 (tolerance가 아니다): m=4 > worklist_max_m=2라
    ex_small도 flow.worklist=False로 떨어져 self._warm_gemm이 그대로
    쓰인다 — 그리고 "gemv_worklist" 커널 키의 gpu_warm 폴백은
    kernels.py에서 torch_bmm과 동일 함수(_warm_gemm_torch_bmm)로 등록돼
    있다. 즉 ex_small과 ex_ref는 (plan.kernels.gpu_warm 문자열만 다를 뿐)
    완전히 같은 GEMM 함수·같은 Dedup grouping 코드를 탄다 — worklist 커널
    자체가 아예 호출되지 않으므로 위 test_worklist_matches_bmm_path의
    tolerance 사유(추가 bf16 라운딩)가 여기엔 존재하지 않는다. 실측으로도
    torch.equal이 성립함을 확인했다."""
    w13, w2 = make_weights(exact=True)
    x, ids, tw = make_inputs(m=4, exact=True)
    plan = make_plan("three_tier", gpu_warm="gemv_worklist")
    ex_small = build_executor(plan, w13, w2, worklist_max_m=2)   # m=4 > 2 → Dedup
    ex_ref = build_executor(make_plan("three_tier"), w13, w2)
    assert torch.equal(
        ex_small.run_layer(0, x.cuda(), ids.cuda(), tw.cuda()).cpu(),
        ex_ref.run_layer(0, x.cuda(), ids.cuda(), tw.cuda()).cpu(),
    )
