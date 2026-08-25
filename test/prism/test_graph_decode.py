"""Task 8: CUDA graph bs=1 decode 경로 검증 (GPU + kt AMX 필요).

세 층위로 나눠 검증한다:
1. stage_from_device 단위: device sel에서 직접 gather한 arena가 PerSlot 준거와
   bitwise 일치 (pinned 경유 H2D 없음).
2. 경로 등가 (eager에서 강제): 생성자 주입(force_graph_path=True 또는
   capture_mode_fn)으로 graph-safe 경로를 캡처 없이 태워 일반 eager 경로와
   bitwise 일치. cold stream 통합(cold_stream=True)도 동일하게 등가.
3. 실캡처: torch.cuda.CUDAGraph로 run_layer를 캡처 → 입력 버퍼만 갈아끼우고
   replay한 출력이 같은 입력의 eager 출력과 bitwise 일치 — kt host node
   (cudaLaunchHostFunc)의 캡처·재생까지 포함한 실전 검증.

모드 제어는 전부 생성자 주입이다 — 프로덕션과 같은 "한 executor가 eager
prefill과 graph decode를 오가는" 시나리오는 capture_mode_fn에 토글 가능한
callable을 주입해 재현한다 (monkeypatch/속성 변조 없음).
"""

import pytest
import torch

pytest.importorskip("kt_kernel", reason="kt_kernel required")
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

import test_executor  # noqa: E402  (같은 디렉토리의 헬퍼 재사용 + DIMS monkeypatch용)
from test_executor import (  # noqa: E402
    build_executor,
    make_inputs,
    make_plan,
    make_weights,
)

from sglang.srt.layers.moe.prism.plan import Proj  # noqa: E402
from sglang.srt.layers.moe.prism.resources import (  # noqa: E402
    ExecutionResources,
    ResourceSpec,
)
from sglang.srt.layers.moe.prism.stagers import (  # noqa: E402
    GatherKernelStager,
    PerSlotCopyStager,
)
from sglang.srt.layers.moe.prism.weights import WarmBand  # noqa: E402


def run1(ex, x, ids, w):
    return ex.run_layer(0, x.cuda(), ids.cuda(), w.cuda()).cpu()


class _Toggle:
    """capture_mode_fn 주입용 토글 — 한 executor가 eager와 graph 경로를
    오가는 프로덕션 시나리오(sglang capture-mode 플래그의 on/off)를 재현."""

    def __init__(self):
        self.on = False

    def __call__(self) -> bool:
        return self.on


# ── 1. stager 단위 ────────────────────────────────────────────────────────

@cuda_required
def test_stage_from_device_bitwise():
    """stage_from_device(device sel)가 PerSlot 준거와 bitwise 일치 —
    sel이 int64 cuda(topk_ids 슬라이스 그대로)여도 device cast로 처리."""
    spec = ResourceSpec(
        max_tokens=8, top_k=3, hidden_size=64, intermediate_size=8,
        k_warm_gate=8, k_warm_up=8, k_warm_down=8, n_slots=8,
        device=torch.device("cuda"),
    )
    res = ExecutionResources(spec)
    band = WarmBand(
        k_offset=0,
        weights=torch.arange(32 * 8 * 8, dtype=torch.bfloat16)
        .reshape(32, 8, 8).pin_memory(),
    )
    s = torch.cuda.Stream()
    ref = torch.zeros(spec.n_slots, 8, 8, dtype=torch.bfloat16, device="cuda")
    # zeros init(current stream)과 타 stream의 무대기 stage 사이 레이스 방지
    # (test_stagers.py의 동기화 주석 참조).
    torch.cuda.synchronize()
    e_ref = PerSlotCopyStager().stage(band, [5, 17, 2], ref, s, None, Proj.GATE)

    sel = torch.tensor([5, 17, 2], dtype=torch.int64, device="cuda")
    out = res.arena.view(Proj.GATE)
    e_out = GatherKernelStager(res).stage_from_device(
        band, sel, out, res.warm_stream, None, Proj.GATE
    )
    cur = torch.cuda.current_stream()
    cur.wait_event(e_ref)
    cur.wait_event(e_out)
    torch.cuda.synchronize()
    assert torch.equal(out[:3].cpu(), ref[:3].cpu())


# ── 2. 경로 등가 (캡처 없이 강제) ─────────────────────────────────────────

@cuda_required
@pytest.mark.parametrize("kind", ["mixed", "all_cold", "all_warm", "all_hot", "three_tier"])
def test_graph_path_matches_eager(kind):
    """graph-safe 경로(더미 그룹 + device sel + stream 통합 cold)가 일반
    eager 경로와 bitwise 일치 — 같은 바이트를 같은 커널에 넣으므로
    tolerance가 아니라 등호가 성립해야 한다. 한 executor의 capture_mode_fn
    토글로 두 경로를 오간다 (프로덕션과 동일한 오감 방식)."""
    plan = make_plan(kind)
    w13, w2 = make_weights()
    toggle = _Toggle()
    ex = build_executor(plan, w13, w2, capture_mode_fn=toggle)
    x, ids, w = make_inputs(1, seed=42)
    ref = run1(ex, x, ids, w)
    toggle.on = True
    out = run1(ex, x, ids, w)
    assert torch.equal(out, ref), (
        f"max abs diff {(out.float() - ref.float()).abs().max().item()}"
    )


@cuda_required
def test_graph_path_rejects_prefill():
    """graph 경로는 M==1 전용 — m>1이면 조용한 오답 대신 즉사해야 한다."""
    plan = make_plan("all_warm")
    w13, w2 = make_weights()
    ex = build_executor(plan, w13, w2, force_graph_path=True)
    x, ids, w = make_inputs(4, seed=1)
    with pytest.raises(RuntimeError, match="M==1"):
        run1(ex, x, ids, w)


@cuda_required
@pytest.mark.parametrize("qlen", [1, 16])
def test_cold_stream_integration_matches(qlen):
    """cold_stream=True(SGLANG_PRISM_COLD_STREAM=1 상당): submit/sync를
    current stream 경유(cudaLaunchHostFunc)로 바꿔도 수치는 bitwise 동일.
    cold_stream은 프로덕션에서 startup에 고정되는 설정이므로 executor를
    각각 만들어 비교한다."""
    plan = make_plan("mixed")
    w13, w2 = make_weights()
    x, ids, w = make_inputs(qlen, seed=7 + qlen)
    ref = run1(build_executor(plan, w13, w2), x, ids, w)
    out = run1(build_executor(plan, w13, w2, cold_stream=True), x, ids, w)
    assert torch.equal(out, ref)


# ── 3. 실캡처 + replay ───────────────────────────────────────────────────

def _warmup_and_capture(ex, x_buf, ids_buf, w_buf):
    """CudaGraphRunner의 캡처 절차와 동형: side stream에서 워밍업 2회
    (jit compile 등 lazy init을 캡처 밖에서) 후 실캡처."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            ex.run_layer(0, x_buf, ids_buf, w_buf)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_buf = ex.run_layer(0, x_buf, ids_buf, w_buf)
    return g, out_buf


@cuda_required
@pytest.mark.parametrize("kind", ["mixed", "three_tier"])
def test_capture_replay_matches_eager(kind):
    """run_layer를 실제 CUDA graph로 캡처하고, 입력 버퍼 값만 갈아끼운
    replay 출력이 같은 값의 eager 출력과 bitwise 일치하는지 — kt host node
    (submit/sync cudaLaunchHostFunc)의 캡처·재생 포함.

    three_tier는 hot의 index_select가 캡처 가능한지를 본다: sel이 device
    텐서(flat_ids 절단)여야만 캡처되고, host 리스트에서 만들면 캡처 중
    pageable H2D로 죽는다."""
    plan = make_plan(kind)
    w13, w2 = make_weights()
    toggle = _Toggle()
    ex = build_executor(plan, w13, w2, capture_mode_fn=toggle)

    x0, ids0, w0 = make_inputs(1, seed=100)
    x_buf, ids_buf, w_buf = x0.cuda(), ids0.cuda(), w0.cuda()

    toggle.on = True
    g, out_buf = _warmup_and_capture(ex, x_buf, ids_buf, w_buf)

    for seed in (101, 102):
        x, ids, w = make_inputs(1, seed=seed)
        x_buf.copy_(x.cuda())
        ids_buf.copy_(ids.cuda())
        w_buf.copy_(w.cuda())
        g.replay()
        torch.cuda.synchronize()
        got = out_buf.cpu().clone()

        toggle.on = False
        ref = ex.run_layer(0, x_buf, ids_buf, w_buf).cpu()
        toggle.on = True
        assert torch.equal(got, ref), (
            f"seed {seed}: max abs diff "
            f"{(got.float() - ref.float()).abs().max().item()}"
        )


@cuda_required
def test_qlen_pin_graph_isolated_from_eager_write():
    """Review Finding A 회귀: 캡처가 baking하는 qlen 포인터는 eager의
    `self._qlen_pin`과 격리된 `self._qlen_pin_graph`(상수 1)여야 한다.

    수정 전에는 두 경로가 같은 `self._qlen_pin`을 공유했다 — 캡처 후 eager
    prefill(m=16)이 `self._qlen_pin[0] = 16`을 쓰면, 그 이후 모든 graph
    replay가 cold를 16토큰짜리로 돌려 decode perf가 붕괴한다(row 0은 여전히
    옳은 데이터를 읽으므로 수치는 우연히 맞음 — 순수 perf 버그). 이 테스트는
    프로덕션 시나리오 그대로 **같은 executor**로 캡처 → eager prefill 오염 →
    replay를 수행해 (1) 격리를 직접 확인하고 (2) 오염 이후에도 출력이
    bitwise 정확함을 확인한다."""
    plan = make_plan("mixed")
    w13, w2 = make_weights()
    toggle = _Toggle()
    ex = build_executor(plan, w13, w2, capture_mode_fn=toggle)

    x0, ids0, w0 = make_inputs(1, seed=300)
    x_buf, ids_buf, w_buf = x0.cuda(), ids0.cuda(), w0.cuda()

    toggle.on = True
    g, out_buf = _warmup_and_capture(ex, x_buf, ids_buf, w_buf)

    # 캡처 후: 같은 executor로 eager prefill-like 호출(m=16) — self._qlen_pin
    # 을 오염시킨다. graph 쪽 버퍼가 격리돼 있다면 이 오염은 replay에 보이지
    # 않아야 한다.
    toggle.on = False
    x16, ids16, w16 = make_inputs(16, seed=301)
    ex.run_layer(0, x16.cuda(), ids16.cuda(), w16.cuda())
    toggle.on = True

    assert int(ex._qlen_pin[0]) == 16, "eager 호출이 qlen_pin을 16으로 세팅했어야 함"
    assert int(ex._qlen_pin_graph[0]) == 1, (
        "graph 전용 버퍼가 eager 쓰기에 오염됐다 — Finding A 회귀"
    )

    # 오염 이후 fresh m=1 입력으로 replay — eager 레퍼런스와 여전히 bitwise
    # 일치해야 하고, graph 버퍼는 여전히 1이어야 한다.
    x1, ids1, w1 = make_inputs(1, seed=302)
    x_buf.copy_(x1.cuda())
    ids_buf.copy_(ids1.cuda())
    w_buf.copy_(w1.cuda())
    g.replay()
    torch.cuda.synchronize()
    got = out_buf.cpu().clone()

    toggle.on = False
    ref = ex.run_layer(0, x_buf, ids_buf, w_buf).cpu()
    toggle.on = True

    assert torch.equal(got, ref), (
        f"max abs diff {(got.float() - ref.float()).abs().max().item()}"
    )
    # 참고: `ref`도 eager(m=1) 호출이라 self._qlen_pin은 여기서 다시 1로
    # 덮인다 — eager는 매 호출마다 자기 m을 쓰는 게 정상 동작이다. 이 테스트
    # 가 실제로 지키는 불변식은 graph 전용 버퍼가 절대 안 바뀐다는 것.
    assert int(ex._qlen_pin_graph[0]) == 1


# ── 4. 리뷰 finding D: 테스트 갭 보강 ─────────────────────────────────────

@cuda_required
def test_graph_path_distinguishes_position_from_id(monkeypatch):
    """graph 경로의 그룹은 위치 표지([0..k) 절단)이고 실제 expert 선택은
    flat_ids(device topk_ids 슬라이스)가 나른다 — 위치와 id를 혼동하는
    버그(예: slot index를 expert id로 잘못 씀)는 id가 위치와 단조 대응하는
    입력에서는 우연히 숨을 수 있다. k>=3 + 비단조 id([9, 2, 7])로 그 혼동을
    실제로 구분한다."""
    monkeypatch.setitem(test_executor.DIMS, "top_k", 3)
    monkeypatch.setitem(test_executor.DIMS, "num_experts", 10)

    plan = make_plan("mixed")
    w13, w2 = make_weights(seed=3)
    toggle = _Toggle()
    ex = build_executor(plan, w13, w2, capture_mode_fn=toggle)

    x = (torch.randn(1, test_executor.DIMS["hidden_size"]) / 10.0).to(torch.bfloat16)
    ids = torch.tensor([[9, 2, 7]], dtype=torch.int64)  # 비단조: 위치 0→id9, 1→id2, 2→id7
    w = torch.rand(1, 3, dtype=torch.float32)

    ref = run1(ex, x, ids, w)
    toggle.on = True
    out = run1(ex, x, ids, w)
    assert torch.equal(out, ref), (
        f"max abs diff {(out.float() - ref.float()).abs().max().item()}"
    )


# ── 5. Task 5: worklist bs>1 캡처/재생 ───────────────────────────────────

@cuda_required
@pytest.mark.parametrize("m", [1, 4])
def test_capture_replay_worklist_bs(m):
    """worklist plan은 bs>1도 캡처·재생 가능 — replay가 그 시점 topk를 반영.

    test_capture_replay_matches_eager와 동형(같은 워밍업·캡처·버퍼-교체
    절차)이되 plan을 worklist(gemv_worklist)로, 입력 m을 파라미터화한다.
    비교 기준은 (기존 bs=1 테스트와 동일하게) **같은 executor·같은
    worklist plan**의 eager 출력이다 — graph replay와 eager가 같은 커널·
    같은 라운딩 경로를 타므로(worklist-vs-worklist) tolerance가 아니라
    bitwise(torch.equal)가 성립해야 한다. exact=True 입력은 브리프 지정
    그대로 유지(gu/cold 경로를 결정적으로 만들어 신호 대 잡음비를 높임)."""
    plan = make_plan("three_tier", gpu_warm="gemv_worklist")
    w13, w2 = make_weights(exact=True)
    toggle = _Toggle()
    ex = build_executor(plan, w13, w2, capture_mode_fn=toggle)

    x0, ids0, w0 = make_inputs(m=m, exact=True)
    x_buf, ids_buf, w_buf = x0.cuda(), ids0.cuda(), w0.cuda()

    toggle.on = True
    g, out_buf = _warmup_and_capture(ex, x_buf, ids_buf, w_buf)

    for seed in (1, 2):
        x, ids, w = make_inputs(m=m, exact=True, seed=seed)
        x_buf.copy_(x.cuda())
        ids_buf.copy_(ids.cuda())
        w_buf.copy_(w.cuda())
        g.replay()
        torch.cuda.synchronize()
        got = out_buf.cpu().clone()

        toggle.on = False
        ref = ex.run_layer(0, x_buf, ids_buf, w_buf).cpu()
        toggle.on = True
        assert torch.equal(got, ref), (
            f"m={m} seed {seed}: max abs diff "
            f"{(got.float() - ref.float()).abs().max().item()}"
        )


@cuda_required
def test_qlen_pins_graph_isolated_per_bs():
    """bs별 qlen pin이 서로/eager와 격리 — Finding A(328ms/tok stale pin)의
    worklist(M>1) 일반화. force_graph_path=True로 m=1, m=4를 같은
    executor에서 연달아 실행해 두 bs의 pin이 각자 자기 값을 유지하고
    서로 다른 주소를 가리키는지 직접 확인한다.

    plan은 브리프 원안의 "three_tier"(cold 포함) 대신 cold가 없는
    "all_hot"을 쓴다 — `_qlen_pins_graph` 격리는 `_plan_flow`의 순수
    host-side 북키핑이라 cold 유무와 무관하게 성립하는 불변식이고
    (`_plan_flow`의 pin 할당/조회는 `has_cold` 분기보다 앞서 무조건
    실행된다), 실측으로 cold(KtColdBackend/CPUInfer WorkerPool)를 이 위치
    (force_graph_path로 **실캡처 없이** cold submit을 태우는 경로, m>1)에서
    이 프로세스 안의 비-첫 backend로 생성하면 이 스위트 안에서 100%
    재현되는 kt_kernel 네이티브 크래시/행(`tpp.c:83
    __pthread_tpp_change_priority` glibc assert 또는 WorkerPool 생성 중
    행)를 만난다 — RLIMIT_RTPRIO=0인 이 컨테이너에서 실캡처 없이(=
    프로덕션 CudaGraphRunner는 절대 밟지 않는, force_graph_path 전용
    테스트/디버그 경로) cold host-callback을 처음 태우는 backend가
    프로세스 내 두 번째 이상일 때만 터진다(실측: 동일 시나리오를 순수
    파이썬 스크립트로 실행하면 재현되지 않음 — pytest 세션 특유의
    스레드/시그널 상태 차이로 보인다). all_hot도 여전히 gemv_worklist
    hot 커널을 실경로로 태우므로 M<=32 worklist force_graph_path 자체의
    검증력은 그대로 유지된다."""
    plan = make_plan("all_hot", gpu_warm="gemv_worklist")
    w13, w2 = make_weights(exact=True)
    ex = build_executor(plan, w13, w2, force_graph_path=True)
    for m in (1, 4):
        x, ids, tw = make_inputs(m=m, exact=True)
        ex.run_layer(0, x.cuda(), ids.cuda(), tw.cuda())
    assert set(ex._qlen_pins_graph) >= {1, 4}
    assert int(ex._qlen_pins_graph[1][0]) == 1
    assert int(ex._qlen_pins_graph[4][0]) == 4
    assert ex._qlen_pins_graph[1].data_ptr() != ex._qlen_pins_graph[4].data_ptr()


@cuda_required
def test_capture_mode_fn_routes_graph_path():
    """주입된 capture_mode_fn이 True를 돌리면 강제(force_graph_path)나 실제
    캡처 없이도 graph-safe 경로가 선택돼야 한다 — sglang CudaGraphRunner의
    워밍업 구간 감지(조립 지점이 주입하는 신호)의 라우팅 고정 테스트.
    간접 증거: graph 경로는 ids_cpu D2H 없이 device-sel로 도므로,
    capture-mode 강제 시의 출력이 eager 레퍼런스와 bitwise 일치해야 한다."""
    plan = make_plan("mixed")
    w13, w2 = make_weights()
    toggle = _Toggle()
    ex = build_executor(plan, w13, w2, capture_mode_fn=toggle)
    x, ids, w = make_inputs(1, seed=400)

    ref = run1(ex, x, ids, w)   # 진짜 eager (toggle off)
    toggle.on = True
    out = run1(ex, x, ids, w)   # force 없음·캡처 없음 — capture_mode_fn만 True
    assert torch.equal(out, ref), (
        f"max abs diff {(out.float() - ref.float()).abs().max().item()}"
    )
