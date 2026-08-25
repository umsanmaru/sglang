import pytest
import torch

from sglang.srt.layers.moe.prism.plan import Proj
from sglang.srt.layers.moe.prism.resources import ExecutionResources, ResourceSpec
from sglang.srt.layers.moe.prism.stagers import (
    BatchedCopyStager,
    GatherKernelStager,
    PerSlotCopyStager,
)
from sglang.srt.layers.moe.prism.weights import WarmBand


def _band(e=16, rows=5, n=7):
    return WarmBand.from_band(
        torch.arange(e * rows * n, dtype=torch.bfloat16).reshape(e, rows, n).pin_memory())


@pytest.fixture
def prism_resources():
    """conftest에 공용 fixture가 없어 이 파일 로컬로 정의 (test_resources.py의
    ResourceSpec 직접 구성 패턴을 재사용)."""
    # intermediate_size/k_warm_*는 8로 맞춘다: gather 커널은 rows*N*2 % 16 == 0
    # (16B-vectorized UVA read)을 요구하므로(8*8*2=128), 이 값이면 gather와
    # per_slot/batched가 동일 spec을 공유하는 3-way 파라미터화 테스트가 성립한다.
    spec = ResourceSpec(
        max_tokens=8, top_k=3, hidden_size=64, intermediate_size=8,
        k_warm_gate=8, k_warm_up=8, k_warm_down=8, n_slots=8,
        device=torch.device("cuda"),
    )
    return ExecutionResources(spec)


def test_per_slot_stager_bitwise():
    band = _band()
    arena = torch.zeros(4, 5, 7, dtype=torch.bfloat16, device="cuda")
    # zeros는 current stream에 enqueue된다 — stager는 다른 stream에서 wait 없이
    # (wait_event=None) 쓰므로, 초기화가 끝나기 전에 copy가 돌면 zeros가 copy를
    # 덮는 레이스가 난다 (부하 시 간헐 재현). 프로덕션 executor는 WAR 시드
    # 이벤트가 이 순서를 보장한다 — 테스트는 여기서 명시 동기화로 대신한다.
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    evt = PerSlotCopyStager().stage(band, [3, 9, 1], arena, s, None, Proj.GATE)
    torch.cuda.current_stream().wait_event(evt)
    torch.cuda.synchronize()
    assert torch.equal(arena[0].cpu(), band.weights[3])
    assert torch.equal(arena[2].cpu(), band.weights[1])


@pytest.mark.parametrize(
    "make",
    [
        lambda r: PerSlotCopyStager(),
        lambda r: BatchedCopyStager(r),
        lambda r: GatherKernelStager(r),
    ],
    ids=["per_slot", "batched", "gather"],
)
def test_stagers_bitwise_equivalent(prism_resources, make):
    """PerSlot을 준거로 Batched/Gather 둘 다 bitwise 일치해야 한다 (Task 6:
    Task 4의 2-way 비교를 3-way로 확장)."""
    band = _band(e=32, rows=prism_resources.spec.k_warm_gate, n=prism_resources.spec.intermediate_size)
    ref = torch.zeros_like(prism_resources.arena.view(Proj.GATE))
    torch.cuda.synchronize()  # 위 per-slot 테스트와 같은 init-vs-stage 레이스 방지
    s = torch.cuda.Stream()
    e_ref = PerSlotCopyStager().stage(band, [5, 17, 2], ref, s, None, Proj.GATE)

    out = prism_resources.arena.view(Proj.GATE)
    stager = make(prism_resources)
    e_out = stager.stage(band, [5, 17, 2], out, s, None, Proj.GATE)

    torch.cuda.current_stream().wait_event(e_ref)
    torch.cuda.current_stream().wait_event(e_out)
    torch.cuda.synchronize()
    assert torch.equal(out[:3].cpu(), ref[:3].cpu())


def test_batched_stager_double_buffer_war(prism_resources):
    """회귀: scratch가 proj당 1벌 재사용이면, 이전 H2D가 아직 scratch를
    읽는 중에 다음 stage() 호출의 host index_select가 그걸 덮어써
    조용한 weight corruption이 난다. warm_stream을 블로커로 지연시켜
    이 WAR 위험을 결정론적으로 재현한다.

    더블버퍼는 slot 0/1 두 개뿐이라 세 번째 호출에서야 call 1의 버퍼가
    실제로 재사용된다 (호출 1→buffer0, 2→buffer1, 3→buffer0). 그래서
    guard-event synchronize()가 없어도 통과하는 2-call 케이스로는 이
    회귀를 못 잡는다 — 3번째 호출로 buffer0 재사용을 강제하고, blocker는
    호출 1에만 걸어 그 H2D가 지연된 채로 buffer0에 남아있을 때 호출 3의
    host index_select가 그 자리를 덮어쓰는지를 검증한다."""
    spec = prism_resources.spec
    band = _band(e=16, rows=spec.k_warm_gate, n=spec.intermediate_size)
    dst1 = torch.zeros(spec.n_slots, spec.k_warm_gate, spec.intermediate_size,
                        dtype=torch.bfloat16, device="cuda")
    dst2 = torch.zeros_like(dst1)
    dst3 = torch.zeros_like(dst1)
    torch.cuda.synchronize()  # init-vs-stage 레이스 방지 (blocker 의미는 불변)

    blocker_stream = torch.cuda.Stream()
    with torch.cuda.stream(blocker_stream):
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        for _ in range(200):
            a = a @ b
    blocker = torch.cuda.Event()
    blocker.record(blocker_stream)

    stager = BatchedCopyStager(prism_resources)
    # 호출 1: buffer0, blocker에 막혀 H2D가 지연된다.
    e1 = stager.stage(band, [0, 1, 2], dst1, prism_resources.warm_stream, blocker, Proj.GATE)
    # 호출 2: buffer1, 지연 없이(wait_event=None) 즉시 진행 — buffer0은
    # 아직 건드리지 않는다.
    e2 = stager.stage(band, [3, 4, 5], dst2, prism_resources.warm_stream, None, Proj.GATE)
    # 호출 3: 다시 buffer0 재사용 (flip이 0으로 돌아옴), wait_event=None.
    # guard.synchronize()가 없으면 이 host index_select가 호출 1의 H2D가
    # 아직 읽고 있는 buffer0을 곧바로 덮어써 조용한 corruption이 난다.
    e3 = stager.stage(band, [6, 7, 8], dst3, prism_resources.warm_stream, None, Proj.GATE)
    torch.cuda.current_stream().wait_event(e1)
    torch.cuda.current_stream().wait_event(e2)
    torch.cuda.current_stream().wait_event(e3)
    torch.cuda.synchronize()
    assert torch.equal(dst1[:3].cpu(), band.weights[[0, 1, 2]].cpu())
    assert torch.equal(dst2[:3].cpu(), band.weights[[3, 4, 5]].cpu())
    assert torch.equal(dst3[:3].cpu(), band.weights[[6, 7, 8]].cpu())


def test_gather_stager_repeated_calls_bitwise(prism_resources):
    """GatherKernelStager의 sel 더블버퍼가 정상적으로 flip해도(0->1->0)
    3연속 호출이 각자 옳은 group_ids를 내는지 확인 — BatchedCopyStager의
    WAR 회귀 테스트와 같은 3-call 구조(더블버퍼 재사용을 강제)를 재사용하되,
    sel 내용은 매 호출 달라지므로 blocker 없이도 flip 정확성을 검증한다."""
    spec = prism_resources.spec
    band = _band(e=16, rows=spec.k_warm_gate, n=spec.intermediate_size)
    dst1 = torch.zeros(spec.n_slots, spec.k_warm_gate, spec.intermediate_size,
                        dtype=torch.bfloat16, device="cuda")
    dst2 = torch.zeros_like(dst1)
    dst3 = torch.zeros_like(dst1)
    torch.cuda.synchronize()  # init-vs-stage 레이스 방지

    stager = GatherKernelStager(prism_resources)
    e1 = stager.stage(band, [0, 1, 2], dst1, prism_resources.warm_stream, None, Proj.GATE)
    e2 = stager.stage(band, [3, 4, 5], dst2, prism_resources.warm_stream, None, Proj.GATE)
    e3 = stager.stage(band, [6, 7, 8], dst3, prism_resources.warm_stream, None, Proj.GATE)
    torch.cuda.current_stream().wait_event(e1)
    torch.cuda.current_stream().wait_event(e2)
    torch.cuda.current_stream().wait_event(e3)
    torch.cuda.synchronize()
    assert torch.equal(dst1[:3].cpu(), band.weights[[0, 1, 2]].cpu())
    assert torch.equal(dst2[:3].cpu(), band.weights[[3, 4, 5]].cpu())
    assert torch.equal(dst3[:3].cpu(), band.weights[[6, 7, 8]].cpu())


def test_gather_kernel_bitwise():
    from sglang.jit_kernel.prism_gather import gather_bands_from_pinned

    band = _band(e=32, rows=8, n=64)  # 8*64*2 = 1024B, multiple of 16
    sel = torch.tensor([5, 17, 2], dtype=torch.int32, device="cuda")
    dst = torch.zeros(3, 8, 64, dtype=torch.bfloat16, device="cuda")
    gather_bands_from_pinned(band.weights, sel, dst, torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.equal(dst.cpu(), band.weights[[5, 17, 2]])


def test_gather_kernel_device_bitwise():
    """device-src 변형(hot 티어용): index_select 참조와 비트 일치."""
    from sglang.jit_kernel.prism_gather import gather_bands_from_device

    src = (torch.arange(32 * 8 * 64, dtype=torch.bfloat16, device="cuda")
           .reshape(32, 8, 64))
    sel = torch.tensor([5, 17, 2], dtype=torch.int32, device="cuda")
    dst = torch.zeros(3, 8, 64, dtype=torch.bfloat16, device="cuda")
    gather_bands_from_device(src, sel, dst, torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = src.index_select(0, sel.long())
    assert torch.equal(dst, ref)


def test_gather_kernel_device_rejects_host_src():
    """device 변형은 CPU/pinned 소스를 즉사시켜야 한다 (pinned 변형과 계약 분리)."""
    from sglang.jit_kernel.prism_gather import gather_bands_from_device

    band = _band(e=8, rows=8, n=64)  # pinned host store
    sel = torch.tensor([1], dtype=torch.int32, device="cuda")
    dst = torch.zeros(1, 8, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(Exception):
        gather_bands_from_device(band.weights, sel, dst, torch.cuda.current_stream())
