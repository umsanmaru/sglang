"""ExecutionResources: 영구 메모리·스트림·이벤트의 유일한 소유자.

계약 ④의 invariant를 이 모듈이 물리적으로 보장한다:
- primitive는 영구 메모리를 절대 할당하지 않는다 — 전부 여기서 빌린다.
- "Stage 4(capture) 이후 graph가 참조하는 storage identity는 바뀌지 않는다"
  → 모든 버퍼는 생성 후 재할당되지 않고, 내용 갱신은 `.copy_()` in-place만.
  (setter를 제공하지 않는 것으로 계약을 API 형태로 강제)

배치:
- DeviceArena: warm 밴드의 step-time 목적지. gate/up이 동시 상주하고,
  down은 같은 storage를 재사용한다 — down 전송은 act 이후에만 일어나므로
  gate/up 내용에 대한 WAR가 데이터 의존으로 자동 충족된다.
- ColdStaging: cold 경로의 pinned 왕복 버퍼 (x D2H / act D2H / partial H2D).
  act는 per-expert 값이므로 [max_tokens, top_k, inter]다 (계약 ② 정오표
  참조 — 초판의 [qlen × inter]는 오기).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.moe.prism.plan import Plan, Proj, Tier


@dataclass(frozen=True)
class ResourceSpec:
    """버퍼 크기를 결정하는 수량 전부. Plan + 실행 설정에서 파생."""

    max_tokens: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    k_warm_gate: int  # 0 = 해당 proj에 warm 없음
    k_warm_up: int
    k_warm_down: int
    n_slots: int  # arena slot 수 = 한 그룹에서 동시에 다루는 distinct expert 수
    device: torch.device

    @classmethod
    def from_plan(
        cls,
        plan: Plan,
        *,
        max_tokens: int,
        device: torch.device,
        n_slots: Optional[int] = None,
    ) -> "ResourceSpec":
        """arena는 전 레이어가 공유하므로 warm 크기는 레이어 전체의 최대치."""
        k_warm = {proj: 0 for proj in Proj}
        seen: set[int] = set()
        for ep in plan.experts.values():
            if id(ep) in seen:
                continue
            seen.add(id(ep))
            for proj in Proj:
                k_warm[proj] = max(k_warm[proj], ep.proj(proj).rows(Tier.WARM))
        return cls(
            max_tokens=max_tokens,
            top_k=plan.dims.top_k,
            hidden_size=plan.dims.hidden_size,
            intermediate_size=plan.dims.intermediate_size,
            k_warm_gate=k_warm[Proj.GATE],
            k_warm_up=k_warm[Proj.UP],
            k_warm_down=k_warm[Proj.DOWN],
            n_slots=n_slots if n_slots is not None else plan.dims.top_k,
            device=torch.device(device),
        )

    def k_warm_of(self, proj: Proj) -> int:
        return {
            Proj.GATE: self.k_warm_gate,
            Proj.UP: self.k_warm_up,
            Proj.DOWN: self.k_warm_down,
        }[proj]

    def n_of(self, proj: Proj) -> int:
        return (
            self.hidden_size if proj is Proj.DOWN else self.intermediate_size
        )


def _view_as(flat: torch.Tensor, byte_offset: int, shape, dtype) -> torch.Tensor:
    numel = 1
    for s in shape:
        numel *= s
    nbytes = numel * dtype.itemsize
    return flat[byte_offset : byte_offset + nbytes].view(dtype).view(shape)


class DeviceArena:
    """warm 밴드의 device 상주 목적지 (bf16).

    gate/up은 동시 상주(서로소 구간), down은 flat buffer 선두를 재사용.
    slot i는 "이번 그룹의 i번째 distinct expert"의 밴드를 담는다.
    """

    def __init__(self, spec: ResourceSpec):
        if spec.device.type != "cuda":
            raise ValueError("DeviceArena requires a CUDA device")
        self.spec = spec
        itemsize = torch.finfo(torch.bfloat16).bits // 8
        gate_bytes = spec.n_slots * spec.k_warm_gate * spec.intermediate_size * itemsize
        up_bytes = spec.n_slots * spec.k_warm_up * spec.intermediate_size * itemsize
        down_bytes = spec.n_slots * spec.k_warm_down * spec.hidden_size * itemsize
        self._flat = torch.empty(
            max(gate_bytes + up_bytes, down_bytes),
            dtype=torch.uint8,
            device=spec.device,
        )
        self._gate = _view_as(
            self._flat, 0,
            (spec.n_slots, spec.k_warm_gate, spec.intermediate_size), torch.bfloat16,
        )
        self._up = _view_as(
            self._flat, gate_bytes,
            (spec.n_slots, spec.k_warm_up, spec.intermediate_size), torch.bfloat16,
        )
        # down은 gate/up storage를 재사용 (act 이후에만 쓰이므로 WAR 자동 충족)
        self._down = _view_as(
            self._flat, 0,
            (spec.n_slots, spec.k_warm_down, spec.hidden_size), torch.bfloat16,
        )

    def view(self, proj: Proj) -> torch.Tensor:
        """[n_slots, k_warm, N] bf16. identity 불변 — 재할당 없음."""
        return {Proj.GATE: self._gate, Proj.UP: self._up, Proj.DOWN: self._down}[proj]

    @property
    def nbytes(self) -> int:
        return self._flat.numel()


class ColdStaging:
    """cold 경로 pinned 왕복 버퍼. C++가 이 버퍼들의 포인터를 들게 되므로
    (K3 이후) 생성 후 재할당 금지 — 내용 갱신은 fill_*의 `.copy_()` 뿐이다."""

    def __init__(self, spec: ResourceSpec, *, pin_memory: bool = True):
        self.spec = spec
        m, k = spec.max_tokens, spec.top_k
        h, i = spec.hidden_size, spec.intermediate_size
        kw = dict(pin_memory=pin_memory)
        self._x = torch.empty(m, h, dtype=torch.bfloat16, **kw)
        self._expert_ids = torch.empty(m, k, dtype=torch.int64, **kw)
        self._act = torch.empty(m, k, i, dtype=torch.bfloat16, **kw)
        # partial은 bf16 (계약 ⑤ 개정: wire dtype = bf16, kt to_mat 재사용;
        # fp32 누산은 GPU rejoin에서 upcast로 수행)
        self._partial_gateup = torch.empty(m, k, 2 * i, dtype=torch.bfloat16, **kw)
        self._partial_down = torch.empty(m, k, h, dtype=torch.bfloat16, **kw)

    def _fill(self, buf: torch.Tensor, value: torch.Tensor,
              non_blocking: bool = False) -> torch.Tensor:
        # non_blocking=True는 cold stream 통합 경로(Task 8)용: D2H를 현재
        # stream에 enqueue만 한다 — 소비자(kt host node)가 같은 stream에
        # 순서대로 올라가므로 host-측 완료 보장이 불필요하고, CUDA graph
        # 캡처도 가능해진다. 기본 False = P0 blocking 동작 그대로.
        if value.shape[0] > self.spec.max_tokens:
            raise ValueError(
                f"{value.shape[0]} tokens exceed staging capacity "
                f"{self.spec.max_tokens}"
            )
        view = buf[: value.shape[0]]
        view.copy_(value, non_blocking=non_blocking)  # in-place만 — 계약 ④
        return view

    def fill_x(self, x: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        return self._fill(self._x, x, non_blocking)

    def fill_expert_ids(self, ids: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        """ids: cpu int64 (eager) 또는 cuda int64 (graph 경로 — 캡처 가능한
        async D2H로 pinned int64 버퍼에 내린다; dtype은 _expert_ids와 동일)."""
        return self._fill(self._expert_ids, ids, non_blocking)

    def fill_act(self, act: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        return self._fill(self._act, act, non_blocking)

    # ── cold submit에 넘기는 원시 주소들 ──────────────────────────────────
    # C++ 경계는 포인터가 곧 인터페이스다 — executor가 내부 버퍼(_x 등)를
    # 직접 만지지 않도록 노출면을 이 다섯 개로 한정한다.
    def x_ptr(self) -> int:
        return self._x.data_ptr()

    def expert_ids_ptr(self) -> int:
        return self._expert_ids.data_ptr()

    def act_ptr(self) -> int:
        return self._act.data_ptr()

    def partial_gateup_ptr(self) -> int:
        return self._partial_gateup.data_ptr()

    def partial_down_ptr(self) -> int:
        return self._partial_down.data_ptr()

    def gateup_out(self, num_tokens: int) -> torch.Tensor:
        return self._partial_gateup[:num_tokens]

    def down_out(self, num_tokens: int) -> torch.Tensor:
        return self._partial_down[:num_tokens]


class ExecutionResources:
    """arena + staging + 전역 스트림/이벤트 묶음. 프로세스에 1벌."""

    def __init__(self, spec: ResourceSpec, *, pin_memory: bool = True):
        self.spec = spec
        self.arena = DeviceArena(spec)
        self.staging = ColdStaging(spec, pin_memory=pin_memory)
        # 전역 warm stream 1개 (kt처럼 레이어당 만들지 않는다 — 문서 §8 규칙)
        self.warm_stream = torch.cuda.Stream(device=spec.device)
        self.evt_staged = torch.cuda.Event()
        self.evt_act = torch.cuda.Event()
        # 계약 ④: BatchedCopyStager의 host-측 결집 스크래치도 리소스가 소유한다.
        # proj당 2벌(더블버퍼) — 이전 H2D가 아직 읽는 중인 버퍼를 다음 호출의
        # host index_select가 덮어쓰는 WAR를 막기 위함 (stager가 이벤트로 가드).
        self._stage_scratch = {
            p: [
                torch.empty(self.arena.view(p).shape, dtype=torch.bfloat16).pin_memory()
                for _ in range(2)
            ]
            for p in Proj
        }
        # stage_index는 1벌만: BatchedCopyStager.stage() 안에서 host
        # index_select가 동기적으로 다 읽고 끝나므로(H2D처럼 비동기로 남는 게
        # 아님) 다음 호출이 덮어써도 WAR가 없다 — 더블버퍼 불필요.
        self._stage_index = torch.empty(spec.n_slots, dtype=torch.int64)
        # GatherKernelStager의 sel 버퍼. host copy_ → 비동기 H2D → gather 커널
        # 순으로 warm_stream에 태우므로 BatchedCopyStager의 scratch와 동일한
        # WAR 위험이 있다 — 2벌(더블버퍼)로 소유한다. sel은 proj별이 아니라
        # 한 그룹의 gate/up/down 3회 stage() 호출에 동일 내용으로 재사용되므로
        # (proj, slot) 키가 아니라 slot만으로 충분하다 (stager가 flip+guard로
        # 관리). sel_device는 디바이스측 쓰기라 스트림 순서로 자동 직렬화되므로
        # 단일 버퍼면 된다.
        self._sel_pinned = [
            torch.empty(spec.n_slots, dtype=torch.int32).pin_memory()
            for _ in range(2)
        ]
        self._sel_device = torch.empty(
            spec.n_slots, dtype=torch.int32, device=spec.device
        )

    def stage_scratch(self, proj: Proj, slot: int) -> torch.Tensor:
        """pinned host 결집 스크래치, arena.view(proj)와 동형 [n_slots, k_warm, N].

        slot은 0/1 — BatchedCopyStager가 소유한 더블버퍼 중 하나를 고른다."""
        return self._stage_scratch[proj][slot]

    def stage_index(self) -> torch.Tensor:
        """pinned int64 [n_slots] — BatchedCopyStager의 index_select 인덱스.
        host에서 동기 소비되므로(index_select가 즉시 다 읽음) 단일 버퍼로 충분."""
        return self._stage_index

    def sel_pinned(self, slot: int) -> torch.Tensor:
        """pinned int32 [n_slots] — GatherKernelStager의 expert-index 스테이징.
        slot은 0/1 더블버퍼 중 하나 (H2D WAR 방지, stagers.py의
        GatherKernelStager 참조)."""
        return self._sel_pinned[slot]

    def sel_device(self) -> torch.Tensor:
        """device int32 [n_slots] — gather 커널의 sel_device 인자.
        device측 쓰기는 stream-ordered이므로 단일 버퍼로 충분."""
        return self._sel_device
