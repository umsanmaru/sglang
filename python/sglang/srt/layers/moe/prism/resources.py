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

    def _fill(self, buf: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] > self.spec.max_tokens:
            raise ValueError(
                f"{value.shape[0]} tokens exceed staging capacity "
                f"{self.spec.max_tokens}"
            )
        view = buf[: value.shape[0]]
        view.copy_(value)  # in-place만 — 계약 ④
        return view

    def fill_x(self, x: torch.Tensor) -> torch.Tensor:
        return self._fill(self._x, x)

    def fill_expert_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self._fill(self._expert_ids, ids)

    def fill_act(self, act: torch.Tensor) -> torch.Tensor:
        return self._fill(self._act, act)

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
