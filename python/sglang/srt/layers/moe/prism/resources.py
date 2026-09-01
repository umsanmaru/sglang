"""ExecutionResources: 영구 메모리·스트림의 유일한 소유자.

계약 ④의 invariant를 이 모듈이 물리적으로 보장한다:
- primitive는 영구 메모리를 절대 할당하지 않는다 — 전부 여기서 빌린다.
- "graph가 참조하는 storage identity는 바뀌지 않는다" → 모든 버퍼는 생성 후
  재할당되지 않고, 내용 갱신은 `.copy_()` in-place만. (setter를 제공하지 않는
  것으로 계약을 API 형태로 강제)

남은 것은 **ColdStaging 하나**다. warm이 제자리 UVA 읽기가 되고 GPU 티어가
pair-native가 되면서 `DeviceArena`(warm/hot gather 목적지), stager 스크래치,
sel 버퍼가 전부 소비자를 잃었다 — 그것들은 bmm이 연속 배치 축을 요구해서
존재했고, 가변 per-expert K가 bmm을 불가능하게 만들면서 함께 사라졌다.

ColdStaging: cold 경로의 pinned 왕복 버퍼 (x D2H / act D2H / partial H2D).
act는 per-expert 값이므로 [max_tokens, top_k, inter]다 (계약 ② 정오표 참조).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.moe.prism.plan import Plan, Proj


@dataclass(frozen=True)
class ResourceSpec:
    """버퍼 크기를 결정하는 수량 전부. Plan + 실행 설정에서 파생.

    티어별 K 치수(k_warm/k_hot)와 n_slots가 사라졌다: 스토어가 flat + offset이
    되면서 크기를 로더가 알고, arena가 없어지면서 slot 개념 자체가 없다.
    """

    max_tokens: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    device: torch.device

    @classmethod
    def from_plan(
        cls, plan: Plan, *, max_tokens: int, device: torch.device,
    ) -> "ResourceSpec":
        return cls(
            max_tokens=max_tokens,
            top_k=plan.dims.top_k,
            hidden_size=plan.dims.hidden_size,
            intermediate_size=plan.dims.intermediate_size,
            device=torch.device(device),
        )

    def n_of(self, proj: Proj) -> int:
        return (
            self.hidden_size if proj is Proj.DOWN else self.intermediate_size
        )


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
        # warm-kt 인스턴스(prefill에서 CPU가 warm 행을 계산하는 모드)의 partial — cold와
        # 같은 (m, j) 행을 전 N열에 쓰므로 버퍼가 따로여야 한다. 지연 할당 (모드가 꺼져
        # 있으면 만들지 않는다; 만든 뒤에는 재할당 금지 — 계약 ④).
        self._warm_partial_gateup: Optional[torch.Tensor] = None
        self._warm_partial_down: Optional[torch.Tensor] = None
        # 라우터 가중 (정규화 전). kt가 threshold를 직접 구하는 데 쓴다:
        # s = clip(p - lam*(g_e - ḡ), 0, pmax) → 격자 조회. _x와 같은 취급이면
        # 충분하다 — 채우기가 stream D2H이므로 다음 레이어의 쓰기가 cold host
        # node와 stream 순서로 직렬화된다.
        self._topk_w = torch.zeros(m, k, dtype=torch.float32, **kw)

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

    def fill_topk_w(self, w: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        """w: [m, k] 라우터 가중. fp32로 캐스팅해 내린다."""
        return self._fill(self._topk_w, w.float(), non_blocking)

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

    def topk_w_ptr(self) -> int:
        return self._topk_w.data_ptr()

    def gateup_out(self, num_tokens: int) -> torch.Tensor:
        return self._partial_gateup[:num_tokens]

    def _ensure_warm(self) -> None:
        if self._warm_partial_gateup is None:
            m, k = self.spec.max_tokens, self.spec.top_k
            h, i = self.spec.hidden_size, self.spec.intermediate_size
            kw = dict(pin_memory=self._partial_gateup.is_pinned())
            self._warm_partial_gateup = torch.empty(m, k, 2 * i, dtype=torch.bfloat16, **kw)
            self._warm_partial_down = torch.empty(m, k, h, dtype=torch.bfloat16, **kw)

    def warm_partial_gateup_ptr(self) -> int:
        self._ensure_warm()
        return self._warm_partial_gateup.data_ptr()

    def warm_partial_down_ptr(self) -> int:
        self._ensure_warm()
        return self._warm_partial_down.data_ptr()

    def warm_gateup_out(self, num_tokens: int) -> torch.Tensor:
        return self._warm_partial_gateup[:num_tokens]

    def warm_down_out(self, num_tokens: int) -> torch.Tensor:
        return self._warm_partial_down[:num_tokens]

    def down_out(self, num_tokens: int) -> torch.Tensor:
        return self._partial_down[:num_tokens]


class ExecutionResources:
    """staging + 전역 스트림 묶음. 프로세스에 1벌."""

    def __init__(self, spec: ResourceSpec, *, pin_memory: bool = True):
        self.spec = spec
        self.staging = ColdStaging(spec, pin_memory=pin_memory)
        # warm 전용 스트림 (2026-08-26 부활 — 용도는 다르다). stager 시절엔
        # 전송용이었고, 지금은 prefill grouped 경로에서 warm(PCIe 바운드)
        # 커널을 hot(compute 바운드)과 겹치기 위한 것이다. executor가 fork/join
        # 이벤트를 관리하고, 여기는 소유만 한다 (스트림 핸들은 storage가 아니라
        # 계약 ④의 재할당 금지 대상이 아니지만, 프로세스에 1개면 족하다).
        self.warm_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=spec.device)
            if spec.device.type == "cuda" and torch.cuda.is_available() else None
        )
        # cold_async 모드의 cold 전용 stream + 완료 플래그 (pinned int32, kt가 쓰고 GPU가 읽음).
        # 플래그는 단조 증가 seq — phase마다 값을 올려 재사용한다 (재할당 금지, 계약 ④).
        self.cold_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=spec.device)
            if spec.device.type == "cuda" and torch.cuda.is_available() else None
        )
        self.cold_flag = torch.zeros(1, dtype=torch.int32, pin_memory=pin_memory) if pin_memory \
            else torch.zeros(1, dtype=torch.int32)
        self.cold_seq = 0
