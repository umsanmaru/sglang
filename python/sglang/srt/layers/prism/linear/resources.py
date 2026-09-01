"""dense cold의 pinned staging — 영구 메모리의 유일한 소유자 (계약 ④).

계약 ④의 invariant를 이 모듈이 물리적으로 보장한다:

- **primitive는 영구 메모리를 절대 할당하지 않는다.** executor와 backend는 여기서
  빌린다. 이유는 주소다 — CUDA graph 캡처는 커널 인자에 **포인터 값을 구워 넣고**
  replay가 그 주소를 다시 쓰며, cold는 `data_ptr()`를 넘긴 뒤 kt host node가
  **나중에** 역참조한다. 호출마다 새로 할당하면 둘 다 조용한 오답이 된다.
- 그래서 setter가 없다. 내용 갱신은 `.copy_()` in-place뿐이다.

MoE `moe/prism/resources.py`와 갈리는 것 둘:

**버퍼가 그룹별이다.** MoE는 (hidden, inter) 두 치수로 전 레이어를 덮지만 dense는
proj마다 K/N이 달라 형상 그룹마다 한 벌이 필요하다. 그룹 안에서는 전 layer가
공유한다 — 한 호출이 submit→sync로 닫히고 같은 그룹의 다음 layer는 그 뒤에 오기
때문이다.

**`expert_ids`가 정적이다.** 슬롯 신원은 로드 타임에 고정이므로 step마다 조달할
것이 없다 (계약 ④ dense 고유). MoE에서 eager/graph를 가르던 유일한 지점이
여기서는 사라진다 — 한 번 채우고 포인터만 넘긴다.

**`qlen`은 bs별 전용이다.** 캡처가 굽는 포인터를 eager의 버퍼와 공유하면 나중의
eager prefill 쓰기에 노출돼 replay마다 cold가 L토큰 분량을 계산하는 stale-share
버그가 된다 (MoE Finding A, 30B decode 328→56 ms/tok). bs가 다른 replay끼리도
절대 공유하지 않는다.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class LinearColdStaging:
    """한 형상 그룹의 pinned 왕복 버퍼. 생성 후 재할당 금지.

    `x`는 D2H 목적지(kt 입력, full-width), `out`은 H2D 원본(kt partial)이다.
    `expert_ids`는 로드 타임에 한 번 채운다.
    """

    def __init__(self, *, max_tokens: int, k: int, out_cols: int, num_experts: int,
                 top_k: int = 1, pin_memory: bool = True):
        kw = dict(pin_memory=pin_memory)
        self.max_tokens = int(max_tokens)
        self._x = torch.empty(max_tokens, k, dtype=torch.bfloat16, **kw)
        # partial은 bf16 (계약 ⑤: wire dtype = bf16, kt `to_mat` 재사용).
        self._out = torch.empty(max_tokens, top_k, out_cols, dtype=torch.bfloat16, **kw)
        # 슬롯 신원 = expert id. 값이 로드 타임에 확정되므로 여기서 굳힌다.
        self._ids = torch.arange(num_experts, dtype=torch.int64).contiguous()
        if pin_memory:
            pinned = torch.empty(num_experts, dtype=torch.int64, pin_memory=True)
            pinned.copy_(self._ids)
            self._ids = pinned
        self._ids_stride = self._ids.element_size()

    # ── 포인터 (kt는 정수 주소만 받는다) ─────────────────────────────────
    def x_ptr(self) -> int:
        return self._x.data_ptr()

    def out_ptr(self) -> int:
        return self._out.data_ptr()

    def expert_ids_ptr(self, expert: int) -> int:
        """`[expert]` 한 칸의 주소. top_k=1이라 kt는 여기서 int64 하나만 읽는다."""
        if not 0 <= expert < int(self._ids.numel()):
            raise IndexError(f"expert {expert} out of range 0..{int(self._ids.numel()) - 1}")
        return self._ids.data_ptr() + expert * self._ids_stride

    # ── 내용 (in-place만) ────────────────────────────────────────────────
    def fill_x(self, x: torch.Tensor, *, non_blocking: bool = True) -> None:
        """GPU `[m, k]` → pinned. cold 호출이 stream host node라 `non_blocking=True`로
        enqueue만 해도 된다 — kt task가 같은 stream에서 이 복사 뒤에 실행된다.
        host 경로(stream=None)로 부를 때는 호출자가 동기를 책임진다."""
        m = x.shape[0]
        if m > self.max_tokens:
            raise ValueError(f"M={m} exceeds cold staging max_tokens={self.max_tokens}")
        if x.shape[1] != self._x.shape[1]:
            raise ValueError(f"x has K={x.shape[1]} but staging holds {self._x.shape[1]}")
        self._x[:m].copy_(x, non_blocking=non_blocking)

    def out_view(self, m: int) -> torch.Tensor:
        """`[m, out_cols]` — top_k=1이라 slot 축이 접힌다. sync 이전의 내용은
        undefined다 (계약 ②-4 완료 계약)."""
        return self._out[:m, 0]


class LinearColdResources:
    """프로세스 전역 1벌. 그룹별 staging과 bs별 qlen 핀의 소유자."""

    def __init__(self, *, max_tokens: int, pin_memory: bool = True):
        self.max_tokens = int(max_tokens)
        self._pin = bool(pin_memory)
        self._staging: Dict[object, LinearColdStaging] = {}
        self._qlen: Dict[Tuple[int, bool], torch.Tensor] = {}

    def attach(self, group) -> LinearColdStaging:
        """그룹의 staging을 만들어 붙인다 (finalize에서 1회). 두 번 부르면 즉사한다
        — 재할당은 캡처가 구운 주소를 무효화한다."""
        if group.key in self._staging:
            raise RuntimeError(f"staging already attached for {group.key.label}")
        st = LinearColdStaging(
            max_tokens=self.max_tokens, k=group.x_width, out_cols=group.out_cols,
            num_experts=group.num_experts, pin_memory=self._pin,
        )
        self._staging[group.key] = st
        group.staging = st
        return st

    def qlen_ptr(self, m: int, *, capture: bool) -> int:
        """`m`을 담은 pinned int32의 주소.

        `capture=True`면 **bs 전용** 버퍼를 쓰고 값을 한 번만 쓴다 — 캡처가 그
        주소를 굽기 때문이다. eager는 공용 버퍼 하나에 매 step 덮어쓴다. 둘을
        섞으면 replay가 나중의 eager 값을 읽는다 (MoE Finding A).
        """
        key = (m if capture else -1, capture)
        buf = self._qlen.get(key)
        if buf is None:
            buf = torch.zeros(1, dtype=torch.int32,
                              pin_memory=self._pin and torch.cuda.is_available())
            self._qlen[key] = buf
        if capture:
            buf.fill_(m)          # 상수 — 캡처 시 1회
        else:
            buf[0] = m            # host 즉시쓰기: stream 순서의 보호를 받지 않는다
        return buf.data_ptr()

    def warmup(self, batch_sizes=()) -> None:
        """캡처 전에 qlen 핀을 미리 잡는다 — 캡처 안에서 처음 할당되면 graph
        전용 풀에 들어간다."""
        self.qlen_ptr(1, capture=False)
        for bs in batch_sizes:
            self.qlen_ptr(int(bs), capture=True)
