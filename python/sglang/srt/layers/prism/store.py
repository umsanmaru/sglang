"""K-인덱스 스토어의 공유 어휘 — dtype, 상한, run 판정.

`moe/prism/index.py`에서 올라왔다 (2026-08-31). expert 축과 무관하다: "티어가
소유한 K행 번호를 uint16으로 담는다"는 MoE든 dense든 같은 결정이고, 두 곳에서
따로 정의하면 **드리프트가 곧 무증상 오답**이다 — 한쪽이 int32로 올라가고 다른
쪽이 안 올라가면 wrap된 인덱스가 전혀 다른 행을 읽는데, 그건 결과만 틀리고
아무 예외도 안 난다.

`geometry.py`가 아니라 여기 있는 이유: torch를 끌어온다. `geometry`는
`plan.py`가 의존하는 순수 stdlib 모듈이고 그 성질을 유지한다.
"""

from __future__ import annotations

import torch

# 인덱스는 uint16 (VRAM·L2 트래픽이 int32의 절반), 오프셋은 int32.
IDX_DTYPE = torch.uint16
OFF_DTYPE = torch.int32

# uint16 인덱스의 상한. K가 이보다 크면 dtype을 올려야 한다 — 조용히 wrap되면
# 전혀 다른 행을 읽는 무증상 오답이므로 로드 시 즉사한다.
MAX_K = (1 << 16) - 1


def is_row_run(rows: torch.Tensor) -> bool:
    """단위 stride 오름차순 구간인가 (길이 0/1은 참).

    참이면 소비자가 gather를 건너뛰고 포인터 오프셋으로 대체할 수 있다 — 밴드
    경로와의 비트일치가 이 판정 위에서 성립한다.
    """
    if int(rows.numel()) <= 1:
        return True
    return bool(torch.equal(rows[1:] - rows[:-1], torch.ones_like(rows[:-1])))
