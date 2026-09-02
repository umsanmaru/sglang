"""dense rejoin — 티어 partial의 fp32 합을 **한 패스**로 (계약 ⑤).

MoE `moe/prism/rejoin.py`의 dense 대응물이고, 훨씬 단순하다. MoE는 두 종류가
필요했다 — gateup은 `Σ → silu·up → bf16`, down은 `Σ → 라우터 가중 k축 합 → bf16`.
dense는 **둘 다 없다**: 활성화는 sglang의 `SiluAndMul`이 `apply` 밖에서 걸고
(`apply`가 `[M, 2I]`를 돌려줘야 하므로 융합할 수 없다 — 융합하려면 모델 파일이
prism을 알아야 한다), 라우터가 없으니 가중합도 없다. 남는 것은 합 하나다::

    rejoin(parts[≤3] bf16 [M, N]) → out bf16 [M, N]
        out = Σ parts (fp32 누산, 최종 1회 라운딩)

수치 계약은 MoE와 같다 — 누산은 fp32, wire(partial)는 bf16, 라운딩은 마지막
한 번. 합산 순서는 part 인덱스 오름차순으로 고정이라 결정적이다.

**티어가 하나면 커널을 부르지 않는다.** partial이 곧 답이므로 복사도 캐스팅도
필요 없다 — hot-only/cold-only plan(벤치의 양 끝)에서 이 경로가 기본이다.

Triton 커널은 graph 캡처 가능하다. 첫 컴파일이 캡처 워밍업에 얽히지 않게
`warmup()`을 startup에서 부른다.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

MAX_PARTS = 3   # hot + warm + cold


@triton.jit
def _rejoin_kernel(p0, p1, p2, out, n, NUM_PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n
    base = row * n
    acc = tl.load(p0 + base + cols, mask=mask, other=0.0).to(tl.float32)
    if NUM_PARTS > 1:
        acc += tl.load(p1 + base + cols, mask=mask, other=0.0).to(tl.float32)
    if NUM_PARTS > 2:
        acc += tl.load(p2 + base + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + base + cols, acc.to(tl.bfloat16), mask=mask)


def rejoin(parts: Sequence[Optional[torch.Tensor]], out: Optional[torch.Tensor] = None
           ) -> torch.Tensor:
    """`[M, N]` bf16 partial들의 fp32 합 → `[M, N]` bf16.

    None 항목은 건너뛴다 (그 티어가 이 proj에 없다). 하나면 그대로 돌려준다 —
    호출자는 **반환값을 소유하지 않는다고 가정하면 안 된다**: 티어가 하나면
    partial 자체가 나오므로, 뒤에서 제자리 수정하면 그 티어의 출력 버퍼를
    건드리는 것이다 (executor는 매 호출 새로 할당하므로 안전하다).
    """
    live = [p for p in parts if p is not None]
    if not live:
        raise ValueError("rejoin got no partials — plan은 [0,k)를 덮어야 한다")
    if len(live) == 1:
        return live[0]
    if len(live) > MAX_PARTS:
        raise ValueError(f"rejoin supports at most {MAX_PARTS} partials, got {len(live)}")

    m, n = live[0].shape
    for p in live[1:]:
        if p.shape != (m, n):
            raise ValueError(f"partial shape mismatch: {tuple(p.shape)} != {(m, n)}")
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=live[0].device)
    pad = list(live) + [live[0]] * (MAX_PARTS - len(live))   # 안 쓰는 포인터 자리
    block = 1024
    _rejoin_kernel[(m, triton.cdiv(n, block))](
        pad[0], pad[1], pad[2], out, n, NUM_PARTS=len(live), BLOCK=block,
    )
    return out


def warmup(device, n: int = 256) -> None:
    """JIT 컴파일을 startup으로 앞당긴다 (캡처 워밍업에 얽히지 않게)."""
    for k in (2, 3):
        parts = [torch.zeros(1, n, dtype=torch.bfloat16, device=device) for _ in range(k)]
        rejoin(parts)
