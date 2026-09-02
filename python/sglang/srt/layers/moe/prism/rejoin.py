"""rejoin 융합 커널 (계약 ⑤의 두 합산을 각각 **한 패스**로).

executor의 rejoin은 원래 torch 연산 사슬이었다 — 티어 partial(bf16) 셋을 각각
fp32로 캐스팅해 더하고, split·silu·mul·bf16 캐스팅을 따로 launch했다. M=1
decode에서는 launch 수(층당 ~30)가 비용이라 CUDA graph가 그것을 지웠지만,
prefill(M=2688)에서는 같은 사슬이 88 MB짜리 텐서를 ~10번 왕복하는 **대역폭**
비용이 되어 층당 2.5 ms(nsys 실측, 2026-08-27)로 노출됐다 — CPU cold 경로에서는
이 GPU 구간 동안 CPU가 놀기 때문에 TTFT에 그대로 더해진다.

여기 두 커널은 partial을 **한 번만** 읽고 결과를 한 번만 쓴다:

    rejoin_gateup(parts[≤4] bf16 [M,k,2I]) → act bf16 [M,k,I]
        acc = Σ parts (fp32) ; act = silu(acc[:I]) · acc[I:]
    rejoin_down(parts[≤4] bf16 [M,k,H], w fp32 [M,k]) → out bf16 [M,H]
        out[m] = Σ_j w[m,j] · Σ parts[m,j] (fp32)

수치 계약 ⑤는 그대로다 — 누산은 fp32, wire(partial)는 bf16, 최종 1회 라운딩.
합산 순서는 (part 0, 1, 2 → j 오름차순)으로 고정이라 결정적이다. torch 사슬과
비트일치하지는 않는다(fp32 합의 결합 순서가 다르다) — 정확표현 입력에서는
일치하고 일반 입력은 tolerance다 (plan 불변성 테스트와 같은 급).

Triton 커널은 graph 캡처 가능하다. 첫 컴파일이 캡처 워밍업에 얽히지 않게
`warmup()`을 startup에서 부른다 (method.py).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def _rejoin_gateup_kernel(
    p0, p1, p2, p3, p4, act,
    inter, n_rows, limit,
    NUM_PARTS: tl.constexpr, BLOCK: tl.constexpr, HAS_LIMIT: tl.constexpr,
):
    row = tl.program_id(0)          # pair (m·k + j)
    cb = tl.program_id(1)           # inter 블록
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    mask = cols < inter
    base = row * (2 * inter)
    g = tl.load(p0 + base + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(p0 + base + inter + cols, mask=mask, other=0.0).to(tl.float32)
    if NUM_PARTS > 1:
        g += tl.load(p1 + base + cols, mask=mask, other=0.0).to(tl.float32)
        u += tl.load(p1 + base + inter + cols, mask=mask, other=0.0).to(tl.float32)
    if NUM_PARTS > 2:
        g += tl.load(p2 + base + cols, mask=mask, other=0.0).to(tl.float32)
        u += tl.load(p2 + base + inter + cols, mask=mask, other=0.0).to(tl.float32)
    if NUM_PARTS > 3:
        g += tl.load(p3 + base + cols, mask=mask, other=0.0).to(tl.float32)
        u += tl.load(p3 + base + inter + cols, mask=mask, other=0.0).to(tl.float32)
    if NUM_PARTS > 4:
        g += tl.load(p4 + base + cols, mask=mask, other=0.0).to(tl.float32)
        u += tl.load(p4 + base + inter + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        # DSV4 swiglu_limit (참조 Expert.forward): up ∈ [−L, L], gate ≤ L — fp32에서, silu 전에.
        u = tl.minimum(tl.maximum(u, -limit), limit)
        g = tl.minimum(g, limit)
    a = g / (1.0 + tl.exp(-g)) * u   # silu(gate) · up, fp32
    tl.store(act + row * inter + cols, a.to(tl.bfloat16), mask=mask)


@triton.jit
def _rejoin_down_kernel(
    p0, p1, p2, p3, p4, w, out,
    hidden, top_k,
    NUM_PARTS: tl.constexpr, BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for j in range(top_k):
        base = (m * top_k + j) * hidden
        s = tl.load(p0 + base + cols, mask=mask, other=0.0).to(tl.float32)
        if NUM_PARTS > 1:
            s += tl.load(p1 + base + cols, mask=mask, other=0.0).to(tl.float32)
        if NUM_PARTS > 2:
            s += tl.load(p2 + base + cols, mask=mask, other=0.0).to(tl.float32)
        if NUM_PARTS > 3:
            s += tl.load(p3 + base + cols, mask=mask, other=0.0).to(tl.float32)
        if NUM_PARTS > 4:
            s += tl.load(p4 + base + cols, mask=mask, other=0.0).to(tl.float32)
        wj = tl.load(w + m * top_k + j)
        acc += s * wj
    tl.store(out + m * hidden + cols, acc.to(tl.bfloat16), mask=mask)


MAX_PARTS = 5  # hot, warm(GPU), cold(GPU), cold(CPU), warm(CPU) — hybrid+warm-cpu에서 공존


def _pad3(parts: Sequence[torch.Tensor]):
    ps = [p for p in parts if p is not None]
    if not 1 <= len(ps) <= MAX_PARTS:
        raise ValueError(f"rejoin expects 1..{MAX_PARTS} partials, got {len(ps)}")
    for p in ps:
        if p.dtype is not torch.bfloat16 or not p.is_contiguous():
            raise TypeError("rejoin partials must be contiguous bf16")
    while len(ps) < MAX_PARTS:
        ps.append(ps[0])  # 미사용 슬롯 — NUM_PARTS가 읽지 않는다
    return ps


def rejoin_gateup(parts: Sequence[Optional[torch.Tensor]], inter: int,
                  swiglu_limit: Optional[float] = None) -> torch.Tensor:
    """parts: bf16 [M, k, 2·inter] (gate 앞 절반, up 뒤 절반). 반환 act bf16 [M, k, inter].

    swiglu_limit(DSV4-Flash 10.0): None이 아니면 fp32 합 뒤 silu 전에 up을 [−L, L], gate를
    ≤ L로 자른다 (참조 `Expert.forward`와 같은 순서·같은 정밀도)."""
    ps = _pad3(parts)
    m, k, two_i = ps[0].shape
    if two_i != 2 * inter:
        raise ValueError(f"partial width {two_i} != 2*inter {2 * inter}")
    act = torch.empty(m, k, inter, dtype=torch.bfloat16, device=ps[0].device)
    n_used = sum(1 for p in parts if p is not None)
    block = 1024
    grid = (m * k, triton.cdiv(inter, block))
    _rejoin_gateup_kernel[grid](ps[0], ps[1], ps[2], ps[3], ps[4], act, inter, m * k,
                                float(swiglu_limit if swiglu_limit is not None else 0.0),
                                NUM_PARTS=n_used, BLOCK=block,
                                HAS_LIMIT=swiglu_limit is not None)
    return act


def rejoin_down(parts: Sequence[Optional[torch.Tensor]], w32: torch.Tensor) -> torch.Tensor:
    """parts: bf16 [M, k, H]; w32: fp32 [M, k] 라우터 가중. 반환 bf16 [M, H]."""
    ps = _pad3(parts)
    m, k, h = ps[0].shape
    if w32.dtype is not torch.float32 or tuple(w32.shape) != (m, k):
        raise TypeError(f"rejoin_down needs fp32 [M,k] weights, got {w32.dtype} {tuple(w32.shape)}")
    w32 = w32.contiguous()
    out = torch.empty(m, h, dtype=torch.bfloat16, device=ps[0].device)
    n_used = sum(1 for p in parts if p is not None)
    block = 1024
    grid = (m, triton.cdiv(h, block))
    _rejoin_down_kernel[grid](ps[0], ps[1], ps[2], ps[3], ps[4], w32, out, h, k,
                              NUM_PARTS=n_used, BLOCK=block)
    return out


def warmup(device: torch.device, inter: int, hidden: int, top_k: int,
           swiglu_limit: Optional[float] = None) -> None:
    """NUM_PARTS 1..MAX 변형을 미리 컴파일한다 (graph 캡처 워밍업과의 얽힘 방지).
    swiglu_limit이 있으면 그 변형(HAS_LIMIT)도 함께."""
    for n in range(1, MAX_PARTS + 1):
        gu = [torch.zeros(1, top_k, 2 * inter, dtype=torch.bfloat16, device=device)] * n
        rejoin_gateup(gu, inter)
        if swiglu_limit is not None:
            rejoin_gateup(gu, inter, swiglu_limit)
        dn = [torch.zeros(1, top_k, hidden, dtype=torch.bfloat16, device=device)] * n
        rejoin_down(dn, torch.ones(1, top_k, dtype=torch.float32, device=device))
    torch.cuda.synchronize(device)
