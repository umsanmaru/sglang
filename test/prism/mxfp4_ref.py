"""MXFP4 테스트 공용 참조 — 체크포인트 규약의 dequant, pair-row 스토어 생성, 정확표현 입력.

규약(DeepSeek-V4-Flash `inference/convert.py`): 바이트 = k 짝수(하위 nibble) + k 홀수(상위),
nibble = bit3 부호 · e2m1 {0,.5,1,1.5,2,3,4,6}; 배율 E8M0 바이트 e → 2^(e−127), 32-k 블록당 1.
"""

from __future__ import annotations

import torch

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def dequant_ckpt(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """체크포인트 방향 [N, K/2] u8 + [N, K/32] u8 → fp32 [N, K] (convert.py의 stack(low, high))."""
    low = (codes & 0xF).long()
    high = (codes >> 4).long()
    vals = torch.stack([FP4_TABLE[low], FP4_TABLE[high]], dim=-1).flatten(-2)  # [N, K]
    sc = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.int() - 127)
    return vals * sc.repeat_interleave(32, dim=-1)


def dequant_pairrow(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """pair-row 스토어 [P, N] u8 + [G, N] u8 → fp32 [2P, N] (k행 순서)."""
    low = (codes & 0xF).long()
    high = (codes >> 4).long()
    vals = torch.stack([FP4_TABLE[low], FP4_TABLE[high]], dim=1).reshape(2 * codes.shape[0], -1)
    sc = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.int() - 127)
    return vals * sc.repeat_interleave(32, dim=0)


def random_expert_ckpt(N: int, K: int, g: torch.Generator, exact: bool = False):
    """체크포인트 형태 expert 하나: codes int8 [N, K/2], scales u8 [N, K/32].

    exact=True: 배율 2^0(e=127)만, 코드는 {0, ±1, ±2}에 해당하는 nibble — 작은 정수 x와의
    누산이 fp32에서 정확해 커널 간 비트일치를 요구할 수 있다."""
    if exact:
        # nibble 값 {0:0, 2:1.0, 4:2.0, 10:-1.0, 12:-2.0}
        pool = torch.tensor([0, 2, 4, 10, 12], dtype=torch.uint8)
        nib = pool[torch.randint(0, 5, (N, K), generator=g)]
        scales = torch.full((N, K // 32), 127, dtype=torch.uint8)
    else:
        nib = torch.randint(0, 16, (N, K), generator=g).to(torch.uint8)
        scales = torch.randint(120, 130, (N, K // 32), generator=g).to(torch.uint8)
    codes = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    return codes.view(torch.int8), scales


def pairrow_store(codes_ckpt: torch.Tensor, scales_ckpt: torch.Tensor, rows: torch.Tensor):
    """expert 하나의 체크포인트 텐서와 32-정렬 k 인덱스 rows → (codes [k/2, N], scales [k/32, N])."""
    pairs = (rows[0::2] // 2).long()
    blocks = (rows[0::32] // 32).long()
    c = codes_ckpt.view(torch.uint8).index_select(1, pairs).t().contiguous()
    s = scales_ckpt.index_select(1, blocks).t().contiguous()
    return c, s


def aligned_index(K: int, k_rows: int, g: torch.Generator) -> torch.Tensor:
    """32-블록 단위로 뽑은 오름차순 k 인덱스 (길이 k_rows, k_rows % 32 == 0)."""
    nb = K // 32
    pick = torch.randperm(nb, generator=g)[: k_rows // 32].sort().values
    return (pick[:, None] * 32 + torch.arange(32)[None, :]).reshape(-1)
