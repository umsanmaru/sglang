"""FP8 테스트 공용 참조 — e4m3 dequant, row 스토어 생성, 정확표현 입력.

규약(DeepSeek blockwise fp8, kt `fp8-moe.hpp`와 같은 것): 원소 = e4m3 바이트 (bit7 부호,
bit6:3 지수 bias 7, bit2:0 가수), 배율 = fp32 `scale_inv` 하나가 원본 128 n × 128 k 블록을
덮고 `w = q · scale_inv`.

디코드는 커널(`prism_fp8.cuh`)·CPU 포팅(`ktf8_fp8_32`)과 **같은 비트 산술**이다: 지수·가수가
모두 0인 코드만 0이고, denormal·NaN 코드는 유한한 다른 값이 된다 (인코더가 그런 코드를 만들지
않는다는 공통 전제 — 테스트도 만들지 않는다).
"""

from __future__ import annotations

import torch

BLK = 128


def e4m3_table() -> torch.Tensor:
    """바이트 256개 → fp32 값 (커널의 비트 산술과 동일)."""
    b = torch.arange(256, dtype=torch.int32)
    mag = (b & 0x7F) << 4
    bits = ((b & 0x80) << 8) | torch.where(mag != 0, mag + 0x3C00, torch.zeros_like(mag))
    return (bits << 16).view(torch.float32).clone()


_TABLE = e4m3_table()


def dequant_ckpt(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """체크포인트 방향 [N, K] u8 + [N/128, K/128] fp32 → fp32 [N, K]."""
    vals = _TABLE[codes.long()]
    sc = scales.repeat_interleave(BLK, dim=0).repeat_interleave(BLK, dim=1)
    return vals * sc


def random_expert_ckpt(N: int, K: int, g: torch.Generator, exact: bool = False):
    """체크포인트 형태 expert 하나: codes u8 [N, K], scales fp32 [N/128, K/128].

    exact=True: 배율 1.0, 코드는 {0, ±1, ±2}에 해당하는 바이트 — 작은 정수 x와의 누산이
    fp32에서 정확해 커널 간 비트일치를 요구할 수 있다."""
    assert N % BLK == 0 and K % BLK == 0
    if exact:
        pool = torch.tensor([0x00, 0x38, 0x40, 0xB8, 0xC0], dtype=torch.uint8)  # 0, 1, 2, -1, -2
        codes = pool[torch.randint(0, 5, (N, K), generator=g)]
        scales = torch.ones(N // BLK, K // BLK, dtype=torch.float32)
    else:
        # 지수 1..13, 가수 임의, 부호 임의 — denormal(e=0)과 NaN(0x7F/0xFF)은 만들지 않는다.
        e = torch.randint(1, 14, (N, K), generator=g)
        m = torch.randint(0, 8, (N, K), generator=g)
        s = torch.randint(0, 2, (N, K), generator=g)
        codes = ((s << 7) | (e << 3) | m).to(torch.uint8)
        scales = torch.ldexp(torch.ones(N // BLK, K // BLK),
                             torch.randint(-3, 4, (N // BLK, K // BLK), generator=g)).float()
    return codes, scales


def row_store(codes_ckpt: torch.Tensor, scales_ckpt: torch.Tensor, rows: torch.Tensor):
    """expert 하나의 체크포인트 텐서와 128-정렬 k 인덱스 rows →
    (codes [k, N] u8, scales [k/128, N/128] fp32) — Prism GPU 스토어 방향(행 = k)."""
    blocks = (rows[0::BLK] // BLK).long()
    c = codes_ckpt.index_select(1, rows.long()).t().contiguous()
    s = scales_ckpt.index_select(1, blocks).t().contiguous()
    return c, s


def aligned_index(K: int, k_rows: int, g: torch.Generator) -> torch.Tensor:
    """128-블록 단위로 뽑은 오름차순 k 인덱스 (길이 k_rows, k_rows % 128 == 0)."""
    nb = K // BLK
    pick = torch.randperm(nb, generator=g)[: k_rows // BLK].sort().values
    return (pick[:, None] * BLK + torch.arange(BLK)[None, :]).reshape(-1)


def ktf8_off(n, k, K):
    """cpu-mm/kt fp8 타일 레이아웃의 바이트 오프셋 (tile_k2_fp8b128_port.hpp `ktf8_off`)."""
    return ((k & 1) + (n & 63) * 2 + ((k >> 1) & 15) * 128 + ((n >> 6) & 3) * 2048
            + (k >> 5) * 8192 + (n >> 8) * 8192 * (K // 32))


def tile_block(codes_ckpt: torch.Tensor, scales_ckpt: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """expert 하나를 kt `GemmKernelTileK2FP8B128::BufferB` 블록(u8 1-D)으로: 타일 코드 N·k B
    (64 B 올림) + 전치 fp32 배율 [k/128][N/128]. rows = 128-정렬 k 인덱스. N은 256 배수."""
    N = codes_ckpt.shape[0]
    k = int(rows.numel())
    assert N % 256 == 0 and k % BLK == 0
    src = codes_ckpt.index_select(1, rows.long())                  # [N, k]
    codes = torch.zeros(N * k, dtype=torch.uint8)
    nn = torch.arange(N).view(N, 1).expand(N, k)
    kk = torch.arange(k).view(1, -1).expand(N, k)
    off = ktf8_off(nn, kk, k)
    codes[off.reshape(-1)] = src.reshape(-1)
    blocks = (rows[0::BLK] // BLK).long()
    sc = scales_ckpt.index_select(1, blocks).t().contiguous().reshape(-1)  # [k/128][N/128] fp32
    pad = (-codes.numel()) % 64
    return torch.cat([codes, torch.zeros(pad, dtype=torch.uint8), sc.view(torch.uint8).reshape(-1)])
