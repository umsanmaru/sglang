"""expert 그룹핑 — prefill의 GPU 티어가 W를 expert당 한 번만 읽기 위한 pair 정렬.

worklist GEMV는 pair (m, j)가 좌표라 그룹핑이 필요 없었다 (블록이 `topk[pair]`를
스스로 읽는다). grouped GEMM은 반대로 **expert가 좌표**라 "이 expert의 pair들이
어디 있나"를 알아야 하고, 그것이 이 모듈이 만드는 세 텐서다:

    pair_sorted [M·k]  int32  expert 오름차순 정렬된 pair 번호 (p = m·k + j)
    pair_off    [E+1]  int32  expert e의 pair 구간
    tile_off    [E+1]  int32  expert e의 토큰 타일(TILE_M pair) 구간 — 커널의
                              blockIdx.y → (e, 타일) 역사상의 재료

전부 **device 연산**이고 host sync가 없다 (bincount는 CUDA에서 max를 .item()으로
읽어 동기화하므로 쓰지 않는다 — index_add_로 대신한다). 레이어당 한 번 만들어
gateup·down·hot·warm이 공유한다 (topk가 같으므로).

한 레이어의 세 GPU 티어 phase가 같은 객체를 받는 것이 이 값 객체의 존재 이유다 —
티어 안에서 만들면 티어 수만큼 정렬을 반복한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.jit_kernel.prism_grouped import TILE_M


@dataclass(frozen=True)
class Grouping:
    pair_sorted: torch.Tensor  # int32 [M·k]
    pair_off: torch.Tensor     # int32 [E+1]
    tile_off: torch.Tensor     # int32 [E+1]
    tile_m: int = TILE_M


def build_grouping(topk_ids: torch.Tensor, num_experts: int,
                   tile_m: int = TILE_M,
                   expert_mask: torch.Tensor | None = None) -> Grouping:
    """topk_ids [M, k] (cuda int32/int64) → Grouping. host sync 없음.

    expert_mask: bool [E] device — False인 expert는 **타일을 받지 않는다**(tile_off
    구간이 빈다). cold hybrid에서 GPU가 맡은 expert만 계산하게 하는 스위치다;
    pair_off는 그대로라 나머지 expert의 pair는 CPU(kt) 쪽이 처리한다."""
    flat = topk_ids.reshape(-1)
    e_sorted, pair_sorted = torch.sort(flat)
    dev = flat.device
    counts = torch.zeros(num_experts, dtype=torch.int32, device=dev)
    counts.index_add_(0, e_sorted, torch.ones_like(e_sorted, dtype=torch.int32))
    pair_off = torch.zeros(num_experts + 1, dtype=torch.int32, device=dev)
    torch.cumsum(counts, 0, dtype=torch.int32, out=pair_off[1:])
    tiles = (counts + (tile_m - 1)) // tile_m
    if expert_mask is not None:
        tiles = tiles * expert_mask.to(torch.int32)
    tile_off = torch.zeros(num_experts + 1, dtype=torch.int32, device=dev)
    torch.cumsum(tiles, 0, dtype=torch.int32, out=tile_off[1:])
    return Grouping(
        pair_sorted=pair_sorted.to(torch.int32),
        pair_off=pair_off, tile_off=tile_off, tile_m=tile_m,
    )
