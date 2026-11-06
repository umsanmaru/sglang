"""Format gather, padding and index helpers; no cache ownership."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sglang.srt.layers.moe.prism.index import TierIndex
from sglang.srt.layers.moe.prism.plan import ModelDims, PlanError, Proj

def _host(t: torch.Tensor) -> torch.Tensor:
    return t.cpu() if t.is_cuda else t


def _check_shapes(where, a, exp_a, b, exp_b):
    if tuple(a.shape) != tuple(exp_a) or tuple(b.shape) != tuple(exp_b):
        raise PlanError(
            f"{where}: weight shape mismatch vs plan dims: "
            f"{tuple(a.shape)} (expected {tuple(exp_a)}), {tuple(b.shape)} (expected {tuple(exp_b)}) — "
            f"plan이 다른 모델에 적용되고 있을 가능성")


def _proj_view(w13: torch.Tensor, w2: torch.Tensor, inter: int, proj: Proj) -> torch.Tensor:
    """proj의 ckpt-방향 소스 [E, N, K…] 뷰 (gate가 w13 앞 절반, up이 뒤 절반)."""
    if proj is Proj.GATE:
        return w13[:, :inter]
    if proj is Proj.UP:
        return w13[:, inter:]
    return w2


def _e8m0_bytes(scales: torch.Tensor) -> torch.Tensor:
    """배율 소스를 E8M0 바이트 [E, N, K/32] u8로. fp32(로더 캐스팅)면 2의 거듭제곱 검증 후
    지수 바이트로, e8m0/u8이면 그대로. 2의 거듭제곱이 아니면 MXFP4가 아니므로 즉사."""
    if scales.dtype == torch.uint8:
        return scales
    if scales.dtype == getattr(torch, "float8_e8m0fnu", None):
        return scales.view(torch.uint8)
    f = scales.float().contiguous()
    bits = f.view(torch.int32)
    if torch.any(bits & 0x807FFFFF):
        bad = int((bits & 0x807FFFFF).ne(0).sum())
        raise PlanError(f"mxfp4 scales: {bad} entries are not powers of two (sign/mantissa bits set)")
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def _half_index(ti: TierIndex, div: int, band_start, uniform_k) -> TierIndex:
    """k 단위 TierIndex → 행 단위(div 배수로 묶인) TierIndex: 페어(div=2) 또는 블록(div=32).
    32-정렬(check_index)이 보장되므로 각 묶음의 첫 원소 / div 가 행 번호다."""
    idx = ti.idx.to(torch.int64)
    return TierIndex(row_off=ti.row_off // div, idx=(idx.reshape(-1, div)[:, 0] // div).to(torch.int64),
                     contiguous=ti.contiguous)


def _gather_rows(src: torch.Tensor, ti: TierIndex, uniform_k, band_start) -> torch.Tensor:
    """[E, N, R] 소스에서 flat 스토어 [Σₑ r[e], N]를 만든다 (CPU). 밴드 퇴화형이면 한 번의
    transpose+contiguous; 일반 경로는 expert 루프 (배치 gather의 int64 인덱스 물질화가 더 비싸다)."""
    E, N, _ = src.shape
    if band_start is not None:
        return (src[:, :, band_start: band_start + uniform_k].transpose(1, 2).contiguous().reshape(-1, N))
    out = torch.empty(int(ti.row_off[-1]), N, dtype=src.dtype)
    for e in range(E):
        o0, o1 = int(ti.row_off[e]), int(ti.row_off[e + 1])
        if o1 > o0:
            rows = ti.idx[o0:o1].to(torch.int64)
            out[o0:o1] = src[e].t().index_select(0, rows)
    return out


def _cold_flat_rows(src: torch.Tensor, ti: TierIndex, real_rows, tile: int, *, pad_value=0, div: int = 1):
    """cold 스토어 [Σₑ N·r_pad(e)]와 패딩된 (row_off, k_index(k 단위), real_rows)를 만든다.

    src는 [E, N, R] (R = K/div: bf16이면 K, nibble이면 K/2, 배율이면 K/32), ti는 **행 단위**
    TierIndex(div로 나눈 인덱스). expert 블록은 ckpt 방향 [N, r_pad(e)]이고 패딩 열은 pad_value.
    반환 row_off/k_index는 항상 **k 단위**(div 배)로 — 소비자(kt KIndex, GPU 로더)가 k를 쓴다.
    """
    E, N, _ = src.shape
    pad = [((int(r) + tile - 1) // tile) * tile for r in real_rows]       # k 단위
    off = torch.zeros(E + 1, dtype=torch.int32)
    for e, kp in enumerate(pad):
        off[e + 1] = int(off[e]) + kp
    total_k = int(off[-1])
    real_t = torch.tensor([int(r) for r in real_rows], dtype=torch.int32)
    # 밴드 퇴화형(전 expert 같은 연속 구간, 패딩 없음)은 슬라이스 한 번의 contiguous 복사다 —
    # expert 루프(256 × index_select)는 DSV4 치수에서 층당 ~20 s였다 (2026-08-28 서버 로그).
    kr0 = int(real_rows[0]) if E else 0
    if (ti.contiguous and E and all(int(r) == kr0 for r in real_rows) and kr0 and pad[0] == kr0):
        start = int(ti.idx[0])   # ti는 이미 행 단위(div로 나눈) 인덱스다
        rlen = kr0 // div
        starts = ti.idx.to(torch.int64)[ti.row_off[:-1].long()]
        if bool(torch.all(ti.row_off[1:] - ti.row_off[:-1] == rlen)) and bool(torch.all(starts == start)):
            flat = src[:, :, start: start + rlen].contiguous().reshape(-1)
            idx = (torch.arange(kr0, dtype=torch.int64) + start).repeat(E) if div == 1 \
                else torch.zeros(total_k, dtype=torch.int64)
            return flat, off, idx, real_t
    from sglang.srt.layers.moe.prism.loading_gather import gather_padded_columns

    return gather_padded_columns(src, ti, real_rows, tile, pad_value=pad_value, div=div)


def _tensor_view(ptr: int, nbytes: int, dtype: torch.dtype) -> torch.Tensor:
    """host 메모리 주소를 1-D 텐서로 본다 (복사 없음). 소유자는 kt."""
    import ctypes

    esz = torch.tensor([], dtype=dtype).element_size()
    n = nbytes // esz
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
    return torch.frombuffer(buf, dtype=torch.uint8, count=nbytes).view(dtype)[:n]


def _padded_kindex(ti: TierIndex, off: torch.Tensor, real_rows) -> torch.Tensor:
    """k 단위 TierIndex → 타일 올림된 오프셋(off)에 맞춘 k 인덱스 (패딩 항목은 0)."""
    idx = torch.zeros(int(off[-1]), dtype=torch.int64)
    for e in range(ti.num_experts):
        kr, o0 = int(real_rows[e]), int(off[e])
        if kr:
            idx[o0: o0 + kr] = ti.idx[int(ti.row_off[e]): int(ti.row_off[e]) + kr].to(torch.int64)
    return idx


