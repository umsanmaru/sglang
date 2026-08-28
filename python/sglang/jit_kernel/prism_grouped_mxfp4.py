"""Prism grouped GEMM — MXFP4 pair-row 스토어 판 (`prism_grouped_mxfp4.cuh`).

호출 규약은 bf16 `prism_grouped.py`와 같고 스토어 인자만 `(codes, scales)`다. 토큰
타일 크기는 bf16과 같은 `TILE_M`(grouping.py가 tile_off를 만들 때 쓰는 값)이어야 한다.
W-resident 변형은 없다 (fp4는 expert K 슬라이스가 smem 예산을 넘고, DSV4 치수에서는
expert당 pair가 타일 하나에 대개 들어가 재읽기가 드물다) — `wres_k_max` 인자는 받되 무시.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.prism_grouped import TILE_M  # noqa: F401  (그룹핑 타일 크기 공유)
from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_WRAPPERS = (
    "grouped_mxfp4_indexed",
    "grouped_mxfp4_indexed_pinned",
    "grouped_mxfp4_indexed_gateup",
    "grouped_mxfp4_indexed_pinned_gateup",
    "grouped_mxfp4_cold",
    "grouped_mxfp4_cold_gateup",
)


@cache_once
def _jit_prism_grouped_mxfp4_module() -> Module:
    return load_jit(
        "prism_grouped_mxfp4",
        cuda_files=["moe/prism_grouped_mxfp4.cuh"],
        cuda_wrappers=[(n, n) for n in _WRAPPERS],
    )


def warmup_jit() -> None:
    _jit_prism_grouped_mxfp4_module()


def _single(name):
    def fn(x2d, grouping, codes, scales, row_off, kidx, out3d, out_col_offset,
           x_row_is_pair, stream, max_blocks=0, wres_k_max=0) -> None:
        module = _jit_prism_grouped_mxfp4_module()
        with torch.cuda.stream(stream):
            getattr(module, name)(
                x2d, grouping.pair_sorted, grouping.pair_off, grouping.tile_off,
                codes, scales, row_off, kidx, out3d, int(out_col_offset),
                int(bool(x_row_is_pair)), int(max_blocks))
    fn.__name__ = name
    return fn


def _gateup(name):
    def fn(x2d, grouping, codes_g, scales_g, row_off_g, kidx_g,
           codes_u, scales_u, row_off_u, kidx_u, out3d,
           out_col_gate, out_col_up, x_row_is_pair, stream, max_blocks=0, wres_k_max=0) -> None:
        module = _jit_prism_grouped_mxfp4_module()
        with torch.cuda.stream(stream):
            getattr(module, name)(
                x2d, grouping.pair_sorted, grouping.pair_off, grouping.tile_off,
                codes_g, scales_g, row_off_g, kidx_g, codes_u, scales_u, row_off_u, kidx_u,
                out3d, int(out_col_gate), int(out_col_up), int(bool(x_row_is_pair)),
                int(max_blocks))
    fn.__name__ = name
    return fn


grouped_mxfp4_indexed = _single("grouped_mxfp4_indexed")
grouped_mxfp4_indexed_pinned = _single("grouped_mxfp4_indexed_pinned")
grouped_mxfp4_indexed_gateup = _gateup("grouped_mxfp4_indexed_gateup")
grouped_mxfp4_indexed_pinned_gateup = _gateup("grouped_mxfp4_indexed_pinned_gateup")


def grouped_mxfp4_cold(x2d, grouping, cold, out3d, out_col_offset, x_row_is_pair,
                       stream, max_blocks=0, wres_k_max=0) -> None:
    """kt fp4 BufferB slab(cudaHostRegister된 host 메모리)을 **제자리에서** 읽는 grouped GEMM.
    `cold`는 `ColdSlab`(cold_gpu.py, fmt=mxfp4) — u8 slab 뷰, blk_off(바이트), 타일 올림된
    row_off/kidx, 노드 N shard 시작. 출력 열은 `out_col_offset + cold.n_start`부터."""
    module = _jit_prism_grouped_mxfp4_module()
    with torch.cuda.stream(stream):
        module.grouped_mxfp4_cold(
            x2d, grouping.pair_sorted, grouping.pair_off, grouping.tile_off,
            cold.slab, cold.blk_off, cold.row_off, cold.k_index, out3d,
            int(out_col_offset) + int(cold.n_start), int(bool(x_row_is_pair)),
            int(max_blocks), int(cold.n))


def grouped_mxfp4_cold_gateup(x2d, grouping, gate, up, out3d, out_col_gate, out_col_up,
                              x_row_is_pair, stream, max_blocks=0, wres_k_max=0) -> None:
    """cold gate+up 한 launch (같은 노드의 두 slab)."""
    if (gate.n_start, gate.n) != (up.n_start, up.n):
        raise ValueError("cold gateup fusion requires gate/up slabs of the same node instance")
    module = _jit_prism_grouped_mxfp4_module()
    with torch.cuda.stream(stream):
        module.grouped_mxfp4_cold_gateup(
            x2d, grouping.pair_sorted, grouping.pair_off, grouping.tile_off,
            gate.slab, gate.blk_off, gate.row_off, gate.k_index,
            up.slab, up.blk_off, up.row_off, up.k_index, out3d,
            int(out_col_gate) + int(gate.n_start), int(out_col_up) + int(up.n_start),
            int(bool(x_row_is_pair)), int(max_blocks), int(gate.n))
