"""Prism grouped GEMM — FP8 e4m3(128×128 블록 배율) 스토어 판 (`prism_grouped_fp8.cuh`).

호출 규약은 mxfp4 `prism_grouped_mxfp4.py`와 같다 (스토어 인자 `(codes, scales)`). 토큰
타일 크기는 bf16과 같은 `TILE_M`이어야 한다. W-resident 변형은 없다 — `wres_k_max`는 받되 무시.

수치: tensor core에 `bf16(code · scale)`을 넣는다 (= 같은 가중치를 dequant해 bf16 스토어로
넣은 plan과 같은 계산). decode GEMV는 fp32 부분합에 fp32 배율을 곱하므로 배율이 2의
거듭제곱이면 둘이 정확히 같다 — 자세한 것은 .cuh 상단 "수치 계약".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.prism_grouped import TILE_M  # noqa: F401  (그룹핑 타일 크기 공유)
from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# cold slab 레이아웃 (C++ enum Layout): fp8 타일 (tile_k2_fp8b128)만.
COLD_LAYOUTS = {"kt_tile8": 1}

_WRAPPERS = (
    "grouped_fp8_indexed",
    "grouped_fp8_indexed_pinned",
    "grouped_fp8_indexed_gateup",
    "grouped_fp8_indexed_pinned_gateup",
    "grouped_fp8_cold",
    "grouped_fp8_cold_gateup",
)


@cache_once
def _jit_prism_grouped_fp8_module() -> Module:
    return load_jit(
        "prism_grouped_fp8",
        cuda_files=["moe/prism_grouped_fp8.cuh"],
        cuda_wrappers=[(n, n) for n in _WRAPPERS],
    )


def warmup_jit() -> None:
    _jit_prism_grouped_fp8_module()


def _single(name):
    def fn(x2d, grouping, codes, scales, row_off, kidx, out3d, out_col_offset,
           x_row_is_pair, stream, max_blocks=0, wres_k_max=0) -> None:
        module = _jit_prism_grouped_fp8_module()
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
        module = _jit_prism_grouped_fp8_module()
        with torch.cuda.stream(stream):
            getattr(module, name)(
                x2d, grouping.pair_sorted, grouping.pair_off, grouping.tile_off,
                codes_g, scales_g, row_off_g, kidx_g, codes_u, scales_u, row_off_u, kidx_u,
                out3d, int(out_col_gate), int(out_col_up), int(bool(x_row_is_pair)),
                int(max_blocks))
    fn.__name__ = name
    return fn


grouped_fp8_indexed = _single("grouped_fp8_indexed")
grouped_fp8_indexed_pinned = _single("grouped_fp8_indexed_pinned")
grouped_fp8_indexed_gateup = _gateup("grouped_fp8_indexed_gateup")
grouped_fp8_indexed_pinned_gateup = _gateup("grouped_fp8_indexed_pinned_gateup")


def grouped_fp8_cold(x2d, grouping, cold, out3d, out_col_offset, x_row_is_pair,
                       stream, max_blocks=0, wres_k_max=0) -> None:
    """kt fp8 타일 BufferB slab(cudaHostRegister된 host 메모리)을 **제자리에서** 읽는 grouped GEMM.
    `cold`는 `ColdSlab`(cold_gpu.py, fmt=fp8) — u8 slab 뷰, blk_off(바이트), 타일 올림된
    row_off/kidx, 노드 N shard 시작. 출력 열은 `out_col_offset + cold.n_start`부터."""
    module = _jit_prism_grouped_fp8_module()
    with torch.cuda.stream(stream):
        module.grouped_fp8_cold(
            x2d, grouping.pair_sorted, grouping.pair_off, grouping.tile_off,
            cold.slab, cold.blk_off, cold.row_off, cold.k_index, out3d,
            int(out_col_offset) + int(cold.n_start), int(bool(x_row_is_pair)),
            int(max_blocks), int(cold.n), COLD_LAYOUTS[cold.layout])


def grouped_fp8_cold_gateup(x2d, grouping, gate, up, out3d, out_col_gate, out_col_up,
                              x_row_is_pair, stream, max_blocks=0, wres_k_max=0) -> None:
    """cold gate+up 한 launch (같은 노드의 두 slab)."""
    if (gate.n_start, gate.n, gate.layout) != (up.n_start, up.n, up.layout):
        raise ValueError("cold gateup fusion requires gate/up slabs of the same node instance")
    module = _jit_prism_grouped_fp8_module()
    with torch.cuda.stream(stream):
        module.grouped_fp8_cold_gateup(
            x2d, grouping.pair_sorted, grouping.pair_off, grouping.tile_off,
            gate.slab, gate.blk_off, gate.row_off, gate.k_index,
            up.slab, up.blk_off, up.row_off, up.k_index, out3d,
            int(out_col_gate) + int(gate.n_start), int(out_col_up) + int(up.n_start),
            int(bool(x_row_is_pair)), int(max_blocks), int(gate.n), COLD_LAYOUTS[gate.layout])
