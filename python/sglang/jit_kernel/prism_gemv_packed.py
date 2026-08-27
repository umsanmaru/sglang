"""kt packed(AMX) 레이아웃을 읽는 worklist GEMV — `prism_gemv_packed.cuh`의 진입점.

warm을 kt 포맷 slab 한 벌(pinned)로 두기 위한 decode 커널이다. 호출 규약은
`prism_gemv.py`의 indexed worklist와 같고(현재 stream에 launch만, out[m, j, off:off+N]),
스토어 인자만 `ColdSlab`(cold_gpu.py) 하나로 받는다 — slab/blk_off/row_off/k_index/n/
n_start/n_block/k_block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_prism_gemv_packed_module() -> Module:
    return load_jit(
        "prism_gemv_packed",
        cuda_files=["moe/prism_gemv_packed.cuh"],
        cuda_wrappers=[
            ("gemv_packed", "gemv_packed"),
            ("gemv_packed_gateup", "gemv_packed_gateup"),
            ("gemv_packed_sparse", "gemv_packed_sparse"),
            ("gemv_packed_sparse_gateup", "gemv_packed_sparse_gateup"),
        ],
    )


def warmup_jit() -> None:
    _jit_prism_gemv_packed_module()


def gemv_packed(x2d, topk_ids, slab, out3d, out_col_offset, x_row_is_pair, stream) -> None:
    """out3d[m, j, off + n_start + n] = x_row · W_slab[topk[m,j]] (dense)."""
    module = _jit_prism_gemv_packed_module()
    with torch.cuda.stream(stream):
        module.gemv_packed(x2d, topk_ids, out3d, slab.slab, slab.blk_off, slab.row_off,
                           slab.k_index, int(out_col_offset) + int(slab.n_start), int(slab.n),
                           int(slab.n_block), int(slab.k_block), int(bool(x_row_is_pair)))


def gemv_packed_gateup(x2d, topk_ids, gate, up, out3d, out_col_gate, out_col_up,
                       x_row_is_pair, stream) -> None:
    if (gate.n, gate.n_start, gate.n_block, gate.k_block) != (up.n, up.n_start, up.n_block, up.k_block):
        raise ValueError("packed gateup fusion requires gate/up slabs of the same node instance")
    module = _jit_prism_gemv_packed_module()
    with torch.cuda.stream(stream):
        module.gemv_packed_gateup(
            x2d, topk_ids, out3d,
            gate.slab, gate.blk_off, gate.row_off, gate.k_index,
            up.slab, up.blk_off, up.row_off, up.k_index,
            int(out_col_gate) + int(gate.n_start), int(out_col_up) + int(up.n_start),
            int(gate.n), int(gate.n_block), int(gate.k_block), int(bool(x_row_is_pair)))


def gemv_packed_sparse(x2d, topk_ids, topk_weights, slab, sp, out3d, out_col_offset,
                       x_row_is_pair, stream) -> None:
    """k2wl2 sparse — 죽은 페어의 64 B 타일 행을 읽지 않는다. sp = tiers.SparseSpec."""
    module = _jit_prism_gemv_packed_module()
    with torch.cuda.stream(stream):
        module.gemv_packed_sparse(
            x2d, topk_ids, out3d, slab.slab, slab.blk_off, slab.row_off, slab.k_index,
            sp.a, sp.c, sp.thr, topk_weights,
            int(out_col_offset) + int(slab.n_start), int(slab.n), int(slab.n_block), int(slab.k_block),
            int(bool(x_row_is_pair)), float(sp.p), float(sp.lam), float(sp.pmax), float(sp.grid),
            int(sp.ng), int(sp.renorm_it))


def gemv_packed_sparse_gateup(x2d, topk_ids, topk_weights, gate, up, sp_gate, sp_up, out3d,
                              out_col_gate, out_col_up, x_row_is_pair, stream) -> None:
    for f in ("pmax", "grid", "ng", "renorm_it"):
        if getattr(sp_gate, f) != getattr(sp_up, f):
            raise ValueError(f"gateup fusion requires a shared sparsity budget; {f} differs")
    module = _jit_prism_gemv_packed_module()
    with torch.cuda.stream(stream):
        module.gemv_packed_sparse_gateup(
            x2d, topk_ids, out3d,
            gate.slab, gate.blk_off, gate.row_off, gate.k_index,
            up.slab, up.blk_off, up.row_off, up.k_index,
            sp_gate.a, sp_gate.c, sp_gate.thr, sp_up.a, sp_up.c, sp_up.thr, topk_weights,
            int(out_col_gate) + int(gate.n_start), int(out_col_up) + int(up.n_start),
            int(gate.n), int(gate.n_block), int(gate.k_block), int(bool(x_row_is_pair)),
            float(sp_gate.p), float(sp_gate.lam), float(sp_up.p), float(sp_up.lam),
            float(sp_gate.pmax), float(sp_gate.grid), int(sp_gate.ng), int(sp_gate.renorm_it))
