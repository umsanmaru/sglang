"""Prism worklist GEMV — MXFP4 pair-row 스토어 판 (`prism_gemv_mxfp4.cuh`).

호출 규약은 bf16 `gemv.py`와 같고, 스토어 인자만 `w_flat` 하나 대신
`(codes, scales)` 둘이다:
  codes  u8 [Σₑ k[e]/2, N]   — 행 = k-페어 (하위 nibble = 짝수 k)
  scales u8 [Σₑ k[e]/32, N]  — 행 = 32-k 블록의 E8M0 배율
row_off(k 단위, 32 배수)/kidx/out3d/out_col_offset/x_row_is_pair/stream은 동일.
누산 fp32, 출력 bf16 (계약 ⑤). 정확표현 입력에서 grouped mxfp4 커널과 비트일치.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.moe.prism.gemv import keep_ctr_arg

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Export name -> C++ entry (prism_gemv_mxfp4.cuh); the placement is the trailing w_on_device
# argument (1 = hot, 0 = warm).
_SCORE_SUFFIX = {"kDense": "", "kPair": "_sparse"}


def _entries() -> list[tuple[str, str]]:
    out = []
    for mode, suffix in _SCORE_SUFFIX.items():
        for fused in (False, True):
            name = "gemv_mxfp4_indexed" + suffix + ("_gateup" if fused else "")
            out.append((name, f"PrismGemvMxfp4<Mode::{mode}, {str(fused).lower()}>::run"))
    return out


_ENTRIES = _entries()


@cache_once
def _jit_prism_gemv_mxfp4_module() -> Module:
    return load_jit(
        "prism_gemv_mxfp4",
        cuda_files=["moe/prism/prism_gemv_mxfp4.cuh"],
        cuda_wrappers=_ENTRIES,
    )


def warmup_jit() -> None:
    """startup에서 강제 컴파일 (첫 호출이 캡처 워밍업에 얽히지 않게)."""
    _jit_prism_gemv_mxfp4_module()


def _dense(name):
    export = name.replace("_pinned", "")
    on_device = int("_pinned" not in name)
    def fn(x2d, topk_ids, codes, scales, row_off, kidx, out3d, out_col_offset,
           x_row_is_pair, stream) -> None:
        module = _jit_prism_gemv_mxfp4_module()
        with torch.cuda.stream(stream):
            getattr(module, export)(x2d, topk_ids, codes, scales, row_off, kidx, out3d,
                                  int(out_col_offset), int(bool(x_row_is_pair)), on_device)
    fn.__name__ = name
    return fn


def _sparse(name):
    export = name.replace("_pinned", "")
    on_device = int("_pinned" not in name)
    def fn(x2d, topk_ids, topk_weights, codes, scales, row_off, kidx, out3d, sp,
           out_col_offset, x_row_is_pair, stream) -> None:
        module = _jit_prism_gemv_mxfp4_module()
        with torch.cuda.stream(stream):
            getattr(module, export)(
                x2d, topk_ids, codes, scales, row_off, kidx, out3d,
                sp.a, sp.c, sp.thr, topk_weights, keep_ctr_arg(sp),
                int(out_col_offset), int(bool(x_row_is_pair)),
                float(sp.p), float(sp.lam), float(sp.pmax), float(sp.grid),
                int(sp.ng), int(sp.renorm_it), on_device)
    fn.__name__ = name
    return fn


def _dense_gateup(name):
    export = name.replace("_pinned", "")
    on_device = int("_pinned" not in name)
    def fn(x2d, topk_ids, codes_g, scales_g, row_off_g, kidx_g,
           codes_u, scales_u, row_off_u, kidx_u, out3d,
           out_col_gate, out_col_up, x_row_is_pair, stream) -> None:
        module = _jit_prism_gemv_mxfp4_module()
        with torch.cuda.stream(stream):
            getattr(module, export)(
                x2d, topk_ids, codes_g, scales_g, row_off_g, kidx_g,
                codes_u, scales_u, row_off_u, kidx_u, out3d,
                int(out_col_gate), int(out_col_up), int(bool(x_row_is_pair)), on_device)
    fn.__name__ = name
    return fn


def _sparse_gateup(name):
    export = name.replace("_pinned", "")
    on_device = int("_pinned" not in name)
    def fn(x2d, topk_ids, topk_weights, codes_g, scales_g, row_off_g, kidx_g,
           codes_u, scales_u, row_off_u, kidx_u, out3d, sp_gate, sp_up,
           out_col_gate, out_col_up, x_row_is_pair, stream) -> None:
        for f in ("pmax", "grid", "ng", "renorm_it"):
            if getattr(sp_gate, f) != getattr(sp_up, f):
                raise ValueError(f"gateup fusion requires a shared sparsity budget; "
                                 f"{f} differs ({getattr(sp_gate, f)} vs {getattr(sp_up, f)})")
        module = _jit_prism_gemv_mxfp4_module()
        with torch.cuda.stream(stream):
            getattr(module, export)(
                x2d, topk_ids, codes_g, scales_g, row_off_g, kidx_g,
                codes_u, scales_u, row_off_u, kidx_u, out3d,
                sp_gate.a, sp_gate.c, sp_gate.thr, sp_up.a, sp_up.c, sp_up.thr, topk_weights,
                keep_ctr_arg(sp_gate), keep_ctr_arg(sp_up),
                int(out_col_gate), int(out_col_up), int(bool(x_row_is_pair)),
                float(sp_gate.p), float(sp_gate.lam), float(sp_up.p), float(sp_up.lam),
                float(sp_gate.pmax), float(sp_gate.grid), int(sp_gate.ng), int(sp_gate.renorm_it),
                on_device)
    fn.__name__ = name
    return fn


# device-상주(HOT) / pinned(WARM) 쌍둥이. 커널 코드는 하나이고 host 검증만 갈린다 (계약 ①).
gemv_mxfp4_indexed = _dense("gemv_mxfp4_indexed")
gemv_mxfp4_indexed_pinned = _dense("gemv_mxfp4_indexed_pinned")
gemv_mxfp4_indexed_sparse = _sparse("gemv_mxfp4_indexed_sparse")
gemv_mxfp4_indexed_pinned_sparse = _sparse("gemv_mxfp4_indexed_pinned_sparse")
gemv_mxfp4_indexed_gateup = _dense_gateup("gemv_mxfp4_indexed_gateup")
gemv_mxfp4_indexed_pinned_gateup = _dense_gateup("gemv_mxfp4_indexed_pinned_gateup")
gemv_mxfp4_indexed_sparse_gateup = _sparse_gateup("gemv_mxfp4_indexed_sparse_gateup")
gemv_mxfp4_indexed_pinned_sparse_gateup = _sparse_gateup("gemv_mxfp4_indexed_pinned_sparse_gateup")
