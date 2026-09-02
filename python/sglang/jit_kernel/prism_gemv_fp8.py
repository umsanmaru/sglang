"""Prism worklist GEMV — FP8 e4m3(128×128 블록 배율) 스토어 판 (`prism_gemv_fp8.cuh`).

호출 규약은 mxfp4 `prism_gemv_mxfp4.py`와 **같다** (스토어 인자가 `(codes, scales)` 둘):
  codes  u8   [Σₑ k[e], N]         — 행 = k (원소 1 B)
  scales fp32 [Σₑ k[e]/128, N/128] — 행 = 128-k 블록, 열 = 128-n 블록
row_off(k 단위, 128 배수)/kidx/out3d/out_col_offset/x_row_is_pair/stream은 동일.
누산 fp32, 출력 bf16 (계약 ⑤). 정확표현 입력에서 grouped fp8 커널과 비트일치.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_WRAPPERS = (
    "gemv_fp8_indexed",
    "gemv_fp8_indexed_pinned",
    "gemv_fp8_indexed_sparse",
    "gemv_fp8_indexed_pinned_sparse",
    "gemv_fp8_indexed_gateup",
    "gemv_fp8_indexed_pinned_gateup",
    "gemv_fp8_indexed_sparse_gateup",
    "gemv_fp8_indexed_pinned_sparse_gateup",
)


@cache_once
def _jit_prism_gemv_fp8_module() -> Module:
    return load_jit(
        "prism_gemv_fp8",
        cuda_files=["moe/prism_gemv_fp8.cuh"],
        cuda_wrappers=[(n, n) for n in _WRAPPERS],
    )


def warmup_jit() -> None:
    """startup에서 강제 컴파일 (첫 호출이 캡처 워밍업에 얽히지 않게)."""
    _jit_prism_gemv_fp8_module()


def _dense(name):
    def fn(x2d, topk_ids, codes, scales, row_off, kidx, out3d, out_col_offset,
           x_row_is_pair, stream) -> None:
        module = _jit_prism_gemv_fp8_module()
        with torch.cuda.stream(stream):
            getattr(module, name)(x2d, topk_ids, codes, scales, row_off, kidx, out3d,
                                  int(out_col_offset), int(bool(x_row_is_pair)))
    fn.__name__ = name
    return fn


def _sparse(name):
    def fn(x2d, topk_ids, topk_weights, codes, scales, row_off, kidx, out3d, sp,
           out_col_offset, x_row_is_pair, stream) -> None:
        module = _jit_prism_gemv_fp8_module()
        with torch.cuda.stream(stream):
            getattr(module, name)(
                x2d, topk_ids, codes, scales, row_off, kidx, out3d,
                sp.a, sp.c, sp.thr, topk_weights,
                int(out_col_offset), int(bool(x_row_is_pair)),
                float(sp.p), float(sp.lam), float(sp.pmax), float(sp.grid),
                int(sp.ng), int(sp.renorm_it))
    fn.__name__ = name
    return fn


def _dense_gateup(name):
    def fn(x2d, topk_ids, codes_g, scales_g, row_off_g, kidx_g,
           codes_u, scales_u, row_off_u, kidx_u, out3d,
           out_col_gate, out_col_up, x_row_is_pair, stream) -> None:
        module = _jit_prism_gemv_fp8_module()
        with torch.cuda.stream(stream):
            getattr(module, name)(
                x2d, topk_ids, codes_g, scales_g, row_off_g, kidx_g,
                codes_u, scales_u, row_off_u, kidx_u, out3d,
                int(out_col_gate), int(out_col_up), int(bool(x_row_is_pair)))
    fn.__name__ = name
    return fn


def _sparse_gateup(name):
    def fn(x2d, topk_ids, topk_weights, codes_g, scales_g, row_off_g, kidx_g,
           codes_u, scales_u, row_off_u, kidx_u, out3d, sp_gate, sp_up,
           out_col_gate, out_col_up, x_row_is_pair, stream) -> None:
        for f in ("pmax", "grid", "ng", "renorm_it"):
            if getattr(sp_gate, f) != getattr(sp_up, f):
                raise ValueError(f"gateup fusion requires a shared sparsity budget; "
                                 f"{f} differs ({getattr(sp_gate, f)} vs {getattr(sp_up, f)})")
        module = _jit_prism_gemv_fp8_module()
        with torch.cuda.stream(stream):
            getattr(module, name)(
                x2d, topk_ids, codes_g, scales_g, row_off_g, kidx_g,
                codes_u, scales_u, row_off_u, kidx_u, out3d,
                sp_gate.a, sp_gate.c, sp_gate.thr, sp_up.a, sp_up.c, sp_up.thr, topk_weights,
                int(out_col_gate), int(out_col_up), int(bool(x_row_is_pair)),
                float(sp_gate.p), float(sp_gate.lam), float(sp_up.p), float(sp_up.lam),
                float(sp_gate.pmax), float(sp_gate.grid), int(sp_gate.ng), int(sp_gate.renorm_it))
    fn.__name__ = name
    return fn


# device-상주(HOT) / pinned(WARM) 쌍둥이. 커널 코드는 하나이고 host 검증만 갈린다 (계약 ①).
gemv_fp8_indexed = _dense("gemv_fp8_indexed")
gemv_fp8_indexed_pinned = _dense("gemv_fp8_indexed_pinned")
gemv_fp8_indexed_sparse = _sparse("gemv_fp8_indexed_sparse")
gemv_fp8_indexed_pinned_sparse = _sparse("gemv_fp8_indexed_pinned_sparse")
gemv_fp8_indexed_gateup = _dense_gateup("gemv_fp8_indexed_gateup")
gemv_fp8_indexed_pinned_gateup = _dense_gateup("gemv_fp8_indexed_pinned_gateup")
gemv_fp8_indexed_sparse_gateup = _sparse_gateup("gemv_fp8_indexed_sparse_gateup")
gemv_fp8_indexed_pinned_sparse_gateup = _sparse_gateup("gemv_fp8_indexed_pinned_sparse_gateup")
