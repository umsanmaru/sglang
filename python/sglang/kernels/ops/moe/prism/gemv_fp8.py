"""Prism worklist GEMV — FP8 e4m3 스토어 판 (`prism_gemv_fp8.cuh`), 배율 두 가지.

호출 규약은 mxfp4 `gemv_mxfp4.py`와 **같다** (스토어 인자가 `(codes, scales)` 둘):
  codes  u8   [Σₑ k[e], N]         — 행 = k (원소 1 B)
  scales fp32 [Σₑ k[e]/128, N/128] — 블록판 (`gemv_fp8_*`): 행 = 128-k 블록, 열 = 128-n 블록
         fp32 [E]                  — per-tensor판 (`gemv_fp8pt_*`): 슬롯당 스칼라 1개
row_off(k 단위; 블록판 128 배수, pt판 32 배수)/kidx/out3d/out_col_offset/x_row_is_pair/stream은 동일.
누산 fp32, 출력 bf16 (계약 ⑤). 정확표현 입력에서 grouped fp8 커널과 비트일치.
pt판은 Mistral-Medium-3.5 류 static-tensor FP8(`weight_block_size: null`)용이다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.moe.prism.gemv import keep_ctr_arg

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Export name -> C++ entry (prism_gemv_fp8.cuh). `pt` = per-tensor scales; the placement is the
# trailing w_on_device argument (1 = hot, 0 = warm). k1w has no fused gate+up entry.
_SCORE_SUFFIX = {
    "kDense": "",
    "kPair": "_sparse",
    "kK1": "_sparsek1",
    "kK1W": "_sparsek1w",
}


def _entries() -> list[tuple[str, str]]:
    out = []
    for pt in (False, True):
        for mode, suffix in _SCORE_SUFFIX.items():
            for fused in (False, True) if mode != "kK1W" else (False,):
                name = (
                    ("gemv_fp8pt" if pt else "gemv_fp8")
                    + "_indexed"
                    + suffix
                    + ("_gateup" if fused else "")
                )
                flags = ", ".join(str(b).lower() for b in (fused, pt))
                out.append((name, f"PrismGemvFp8<Mode::{mode}, {flags}>::run"))
    return out


_ENTRIES = _entries()


def _per_k(sp) -> bool:
    return bool(getattr(sp, "per_k", False))


def _per_k_w(sp) -> bool:
    """score k1w1 — per-k에 a(wn²)를 곱한다. a의 유무가 표식이다 (커널도 같은 규칙)."""
    return _per_k(sp) and getattr(sp, "a", None) is not None


@cache_once
def _jit_prism_gemv_fp8_module() -> Module:
    return load_jit(
        "prism_gemv_fp8",
        cuda_files=["moe/prism/prism_gemv_fp8.cuh"],
        cuda_wrappers=_ENTRIES,
    )


def warmup_jit() -> None:
    """startup에서 강제 컴파일 (첫 호출이 캡처 워밍업에 얽히지 않게)."""
    _jit_prism_gemv_fp8_module()


def _dense(name):
    def fn(x2d, topk_ids, codes, scales, row_off, kidx, out3d, out_col_offset,
           x_row_is_pair, stream, *, on_device: bool = True) -> None:
        module = _jit_prism_gemv_fp8_module()
        with torch.cuda.stream(stream):
            getattr(module, name)(x2d, topk_ids, codes, scales, row_off, kidx, out3d,
                                  int(out_col_offset), int(bool(x_row_is_pair)),
                                  int(bool(on_device)))
    fn.__name__ = name
    return fn


def _sparse(name):
    def fn(x2d, topk_ids, topk_weights, codes, scales, row_off, kidx, out3d, sp,
           out_col_offset, x_row_is_pair, stream, *, on_device: bool = True) -> None:
        module = _jit_prism_gemv_fp8_module()
        tail = (int(out_col_offset), int(bool(x_row_is_pair)),
                float(sp.p), float(sp.lam), float(sp.pmax), float(sp.grid), int(sp.ng), int(sp.renorm_it),
                int(bool(on_device)))
        with torch.cuda.stream(stream):
            if _per_k_w(sp):
                getattr(module, name.replace("sparse", "sparsek1w"))(
                    x2d, topk_ids, codes, scales, row_off, kidx, out3d, sp.a, sp.thr,
                    topk_weights, keep_ctr_arg(sp), *tail)
                return
            if _per_k(sp):
                getattr(module, name.replace("sparse", "sparsek1"))(
                    x2d, topk_ids, codes, scales, row_off, kidx, out3d, sp.thr, topk_weights,
                    keep_ctr_arg(sp), *tail)
            else:
                getattr(module, name)(
                    x2d, topk_ids, codes, scales, row_off, kidx, out3d,
                    sp.a, sp.c, sp.thr, topk_weights, keep_ctr_arg(sp), *tail)
    fn.__name__ = name
    return fn


def _dense_gateup(name):
    def fn(x2d, topk_ids, codes_g, scales_g, row_off_g, kidx_g,
           codes_u, scales_u, row_off_u, kidx_u, out3d,
           out_col_gate, out_col_up, x_row_is_pair, stream, *, on_device: bool = True) -> None:
        module = _jit_prism_gemv_fp8_module()
        with torch.cuda.stream(stream):
            getattr(module, name)(
                x2d, topk_ids, codes_g, scales_g, row_off_g, kidx_g,
                codes_u, scales_u, row_off_u, kidx_u, out3d,
                int(out_col_gate), int(out_col_up), int(bool(x_row_is_pair)), int(bool(on_device)))
    fn.__name__ = name
    return fn


def _sparse_gateup(name):
    def fn(x2d, topk_ids, topk_weights, codes_g, scales_g, row_off_g, kidx_g,
           codes_u, scales_u, row_off_u, kidx_u, out3d, sp_gate, sp_up,
           out_col_gate, out_col_up, x_row_is_pair, stream, *, on_device: bool = True) -> None:
        for f in ("pmax", "grid", "ng", "renorm_it", "per_k"):
            if getattr(sp_gate, f, False) != getattr(sp_up, f, False):
                raise ValueError(f"gateup fusion requires a shared sparsity budget; "
                                 f"{f} differs ({getattr(sp_gate, f, None)} vs {getattr(sp_up, f, None)})")
        module = _jit_prism_gemv_fp8_module()
        tail = (int(out_col_gate), int(out_col_up), int(bool(x_row_is_pair)),
                float(sp_gate.p), float(sp_gate.lam), float(sp_up.p), float(sp_up.lam),
                float(sp_gate.pmax), float(sp_gate.grid), int(sp_gate.ng), int(sp_gate.renorm_it),
                int(bool(on_device)))
        with torch.cuda.stream(stream):
            if _per_k(sp_gate):
                getattr(module, name.replace("sparse", "sparsek1"))(
                    x2d, topk_ids, codes_g, scales_g, row_off_g, kidx_g,
                    codes_u, scales_u, row_off_u, kidx_u, out3d,
                    sp_gate.thr, sp_up.thr, topk_weights,
                    keep_ctr_arg(sp_gate), keep_ctr_arg(sp_up), *tail)
            else:
                getattr(module, name)(
                    x2d, topk_ids, codes_g, scales_g, row_off_g, kidx_g,
                    codes_u, scales_u, row_off_u, kidx_u, out3d,
                    sp_gate.a, sp_gate.c, sp_gate.thr, sp_up.a, sp_up.c, sp_up.thr, topk_weights,
                    keep_ctr_arg(sp_gate), keep_ctr_arg(sp_up), *tail)
    fn.__name__ = name
    return fn


# HOT(device W)과 WARM(pinned W)은 같은 진입점이고 `on_device`로 host 검증만 갈린다 (계약 ①).
gemv_fp8_indexed = _dense("gemv_fp8_indexed")
gemv_fp8_indexed_sparse = _sparse("gemv_fp8_indexed_sparse")
gemv_fp8_indexed_gateup = _dense_gateup("gemv_fp8_indexed_gateup")
gemv_fp8_indexed_sparse_gateup = _sparse_gateup("gemv_fp8_indexed_sparse_gateup")

# per-tensor 배율 쌍둥이 — 같은 호출 규약, scales만 [E].
gemv_fp8pt_indexed = _dense("gemv_fp8pt_indexed")
gemv_fp8pt_indexed_sparse = _sparse("gemv_fp8pt_indexed_sparse")
gemv_fp8pt_indexed_gateup = _dense_gateup("gemv_fp8pt_indexed_gateup")
gemv_fp8pt_indexed_sparse_gateup = _sparse_gateup("gemv_fp8pt_indexed_sparse_gateup")
