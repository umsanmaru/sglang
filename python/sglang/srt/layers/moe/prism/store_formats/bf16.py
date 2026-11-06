"""Bf16Format implementation, mechanically extracted."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sglang.srt.layers.moe.prism.index import TierIndex
from sglang.srt.layers.moe.prism.plan import ModelDims, PlanError, Proj

from .base import FullWeights, ProjSource, StoreFormat
from .loading import (
    _host,
    _check_shapes,
    _proj_view,
    _gather_rows,
    _cold_flat_rows,
    _tensor_view,
)

class Bf16Format(StoreFormat):
    name = "bf16"
    k_align = 2
    has_scales = False
    supports_cold = True

    def create_params(self, layer, num_experts, hidden, inter, params_dtype, extra_weight_attrs):
        from sglang.srt.utils import set_weight_attrs

        if params_dtype != torch.bfloat16:
            raise NotImplementedError(f"Prism bf16 store supports bf16 params only, got {params_dtype}")
        w13 = torch.nn.Parameter(
            torch.empty(num_experts, 2 * inter, hidden, dtype=params_dtype, device="cpu"),
            requires_grad=False)
        w2 = torch.nn.Parameter(
            torch.empty(num_experts, hidden, inter, dtype=params_dtype, device="cpu"),
            requires_grad=False)
        layer.register_parameter("w13_weight", w13)
        layer.register_parameter("w2_weight", w2)
        set_weight_attrs(w13, extra_weight_attrs)
        set_weight_attrs(w2, extra_weight_attrs)

    def take_full(self, layer) -> FullWeights:
        return FullWeights(w13=_host(layer.w13_weight.data), w2=_host(layer.w2_weight.data))

    def release(self, layer) -> None:
        layer.w13_weight.data = torch.empty(0, dtype=layer.w13_weight.dtype)
        layer.w2_weight.data = torch.empty(0, dtype=layer.w2_weight.dtype)

    def check_full_shapes(self, full, dims, where):
        exp13 = (dims.num_experts, 2 * dims.intermediate_size, dims.hidden_size)
        exp2 = (dims.num_experts, dims.hidden_size, dims.intermediate_size)
        _check_shapes(where, full.w13, exp13, full.w2, exp2)

    def proj_source(self, full, inter, proj):
        return ProjSource(w=_proj_view(full.w13, full.w2, inter, proj), scales=None)

    def gather(self, src, ti, uniform_k, band_start):
        return _gather_rows(src.w, ti, uniform_k, band_start), None

    def cold_source(self, src, where):
        return src.w

    cold_kernels = ("kt_amx_bf16", "kt_tile_k2_bf16", "kt_tile_k2_bf16_nopf")
    cold_slab_dtype = torch.bfloat16

    def cold_flat(self, src, ti, real_rows, tile):
        flat, off, idx, real_t = _cold_flat_rows(src.w, ti, real_rows, tile, pad_value=0)
        return flat, None, off, idx, real_t

    def cold_slab(self, ptr, nbytes, expert_off, device):
        slab = _tensor_view(ptr, nbytes, torch.bfloat16)
        return slab, torch.tensor([o // 2 for o in expert_off[:-1]], dtype=torch.int64, device=device)

    def grouped_cold(self):
        from sglang.jit_kernel import prism_grouped as k

        return k.grouped_gemm_cold

    def grouped_cold_gateup(self):
        from sglang.jit_kernel import prism_grouped as k

        return k.grouped_gemm_cold_gateup

    def gemv(self, *, pinned, sparse):
        from sglang.jit_kernel import prism_gemv as k

        table = {
            (False, False): k.gemv_worklist_indexed,
            (True, False): k.gemv_worklist_indexed_pinned,
            (False, True): k.gemv_worklist_indexed_sparse,
            (True, True): k.gemv_worklist_indexed_pinned_sparse,
        }
        return table[(pinned, sparse)]

    def gemv_gateup(self, *, pinned, sparse):
        # 네 조합 전부 (mxfp4/fp8과 같은 표). warm dense와 hot sparse는 2026-08-29에
        # 채웠다 — 그 전에는 같은 스텝이 조용히 2회 launch로 떨어졌다.
        from sglang.jit_kernel import prism_gemv as k

        table = {
            (False, False): k.gemv_worklist_indexed_gateup,
            (True, False): k.gemv_worklist_indexed_pinned_gateup,
            (False, True): k.gemv_worklist_indexed_sparse_gateup,
            (True, True): k.gemv_worklist_indexed_pinned_sparse_gateup,
        }
        return table[(pinned, sparse)]

    def grouped(self, *, pinned):
        from sglang.jit_kernel import prism_grouped as k

        return k.grouped_gemm_indexed_pinned if pinned else k.grouped_gemm_indexed

    def grouped_gateup(self, *, pinned):
        from sglang.jit_kernel import prism_grouped as k

        return k.grouped_gemm_indexed_pinned_gateup if pinned else k.grouped_gemm_indexed_gateup

    def wres_k_max(self, *shards) -> int:
        from sglang.jit_kernel import prism_grouped

        if not prism_grouped.WRES_PCIE:
            return 0
        k = max(int(getattr(sh, "k_max", 0) or 0) for sh in shards)
        return (k + 31) // 32 * 32

    def warmup(self) -> None:
        from sglang.jit_kernel.prism_gemv import warmup_jit
        from sglang.jit_kernel.prism_grouped import warmup_jit as warmup_grouped

        warmup_jit()
        warmup_grouped()


