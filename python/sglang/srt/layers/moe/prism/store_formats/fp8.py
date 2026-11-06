"""Fp8Format implementation, mechanically extracted."""
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
    _half_index,
    _gather_rows,
    _cold_flat_rows,
    _tensor_view,
)

class Fp8Format(StoreFormat):
    """DeepSeek류 blockwise FP8 (e4m3 + 128×128 fp32 `scale_inv`) routed expert.

    파라미터 이름/attrs는 sglang의 blockwise fp8 MoE와 같다 (w13_weight fp8_e4m3
    [E, 2I, H], w13_weight_scale_inv fp32 [E, 2I/128, H/128], BLOCK quant_method) —
    그 이름으로 로더가 채운다.

    K 정렬이 **128**인 것이 mxfp4(32)와 갈리는 유일한 계약 차이다: 배율 하나가 원본
    128 k × 128 n 블록을 덮으므로 티어 K-인덱스가 블록을 쪼개면 "블록당 배율 1"이
    깨진다 (재양자화 없이 체크포인트 수치를 보존하는 유일한 선택 — mxfp4의 32와
    같은 삼자택일, 한 단계 거칠 뿐이다)."""

    name = "fp8"
    k_align = 128
    has_scales = True
    supports_cold = True
    # kt_tile_k2_fp8b128: cpu-mm tile_k2_fp8b128 포팅 (fp8 타일 레이아웃, 프리페치 커서,
    # sparse plan 경로; N shard 256 배수). kt 원본 fp8 커널(AMX_FP8_MOE_TP)은 Prism partial을
    # 지원하지 않아 cold 후보가 아니다.
    # kt_tile_k1_fp8b128: 같은 타일의 k=1 판 — per-k 마스크(score k1), GPU 로더 KT_TILE8_K1.
    cold_kernels = ("kt_tile_k2_fp8b128", "kt_tile_k1_fp8b128")
    cold_slab_dtype = torch.uint8
    BLOCK = 128

    @property
    def _wdtype(self):
        return getattr(torch, "float8_e4m3fn")

    def default_cold_gpu_min_m(self, executor_default, grouped_min_m):
        # 사용자 결정(2026-08-28): fp8 prefill은 AMX를 쓰지 않고 전부 GPU — grouped 경계부터.
        return grouped_min_m

    def create_params(self, layer, num_experts, hidden, inter, params_dtype, extra_weight_attrs):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported
        from sglang.srt.utils import set_weight_attrs

        B = self.BLOCK
        if hidden % B or inter % B:
            raise PlanError(f"fp8 needs hidden/inter multiples of {B}, got {hidden}/{inter}")
        w13 = torch.nn.Parameter(
            torch.empty(num_experts, 2 * inter, hidden, dtype=self._wdtype, device="cpu"),
            requires_grad=False)
        w2 = torch.nn.Parameter(
            torch.empty(num_experts, hidden, inter, dtype=self._wdtype, device="cpu"),
            requires_grad=False)
        layer.register_parameter("w13_weight", w13)
        set_weight_attrs(w13, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2)
        set_weight_attrs(w2, extra_weight_attrs)
        w13_s = torch.nn.Parameter(
            torch.ones(num_experts, 2 * inter // B, hidden // B, dtype=torch.float32, device="cpu"),
            requires_grad=False)
        w2_s = torch.nn.Parameter(
            torch.ones(num_experts, hidden // B, inter // B, dtype=torch.float32, device="cpu"),
            requires_grad=False)
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter("w13_weight_scale_inv", w13_s)
        set_weight_attrs(w13_s, scale_attrs)
        layer.register_parameter("w2_weight_scale_inv", w2_s)
        set_weight_attrs(w2_s, scale_attrs)

    def take_full(self, layer) -> FullWeights:
        return FullWeights(
            w13=_host(layer.w13_weight.data), w2=_host(layer.w2_weight.data),
            w13_scale=_host(layer.w13_weight_scale_inv.data),
            w2_scale=_host(layer.w2_weight_scale_inv.data))

    def release(self, layer) -> None:
        for name in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
            p = getattr(layer, name)
            p.data = torch.empty(0, dtype=p.dtype)

    def check_full_shapes(self, full, dims, where):
        E, H, I, B = dims.num_experts, dims.hidden_size, dims.intermediate_size, self.BLOCK
        _check_shapes(where, full.w13, (E, 2 * I, H), full.w2, (E, H, I))
        if full.w13_scale is None or full.w2_scale is None:
            raise PlanError(f"{where}: fp8 store needs w13/w2 scales")
        _check_shapes(where + " scales", full.w13_scale, (E, 2 * I // B, H // B),
                      full.w2_scale, (E, H // B, I // B))
        if full.w13.dtype not in (self._wdtype, torch.int8, torch.uint8):
            raise PlanError(f"{where}: fp8 codes must be float8_e4m3fn (or raw bytes), got "
                            f"{full.w13.dtype}")

    def proj_source(self, full, inter, proj):
        # 배율의 N축은 128-블록이라 gate/up 경계도 inter/128이다.
        B = self.BLOCK
        return ProjSource(w=_proj_view(full.w13, full.w2, inter, proj).view(torch.uint8),
                          scales=_proj_view(full.w13_scale, full.w2_scale, inter // B, proj))

    def check_index(self, ti, where):
        """모든 expert의 인덱스가 128-블록 단위여야 한다 (mxfp4의 32와 같은 검사)."""
        B = self.BLOCK
        idx = ti.idx.to(torch.int64)
        for e in range(ti.num_experts):
            o0, o1 = int(ti.row_off[e]), int(ti.row_off[e + 1])
            if o0 % B or (o1 - o0) % B:
                raise PlanError(f"{where}: expert {e} rows [{o0},{o1}) are not {B}-aligned — "
                                f"fp8 tier index must move in whole scale blocks")
            if o1 == o0:
                continue
            rows = idx[o0:o1].reshape(-1, B)
            starts = rows[:, 0]
            if torch.any(starts % B) or not torch.equal(rows, starts[:, None] + torch.arange(B)[None, :]):
                raise PlanError(f"{where}: expert {e} index does not consist of whole {B}-row "
                                f"scale blocks — fp8 cannot split a block across tiers")

    def gather(self, src, ti, uniform_k, band_start):
        codes = _gather_rows(src.w, ti, uniform_k, band_start)          # [Σₑ k[e], N] u8
        scales = _gather_rows(src.scales, _half_index(ti, self.BLOCK, band_start, uniform_k),
                              None if uniform_k is None else uniform_k // self.BLOCK,
                              None if band_start is None else band_start // self.BLOCK)
        return codes, scales

    def cold_source(self, src, where):
        raise PlanError(f"{where}: fp8 has no warm-kt (bf16 slab) mode")

    def cold_flat(self, src, ti, real_rows, tile):
        """cold 블록 = e4m3 행 [N, k_pad] u8 + 배율 fp32 [N/128, k_pad/128] (kt `scale_inv`의
        n-major 방향 그대로). 패딩 행은 코드 0x00(= +0.0), 패딩 배율은 1.0 — weight가 0이라
        무해하고, sparse는 마스크가 tail 비트를 꺼서 읽지도 않는다."""
        codes, off, idx, real_t = _cold_flat_rows(src.w, ti, real_rows, tile, pad_value=0)
        scales, _, _, _ = _cold_flat_rows(src.scales, _half_index(ti, self.BLOCK, None, None),
                                          real_rows, tile, pad_value=1.0, div=self.BLOCK)
        return codes, scales, off, idx, real_t

    def cold_load_kwargs(self, cold):
        return dict(gate_scale=cold.gate.s_flat, up_scale=cold.up.s_flat, down_scale=cold.down.s_flat)

    def cold_slab(self, ptr, nbytes, expert_off, device):
        slab = _tensor_view(ptr, nbytes, torch.uint8)
        return slab, torch.tensor(list(expert_off[:-1]), dtype=torch.int64, device=device)

    def grouped_cold(self):
        from sglang.jit_kernel import prism_grouped_fp8 as k

        return k.grouped_fp8_cold

    def grouped_cold_gateup(self):
        from sglang.jit_kernel import prism_grouped_fp8 as k

        return k.grouped_fp8_cold_gateup

    def grouped(self, *, pinned):
        from sglang.jit_kernel import prism_grouped_fp8 as k

        return k.grouped_fp8_indexed_pinned if pinned else k.grouped_fp8_indexed

    def grouped_gateup(self, *, pinned):
        from sglang.jit_kernel import prism_grouped_fp8 as k

        return k.grouped_fp8_indexed_pinned_gateup if pinned else k.grouped_fp8_indexed_gateup

    def gemv(self, *, pinned, sparse):
        from sglang.jit_kernel import prism_gemv_fp8 as k

        table = {
            (False, False): k.gemv_fp8_indexed,
            (True, False): k.gemv_fp8_indexed_pinned,
            (False, True): k.gemv_fp8_indexed_sparse,
            (True, True): k.gemv_fp8_indexed_pinned_sparse,
        }
        return table[(pinned, sparse)]

    def gemv_gateup(self, *, pinned, sparse):
        from sglang.jit_kernel import prism_gemv_fp8 as k

        table = {
            (False, False): k.gemv_fp8_indexed_gateup,
            (True, False): k.gemv_fp8_indexed_pinned_gateup,
            (False, True): k.gemv_fp8_indexed_sparse_gateup,
            (True, True): k.gemv_fp8_indexed_pinned_sparse_gateup,
        }
        return table[(pinned, sparse)]

    def warmup(self) -> None:
        from sglang.jit_kernel.prism_gemv_fp8 import warmup_jit
        from sglang.jit_kernel.prism_grouped_fp8 import warmup_jit as warmup_grouped

        warmup_jit()
        warmup_grouped()


