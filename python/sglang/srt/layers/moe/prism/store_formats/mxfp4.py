"""Mxfp4Format implementation, mechanically extracted."""
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
    _e8m0_bytes,
    _half_index,
    _gather_rows,
    _cold_flat_rows,
    _tensor_view,
    _padded_kindex,
)

class Mxfp4Format(StoreFormat):
    """MXFP4 g32 routed expert (e2m1 nibble 코드 + E8M0 32-블록 배율). 파라미터 이름/shape/dtype/
    attrs는 **감싸는 GPU quant method가 만들었을 것과 동일**해야 한다 — 로더가 그 이름으로 채우고,
    모델 load_weights는 이름이 없으면 조용히 건너뛴다 (kimi_k3.py: `if name not in params_dict: break`).
    두 가지 명명이 있다 (`layer._prism_mxfp4_naming`, method.create_weights가 gpu_method로 정한다):

      "deepseek" — DeepSeek-V4-Flash, `DeepSeekMxfp4MoEMethod`:
                   w13_weight int8 [E, 2I, H/2], w13_weight_scale_inv fp32(=2^e) [E, 2I, H/32], BLOCK
      "ct"       — Kimi K3 (compressed-tensors mxfp4-pack-quantized → `Mxfp4MoEMethod`):
                   w13_weight uint8 [E, 2I, H/2], w13_weight_scale uint8(E8M0) [E, 2I, H/32], GROUP

    둘의 차이는 파라미터 컨테이너까지다 — 코드 nibble 배치와 E8M0 배율은 같은 것이라 인출
    뒤(`take_full`)의 경로(gather / cold_flat / 커널)는 하나다 (`_e8m0_bytes`가 fp32·u8 둘을 받는다)."""

    NAMINGS = ("deepseek", "ct")

    name = "mxfp4"
    k_align = 32
    has_scales = True
    supports_cold = True
    # kt_amx_fp4: kt 원본 fp4 커널(dense, 행우선 nibble) / kt_tile_k2_mxfp4: cpu-mm tile_k2_mxfp4
    # 포팅(fp4 타일 레이아웃, 프리페치 커서, sparse plan 경로; N shard 256 배수).
    cold_kernels = ("kt_amx_fp4", "kt_tile_k2_mxfp4")
    cold_slab_dtype = torch.uint8
    BLOCK_K = 32

    def default_cold_gpu_min_m(self, executor_default, grouped_min_m):
        # 사용자 결정(2026-08-27): mxfp4 prefill은 AMX를 쓰지 않고 전부 GPU — grouped 경계부터 GPU.
        return grouped_min_m

    @classmethod
    def naming(cls, layer) -> str:
        n = getattr(layer, "_prism_mxfp4_naming", "deepseek")
        if n not in cls.NAMINGS:
            raise PlanError(f"unknown mxfp4 param naming {n!r} (known: {cls.NAMINGS})")
        return n

    @classmethod
    def naming_for_method(cls, gpu_method) -> str:
        """감싸는 GPU method → 명명. compressed-tensors mxfp4(Mxfp4MoEMethod, K3)만 "ct"."""
        return "ct" if gpu_method.__class__.__name__ == "Mxfp4MoEMethod" else "deepseek"

    @classmethod
    def scale_names(cls, layer) -> tuple[str, str]:
        if cls.naming(layer) == "ct":
            return ("w13_weight_scale", "w2_weight_scale")
        return ("w13_weight_scale_inv", "w2_weight_scale_inv")

    def create_params(self, layer, num_experts, hidden, inter, params_dtype, extra_weight_attrs):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported
        from sglang.srt.utils import set_weight_attrs

        if hidden % self.BLOCK_K or inter % self.BLOCK_K:
            raise PlanError(f"mxfp4 needs hidden/inter multiples of {self.BLOCK_K}, got {hidden}/{inter}")
        ct = self.naming(layer) == "ct"
        code_dtype = torch.uint8 if ct else torch.int8
        w13 = torch.nn.Parameter(
            torch.empty(num_experts, 2 * inter, hidden // 2, dtype=code_dtype, device="cpu"),
            requires_grad=False)
        w2 = torch.nn.Parameter(
            torch.empty(num_experts, hidden, inter // 2, dtype=code_dtype, device="cpu"),
            requires_grad=False)
        layer.register_parameter("w13_weight", w13)
        set_weight_attrs(w13, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2)
        set_weight_attrs(w2, extra_weight_attrs)
        if ct:
            # E8M0 바이트 그대로 (127 = 2^0). 로더의 GROUP 경로(_load_w13/_load_w2)가 shard를 narrow해
            # copy_ 하므로 u8→u8 무손실. dummy 로더(initialize_dummy_weights)는 비부동 파라미터를
            # 건드리지 않아 배율이 1.0으로 남는다 — 랜덤 nibble 코드만으로도 유한한 가중치가 된다.
            w13_s = torch.nn.Parameter(
                torch.full((num_experts, 2 * inter, hidden // self.BLOCK_K), 127, dtype=torch.uint8, device="cpu"),
                requires_grad=False)
            w2_s = torch.nn.Parameter(
                torch.full((num_experts, hidden, inter // self.BLOCK_K), 127, dtype=torch.uint8, device="cpu"),
                requires_grad=False)
            quant_method = FusedMoeWeightScaleSupported.GROUP.value
            ue8m0 = True
        else:
            w13_s = torch.nn.Parameter(
                torch.ones(num_experts, 2 * inter, hidden // self.BLOCK_K, dtype=torch.float32, device="cpu"),
                requires_grad=False)
            w2_s = torch.nn.Parameter(
                torch.ones(num_experts, hidden, inter // self.BLOCK_K, dtype=torch.float32, device="cpu"),
                requires_grad=False)
            quant_method = FusedMoeWeightScaleSupported.BLOCK.value
            ue8m0 = False
        w13_s.format_ue8m0 = ue8m0
        w2_s.format_ue8m0 = ue8m0
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = quant_method
        n13, n2 = self.scale_names(layer)
        layer.register_parameter(n13, w13_s)
        set_weight_attrs(w13_s, scale_attrs)
        layer.register_parameter(n2, w2_s)
        set_weight_attrs(w2_s, scale_attrs)

    def take_full(self, layer) -> FullWeights:
        n13, n2 = self.scale_names(layer)
        return FullWeights(
            w13=_host(layer.w13_weight.data), w2=_host(layer.w2_weight.data),
            w13_scale=_host(getattr(layer, n13).data),
            w2_scale=_host(getattr(layer, n2).data))

    def release(self, layer) -> None:
        for name in ("w13_weight", "w2_weight", *self.scale_names(layer)):
            p = getattr(layer, name)
            p.data = torch.empty(0, dtype=p.dtype)

    def check_full_shapes(self, full, dims, where):
        E, H, I = dims.num_experts, dims.hidden_size, dims.intermediate_size
        _check_shapes(where, full.w13, (E, 2 * I, H // 2), full.w2, (E, H, I // 2))
        if full.w13_scale is None or full.w2_scale is None:
            raise PlanError(f"{where}: mxfp4 store needs w13/w2 scales")
        _check_shapes(where + " scales", full.w13_scale, (E, 2 * I, H // self.BLOCK_K),
                      full.w2_scale, (E, H, I // self.BLOCK_K))
        if full.w13.dtype not in (torch.int8, torch.uint8) or full.w2.dtype not in (torch.int8, torch.uint8):
            raise PlanError(f"{where}: mxfp4 codes must be int8/uint8 nibble bytes, got "
                            f"{full.w13.dtype}/{full.w2.dtype}")

    def proj_source(self, full, inter, proj):
        return ProjSource(w=_proj_view(full.w13, full.w2, inter, proj).view(torch.uint8),
                          scales=_proj_view(full.w13_scale, full.w2_scale, inter, proj))

    def check_index(self, ti, where):
        """모든 expert의 인덱스가 32-블록 단위여야 한다: 길이 % 32 == 0, 오프셋 % 32 == 0,
        각 32-묶음이 어떤 원본 블록 b의 [32b, 32b+32) 오름차순 전체와 일치."""
        B = self.BLOCK_K
        idx = ti.idx.to(torch.int64)
        for e in range(ti.num_experts):
            o0, o1 = int(ti.row_off[e]), int(ti.row_off[e + 1])
            if o0 % B or (o1 - o0) % B:
                raise PlanError(f"{where}: expert {e} rows [{o0},{o1}) are not {B}-aligned — "
                                f"mxfp4 tier index must move in whole scale blocks")
            if o1 == o0:
                continue
            rows = idx[o0:o1].reshape(-1, B)
            starts = rows[:, 0]
            if torch.any(starts % B) or not torch.equal(rows, starts[:, None] + torch.arange(B)[None, :]):
                raise PlanError(f"{where}: expert {e} index does not consist of whole {B}-row "
                                f"scale blocks — mxfp4 cannot split a block across tiers")

    def gather(self, src, ti, uniform_k, band_start):
        codes = _gather_rows(src.w, _half_index(ti, 2, band_start, uniform_k),
                             None if uniform_k is None else uniform_k // 2,
                             None if band_start is None else band_start // 2)
        scales = _gather_rows(_e8m0_bytes(src.scales),
                              _half_index(ti, self.BLOCK_K, band_start, uniform_k),
                              None if uniform_k is None else uniform_k // self.BLOCK_K,
                              None if band_start is None else band_start // self.BLOCK_K)
        return codes, scales

    def cold_source(self, src, where):
        raise PlanError(f"{where}: mxfp4 has no warm-kt (bf16 slab) mode")

    def cold_flat(self, src, ti, real_rows, tile):
        """cold 블록 = nibble 행 [N, k_pad/2] u8 + 배율 bf16 [N, k_pad/32] (E8M0 → bf16 = e<<7,
        정확). 패딩 블록은 코드 0, 배율 1.0(0x3F80) — weight 0이라 무해하고 denormal을 피한다."""
        codes, off, idx, real_t = _cold_flat_rows(src.w, _half_index(ti, 2, None, None), real_rows, tile,
                                                  pad_value=0, div=2)
        e8 = _e8m0_bytes(src.scales)
        s16 = (e8.to(torch.int16) << 7).view(torch.bfloat16)   # 2^(e−127) exact in bf16
        scales, _, _, _ = _cold_flat_rows(s16, _half_index(ti, self.BLOCK_K, None, None), real_rows, tile,
                                          pad_value=0x3F80, div=self.BLOCK_K)
        idx = _padded_kindex(ti, off, real_rows)   # k 단위 인덱스 (패딩 항목 0)
        return codes, scales, off, idx, real_t

    def cold_load_kwargs(self, cold):
        return dict(gate_scale=cold.gate.s_flat, up_scale=cold.up.s_flat, down_scale=cold.down.s_flat)

    def cold_slab(self, ptr, nbytes, expert_off, device):
        slab = _tensor_view(ptr, nbytes, torch.uint8)
        return slab, torch.tensor(list(expert_off[:-1]), dtype=torch.int64, device=device)

    def grouped_cold(self):
        from sglang.jit_kernel import prism_grouped_mxfp4 as k

        return k.grouped_mxfp4_cold

    def grouped_cold_gateup(self):
        from sglang.jit_kernel import prism_grouped_mxfp4 as k

        return k.grouped_mxfp4_cold_gateup

    def gemv(self, *, pinned, sparse):
        from sglang.jit_kernel import prism_gemv_mxfp4 as k

        table = {
            (False, False): k.gemv_mxfp4_indexed,
            (True, False): k.gemv_mxfp4_indexed_pinned,
            (False, True): k.gemv_mxfp4_indexed_sparse,
            (True, True): k.gemv_mxfp4_indexed_pinned_sparse,
        }
        return table[(pinned, sparse)]

    def gemv_gateup(self, *, pinned, sparse):
        from sglang.jit_kernel import prism_gemv_mxfp4 as k

        table = {
            (False, False): k.gemv_mxfp4_indexed_gateup,
            (True, False): k.gemv_mxfp4_indexed_pinned_gateup,
            (False, True): k.gemv_mxfp4_indexed_sparse_gateup,
            (True, True): k.gemv_mxfp4_indexed_pinned_sparse_gateup,
        }
        return table[(pinned, sparse)]

    def grouped(self, *, pinned):
        from sglang.jit_kernel import prism_grouped_mxfp4 as k

        return k.grouped_mxfp4_indexed_pinned if pinned else k.grouped_mxfp4_indexed

    def grouped_gateup(self, *, pinned):
        from sglang.jit_kernel import prism_grouped_mxfp4 as k

        return k.grouped_mxfp4_indexed_pinned_gateup if pinned else k.grouped_mxfp4_indexed_gateup

    def warmup(self) -> None:
        from sglang.jit_kernel.prism_gemv_mxfp4 import warmup_jit
        from sglang.jit_kernel.prism_grouped_mxfp4 import warmup_jit as warmup_grouped

        warmup_jit()
        warmup_grouped()


