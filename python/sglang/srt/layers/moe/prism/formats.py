"""스토어 포맷 — GPU 티어 스토어가 bf16이냐 MXFP4냐 FP8이냐를 **객체 하나**로 표현한다.

계약 ①에서 커널 선택은 model-global(`KernelSpec.gpu_warm`)이고, 그 키가 함의하는 것이
스토어 형식·K 정렬·커널 진입점·로더 파라미터 형태 전부다. 그것을 `if fmt == ...`로
곳곳에 흩뿌리지 않고 여기 한 객체에 모은다 — weights.py(gather·정렬 검증), method.py
(파라미터 등록·full 텐서 인출), tiers.py(커널 진입점 선택)는 이 객체의 메서드를 부를 뿐
형식 이름을 모른다.

세 포맷:
  bf16  — `w_flat bf16 [Σₑ k[e], N]`. 정렬 = 페어(2). 기존 경로 그대로.
  mxfp4 — `w_flat u8 [Σₑ k[e]/2, N]`(코드, 행 = k-페어) + `s_flat u8 [Σₑ k[e]/32, N]`(E8M0
          배율, 행 = 32-k 블록). 정렬 = 32 (배율 블록이 원본 32행 블록이라 티어 K-인덱스가
          블록을 쪼개면 "블록당 배율 1"이 깨진다 — 재양자화 없이 체크포인트 수치를 보존하는
          유일한 선택).
  fp8   — `w_flat u8 [Σₑ k[e], N]`(e4m3 코드, 행 = k) + `s_flat fp32 [Σₑ k[e]/128, N/128]`
          (blockwise `scale_inv`, 행 = 128-k 블록, 열 = 128-n 블록). 정렬 = 128, 같은 이유로
          한 단계 거칠다. mxfp4·fp8 둘 다 prefill은 CPU(AMX)를 거치지 않고 전부 GPU다.

hot/warm의 계산 계약은 포맷 안에서도 같다: 갈리는 것은 `pinned`(스토어 거처) 하나이고
그것은 커널 진입점의 host 검증 차이일 뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sglang.srt.layers.moe.prism.index import TierIndex
from sglang.srt.layers.moe.prism.plan import ModelDims, PlanError, Proj


@dataclass(frozen=True)
class FullWeights:
    """process_weights_after_loading 시점의 한 레이어 full expert weight (CPU).

    bf16: w13 [E, 2I, H] bf16, w2 [E, H, I] bf16, scale 둘은 None.
    mxfp4: w13 [E, 2I, H/2] int8(nibble), w2 [E, H, I/2] int8, w13_scale [E, 2I, H/32],
           w2_scale [E, H, I/32] (fp32 = E8M0를 로더가 캐스팅한 값, 또는 u8/e8m0).
    fp8:   w13 [E, 2I, H] float8_e4m3fn, w2 [E, H, I], w13_scale [E, 2I/128, H/128] fp32,
           w2_scale [E, H/128, I/128] fp32 (blockwise `scale_inv`)."""

    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ProjSource:
    """한 proj의 ckpt-방향 소스 `[E, N, K…]` 뷰 묶음. bf16은 w만, mxfp4는 (codes, scales)."""

    w: torch.Tensor                 # bf16 [E, N, K] / u8 [E, N, K/2]
    scales: Optional[torch.Tensor]  # mxfp4: [E, N, K/32] (fp32 또는 u8)

    @property
    def num_experts(self) -> int:
        return int(self.w.shape[0])

    @property
    def n(self) -> int:
        return int(self.w.shape[1])


class StoreFormat:
    """포맷 다형성의 기저. 서브클래스가 전부를 채운다 (여기 구현은 공통 헬퍼만)."""

    name: str = ""
    k_align: int = 2          # 티어 K-인덱스가 지켜야 할 정렬 (계약 ①: 커널 키가 함의)
    has_scales: bool = False
    supports_cold: bool = True
    cold_kernels: tuple = ()  # 이 포맷의 cold 스토어를 소비할 수 있는 cpu_cold 커널 키
    # cold(kt) slab의 dtype과 blk_off 단위 — GPU 제자리 읽기 로더가 해석하는 형태.
    cold_slab_dtype = torch.bfloat16

    def check_cold_kernel(self, name: str) -> None:
        if name not in self.cold_kernels:
            raise PlanError(f"cpu_cold kernel '{name}' cannot consume a {self.name} store "
                            f"(compatible: {list(self.cold_kernels)})")

    def default_cold_gpu_min_m(self, executor_default: int, grouped_min_m: int) -> int:
        """cold를 GPU가 읽는 최소 M의 기본값. bf16은 실측 교차점(executor 기본), mxfp4는 CPU
        prefill(AMX) 경로를 쓰지 않으므로 grouped 경계 그대로 (사용자 결정 2026-08-27)."""
        return executor_default

    def cold_flat(self, src: ProjSource, ti: TierIndex, real_rows, tile: int):
        """cold 스토어: expert 블록 [N, k_pad(e)…]를 이어 붙인 flat들 + 패딩된 (row_off, k_index,
        real_rows). 반환 (w_flat, s_flat|None, off, idx, real_t)."""
        raise NotImplementedError

    def cold_load_kwargs(self, cold) -> dict:
        """PartialMoEWrapper.load_weights_from_tensors에 넘길 포맷별 추가 인자 (배율 등)."""
        return {}

    def cold_slab(self, ptr: int, nbytes: int, expert_off, device) -> tuple:
        """kt slab 기술자 → (slab 텐서 뷰, blk_off device 텐서). bf16은 원소 단위, u8은 바이트."""
        raise NotImplementedError

    def grouped_cold(self):
        raise NotImplementedError

    def grouped_cold_gateup(self):
        raise NotImplementedError

    # ── 로딩: 파라미터 ───────────────────────────────────────────────────
    def create_params(self, layer, num_experts: int, hidden: int, inter: int,
                      params_dtype, extra_weight_attrs: dict) -> None:
        raise NotImplementedError

    def take_full(self, layer) -> FullWeights:
        """레이어의 full 파라미터를 CPU 텐서로 인출한다 (device_loading_context가 CUDA로
        옮겨둔 상태일 수 있다 — cold/pinned 슬라이스는 host 메모리에서 해야 한다)."""
        raise NotImplementedError

    def release(self, layer) -> None:
        """full 파라미터 소멸 (계약 ③)."""
        raise NotImplementedError

    def check_full_shapes(self, full: FullWeights, dims: ModelDims, where: str) -> None:
        raise NotImplementedError

    def proj_source(self, full: FullWeights, inter: int, proj: Proj) -> ProjSource:
        raise NotImplementedError

    # ── 로딩: 티어 스토어 ────────────────────────────────────────────────
    def check_index(self, ti: TierIndex, where: str) -> None:
        """티어 K-인덱스의 정렬 검증. 기본(bf16)은 index.validate_layer의 페어 검증으로 충분."""

    def gather(self, src: ProjSource, ti: TierIndex, uniform_k: Optional[int],
               band_start: Optional[int]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """[E, N, K…] 소스 → (w_flat, s_flat) CPU. hot/warm 공통 (거처는 호출자가 옮긴다)."""
        raise NotImplementedError

    def cold_source(self, src: ProjSource, where: str) -> torch.Tensor:
        """cold(CPU kt) 경로가 받는 bf16 [E, N, K] 소스. 지원하지 않는 포맷은 즉사."""
        raise NotImplementedError

    # ── 커널 진입점 ──────────────────────────────────────────────────────
    def store_args(self, shard) -> tuple:
        """커널 래퍼의 스토어 인자 (bf16: (w_flat,), mxfp4: (w_flat, s_flat))."""
        return (shard.w_flat,) if not self.has_scales else (shard.w_flat, shard.s_flat)

    def gemv(self, *, pinned: bool, sparse: bool):
        raise NotImplementedError

    def gemv_gateup(self, *, pinned: bool, sparse: bool):
        """gate+up 융합 진입점. 그 조합의 진입점이 없으면 None (2회 launch)."""
        return None

    def grouped(self, *, pinned: bool):
        raise NotImplementedError

    def grouped_gateup(self, *, pinned: bool):
        raise NotImplementedError

    def wres_k_max(self, *shards) -> int:
        """PCIe grouped launch을 W-resident 커널로 보낼 때의 k_max (0 = 스트리밍 커널)."""
        return 0

    def warmup(self) -> None:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
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

    cold_kernels = ("kt_amx_bf16", "kt_tile_k2_bf16")
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


# ─────────────────────────────────────────────────────────────────────────────
class Mxfp4Format(StoreFormat):
    """DeepSeek-V4-Flash류 MXFP4 g32 routed expert. 파라미터 이름/shape/dtype/attrs는 sglang의
    `DeepSeekMxfp4MoEMethod.create_weights`와 **동일**해야 한다 — 그 이름으로 로더가 채운다
    (w13_weight int8 [E, 2I, H/2], w13_weight_scale_inv fp32 [E, 2I, H/32], BLOCK quant_method)."""

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

    def create_params(self, layer, num_experts, hidden, inter, params_dtype, extra_weight_attrs):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported
        from sglang.srt.utils import set_weight_attrs

        if hidden % self.BLOCK_K or inter % self.BLOCK_K:
            raise PlanError(f"mxfp4 needs hidden/inter multiples of {self.BLOCK_K}, got {hidden}/{inter}")
        w13 = torch.nn.Parameter(
            torch.empty(num_experts, 2 * inter, hidden // 2, dtype=torch.int8, device="cpu"),
            requires_grad=False)
        w2 = torch.nn.Parameter(
            torch.empty(num_experts, hidden, inter // 2, dtype=torch.int8, device="cpu"),
            requires_grad=False)
        layer.register_parameter("w13_weight", w13)
        set_weight_attrs(w13, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2)
        set_weight_attrs(w2, extra_weight_attrs)
        w13_s = torch.nn.Parameter(
            torch.ones(num_experts, 2 * inter, hidden // self.BLOCK_K, dtype=torch.float32, device="cpu"),
            requires_grad=False)
        w2_s = torch.nn.Parameter(
            torch.ones(num_experts, hidden, inter // self.BLOCK_K, dtype=torch.float32, device="cpu"),
            requires_grad=False)
        w13_s.format_ue8m0 = False
        w2_s.format_ue8m0 = False
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


# ─────────────────────────────────────────────────────────────────────────────
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
    cold_kernels = ("kt_tile_k2_fp8b128",)
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


BF16 = Bf16Format()
MXFP4 = Mxfp4Format()
FP8 = Fp8Format()
FORMATS = {BF16.name: BF16, MXFP4.name: MXFP4, FP8.name: FP8}


# ─── 공통 헬퍼 ───────────────────────────────────────────────────────────────
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
        if bool(torch.all(ti.row_off[1:] - ti.row_off[:-1] == rlen)):
            flat = src[:, :, start: start + rlen].contiguous().reshape(-1)
            idx = (torch.arange(kr0, dtype=torch.int64) + start).repeat(E) if div == 1 \
                else torch.zeros(total_k, dtype=torch.int64)
            return flat, off, idx, real_t
    flat = torch.full((total_k // div * N,), pad_value, dtype=src.dtype)
    idx = torch.zeros(total_k, dtype=torch.int64)
    for e in range(E):
        kr, kp, o0 = int(real_rows[e]), pad[e], int(off[e])
        if kr == 0:
            continue
        rows = ti.idx[int(ti.row_off[e]): int(ti.row_off[e]) + kr // div].to(torch.int64)
        if div == 1:
            idx[o0: o0 + kr] = rows
        blk = flat[(o0 // div) * N: ((o0 + kp) // div) * N].view(N, kp // div)
        blk[:, : kr // div] = src[e].index_select(1, rows)
    return flat.contiguous(), off, idx, real_t


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
