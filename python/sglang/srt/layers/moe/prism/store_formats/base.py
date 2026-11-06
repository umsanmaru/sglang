"""Storage contracts and borrowed checkpoint views."""
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


