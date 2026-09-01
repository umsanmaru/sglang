"""dense 스토어 포맷 — bf16과 blockwise fp8.

`moe/prism/formats.py`의 `StoreFormat`이 하던 일 중 **절단·검증에 필요한 만큼만**
가져왔다. MoE 포맷은 파라미터 생성(`create_params`)과 full 텐서 인출(`take_full`)까지
들고 있는데 그 둘은 `w13_weight`/`w2_weight`라는 MoE 어휘에 묶여 있어 dense에 못 온다.
dense에서 그 역할은 `method.py`가 `LinearBase`의 `weight`/`weight_scale_inv`를 직접
읽어 넘기는 것으로 대체된다 — 그래서 이 모듈은 텐서만 보고 layer를 모른다.

**포맷이 정하는 것은 결국 하나다: K축을 어디서 자를 수 있는가.**

| | bf16 | fp8 (blockwise) |
|---|---|---|
| 코드 | `[N, K]` bf16 | `[N, K]` e4m3 (u8로 다룬다) |
| 배율 | 없음 | `[N/128, K/128]` fp32 `scale_inv` |
| K 정렬 | 2 (페어) | **128** |
| cold 커널 | `kt_amx_bf16`, `kt_tile_k2_bf16` | `kt_tile_k2_fp8b128` |

fp8의 128은 임의 상수가 아니다: 배율 하나가 원본 128k × 128n 블록을 덮으므로, 티어
경계가 블록을 쪼개면 두 티어가 같은 배율을 나눠 갖게 되어 "블록당 배율 1"이 깨진다.
재양자화 없이 체크포인트 수치를 보존하려면 블록 단위로만 자를 수 있다 (MoE fp8과
같은 계약 — `moe/prism/formats.py`의 `Fp8Format` 참조).

**mxfp4는 없다.** dense 대상(qkvo·dense MLP)에 mxfp4 체크포인트가 없다는 사용자
확인(2026-08-31)에 따른다. 필요해지면 `k_align=32`짜리 구현을 여기 더하면 되고,
배관은 이미 배율 있는 포맷을 지원한다.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.layers.prism.geometry import PAIR_GROUP, PlanError
from sglang.srt.layers.prism.kernels import cold_slab_layout, gpu_store_format_tag
from sglang.srt.layers.prism.store import IDX_DTYPE

# 이 포맷들이 받아들이는 cold slab 레이아웃 태그 (`kernels.cold_slab_layout`).
_BF16_COLD_LAYOUTS = ("kt_bf16",)
_FP8_COLD_LAYOUTS = ("kt_tile8",)


class LinearStoreFormat:
    """한 스토어 형식의 절단 규칙. 인스턴스는 상태가 없다 (모듈 전역 1벌)."""

    name: str = ""
    k_align: int = PAIR_GROUP
    has_scales: bool = False
    cold_layouts: Tuple[str, ...] = ()

    # ── 검증 ────────────────────────────────────────────────────────────
    def check(self, weight, scale, k: int, n: int, where: str) -> None:
        raise NotImplementedError

    def check_rows(self, rows: torch.Tensor, where: str) -> None:
        """티어가 가져가는 K행이 이 포맷의 절단 단위를 지키는가.

        로드 타임 검사다 (MoE `StoreFormat.check_index`와 같은 자리). plan의 밴드
        정렬은 `ROW_GROUP=2`까지만 보므로, 128 정렬 같은 포맷 요구는 여기가 유일한
        게이트다 — 통과하지 못한 절단은 배율을 쪼개 **조용히 다른 수치**를 만든다.
        """
        a = self.k_align
        if a <= PAIR_GROUP:
            return
        n_rows = int(rows.numel())
        if n_rows % a:
            raise PlanError(
                f"{where}: row count {n_rows} is not a multiple of k_align={a} "
                f"({self.name} tier index must move in whole scale blocks)"
            )
        if n_rows == 0:
            return
        blocks = rows.to(torch.int64).reshape(-1, a)
        starts = blocks[:, 0]
        expected = starts[:, None] + torch.arange(a)[None, :]
        if bool((starts % a).any()) or not torch.equal(blocks, expected):
            raise PlanError(
                f"{where}: index does not consist of whole {a}-row scale blocks — "
                f"{self.name} cannot split a block across tiers"
            )

    # ── 절단 ────────────────────────────────────────────────────────────
    def gather(self, weight, scale, rows, contiguous, k_start):
        """`[N, K]` → GPU 티어 스토어 `([k_tier, N], scales|None)` (CPU에서)."""
        raise NotImplementedError

    def cold_flat(self, weight, scale, rows, tile: int):
        """`[N, K]` → cold 스토어 `([N, k_pad], scales|None, k_index, real_rows)`."""
        raise NotImplementedError

    # ── 커널 진입점 ──────────────────────────────────────────────────────
    # MoE 커널을 E=1로 퇴화시켜 쓴다 (`tiers.py` docstring). 스토어 레이아웃이
    # MoE와 같으므로 진입점도 같은 것을 부른다 — 갈리는 것은 인자의 모양뿐이다.
    def store_args(self, shard) -> tuple:
        """커널 래퍼의 스토어 인자 (bf16: (w_flat,), fp8: (w_flat, s_flat))."""
        return (shard.w_flat,) if not self.has_scales else (shard.w_flat, shard.s_flat)

    def gemv(self, *, pinned: bool, sparse: bool):
        raise NotImplementedError

    def grouped(self, *, pinned: bool):
        """prefill 진입점 — W를 **한 번만** 읽는다.

        worklist(`gemv`)는 pair마다 W를 다시 읽는데, dense는 E=1이라 중복도가 곧 M이다
        (MoE는 M·k/E). warm은 그 재읽기가 전부 PCIe라 M=2048에서 층당 수 초가 된다 —
        실측 2026-09-01: Qwen3.8 전체 warm 43.8 GB × 2048 = 89.7 TB, 30분/forward.
        """
        raise NotImplementedError

    def warmup(self) -> None:
        """JIT 컴파일을 startup으로 앞당긴다 (캡처 워밍업에 얽히지 않게)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 공통 헬퍼 — 방향 규약이 두 포맷에서 같으므로 코드도 같다.
# ---------------------------------------------------------------------------


def _gather_km(src: torch.Tensor, rows: torch.Tensor, contiguous: bool,
               k_start: Optional[int]) -> torch.Tensor:
    """`[N, R]` → `[r_tier, N]` K-major. contiguous면 slice+transpose 한 번."""
    if contiguous and k_start is not None:
        return src[:, k_start : k_start + int(rows.numel())].t().contiguous()
    return src.t().index_select(0, rows.to(torch.int64)).contiguous()


def _pad_ckpt(src: torch.Tensor, rows: torch.Tensor, pad_to: int, pad_value) -> torch.Tensor:
    """`[N, R]` → cold 방향 `[N, pad_to]`, 남는 열은 pad_value."""
    real = int(rows.numel())
    out = torch.full((int(src.shape[0]), pad_to), pad_value, dtype=src.dtype)
    if real:
        out[:, :real] = src.index_select(1, rows.to(torch.int64))
    return out.contiguous()


def _pad_index(rows: torch.Tensor, pad_to: int) -> torch.Tensor:
    """패딩된 K-인덱스. 패딩 항목은 0을 가리킨다 — weight가 0이라 무해하지만
    kt가 축 범위를 검증하므로 유효한 값이어야 한다."""
    idx = torch.zeros(pad_to, dtype=torch.int64)
    real = int(rows.numel())
    if real:
        idx[:real] = rows.to(torch.int64)
    return idx.to(IDX_DTYPE)


def _check_2d(t: torch.Tensor, shape, what: str, where: str) -> None:
    if tuple(t.shape) != tuple(shape):
        raise PlanError(
            f"{where}: {what} shape {tuple(t.shape)} != expected {tuple(shape)} — "
            f"plan이 다른 모델/설정에 적용되고 있을 가능성"
        )


# ---------------------------------------------------------------------------


class Bf16LinearFormat(LinearStoreFormat):
    name = "bf16"
    k_align = PAIR_GROUP
    has_scales = False
    cold_layouts = _BF16_COLD_LAYOUTS

    def check(self, weight, scale, k, n, where):
        if weight.dtype != torch.bfloat16:
            raise PlanError(f"{where}: bf16 store needs a bfloat16 weight, got {weight.dtype}")
        _check_2d(weight, (n, k), "weight", where)
        if scale is not None:
            raise PlanError(f"{where}: bf16 store takes no scales but one was given")

    def gather(self, weight, scale, rows, contiguous, k_start):
        return _gather_km(weight, rows, contiguous, k_start), None

    def cold_flat(self, weight, scale, rows, tile):
        real = int(rows.numel())
        k_pad = ((real + tile - 1) // tile) * tile
        return _pad_ckpt(weight, rows, k_pad, 0), None, _pad_index(rows, k_pad), real

    def gemv(self, *, pinned, sparse):
        from sglang.jit_kernel import prism_gemv as k

        return {
            (False, False): k.gemv_worklist_indexed,
            (True, False): k.gemv_worklist_indexed_pinned,
            (False, True): k.gemv_worklist_indexed_sparse,
            (True, True): k.gemv_worklist_indexed_pinned_sparse,
        }[(pinned, sparse)]

    def grouped(self, *, pinned):
        from sglang.jit_kernel import prism_grouped as k

        return k.grouped_gemm_indexed_pinned if pinned else k.grouped_gemm_indexed

    def warmup(self) -> None:
        from sglang.jit_kernel.prism_gemv import warmup_jit
        from sglang.jit_kernel.prism_grouped import warmup_jit as warmup_grouped

        warmup_jit()
        warmup_grouped()


class Fp8LinearFormat(LinearStoreFormat):
    """DeepSeek류 blockwise FP8 — e4m3 코드 + 128×128 fp32 `scale_inv`.

    `weight`는 `float8_e4m3fn`인데 대부분의 torch 연산(index_select 등)이 그 dtype을
    안 받으므로 **uint8 뷰로 다룬다**. 비트를 옮기는 것뿐이라 값은 보존된다.

    배율은 `[N/128, K/128]`이고 **N축도 블록**이다 — 그래서 배율의 gather는 K 블록
    번호(`rows[::128] // 128`)로 하고, 결과의 열이 `N/128`이 된다.
    """

    name = "fp8"
    k_align = 128
    has_scales = True
    cold_layouts = _FP8_COLD_LAYOUTS
    BLOCK = 128

    def check(self, weight, scale, k, n, where):
        B = self.BLOCK
        if weight.dtype not in (torch.float8_e4m3fn, torch.uint8, torch.int8):
            raise PlanError(
                f"{where}: fp8 codes must be float8_e4m3fn (or raw bytes), got {weight.dtype}"
            )
        _check_2d(weight, (n, k), "weight", where)
        if k % B or n % B:
            raise PlanError(
                f"{where}: fp8 needs k and n to be multiples of {B}, got k={k} n={n} — "
                f"부분 배율 블록은 표현할 수 없다"
            )
        if scale is None:
            raise PlanError(f"{where}: fp8 store needs weight_scale_inv")
        _check_2d(scale, (n // B, k // B), "weight_scale_inv", where)
        if scale.dtype != torch.float32:
            raise PlanError(
                f"{where}: fp8 scale_inv must be fp32, got {scale.dtype} "
                f"(ue8m0/mxfp8 체크포인트는 아직 지원하지 않는다)"
            )

    def _codes(self, weight):
        return weight if weight.dtype == torch.uint8 else weight.view(torch.uint8)

    def _scale_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """K행 → 배율의 K-블록 행. `check_rows`가 통째 블록임을 보장한 뒤에만 유효하다."""
        return rows.to(torch.int64).reshape(-1, self.BLOCK)[:, 0] // self.BLOCK

    def gather(self, weight, scale, rows, contiguous, k_start):
        B = self.BLOCK
        codes = _gather_km(self._codes(weight), rows, contiguous, k_start)
        srows = self._scale_rows(rows)
        s = _gather_km(
            scale, srows, contiguous, None if k_start is None else k_start // B
        )
        return codes, s

    def cold_flat(self, weight, scale, rows, tile):
        B = self.BLOCK
        real = int(rows.numel())
        # 타일이 블록의 배수여야 배율 패딩이 통째 블록으로 떨어진다.
        # (`kt_tile_k2_fp8b128`의 tile은 128 — kernels._CPU_COLD_TILE_ROWS)
        if tile % B:
            raise PlanError(
                f"fp8 cold tile rows {tile} is not a multiple of the {B}-scale block"
            )
        k_pad = ((real + tile - 1) // tile) * tile
        codes = _pad_ckpt(self._codes(weight), rows, k_pad, 0)
        # 패딩 코드가 0x00(= +0.0)이라 배율은 아무 유한값이어도 되지만 1.0을 넣는다.
        s = _pad_ckpt(scale, self._scale_rows(rows), k_pad // B, 1.0)
        return codes, s, _pad_index(rows, k_pad), real

    def gemv(self, *, pinned, sparse):
        from sglang.jit_kernel import prism_gemv_fp8 as k

        return {
            (False, False): k.gemv_fp8_indexed,
            (True, False): k.gemv_fp8_indexed_pinned,
            (False, True): k.gemv_fp8_indexed_sparse,
            (True, True): k.gemv_fp8_indexed_pinned_sparse,
        }[(pinned, sparse)]

    def grouped(self, *, pinned):
        from sglang.jit_kernel import prism_grouped_fp8 as k

        return k.grouped_fp8_indexed_pinned if pinned else k.grouped_fp8_indexed

    def warmup(self) -> None:
        from sglang.jit_kernel.prism_gemv_fp8 import warmup_jit
        from sglang.jit_kernel.prism_grouped_fp8 import warmup_jit as warmup_grouped

        warmup_jit()
        warmup_grouped()


BF16 = Bf16LinearFormat()
FP8 = Fp8LinearFormat()

FORMATS = {BF16.name: BF16, FP8.name: FP8}


def store_format(gpu_warm: str, cpu_cold: str) -> LinearStoreFormat:
    """plan의 커널 쌍 → 스토어 포맷. 두 커널의 형식이 어긋나면 즉사.

    GPU 커널 키가 스토어 형식을 함의하고(계약 ①) cold 커널 키가 slab 레이아웃을
    함의하므로, 둘이 다른 형식을 가리키는 plan은 startup에서 죽어야 한다 — 안 죽으면
    한쪽이 남의 바이트를 자기 형식으로 읽는다.
    """
    tag = gpu_store_format_tag(gpu_warm)
    fmt = FORMATS.get(tag)
    if fmt is None:
        raise PlanError(
            f"prism dense supports {sorted(FORMATS)} stores, but gpu_warm "
            f"'{gpu_warm}' implies '{tag}'"
        )
    layout = cold_slab_layout(cpu_cold)
    if layout not in fmt.cold_layouts:
        raise PlanError(
            f"cpu_cold '{cpu_cold}' (slab layout '{layout}') is not compatible with the "
            f"{fmt.name} store (expected one of {list(fmt.cold_layouts)})"
        )
    return fmt
