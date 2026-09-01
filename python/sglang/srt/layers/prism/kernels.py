"""Prism GPU 커널 이름 registry (계약 ①).

커널 이름은 startup에 검증되고, 이후 런타임에 문자열/enum 분기는 존재하지
않는다. 2026-08-25 이후 GPU 티어의 구현은 **하나**다 — `tiers.py`의 인덱스
worklist GEMV이고, hot/warm은 스토어의 거처로만 갈린다(계약 ①). 그래서 이
모듈에 남은 책임은 "Plan이 아는 이름인가"의 검증뿐이고, 구현 선택은 여기서
일어나지 않는다.

`torch_bmm`은 밴드 시절의 이름이다. 기존 plan 40개가 이 키를 쓰므로 유효한
이름으로 남기되, 가리키는 구현은 `gemv_worklist`와 같다 — bmm 경로는 가변
per-expert K를 표현할 수 없어(연속 배치 축 요구) 폐기됐다.

cpu_cold의 실체는 kt-kernel의 CRTP 클래스다. 여기서는 유효한 키 목록만 안다.

이 모듈은 `layers/prism/`의 공유 코어다 (2026-08-31 승격) — 커널 이름과 그 이름이
함의하는 정렬·타일·slab 레이아웃·스토어 태그는 expert 축과 무관하므로 MoE와 dense가
같은 표를 본다. 유일하게 갈리던 `gpu_store_format`(태그 → StoreFormat 객체)만
`gpu_store_format_tag`로 낮추고 객체 해석은 각 오프로드에 남겼다.
"""

from __future__ import annotations

from typing import Tuple

_GPU_WARM_KERNELS: Tuple[str, ...] = ("gemv_worklist", "torch_bmm", "gemv_worklist_mxfp4",
                                      "gemv_worklist_fp8")
_CPU_COLD_KERNELS: Tuple[str, ...] = ("kt_amx_bf16", "kt_tile_k2_bf16", "kt_amx_fp4",
                                      "kt_tile_k2_mxfp4", "kt_tile_k2_fp8b128")

# cold 커널 키가 함의하는 **slab 레이아웃**(GPU 제자리 읽기 로더가 해석) 과 노드 N shard 정렬.
#   kt_bf16  — kt BufferBBF16Impl packed 6D (prism_grouped.cuh COLD)
#   kt_fp4   — kt BufferBInt4KGroupImpl: 행우선 nibble + fp32 d (prism_grouped_mxfp4.cuh KT_FP4)
#   kt_tile4 — GemmKernelTileK2MXFP4::BufferB: fp4 32k×256n 타일 + 전치 E8M0 (KT_TILE4); N shard 256 배수
#   kt_tile8 — GemmKernelTileK2FP8B128::BufferB: e4m3 32k×256n 타일 + 전치 fp32 128×128 배율
#              (KT_TILE8); N shard 256 배수, K 128 배수
_CPU_COLD_SLAB_LAYOUT: dict = {"kt_amx_bf16": "kt_bf16", "kt_tile_k2_bf16": "kt_bf16",
                               "kt_amx_fp4": "kt_fp4", "kt_tile_k2_mxfp4": "kt_tile4",
                               "kt_tile_k2_fp8b128": "kt_tile8"}
_CPU_COLD_N_ALIGN: dict = {"kt_amx_bf16": 32, "kt_tile_k2_bf16": 32, "kt_amx_fp4": 32,
                           "kt_tile_k2_mxfp4": 256, "kt_tile_k2_fp8b128": 256}

# GPU 커널 키가 함의하는 **스토어 포맷** (formats.py). 키 하나가 스토어 형식·K 정렬·커널
# 진입점·로더 파라미터 형태를 전부 정한다 (계약 ①) — 이 dict가 그 유일한 대응표다.
#   gemv_worklist / torch_bmm → bf16 [Σₖ, N]  (정렬 2)
#   gemv_worklist_mxfp4       → mxfp4 pair-row codes u8 [Σₖ/2, N] + E8M0 scales u8 [Σₖ/32, N] (정렬 32)
#   gemv_worklist_fp8         → fp8 e4m3 codes u8 [Σₖ, N] + fp32 scale_inv [Σₖ/128, N/128] (정렬 128)
_GPU_STORE_FORMAT: dict = {
    "gemv_worklist": "bf16",
    "torch_bmm": "bf16",
    "gemv_worklist_mxfp4": "mxfp4",
    "gemv_worklist_fp8": "fp8",
}

# cold packed 저장의 K축 타일 행 수 — **커널 키가 함의하는 값**이다 (계약 ①:
# "cold의 저장 형식(pack)은 커널 키가 함의한다 — 별도 codec 필드 없음").
# plan/자산이 지키는 정렬은 페어(%2)뿐이므로, 로더가 여기까지 올리고 0 행을
# 채운다. 새 cold 커널은 자기 타일 크기를 여기 등록한다.
# fp8 타일은 배율 블록이 128 k라 타일 올림도 128이다 (32로 올리면 마지막 블록의 배율이 없다).
_CPU_COLD_TILE_ROWS: dict = {"kt_amx_bf16": 32, "kt_tile_k2_bf16": 32, "kt_amx_fp4": 32,
                             "kt_tile_k2_mxfp4": 32, "kt_tile_k2_fp8b128": 128}


class KernelError(ValueError):
    """커널 resolve 실패 또는 커널 입력 계약 위반."""


def known_gpu_kernels() -> Tuple[str, ...]:
    return _GPU_WARM_KERNELS


def known_cpu_kernels() -> Tuple[str, ...]:
    return _CPU_COLD_KERNELS


def resolve_gpu_kernel(name: str) -> str:
    """이름을 검증하고 그대로 돌려준다 (startup 1회)."""
    if name not in _GPU_WARM_KERNELS:
        raise KernelError(
            f"unknown gpu_warm kernel '{name}' (known: {sorted(_GPU_WARM_KERNELS)})"
        )
    return name


def gpu_store_format_tag(name: str) -> str:
    """GPU 커널 키 → 스토어 포맷 **태그** ("bf16"/"mxfp4"/"fp8"). 이름 검증 겸.

    태그를 실제 `StoreFormat` 객체로 바꾸는 것은 각 오프로드의 몫이다 — 파라미터
    이름과 full 텐서 인출이 포맷 객체에 딸려 있고 그것이 MoE(w13/w2)와
    dense(weight)에서 갈리기 때문이다. 커널 키가 함의하는 *형식*은 같고
    *컨테이너*만 다르므로, 공유되는 것은 정확히 이 대응표까지다.
    """
    try:
        return _GPU_STORE_FORMAT[name]
    except KeyError:
        raise KernelError(
            f"unknown gpu_warm kernel '{name}' (known: {sorted(_GPU_WARM_KERNELS)})"
        ) from None


def cold_pack_tile_rows(name: str) -> int:
    """cold 스토어를 올림해야 하는 타일 행 수. 이름 검증도 겸한다."""
    try:
        return _CPU_COLD_TILE_ROWS[name]
    except KeyError:
        raise KernelError(
            f"unknown cpu_cold kernel '{name}' (known: {sorted(_CPU_COLD_KERNELS)})"
        ) from None


def cold_slab_layout(name: str) -> str:
    """cold 커널 키 → slab 레이아웃 태그 (cold_gpu.ColdSlab.layout). 이름 검증 겸."""
    try:
        return _CPU_COLD_SLAB_LAYOUT[name]
    except KeyError:
        raise KernelError(f"unknown cpu_cold kernel '{name}' (known: {sorted(_CPU_COLD_KERNELS)})") from None


def cold_n_align(name: str) -> int:
    """cold 커널 키가 요구하는 노드 N shard 정렬 (행 수)."""
    return _CPU_COLD_N_ALIGN[resolve_cpu_kernel(name)]


def resolve_cpu_kernel(name: str) -> str:
    """이름을 검증하고 kt-kernel 클래스 키를 반환한다. 인스턴스화는 cold
    backend의 몫 (kt-kernel이 Plan 어휘를 모르게 하는 번역 경계)."""
    if name not in _CPU_COLD_KERNELS:
        raise KernelError(
            f"unknown cpu_cold kernel '{name}' (known: {sorted(_CPU_COLD_KERNELS)})"
        )
    return name
