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
"""

from __future__ import annotations

from typing import Tuple

_GPU_WARM_KERNELS: Tuple[str, ...] = ("gemv_worklist", "torch_bmm")
_CPU_COLD_KERNELS: Tuple[str, ...] = ("kt_amx_bf16",)

# cold packed 저장의 K축 타일 행 수 — **커널 키가 함의하는 값**이다 (계약 ①:
# "cold의 저장 형식(pack)은 커널 키가 함의한다 — 별도 codec 필드 없음").
# plan/자산이 지키는 정렬은 페어(%2)뿐이므로, 로더가 여기까지 올리고 0 행을
# 채운다. 새 cold 커널은 자기 타일 크기를 여기 등록한다.
_CPU_COLD_TILE_ROWS: dict = {"kt_amx_bf16": 32}


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


def cold_pack_tile_rows(name: str) -> int:
    """cold 스토어를 올림해야 하는 타일 행 수. 이름 검증도 겸한다."""
    try:
        return _CPU_COLD_TILE_ROWS[name]
    except KeyError:
        raise KernelError(
            f"unknown cpu_cold kernel '{name}' (known: {sorted(_CPU_COLD_KERNELS)})"
        ) from None


def resolve_cpu_kernel(name: str) -> str:
    """이름을 검증하고 kt-kernel 클래스 키를 반환한다. 인스턴스화는 cold
    backend의 몫 (kt-kernel이 Plan 어휘를 모르게 하는 번역 경계)."""
    if name not in _CPU_COLD_KERNELS:
        raise KernelError(
            f"unknown cpu_cold kernel '{name}' (known: {sorted(_CPU_COLD_KERNELS)})"
        )
    return name
