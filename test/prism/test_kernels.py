"""Prism 커널 이름 registry 테스트.

2026-08-25 이후 GPU 티어의 **구현은 하나**다 (인덱스 worklist GEMV). 이름은
검증만 되고 구현 선택은 스토어의 거처(device/pinned)가 한다 — 그래서 이 파일에
남은 것은 registry뿐이다.

수치 계약(⑤: fp32 누산, bf16 재료화, 정확표현 입력에서 비트일치)의 커버리지는
`test_gemv_worklist.py`로 옮겨갔다 — 레퍼런스 대비 tolerance와 정수 비트일치를
커널 단위에서 직접 검증한다. 이 파일이 이전에 담당하던 `torch_bmm` 구현 테스트
세 개는 그 구현과 함께 사라졌다.
"""

import pytest

from sglang.srt.layers.moe.prism.kernels import (
    KernelError,
    known_cpu_kernels,
    known_gpu_kernels,
    resolve_cpu_kernel,
    resolve_gpu_kernel,
)


def test_registry_and_resolution():
    assert "kt_amx_bf16" in known_cpu_kernels()
    assert resolve_cpu_kernel("kt_amx_bf16") == "kt_amx_bf16"
    with pytest.raises(KernelError, match="unknown gpu_warm"):
        resolve_gpu_kernel("nope")
    with pytest.raises(KernelError, match="unknown cpu_cold"):
        resolve_cpu_kernel("nope")


def test_gpu_kernel_names_validated_only():
    """`torch_bmm`은 기존 plan 40개를 위해 유효한 이름으로 남지만 같은 구현을
    가리킨다 — bmm 경로는 가변 per-expert K를 표현할 수 없어 폐기됐다."""
    assert set(known_gpu_kernels()) == {"gemv_worklist", "torch_bmm", "gemv_worklist_mxfp4"}
    assert resolve_gpu_kernel("gemv_worklist") == "gemv_worklist"
    assert resolve_gpu_kernel("torch_bmm") == "torch_bmm"


def test_gpu_kernel_key_implies_store_format():
    """커널 키 하나가 스토어 포맷(정렬·진입점·파라미터 형태)을 함의한다 (계약 ①)."""
    from sglang.srt.layers.moe.prism.kernels import gpu_store_format

    assert gpu_store_format("gemv_worklist").name == "bf16"
    assert gpu_store_format("torch_bmm").name == "bf16"
    fmt = gpu_store_format("gemv_worklist_mxfp4")
    assert fmt.name == "mxfp4" and fmt.k_align == 32 and fmt.cold_kernels == ("kt_amx_fp4", "kt_tile_k2_mxfp4")
    with pytest.raises(KernelError):
        gpu_store_format("nope")
