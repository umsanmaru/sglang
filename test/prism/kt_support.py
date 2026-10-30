"""설치된 kt-kernel 빌드가 어떤 cold 커널을 **실제로** 제공하는지 묻는 게이트.

plan 스키마가 받아들이는 이름표(`known_cpu_kernels()`)와, 지금 링크된 kt 빌드가
담고 있는 연산자는 **다른 집합**이다. 레포의 plan·테스트는 앞을 기준으로 쓰이고
빌드는 뒤를 정하므로, 구버전 kt로 레포를 받으면 새 커널 테스트가 통째로 빨갛게
뜬다 — "코드가 깨졌다"와 "빌드가 옛것이다"를 구분할 수 없는 상태다.

여기서 그 둘을 갈라 뒤쪽은 **skip**으로 보낸다. 새 커널을 하나 추가할 때
테스트가 해야 할 일은 그 키를 `requires_cold_kernel`에 적는 것뿐이다.

    @requires_cold_kernel("kt_tile_k1_fp8b128")
    def test_something(): ...

    # 커널 키 자체가 파라미터인 테스트
    @pytest.mark.parametrize("cpu_kernel", [cold_kernel_param(k) for k in KEYS])
    def test_each(cpu_kernel): ...
"""

import pytest


def cold_kernel_available(kernel_key: str) -> bool:
    """이 프로세스에 링크된 kt 빌드가 `kernel_key` 연산자를 갖고 있는가.

    kt의 표는 키 → C++ 클래스를 **지연** 해석하므로(키가 있어도 확장 모듈에
    심볼이 없을 수 있다) 해석까지 시켜봐야 확실하다.
    """
    try:
        from kt_kernel.experts_partial import _PARTIAL_MOE_CLASSES
    except Exception:       # kt_kernel 자체가 없는 환경
        return False
    resolve = _PARTIAL_MOE_CLASSES.get(kernel_key)
    if resolve is None:     # 이 빌드의 표에 키가 없다 (구버전)
        return False
    try:
        resolve()
    except Exception:       # 표에는 있는데 확장 모듈이 그 클래스를 안 낸다
        return False
    return True


def requires_cold_kernel(kernel_key: str):
    """커널 키가 이 빌드에 없으면 skip하는 마커."""
    return pytest.mark.skipif(
        not cold_kernel_available(kernel_key),
        reason=f"kt-kernel build has no {kernel_key!r} cold operator",
    )


def cold_kernel_param(kernel_key: str):
    """`parametrize` 값 하나 — 그 키가 없는 빌드에서는 그 파라미터만 skip된다."""
    return pytest.param(kernel_key, marks=requires_cold_kernel(kernel_key))


def profile_dtype_param(dtype: str):
    """프로파일러 dtype 파라미터 — 그 dtype이 쓰는 cold 커널이 이 빌드에 없으면 skip.

    dtype → cold 커널 대응은 프로파일러의 `Store`가 갖고 있다. 테스트가 그 표를
    베껴 적지 않도록 여기서 물어본다.
    """
    from sglang.srt.layers.moe.prism.profile import store_of

    return pytest.param(dtype, marks=requires_cold_kernel(store_of(dtype).cpu_kernel))
