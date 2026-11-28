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
    assert set(known_gpu_kernels()) == {"gemv_worklist", "torch_bmm", "gemv_worklist_mxfp4",
                                       "gemv_worklist_fp8", "gemv_worklist_fp8pt"}
    assert resolve_gpu_kernel("gemv_worklist") == "gemv_worklist"
    assert resolve_gpu_kernel("torch_bmm") == "torch_bmm"


def test_gpu_kernel_key_implies_store_format():
    """커널 키 하나가 스토어 포맷(정렬·진입점·파라미터 형태)을 함의한다 (계약 ①)."""
    from sglang.srt.layers.moe.prism.kernels import gpu_store_format

    assert gpu_store_format("gemv_worklist").name == "bf16"
    assert gpu_store_format("torch_bmm").name == "bf16"
    fmt = gpu_store_format("gemv_worklist_mxfp4")
    assert fmt.name == "mxfp4" and fmt.k_align == 32 and fmt.cold_kernels == ("kt_amx_fp4", "kt_tile_k2_mxfp4")
    fmt8 = gpu_store_format("gemv_worklist_fp8")
    assert fmt8.name == "fp8" and fmt8.k_align == 128
    assert fmt8.cold_kernels == ("kt_tile_k2_fp8b128", "kt_tile_k1_fp8b128")  # k2(페어) / k1(per-k) 타일
    with pytest.raises(KernelError):
        gpu_store_format("nope")


def test_kernel_keys_cover_exactly_the_built_exports():
    """Every GEMV/grouped export is reachable from a supported kernel key and every supported
    key is built, so adding an export or a key without the other fails here."""
    import itertools

    from sglang.kernels.ops.moe.prism import key as K

    unkeyed = {"gemv_worklist", "prism_wait_flag"}
    modules = ("gemv", "gemv_fp8", "gemv_mxfp4", "grouped", "grouped_fp8", "grouped_mxfp4")
    built = set().union(*(K.module_exports(m) for m in modules)) - unkeyed
    keyed = set()
    for op, fmt, mode, fused in itertools.product(K.OPS, K.FMTS, K.MODES, (False, True)):
        key = K.PrismKernelKey(op=op, fmt=fmt, mode=mode, fused=fused)
        if K.supports(key):
            keyed.add(key.export_name())
            assert callable(K.wrapper(key)), key
    assert keyed == built

    no_k1 = K.unsupported_reason(K.PrismKernelKey(op="gemv", fmt="mxfp4", mode="k1"))
    assert no_k1 is not None and "gemv_mxfp4_indexed_sparsek1" in no_k1
    assert not K.supports(K.PrismKernelKey(op="gemv", fmt="fp8", mode="k1w", fused=True))


def test_plan_precheck_flags_unbuilt_entries():
    """The load-time pre-check must flag a plan whose reachable entries are not built, and treat
    a missing fused gate+up entry as a fallback (two launches) rather than a missing kernel."""
    from sglang.srt.layers.prism.kernels import unsupported_plan_kernels

    kw = dict(moe=True, has_warm=True, cold_on_gpu=True)
    assert unsupported_plan_kernels("gemv_worklist", sparse_score="k2wl2", **kw) == ([], [])
    required, _ = unsupported_plan_kernels("gemv_worklist_mxfp4", sparse_score="k1", **kw)
    assert len(required) == 1 and "gemv_mxfp4_indexed_sparsek1" in required[0]
    required, optional = unsupported_plan_kernels("gemv_worklist_fp8", sparse_score="k1w1", **kw)
    assert required == [] and len(optional) == 1 and "sparsek1w_gateup" in optional[0]
    required, _ = unsupported_plan_kernels("torch_bmm", sparse_score="k1", **kw)
    assert len(required) == 1 and "pair score" in required[0]


def test_launch_tuning_configs_decode():
    """Per-GPU tuning files must parse, and a misspelled knob must be rejected rather than
    silently ignored (the GPU would then run with the default)."""
    import msgspec

    from sglang.kernels.ops.moe.prism import tuning

    for path in sorted(tuning._CONFIG_DIR.glob("*.json")):
        msgspec.json.decode(path.read_bytes(), type=tuning.LaunchTuning)
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(b'{"warm_max_block": 64}', type=tuning.LaunchTuning)
