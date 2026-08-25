"""Prism GPU warm GEMM 커널 테스트 (CUDA 필요).

계약 ⑤: fp32 누산 — fp64 레퍼런스 대비 fp32 오차 이내여야 하고,
전역 TF32 설정에 영향받지 않아야 한다.
"""

import pytest
import torch

from sglang.srt.layers.moe.prism.kernels import (
    KernelError,
    known_cpu_kernels,
    known_gpu_kernels,
    resolve_cpu_kernel,
    resolve_gpu_kernel,
)

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def reference(x_full, w_stack, k_offset):
    """fp64 CPU 레퍼런스 (동일 bf16 입력에서 출발)."""
    k_rows = w_stack.shape[1]
    xs = x_full[:, k_offset : k_offset + k_rows].cpu().double()
    ws = w_stack.cpu().double()
    return torch.einsum("mk,ekn->emn", xs, ws)


@cuda_required
@pytest.mark.parametrize("m,k_full,k_rows,k_offset,n,e", [
    (1, 256, 64, 0, 128, 8),      # decode, 밴드가 선두
    (4, 256, 64, 64, 96, 5),      # 밴드가 중간
    (7, 512, 128, 384, 256, 3),   # 밴드가 꼬리
])
def test_torch_bmm_matches_fp64_reference(m, k_full, k_rows, k_offset, n, e):
    torch.manual_seed(0)
    x = torch.randn(m, k_full, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(e, k_rows, n, dtype=torch.bfloat16, device="cuda")
    kernel = resolve_gpu_kernel("torch_bmm")
    out = kernel(x, w, k_offset)
    # 계약 ⑤: 출력은 bf16, 오차는 출력 재료화 1회 라운딩 수준이어야 함
    # (내부 누산이 bf16이었다면 K에 비례해 오차가 커져 이 tol을 벗어난다)
    assert out.dtype == torch.bfloat16 and out.shape == (e, m, n)
    ref = reference(x, w, k_offset)
    torch.testing.assert_close(out.cpu().double(), ref, rtol=1.6e-2, atol=1e-2)


@cuda_required
def test_reduced_precision_flag_guarded_and_restored():
    torch.manual_seed(1)
    x = torch.randn(2, 512, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(4, 256, 128, dtype=torch.bfloat16, device="cuda")
    kernel = resolve_gpu_kernel("torch_bmm")
    flags = torch.backends.cuda.matmul
    prev = flags.allow_bf16_reduced_precision_reduction
    try:
        flags.allow_bf16_reduced_precision_reduction = True  # 전역이 켜져 있어도
        out = kernel(x, w, 128)
        assert flags.allow_bf16_reduced_precision_reduction is True  # 복원 확인
    finally:
        flags.allow_bf16_reduced_precision_reduction = prev
    ref = reference(x, w, 128)
    torch.testing.assert_close(out.cpu().double(), ref, rtol=1.6e-2, atol=1e-2)


@cuda_required
def test_out_of_range_band_rejected():
    x = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(2, 64, 32, dtype=torch.bfloat16, device="cuda")
    kernel = resolve_gpu_kernel("torch_bmm")
    with pytest.raises(KernelError, match="out of K_full"):
        kernel(x, w, 96)


def test_registry_and_resolution():
    assert "torch_bmm" in known_gpu_kernels()
    assert "kt_amx_bf16" in known_cpu_kernels()
    assert resolve_cpu_kernel("kt_amx_bf16") == "kt_amx_bf16"
    with pytest.raises(KernelError, match="unknown gpu_warm"):
        resolve_gpu_kernel("nope")
    with pytest.raises(KernelError, match="unknown cpu_cold"):
        resolve_cpu_kernel("nope")


def test_worklist_kernel_key_registered():
    from sglang.srt.layers.moe.prism.kernels import (
        known_gpu_kernels, resolve_gpu_kernel, resolve_worklist_kernels,
    )
    assert "gemv_worklist" in known_gpu_kernels()
    assert resolve_worklist_kernels("torch_bmm") is None
    fns = resolve_worklist_kernels("gemv_worklist")
    assert fns is not None and len(fns) == 2
    # prefill 폴백: worklist plan도 bmm형 커널을 반환해야 한다 (Dedup 경로용)
    assert resolve_gpu_kernel("gemv_worklist") is resolve_gpu_kernel("torch_bmm")


def test_worklist_kernel_unknown_key_raises():
    import pytest as _pytest
    from sglang.srt.layers.moe.prism.kernels import KernelError, resolve_worklist_kernels
    with _pytest.raises(KernelError):
        resolve_worklist_kernels("nope")
