"""Prism GPU 커널 registry와 resolve.

계약 ① (CONTRACTS.md): 커널 이름은 startup에 구현체로 resolve되고, 이후
런타임에 문자열/enum 분기는 존재하지 않는다. Plan의 `kernels.gpu_warm`이
여기의 registry 키이고, `kernels.cpu_cold`는 kt-kernel CRTP 클래스의 키다
(실체 선택은 cold backend가 kt-kernel 쪽에서 수행 — 이 모듈은 이름 검증만).

warm GEMM 커널 계약 (계약 ④ run_warm의 계산 코어):

    fn(x_full, w_stack, k_offset) -> Tensor

    x_full  : bf16 [M, K_full] (device) — full-width activation.
              커널이 자기 밴드 [k_offset, k_offset + k_rows) 만 읽는다.
    w_stack : bf16 [E, k_rows, N] (device) — arena에 상주하는 GEMM-ready
              warm 밴드 (선택된 expert 순서로 적층).
    k_offset: warm 밴드 시작 row (P0: proj당 단일 warm 밴드).
    반환    : bf16 [E, M, N] partial. 계약 ⑤: 내부 누산은 fp32
              (bf16 split-K 환원 금지), 출력 재료화만 bf16 —
              rejoin이 upcast해서 fp32로 합산한다.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

# warm GEMM 커널의 시그니처 타입 (클래스 없는 커널 인터페이스 — 구현이
# 이 꼴만 맞추면 registry에 꽂힌다). 인자 순서대로:
#   x_full   : torch.Tensor — bf16 [M, K_full] (device), full-width activation
#   w_stack  : torch.Tensor — bf16 [E, k_rows, N] (device), arena의 warm 밴드
#   k_offset : int          — warm 밴드 시작 row (x에서 읽을 구간의 시작)
#   반환     : torch.Tensor — bf16 [E, M, N] partial (내부 누산은 fp32 — 계약 ⑤)
# (Callable은 shape/dtype까지 표현 못 하므로 상세 계약은 모듈 docstring,
#  런타임 검증은 각 구현의 가드가 담당)
WarmGemmFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


class KernelError(ValueError):
    """커널 resolve 실패 또는 커널 입력 계약 위반."""


def _warm_gemm_torch_bmm(
    x_full: torch.Tensor, w_stack: torch.Tensor, k_offset: int
) -> torch.Tensor:
    if x_full.dim() != 2 or w_stack.dim() != 3:
        raise KernelError(
            f"expected x [M, K_full] and w [E, k_rows, N], "
            f"got {tuple(x_full.shape)} / {tuple(w_stack.shape)}"
        )
    num_experts, k_rows, _ = w_stack.shape
    if k_offset < 0 or k_offset + k_rows > x_full.shape[1]:
        raise KernelError(
            f"warm band [{k_offset}, {k_offset + k_rows}) out of K_full="
            f"{x_full.shape[1]}"
        )

    x_slice = x_full[:, k_offset : k_offset + k_rows]
    # 계약 ⑤: bf16 in/out에 내부 누산 fp32 (cuBLAS compute type 32F).
    # 단 split-K 환원을 bf16으로 허용하는 전역 플래그가 켜져 있어도 이
    # GEMM의 누산은 fp32여야 하므로 저장/복원한다.
    matmul_flags = torch.backends.cuda.matmul
    prev_reduced = matmul_flags.allow_bf16_reduced_precision_reduction
    matmul_flags.allow_bf16_reduced_precision_reduction = False
    try:
        out = torch.bmm(
            x_slice.unsqueeze(0).expand(num_experts, -1, -1), w_stack
        )
    finally:
        matmul_flags.allow_bf16_reduced_precision_reduction = prev_reduced
    return out


_GPU_WARM_KERNELS: dict[str, WarmGemmFn] = {
    "torch_bmm": _warm_gemm_torch_bmm,
}

# cpu_cold의 실체는 kt-kernel의 CRTP 클래스. 여기서는 유효한 키 목록만 안다.
# ("kt_amx_bf16" ↔ kt-kernel bf16-moe.hpp 계열; sparse 커널이 생기면 여기 추가)
_CPU_COLD_KERNELS: Tuple[str, ...] = ("kt_amx_bf16",)


def known_gpu_kernels() -> Tuple[str, ...]:
    return tuple(_GPU_WARM_KERNELS)


def known_cpu_kernels() -> Tuple[str, ...]:
    return _CPU_COLD_KERNELS


def resolve_gpu_kernel(name: str) -> WarmGemmFn:
    """startup 1회 호출. 이후에는 반환된 callable만 존재한다."""
    try:
        return _GPU_WARM_KERNELS[name]
    except KeyError:
        raise KernelError(
            f"unknown gpu_warm kernel '{name}' (known: {sorted(_GPU_WARM_KERNELS)})"
        ) from None


def resolve_cpu_kernel(name: str) -> str:
    """이름을 검증하고 kt-kernel 클래스 키를 반환한다. 인스턴스화는 cold
    backend의 몫 (kt-kernel이 Plan 어휘를 모르게 하는 번역 경계)."""
    if name not in _CPU_COLD_KERNELS:
        raise KernelError(
            f"unknown cpu_cold kernel '{name}' (known: {sorted(_CPU_COLD_KERNELS)})"
        )
    return name


# ── worklist GEMV (decode 전용, spec 2026-08-25) ─────────────────────────
# plan `kernels.gpu_warm: "gemv_worklist"` 선택 시 decode(M ≤ 임계치)는
# worklist 경로(gather/bmm/scatter 우회), prefill은 아래 폴백 bmm을 그대로
# 탄다. 래퍼 시그니처는 jit_kernel/prism_gemv.py docstring이 정본.
def _worklist_fns():
    from sglang.jit_kernel.prism_gemv import (  # 지연 import — CPU-only 환경 보호
        gemv_worklist, gemv_worklist_pinned,
    )
    return (gemv_worklist, gemv_worklist_pinned)


_GPU_WARM_KERNELS["gemv_worklist"] = _warm_gemm_torch_bmm  # prefill/Dedup 폴백

_GPU_WORKLIST_KERNELS: dict[str, Callable[[], tuple]] = {
    "gemv_worklist": _worklist_fns,
}


def resolve_worklist_kernels(name: str) -> Optional[tuple]:
    """worklist 커널 쌍 (device_fn, pinned_fn) 또는 None (bmm 전용 키).

    KernelError는 아예 모르는 키일 때만 — 유효하지만 worklist가 아닌 키
    ("torch_bmm")는 None이다 (호출자가 경로를 가른다)."""
    if name in _GPU_WORKLIST_KERNELS:
        return _GPU_WORKLIST_KERNELS[name]()
    if name in _GPU_WARM_KERNELS:
        return None
    raise KernelError(
        f"unknown gpu_warm kernel '{name}' (known: {sorted(_GPU_WARM_KERNELS)})"
    )
