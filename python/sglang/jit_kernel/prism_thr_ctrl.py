"""Launcher for the online threshold controller step (`csrc/moe/prism_thr_ctrl.cuh`).

The driver and the buffer contract live in `srt/layers/moe/prism/thr_ctrl.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_prism_thr_ctrl_module() -> Module:
    return load_jit(
        "prism_thr_ctrl",
        cuda_files=["moe/prism_thr_ctrl.cuh"],
        cuda_wrappers=[("thr_ctrl_step", "thr_ctrl_step")],
    )


def warmup_jit() -> None:
    _jit_prism_thr_ctrl_module()


def thr_ctrl_step(
    *,
    live_dev: torch.Tensor,
    calib_dev: torch.Tensor,
    live_host: torch.Tensor,
    gpu_ctr: torch.Tensor,
    cpu_ctr: torch.Tensor,
    snap: torch.Tensor,
    tokens: torch.Tensor,
    alpha: torch.Tensor,
    log: torch.Tensor,
    windows_host: torch.Tensor,
    window: int,
    mu: float,
    alpha_min: float,
    alpha_max: float,
    targets: Sequence[float],
    units_per_token: Sequence[float],
) -> None:
    """Launches one controller step for one layer on the current stream (no host sync)."""
    _jit_prism_thr_ctrl_module().thr_ctrl_step(
        live_dev, calib_dev, live_host, gpu_ctr, cpu_ctr, snap, tokens, alpha, log, windows_host,
        int(window), float(mu), float(alpha_min), float(alpha_max),
        *(float(t) for t in targets), *(float(u) for u in units_per_token),
    )
