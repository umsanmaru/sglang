from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_prism_gather_module() -> Module:
    return load_jit(
        "prism_gather",
        cuda_files=["moe/prism_gather.cuh"],
        cuda_wrappers=[("gather_bands_from_pinned", "gather_bands_from_pinned")],
    )


def gather_bands_from_pinned(
    pinned_src: torch.Tensor,
    sel_device: torch.Tensor,
    dst: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    """dst[g] = pinned_src[sel_device[g]], gathered directly from pinned host
    memory over UVA using a 16-byte (uint4) vectorized kernel.

    Ported from planir's `WarmBandGatherPinnedKernel` (kernels.cu:114).
    `sel_device` is an int32 CUDA tensor -- because the index lives on the
    device, a CUDA-graph capture of the call site needs no per-token host
    repatch; the graph simply replays whichever indices `sel_device` holds
    at replay time.

    Parameters
    ----------
    pinned_src : Tensor[E, rows, N], bf16, CPU, pinned (`.pin_memory()`)
        Warm expert-weight bands in a pinned host store, read by the GPU
        directly over UVA.
    sel_device : Tensor[g], int32, CUDA
        Per-gathered-slot expert index into `pinned_src`'s first dimension.
    dst : Tensor[g, rows, N], bf16, CUDA
        Destination arena slots. `g == sel_device.numel() == dst.shape[0]`.
    stream : torch.cuda.Stream
        Stream the gather is launched on.

        Stream-handling convention (binding for later tasks): the underlying
        JIT kernel has no explicit stream parameter of its own -- like the
        rest of this JIT module (see `cuda_wait_value.py`), it launches on
        whatever CUDA stream is "current" at call time. This wrapper
        enforces the caller's requested `stream` by entering
        `with torch.cuda.stream(stream):` around the launch. A later
        CUDA-graph-compatible stager built on top of this kernel must invoke
        it the same way (or already be inside a `torch.cuda.stream(...)`
        context when it calls in) -- there is no way to pass a raw stream
        handle through to the kernel itself.

    Requires `rows * N * 2 % 16 == 0` (bf16 is 2 bytes/elem); this is
    enforced on the C++ side with a clear error.
    """
    module = _jit_prism_gather_module()
    with torch.cuda.stream(stream):
        module.gather_bands_from_pinned(pinned_src, sel_device, dst)
