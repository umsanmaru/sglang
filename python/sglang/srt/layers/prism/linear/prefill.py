"""Bounded H2D staging for dense BF16 prefill of host-resident projections."""
from __future__ import annotations

import torch


class DenseGpuPrefill:
    """One scratch buffer per executor; calls are serialized across CUDA streams.

    Calls originate from one scheduler thread. Weight copies stay pinned on the CPU. Decode never enters this path. Prefill
    CUDA graph capture is unsupported because the scratch is shared across calls.
    """

    def __init__(self, device: torch.device, buffer_bytes: int):
        if buffer_bytes < 2 or buffer_bytes % 2:
            raise ValueError("prefill buffer_bytes must be a positive multiple of two")
        self.device = torch.device(device)
        self.buffer = torch.empty(buffer_bytes // 2, dtype=torch.bfloat16, device=device)
        self.done = torch.cuda.Event()
        self.used = False

    def retain(self, weight: torch.Tensor) -> torch.Tensor:
        from sglang.srt.layers.prism.numa import alloc_pinned_on_node, gpu_numa_node

        if weight.device.type != "cpu" or weight.dtype != torch.bfloat16 or weight.ndim != 2:
            raise ValueError("GPU prefill currently requires CPU BF16 [N,K] weights")
        if weight.shape[1] > self.buffer.numel():
            raise ValueError("prefill buffer must hold at least one weight row")
        host = alloc_pinned_on_node(weight.shape, weight.dtype, gpu_numa_node(self.device),
                                    "linear GPU prefill weight")
        host.copy_(weight)
        return host

    def run(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("staged GPU prefill requires eager prefill (disable prefill graphs)")
        if x.dtype != torch.bfloat16 or x.device != self.buffer.device:
            raise ValueError("GPU prefill input must be BF16 on the staging device")
        n, k = weight.shape
        if x.ndim != 2 or x.shape[1] != k:
            raise ValueError("GPU prefill input K does not match weight")
        stream = torch.cuda.current_stream(x.device)
        self.buffer.record_stream(stream)
        if self.used:
            stream.wait_event(self.done)
        output = torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)
        rows = self.buffer.numel() // k
        if rows >= 32:
            rows = rows // 32 * 32
        # Narrow output panels can select cuBLAS algorithms that truncate the
        # running sum to BF16. Keep FP32 accumulation, as in the CPU path.
        reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        try:
            for start in range(0, n, rows):
                end = min(start + rows, n)
                panel = self.buffer[:(end - start) * k].view(end - start, k)
                panel.copy_(weight[start:end], non_blocking=True)
                torch.mm(x, panel.t(), out=output[:, start:end])
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
            self.done.record(stream)
            self.used = True
        return output
