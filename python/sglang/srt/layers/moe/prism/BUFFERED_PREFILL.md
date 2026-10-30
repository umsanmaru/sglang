# Bounded bf16 prefill (2026-09-09)

The bf16 runtime defaults to packed warm weights (`SGLANG_PRISM_WARM_KT=1`).
Warm prefill runs on CPU for `1 < M < 1536`, including stream and graph execution.
At `M >= 1536`, warm and cold packed weights use GPU staging and grouped GEMM.
The existing decode path at `M=1` remains in place. Explicit cold-GPU thresholds,
warm-KT opt-out, and the legacy warm-CPU minimum override retain their meanings.

`prism_buffered.grouped_packed_buffered` allocates one arena with capacity
`top_k * aligned(max_expert_projection_elements)` bf16 elements. Each batch
contains at most top-k consecutive experts; their contiguous host range is copied
with one DMA. Gate and up reuse the same arena, as do subsequent batches. Copies
and GEMMs run on the caller's stream, so a slot is overwritten only after its
consumer finishes. There is no second weight buffer and no copy/compute overlap
within this stream.

The packed GEMM reads the device arena using its original layout. Grouping retains
global pair coordinates, but tile offsets and weight offsets are rebased for each
batch. Variable K, empty experts, expert masks and partial final batches are
supported. Shared memory stays at the ordinary fixed GEMM tile size (about 35 KiB),
and the Wres `[M * top_k, output_width]` fp32 scratch is not allocated. Normal
activation/output tensors and small grouping metadata still scale with token count.

This implementation targets **bf16 packed stores**. FP8/MXFP4 dispatch is unchanged.
Warm-KT retains its existing requirement for warm rows on gate, up and down;
plans with warm on only some projections must keep `SGLANG_PRISM_WARM_KT=0`.
Staging currently copies inactive experts too; it avoids a routing D2H sync.

## Validation

CUDA tests compare staged packed GEMM against independently packed indexed weights
with exact integer inputs, variable/zero K, skewed routing, E not divisible by
top-k, gate/up reuse, pair-indexed down input, and graph replay. An allocation
check verifies that fused gate/up requests only one top-k-sized weight arena.
Executor tests check the CPU/GPU boundary at M=1, 7, 1535 and 1536 in eager and
stream mode, plus small-prefill CPU graph replay.

Synthetic benchmark on RTX 5090, bf16, E=32, top-k=8, K=N=1024, one projection:

| M | Wres time | Buffered time | Wres temporary GPU memory | Buffered temporary GPU memory |
|---|---:|---:|---:|---:|
| 1536 | 1.935 ms | 2.418 ms | 48 MiB | 16.002 MiB |
| 3072 | 3.253 ms | 3.129 ms | 96 MiB | 16.002 MiB |

Outputs matched exactly in both benchmark cases. These are five-iteration CUDA
event averages after two warmups, not full-model TTFT results. Staging bounds
memory but does not improve latency at every M.

Reproduce with the repository Python package on `PYTHONPATH` and a CUDA/kt-capable
environment:

```bash
python -m pytest -q test/prism/test_buffered_prefill.py test/prism/test_grouped_gemm.py test/prism/test_executor.py
python test/prism/bench_buffered_prefill.py
```
