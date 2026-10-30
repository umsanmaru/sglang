# BF16 prefill in the Qwen runtime

Implementation: `sglang.jit_kernel.prism_prefill`, dispatched by
`prism_grouped`. This change is in **sglang-qwen**, the tree used by the measured
Qwen lane. No checkpoint loading is needed for its microbenchmarks.

## Execution and memory

The eager BF16 path builds one CPU schedule from the existing device Grouping
(offsets and hybrid masks). The schedule is shared across projections/tier
calls for that forward. It sorts active experts by token count to reduce BMM
padding. Inactive or CPU-masked experts are never staged.

One device arena holds exactly `top_k` expert slots. Its element capacity is
`top_k * max(max(32, K_max) * N)` over the projections in a call. For the measured
K=1952, N=320, top-k=10 projection it holds 6,246,400 BF16 elements (12,492,800
bytes). GU projections reuse the arena. This bound describes **weight staging**;
inputs, outputs and activation workspaces are separate.

A Triton staging kernel reads registered host storage in physical byte order
and writes row-major KxN device weights. It decodes either legacy KT tiles or
BufferBNC tiles during this copy. There is no full device weight bank, second
packed device buffer, or extra host weight copy. Gathering happens once per
expert/row chunk, using that expert's own K indices. cuBLAS BMM consumes the
contiguous inputs and weights, and scatter writes the original pair positions.
K differs across experts and GU projections; padding is zero-filled. Activation
chunks have at most 1024 rows per slot, independent of prompt length.

Optional overlap splits the **same** arena into ceil(top-k/2) and floor(top-k/2)
slots. CUDA events prevent overwriting a bank still consumed by GEMM. Sequential
staging is the default because 5+5 overlap did not improve the measured cases.
The caller stream owns temporary allocations and joins every staging write
before reuse/release. CUDA graph capture keeps the previous GPU kernels.

## Default policy

- Eligible BF16 plans automatically prepare warm KT storage when cold GPU views
  are enabled and all three projections have some warm rows. Explicit WARM_KT
  settings take precedence. This replaces the previous row-major warm store.
- Eager grouped prefill below M=1536 computes warm on CPU. Decode keeps the GPU
  GEMV path. Existing stream/graph restrictions of the CPU path still apply.
- At M>=1536 warm and cold compute on GPU. Buffered BMM is selected when the
  largest active expert has at least 256 routed pairs. If every expert has fewer
  pairs, the existing UVA Wres kernel remains faster in measurements and is
  retained. The 256 threshold is an empirical heuristic, not a universal
  crossover claim.

The installed TileK2 BF16 MoE producer uses N-contiguous tile order. ColdSlab now
records this explicitly, and both grouped and packed GEMV (dense/sparse) loaders
interpret it correctly, including paths that do not use buffering.

For an expert with warm rows on only one GU side, the absent side is represented
by a zero weight tile. The KT adapter exposes a zero-valued dummy pair to satisfy
its `padding < one tile` validation; calibration and shard real_rows remain zero.
No nonzero weight or contribution is introduced. This handles the warm-CPU
startup failure observed in the nonuniform Qwen plan without rebuilding KT.

## Controls

| Environment variable | Default | Meaning |
|---|---|---|
| `SGLANG_PRISM_BUFFERED_PREFILL` | `auto` | `0`: old GPU path; `1`: force new path above minimum M; `auto`: apply expert-row heuristic |
| `SGLANG_PRISM_BUFFERED_MIN_M` | `1536` | Minimum M for buffered GPU dispatch |
| `SGLANG_PRISM_BUFFERED_OVERLAP` | `0` | `1`: split the same top-k slots into two banks |
| `SGLANG_PRISM_BUFFERED_STAGE_BLOCKS` | `0` | Optional staging CTA cap; 0 is unlimited. GEMM itself is not capped. |
| `SGLANG_PRISM_WARM_KT` | automatic | `0` retains row-major warm GPU storage; `1` requires warm KT storage |
| `SGLANG_PRISM_WARM_CPU_MAX_M` | `1536` | Exclusive upper bound for eager grouped warm CPU; 0 disables it |

The existing explicit `WARM_CPU_MIN_M` setting preserves its legacy behavior and
suppresses the default maximum. Cold GPU switching still honors
`SGLANG_PRISM_COLD_GPU_MIN_M`.

## Validation and limitations

`test_buffered_bmm.py` covers shuffled expert-specific K, variable/zero K, both KT
layouts, tail batches, top-k=1, odd top-k overlap, row chunking, projection reuse,
hybrid masks, caller side streams, dense/sparse packed runtime, and the warm
CPU/GPU boundary with one-sided empty GU. The relevant suite passed 97 tests;
6 pre-existing cold_async tests are explicitly disabled for their ordering bug.

Microbenchmarks include actual host weight reads, decode-to-row-major, gather,
BMM and scatter. Fresh-schedule results also include the one D2H scheduling
step. Grouping construction is shared by both paths and outside these timings.
Timings are after JIT/warmup. First-use Triton compilation is not represented.
The benchmarks use synthetic routing/weights; no full-model TTFT is claimed.

From the workspace root, run with the prism-glm Python environment:

```bash
PYTHONPATH=sglang-qwen/python python scripts/bench_buffered_bmm.py \
  --m 4148 --active 64 --output results/prefill-diagnosis/buffered_bmm_final_m4148.json
PYTHONPATH=sglang-qwen/python python scripts/bench_buffered_bmm.py \
  --m 8244 --active 512 --hot-active 64 \
  --output results/prefill-diagnosis/buffered_bmm_all512_hot64.json
```

Bind the process/memory to the GPU-local NUMA node for comparable PCIe timings.
The benchmark explicitly calls each implementation and disables auto dispatch
for the Wres baseline; the new implementation is measured even where `auto`
would retain Wres. Full measurements and rejected tuning variants are in
`results/prefill-diagnosis/` at the workspace root.
