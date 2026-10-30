# Loading optimization

The Qwen runtime now uses bounded batched CPU gathering and persistent plan/index
caches. This complements the prefill GPU work; it changes startup preparation.

## Weight preparation

`loading_gather.gather_padded_columns` replaces the expert-by-expert
`index_select` column loop in cold formats and warm KT preparation. It gathers
contiguous expert batches with expanded, zero-stride index views. It preserves
individual K padding and original expert order; variable-size results are copied
into their final flat offsets. Uniform-size batches gather directly into the
final output. Zero-K experts, dummy zero GU tiles, BF16, packed uint8 and scale
storage divisors are supported.

Temporary result/index/mask batches are bounded to 64 MiB, or one expert when a
single expert exceeds that size. The final weight storage is separate. No
E*N*K int64 index tensor is materialized. The existing contiguous-band shortcut
remains, with an additional check that all expert band starts really match.

A synthetic E=512, N=640, K=2560 BF16 projection with variable, padded K and
expert-specific indices measured 3.587s for the previous loop and 0.962s for the
new path, bitwise equal (16 CPU threads, NUMA-local allocation). This is an
isolated projection benchmark, not an end-to-end startup prediction.

## Plan and index cache: enabled by default

- `SGLANG_PRISM_LOAD_CACHE=0` disables plan/index caching.
- `SGLANG_PRISM_LOAD_CACHE_DIR` overrides the root, which otherwise is
  `$XDG_CACHE_HOME/prism/loading` or `~/.cache/prism/loading`.
- Source plan SHA-256 and a cache format revision identify entries. Repeated
  BandSpec objects are interned before serializing the parsed plan. Cache reads
  use a restricted class whitelist, not unrestricted pickle loading.
- Per-layer index tensors are serialized separately. Calibration is still
  loaded and verified, and plan/index validation is retained on cache hits.
- Writes are atomic. Missing, unreadable or structurally corrupt entries rebuild
  from their source. A changed source plan naturally uses a different cache key.

For the actual 406 MiB Qwen block-2 plan, initial parse/intern/cache creation took
145.58s; cached loading took 1.42s. Validation separately took 3.62/3.93s. Layer-0
index creation took 0.693s versus 0.106s on a hit. The parsed plan cache is about
121 MiB. All 48 layer indices have also been prepared in this workspace's default
cache; total plan/index disk use is about 392 MiB. No checkpoint was loaded.

`method.py` logs plan loading, calibration loading and metadata validation times
separately, in addition to the existing per-layer preparation/KT loading times.

## Prepared column cache: optional

Set `SGLANG_PRISM_WEIGHT_CACHE_DIR` to opt in. Set
`SGLANG_PRISM_WEIGHT_CACHE_MAX_GB` for its disk budget (GiB, default 4; <=0 disables
it). The cache stores gathered cold/warm-KT columns and their metadata before
AMX packing; it does not bypass checkpoint reading or KT packing.

Keys hash actual source tensor bytes, dtype/shape, K indices/offsets, real K,
padding rules, storage divisor and implementation revision. Thus a changed
checkpoint at the same pathname cannot reuse stale columns. Reads use
`torch.load(weights_only=True, mmap=True)`; writes are atomic, and writer-locked
LRU eviction bounds disk usage. Entries larger than the budget are not written.

This cache is **off by default**: source hashing and disk I/O can cost more than
the optimized gather. The current filesystem has only about 64 GiB free, so a
full prepared model cache is inappropriate here. Plan/index caching and fast
gather work without it. Raising its budget is useful only after measuring on
storage with enough capacity and bandwidth. Temporary files and active mmap
readers may keep underlying disk blocks alive until they are released.

Bump `loading_cache.REVISION` when changing persisted plan/index interpretation
or gathered-weight representation.

## Validation

- 14 cache/gather tests cover variable K, padding, zero K, shuffled indices,
  noncontiguous sources, divisor/dtype variants, cache hits, source/index changes,
  broken entries and disk limits.
- Existing weight, prefill, nonuniform geometry and sparse regression suite:
  70 passed.
- Final loading/weight suite after the last functional edits: 38 passed (overlaps
  the suites above; counts should not be added).

Reproduction scripts at the workspace root:
`bench_loading_optimized.py`, `bench_plan_cache.py`, and
`prepare_loading_index_cache.py` under `scripts/`. Results are under
`results/prefill-diagnosis/`. The plan benchmark uses existing cache entries if
present; its first-run label does not force deletion of a populated cache.
