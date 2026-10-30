"""Synthetic bf16 Wres versus bounded staging benchmark; run from any directory."""

import json
from dataclasses import replace
import torch
from test_buffered_prefill import _packed_store
from sglang.jit_kernel.prism_grouped import grouped_gemm_cold
from sglang.srt.layers.moe.prism.grouping import build_grouping

torch.set_num_threads(4)
slab, dense = _packed_store([1024] * 32, n=1024)
del dense
legacy = replace(slab, expert_offsets=())
stream = torch.cuda.current_stream()
results = []
for m in [1536, 3072]:
    ids = torch.randint(0, 32, (m, 8), device='cuda')
    grouping = build_grouping(ids, 32)
    x = torch.randn(m, 1024, dtype=torch.bfloat16, device='cuda')
    out = torch.empty(m, 8, 1024, dtype=torch.bfloat16, device='cuda')
    ref = torch.empty_like(out)
    for name, store, target in [('wres', legacy, ref), ('buffered', slab, out)]:
        def run():
            grouped_gemm_cold(x, grouping, store, target, 0, False, stream, 128, 1024)
        for _ in range(2): run()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5): run()
        b.record()
        b.synchronize()
        results.append(dict(m=m, path=name, ms=a.elapsed_time(b)/5,
                            temporary_MiB=(torch.cuda.max_memory_allocated()-baseline)/2**20))
    results[-1]['max_abs_diff'] = (out.float()-ref.float()).abs().max().item()
print(json.dumps(results, indent=2))
