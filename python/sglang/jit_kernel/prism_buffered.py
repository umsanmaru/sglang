"""Bounded bf16 packed-weight prefill staging.

One arena contains exactly top_k expert slots. Projections and expert batches
reuse it on the caller's stream, after the preceding GEMM has consumed it.
The GEMM uses fixed shared-memory tiles and needs no W-resident fp32 scratch.
"""

from dataclasses import replace

import torch


def grouped_packed_buffered(x2d, grouping, projections, out3d,
                            x_row_is_pair, stream):
    from sglang.jit_kernel.prism_grouped import grouped_gemm_cold
    from sglang.srt.layers.moe.prism.grouping import Grouping

    slots = out3d.shape[1]
    if slots < 1:
        raise ValueError("buffered prefill requires top_k >= 1")
    width = 0
    for slab, _ in projections:
        if slab.slab.dtype != torch.bfloat16 or not slab.expert_offsets:
            raise ValueError("buffered prefill requires bf16 packed expert offsets")
        width = max(width, max(b - a for a, b in
                               zip(slab.expert_offsets, slab.expert_offsets[1:])))
    # Keep each slot aligned for the packed kernel's vector loads.
    width = max(8, (width + 7) // 8 * 8)
    with torch.cuda.stream(stream):
        arena = torch.empty(slots * width, dtype=torch.bfloat16, device=out3d.device)
        for slab, col in projections:
            experts = len(slab.expert_offsets) - 1
            for start in range(0, experts, slots):
                end = min(start + slots, experts)
                # Adjacent experts already occupy a contiguous host slab. One
                # DMA per batch avoids top_k individual copy launches; compact
                # offsets still fit the same top_k-expert arena capacity.
                a, b = slab.expert_offsets[start], slab.expert_offsets[end]
                arena[:b - a].copy_(slab.slab[a:b], non_blocking=True)
                batch = Grouping(
                    grouping.pair_sorted, grouping.pair_off[start:end + 1],
                    grouping.tile_off[start:end + 1] - grouping.tile_off[start],
                    grouping.tile_m)
                staged = replace(slab, slab=arena,
                                 blk_off=slab.blk_off[start:end] - a,
                                 row_off=slab.row_off[start:end + 1],
                                 expert_offsets=())
                grouped_gemm_cold(x2d, batch, staged, out3d, col,
                                  x_row_is_pair, stream, wres_k_max=0)
