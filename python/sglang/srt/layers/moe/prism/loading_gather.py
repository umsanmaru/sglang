"""CPU expert-column gather with bounded temporary storage.

The expanded gather index is a zero-stride view, not an E*N*K int64 allocation.
Expert output blocks retain their individual padded K and original order.
"""
import torch


def _gather_padded_columns(src, ti, real_rows, tile, *, pad_value=0, div=1,
                          minimum_rows=None, workspace_bytes=64 << 20):
    e_count, n, _ = src.shape
    real = [int(r) for r in real_rows]
    pad = [((r + tile - 1) // tile) * tile for r in real]
    if minimum_rows is not None:
        pad = [max(k, int(lo)) for k, lo in zip(pad, minimum_rows)]
    if len(real) != e_count or len(pad) != e_count:
        raise ValueError('one real/padded K count is required per expert')
    if any(r % div or k % div for r, k in zip(real, pad)):
        raise ValueError('K counts must be divisible by the storage divisor')
    offsets = [0]
    for k in pad:
        offsets.append(offsets[-1] + k)
    off = torch.tensor(offsets, dtype=torch.int32)
    flat = torch.empty(offsets[-1] // div * n, dtype=src.dtype)
    idx = torch.zeros(offsets[-1], dtype=torch.int64)
    source_offsets = ti.row_off.tolist()
    index = ti.idx.to(torch.int64)
    start = 0
    while start < e_count:
        end, max_k = start, 0
        # At least one expert; an exceptionally wide expert is the irreducible
        # minimum. Bound result + index + mask, excluding the final output.
        while end < e_count:
            next_k = max(max_k, pad[end] // div)
            per_expert = next_k * (n * src.element_size() + 9)
            if end > start and (end - start + 1) * per_expert > workspace_bytes:
                break
            max_k = next_k
            end += 1
        if max_k:
            indices = torch.zeros(end - start, max_k, dtype=torch.int64)
            lengths = []
            for e in range(start, end):
                r = real[e] // div
                indices[e - start, :r] = index[source_offsets[e]:source_offsets[e] + r]
                lengths.append(r)
                if div == 1:
                    idx[offsets[e]:offsets[e] + real[e]] = indices[e - start, :r]
            expanded = indices[:, None, :].expand(-1, n, -1)
            uniform = all(pad[e] // div == max_k for e in range(start, end))
            if uniform:
                dest = flat[offsets[start] // div * n:offsets[end] // div * n].view(end - start, n, max_k)
                torch.gather(src[start:end], 2, expanded, out=dest)
            else:
                dest = torch.gather(src[start:end], 2, expanded)
            if any(r != max_k for r in lengths):
                mask = torch.arange(max_k)[None, :] >= torch.tensor(lengths)[:, None]
                dest.masked_fill_(mask[:, None, :], pad_value)
            if not uniform:
                for e in range(start, end):
                    k = pad[e] // div
                    flat[offsets[e] // div * n:offsets[e + 1] // div * n].view(n, k).copy_(dest[e - start, :, :k])
        start = end
    return flat, off, idx, torch.tensor(real, dtype=torch.int32)


def gather_padded_columns(src, ti, real_rows, tile, *, pad_value=0, div=1,
                          minimum_rows=None, workspace_bytes=64 << 20):
    from sglang.srt.layers.moe.prism.loading_cache import cached_columns

    real_rows = tuple(int(r) for r in real_rows)
    minimum_rows = None if minimum_rows is None else tuple(int(r) for r in minimum_rows)
    return cached_columns(src, ti, real_rows, tile, (pad_value, div, minimum_rows),
                          lambda: _gather_padded_columns(
                              src, ti, real_rows, tile, pad_value=pad_value, div=div,
                              minimum_rows=minimum_rows, workspace_bytes=workspace_bytes))
