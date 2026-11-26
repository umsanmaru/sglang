"""BF16 prefill: one top-k weight arena, gather, cuBLAS BMM, scatter.

Registered host weights are read once into row-major device slots. Packed KT
weights are decoded during staging, without another device weight allocation
or a second host copy. This is an eager path: its routing schedule performs one
small device-to-host synchronization, shared by all projections of a Grouping.
"""
from dataclasses import dataclass
from functools import cached_property
import os

import torch
import triton
import triton.language as tl


import contextlib as _ctx
import os as _os

import torch as _torch

# nsys 구간 태그 — executor와 같은 스위치(SGLANG_PRISM_NVTX=1)를 쓴다. 평시 no-op.
_NVTX = _os.environ.get("SGLANG_PRISM_NVTX") == "1"


@_ctx.contextmanager
def _nvtx(msg: str):
    if not _NVTX:
        yield
        return
    _torch.cuda.nvtx.range_push(msg)
    try:
        yield
    finally:
        _torch.cuda.nvtx.range_pop()


def enabled(out, grouping, stream=None):
    with torch.cuda.stream(stream or torch.cuda.current_stream(out.device)):
        mode = os.environ.get("SGLANG_PRISM_BUFFERED_PREFILL", "auto")
        if (out.shape[0] < int(os.environ.get("SGLANG_PRISM_BUFFERED_MIN_M", "1536"))
                or mode == "0" or torch.cuda.is_current_stream_capturing()):
            return False
        # For uniformly tiny expert GEMMs, UVA Wres already hides compute under
        # PCIe reads; staging + BMM adds overhead. Keep that path until an expert
        # has enough rows for the new GEMM path to pay off. "1" forces benchmarking.
        return mode == "1" or max(grouping.prefill_schedule[2], default=0) >= 256


class _MappedHost:
    def __init__(self, host):
        self.owner = host
        self.__cuda_array_interface__ = {
            "shape": (host.numel(),), "strides": None, "typestr": "<u2",
            "data": (host.data_ptr(), False), "version": 3,
        }


def mapped_bf16(w, device):
    if w.dtype != torch.bfloat16 or not w.is_contiguous():
        raise ValueError("prefill weights must be contiguous BF16")
    if w.is_cuda:
        return w.view(-1)
    if not w.is_pinned():
        raise ValueError("prefill weights must be pinned or cudaHostRegister'ed")
    # UVA alias only; torch retains the interface object's owner. No allocation
    # and no copy. In particular, .to(device) here would violate the arena bound.
    return torch.as_tensor(_MappedHost(w), device=device).view(torch.bfloat16)


@dataclass
class Projection:
    source: torch.Tensor
    row_off: torch.Tensor
    kidx: torch.Tensor
    n: int
    layout: str = "rowmajor"
    blk_off: torch.Tensor | None = None
    n_block: int = 256
    k_block: int = 1792
    rows_host: tuple = ()

    @cached_property
    def mapped(self):
        return mapped_bf16(self.source, self.row_off.device)

    @cached_property
    def ks(self):
        ro = self.rows_host or tuple(self.row_off.cpu().tolist())
        return tuple(b - a for a, b in zip(ro, ro[1:]))


@triton.jit
def _stage(W, B, RO, BO, IDS, start, N: tl.constexpr, K,
           LAYOUT: tl.constexpr, NB: tl.constexpr, KB: tl.constexpr,
           BLOCK: tl.constexpr):
    slot = tl.program_id(1)
    e = tl.load(IDS + start + slot)
    o = tl.load(RO + e)
    kr = tl.load(RO + e + 1) - o
    for block in range(tl.program_id(0), tl.cdiv(K * N, BLOCK), tl.num_programs(0)):
        q = block * BLOCK + tl.arange(0, BLOCK)
        if LAYOUT == 0:
            address = o.to(tl.int64) * N + q
            value = tl.load(W + address, (q < K * N) & (q < kr * N), other=0)
            tl.store(B + slot * K * N + q, value, q < K * N)
        else:
            # Read packed bytes in physical order for fully coalesced PCIe loads.
            # Only the device stores transpose; row-major reads would fetch each
            # host cache line in several narrow transactions.
            safe_kr = tl.maximum(kr, 32)
            nb = q // (NB * safe_kr) * NB
            ns = tl.minimum(NB, N - nb)
            safe_ns = tl.maximum(ns, 32)
            rem = q - nb * safe_kr
            kb = rem // (KB * safe_ns) * KB
            ks = tl.maximum(tl.minimum(KB, kr - kb), 32)
            within = rem - kb * safe_ns
            tile = within // 1024
            if LAYOUT == 2:
                kt, nt = tile // (safe_ns // 32) * 32, tile % (safe_ns // 32) * 32
            else:
                nt, kt = tile // (ks // 32) * 32, tile % (ks // 32) * 32
            inner = within % 1024
            n = nb + nt + inner // 512 * 16 + inner % 32 // 2
            k = kb + kt + inner % 512 // 32 * 2 + inner % 2
            value = tl.load(W + tl.load(BO + e) + q, q < kr * N, other=0)
            tl.store(B + slot * K * N + k * N + n, value, q < kr * N)
            tl.store(B + slot * K * N + q, 0, (q < K * N) & (q >= kr * N))



@triton.jit
def _gather(X, A, RO, KI, PS, PO, IDS, start, row_start,
            H: tl.constexpr, TOPK: tl.constexpr, PAIR: tl.constexpr,
            R, K, BLOCK: tl.constexpr):
    slot = tl.program_id(1)
    e = tl.load(IDS + start + slot)
    o = tl.load(RO + e)
    kr = tl.load(RO + e + 1) - o
    p0, p1 = tl.load(PO + e), tl.load(PO + e + 1)
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r, k = q // K, q % K
    valid = (r < R) & (p0 + row_start + r < p1) & (k < kr)
    pair = tl.load(PS + p0 + row_start + r, valid, other=0)
    row = pair if PAIR else pair // TOPK
    col = tl.load(KI + o + k, valid, other=0).to(tl.int32)
    val = tl.load(X + row.to(tl.int64) * H + col, valid, other=0)
    tl.store(A + slot * R * K + q, val, q < R * K)


@triton.jit
def _scatter(C, OUT, PS, PO, IDS, start, row_start,
             R, N: tl.constexpr, WIDTH: tl.constexpr,
             COL: tl.constexpr, BLOCK: tl.constexpr):
    slot = tl.program_id(1)
    e = tl.load(IDS + start + slot)
    p0, p1 = tl.load(PO + e), tl.load(PO + e + 1)
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r, n = q // N, q % N
    valid = (q < R * N) & (p0 + row_start + r < p1)
    pair = tl.load(PS + p0 + row_start + r, valid, other=0)
    value = tl.load(C + slot * R * N + q, valid, other=0)
    tl.store(OUT + pair.to(tl.int64) * WIDTH + COL + n, value, valid)


def grouped_prefill(x, grouping, projections, out, pair_input, stream, *, stats=None):
    """projections = [(Projection, output_column)]. Share exactly top_k slots.

    R is capped independently of M to bound gathered activations. All temporary
    tensors are allocated and consumed on the caller's stream. No global buffer
    or cross-request mutable state is shared.
    """
    with torch.cuda.stream(stream):
        with _nvtx("pf.schedule(D2H on first use)"):
            ids, experts, counts = grouping.prefill_schedule
        if not counts:
            return
        slots = out.shape[1]
        width = max(max(32, max(p.ks, default=0)) * p.n for p, _ in projections)
        # Even all-zero projections need a valid B pointer for BMM.
        width = max(32, width)
        with _nvtx(f"pf.arena_alloc({slots}x{width})"):
            arena = torch.empty(slots * width, dtype=torch.bfloat16, device=out.device)
        if stats is not None:
            stats.update(weight_slots=slots, weight_elements=arena.numel(),
                         max_expert_elements=width)
        for p, col in projections:
            layout = {"rowmajor": 0, "kt_bf16": 1, "kt_bf16_nc": 2}[p.layout]
            overlap = (slots > 1 and not p.source.is_cuda
                       and os.environ.get("SGLANG_PRISM_BUFFERED_OVERLAP", "0") != "0")
            capacities = ((slots + 1) // 2, slots // 2) if overlap else (slots,)
            offsets = (0, capacities[0] * width) if overlap else (0,)
            jobs = []
            start = 0
            while start < len(counts):
                bank = len(jobs) % len(capacities)
                s = min(capacities[bank], len(counts) - start)
                k = max(32, max(p.ks[e] for e in experts[start:start + s]))
                b = arena[offsets[bank]:offsets[bank] + s * k * p.n].view(s, k, p.n)
                jobs.append((start, s, k, b))
                start += s
            copy_stream = torch.cuda.Stream(device=out.device) if overlap else stream
            if overlap:
                copy_stream.wait_stream(stream)
            ready = [torch.cuda.Event() for _ in jobs] if overlap else None

            def stage(i, consumed=None):
                start, s, k, b = jobs[i]
                with torch.cuda.stream(copy_stream):
                    if consumed is not None:
                        copy_stream.wait_event(consumed)
                    cap = int(os.environ.get("SGLANG_PRISM_BUFFERED_STAGE_BLOCKS", "0"))
                    grid_x = min(triton.cdiv(k * p.n, 1024), max(1, cap // s)) if cap else triton.cdiv(k * p.n, 1024)
                    with _nvtx(f"pf.stage(s={s},k={k})"):
                        _stage[(grid_x, s)](
                            p.mapped, b, p.row_off,
                            p.blk_off if p.blk_off is not None else p.row_off,
                            ids, start, p.n, k, layout, p.n_block, p.k_block, 1024)
                    if overlap:
                        ready[i].record(copy_stream)

            if overlap:
                for i in range(min(2, len(jobs))):
                    stage(i)
            for i, (start, s, k, b) in enumerate(jobs):
                if overlap:
                    stream.wait_event(ready[i])
                else:
                    stage(i)
                max_rows = max(counts[start:start + s])
                for r0 in range(0, max_rows, 1024):
                    r = min(1024, max_rows - r0)
                    with _nvtx(f"pf.ac_alloc(s={s},r={r})"):
                        a = torch.empty((s, r, k), dtype=x.dtype, device=x.device)
                        c = torch.empty((s, r, p.n), dtype=out.dtype, device=out.device)
                    with _nvtx(f"pf.gather(r={r},k={k})"):
                        _gather[(triton.cdiv(r * k, 1024), s)](
                            x, a, p.row_off, p.kidx, grouping.pair_sorted, grouping.pair_off,
                            ids, start, r0, x.shape[1], slots, pair_input, r, k, 1024)
                    with _nvtx(f"pf.bmm({s}x{r}x{k})"):
                        torch.bmm(a, b, out=c)
                    with _nvtx("pf.scatter"):
                        _scatter[(triton.cdiv(r * p.n, 1024), s)](
                            c, out, grouping.pair_sorted, grouping.pair_off, ids, start, r0,
                            r, p.n, out.shape[2], col, 1024)
                if overlap and i + 2 < len(jobs):
                    consumed = torch.cuda.Event()
                    consumed.record(stream)
                    stage(i + 2, consumed)
            # All writes on copy_stream have corresponding waits on stream.
            # The arena can therefore be reused by the next projection, and
            # freed on its allocation stream without a cross-stream lifetime gap.
