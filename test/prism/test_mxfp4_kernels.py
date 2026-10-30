"""MXFP4 GPU 커널(worklist GEMV·grouped GEMM) — 참조 dequant GEMV와의 등가성 (계약 ⑤ 이원화).

- 랜덤 입력: fp32 dequant 참조와 tolerance
- 정확표현 입력(작은 정수 x, 배율 2^0, 코드 {0,±1,±2}): GEMV ↔ grouped ↔ sparse(thr=0) 비트일치
- pinned(UVA) 쌍둥이는 device 변형과 비트일치
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from mxfp4_ref import aligned_index, dequant_ckpt, pairrow_store, random_expert_ckpt  # noqa: E402

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

E, TOPK = 8, 4


def _store(N, K, k_rows, exact, seed):
    """E expert의 pair-row 스토어 (device) + 참조 dequant W [E, N, K] fp32 + kidx."""
    g = torch.Generator().manual_seed(seed)
    cs, ss, kidx, wref = [], [], [], []
    for _ in range(E):
        c_ck, s_ck = random_expert_ckpt(N, K, g, exact=exact)
        rows = aligned_index(K, k_rows, g)
        c, s = pairrow_store(c_ck, s_ck, rows)
        cs.append(c); ss.append(s); kidx.append(rows)
        wref.append(dequant_ckpt(c_ck.view(torch.uint8), s_ck))  # [N, K]
    codes = torch.cat(cs).cuda()
    scales = torch.cat(ss).cuda()
    row_off = (torch.arange(E + 1, dtype=torch.int32) * k_rows).cuda()
    kidx_t = torch.cat(kidx).to(torch.uint16).cuda()
    return codes, scales, row_off, kidx_t, torch.stack(wref)


def _inputs(M, K, exact, pair, seed):
    g = torch.Generator().manual_seed(seed)
    rx = M * TOPK if pair else M
    if exact:
        x = torch.randint(-2, 3, (rx, K), generator=g).to(torch.bfloat16)
    else:
        x = (torch.randn(rx, K, generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.randint(0, E, (M, TOPK), generator=g)
    return x, ids


def _ref(x, ids, wref, kidx_cpu, row_off_cpu, pair):
    """fp32 참조: out[m, j, n] = Σ_{k∈kidx(e)} x[row, k] · W_e[n, k]."""
    M = ids.shape[0]
    N = wref.shape[1]
    out = torch.zeros(M, TOPK, N, dtype=torch.float32)
    xf = x.float()
    for m in range(M):
        for j in range(TOPK):
            e = int(ids[m, j])
            rows = kidx_cpu[int(row_off_cpu[e]):int(row_off_cpu[e + 1])].long()
            r = m * TOPK + j if pair else m
            out[m, j] = wref[e][:, rows] @ xf[r, rows]
    return out


@cuda_required
@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_gemv_mxfp4_matches_reference(pair, exact):
    from sglang.jit_kernel.prism_gemv_mxfp4 import gemv_mxfp4_indexed

    N, K, k_rows, M = 160, 256, 128, 3
    codes, scales, row_off, kidx, wref = _store(N, K, k_rows, exact, seed=1)
    x, ids = _inputs(M, K, exact, pair, seed=2)
    out = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv_mxfp4_indexed(x.cuda(), ids.int().cuda(), codes, scales, row_off, kidx, out, 0, pair,
                       torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = _ref(x, ids, wref, kidx.cpu(), row_off.cpu(), pair)
    if exact:
        assert torch.equal(out.cpu(), ref.to(torch.bfloat16))
    else:
        torch.testing.assert_close(out.float().cpu(), ref, rtol=2e-2, atol=2e-2)


@cuda_required
def test_gemv_mxfp4_pinned_bitwise_and_offset():
    from sglang.jit_kernel.prism_gemv_mxfp4 import gemv_mxfp4_indexed, gemv_mxfp4_indexed_pinned

    N, K, k_rows, M = 128, 512, 256, 2
    codes, scales, row_off, kidx, _ = _store(N, K, k_rows, False, seed=3)
    x, ids = _inputs(M, K, False, False, seed=4)
    stream = torch.cuda.current_stream()
    out_d = torch.zeros(M, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
    out_p = torch.zeros_like(out_d)
    gemv_mxfp4_indexed(x.cuda(), ids.cuda(), codes, scales, row_off, kidx, out_d, N, False, stream)
    cp = codes.cpu().pin_memory(); sp = scales.cpu().pin_memory()
    gemv_mxfp4_indexed_pinned(x.cuda(), ids.cuda(), cp, sp, row_off, kidx, out_p, N, False, stream)
    torch.cuda.synchronize()
    assert torch.equal(out_d, out_p)
    assert torch.all(out_d[:, :, :N] == 0)  # 오프셋 밖은 건드리지 않는다


@cuda_required
def test_gemv_mxfp4_gateup_fused_bitwise():
    from sglang.jit_kernel.prism_gemv_mxfp4 import gemv_mxfp4_indexed, gemv_mxfp4_indexed_gateup

    N, K, k_rows, M = 128, 256, 128, 1
    c1, s1, ro1, ki1, _ = _store(N, K, k_rows, False, seed=5)
    c2, s2, ro2, ki2, _ = _store(N, K, k_rows, False, seed=6)
    x, ids = _inputs(M, K, False, False, seed=7)
    stream = torch.cuda.current_stream()
    ref = torch.zeros(M, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
    gemv_mxfp4_indexed(x.cuda(), ids.cuda(), c1, s1, ro1, ki1, ref, 0, False, stream)
    gemv_mxfp4_indexed(x.cuda(), ids.cuda(), c2, s2, ro2, ki2, ref, N, False, stream)
    out = torch.zeros_like(ref)
    gemv_mxfp4_indexed_gateup(x.cuda(), ids.cuda(), c1, s1, ro1, ki1, c2, s2, ro2, ki2, out, 0, N,
                              False, stream)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@cuda_required
def test_gemv_mxfp4_wide_tile_bitwise():
    """타일 heuristic이 갈리는 치수 — pinned/device와 융합/2회 launch가 여전히 비트일치.

    2026-08-29에 들어온 타일 선택(열 타일 kV·행 슬롯 kNY를 치수에서 고른다)의 회귀 방어선.
    **kNY는 ty 트리의 누산 순서를 정하므로 수치 계약의 일부다** — hot(device)/warm(pinned)이나
    융합/단일 launch가 서로 다른 kNY를 고르면 일반 입력에서 여기서 비트가 갈린다.
    위 테스트들은 k_rows가 작아(≤ 512) 넓은 분기(kNY=64)를 한 번도 타지 않는다.
    """
    from sglang.jit_kernel.prism_gemv_mxfp4 import (
        gemv_mxfp4_indexed, gemv_mxfp4_indexed_gateup, gemv_mxfp4_indexed_pinned,
    )

    N, K, k_rows, M = 128, 2048, 2048, 1  # k_rows >= 2048 이 넓은 kNY의 조건
    c1, s1, ro1, ki1, _ = _store(N, K, k_rows, False, seed=41)
    c2, s2, ro2, ki2, _ = _store(N, K, k_rows, False, seed=42)
    x, ids = _inputs(M, K, False, False, seed=43)
    xd, idsd, stream = x.cuda(), ids.int().cuda(), torch.cuda.current_stream()

    dev = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    pin = torch.zeros_like(dev)
    gemv_mxfp4_indexed(xd, idsd, c1, s1, ro1, ki1, dev, 0, False, stream)
    gemv_mxfp4_indexed_pinned(xd, idsd, c1.cpu().pin_memory(), s1.cpu().pin_memory(),
                              ro1, ki1, pin, 0, False, stream)
    torch.cuda.synchronize()
    assert torch.equal(dev, pin), "pinned가 device와 다른 kNY를 골랐다"

    ref = torch.zeros(M, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
    gemv_mxfp4_indexed(xd, idsd, c1, s1, ro1, ki1, ref, 0, False, stream)
    gemv_mxfp4_indexed(xd, idsd, c2, s2, ro2, ki2, ref, N, False, stream)
    fused = torch.zeros_like(ref)
    gemv_mxfp4_indexed_gateup(xd, idsd, c1, s1, ro1, ki1, c2, s2, ro2, ki2, fused, 0, N,
                              False, stream)
    torch.cuda.synchronize()
    assert torch.equal(fused, ref), "융합(grid.z=2)이 단일 launch와 다른 타일을 골랐다"


@cuda_required
def test_gemv_mxfp4_sparse_thr0_bitwise_and_masked():
    """thr=0(전부 keep) ↔ dense 비트일치; thr>0 ↔ 마스크 적용 참조."""
    from sglang.jit_kernel.prism_gemv_mxfp4 import gemv_mxfp4_indexed, gemv_mxfp4_indexed_sparse
    from sglang.srt.layers.moe.prism.tiers import SparseSpec

    N, K, k_rows, M = 128, 256, 128, 2
    codes, scales, row_off, kidx, wref = _store(N, K, k_rows, False, seed=8)
    x, ids = _inputs(M, K, False, False, seed=9)
    stream = torch.cuda.current_stream()
    total = E * k_rows
    g = torch.Generator().manual_seed(10)
    a = torch.rand(total, generator=g).cuda()
    c = (torch.rand(total // 2, generator=g) * 0.1).cuda()
    w = torch.rand(M, TOPK, generator=g).cuda()

    def spec(thr_val):
        return SparseSpec(a=a, c=c, thr=torch.full((E, 4), thr_val, device="cuda"),
                          p=0.5, lam=0.0, pmax=0.9, grid=0.1, ng=4, renorm_it=1)

    dense = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv_mxfp4_indexed(x.cuda(), ids.cuda(), codes, scales, row_off, kidx, dense, 0, False, stream)
    sp0 = torch.zeros_like(dense)
    gemv_mxfp4_indexed_sparse(x.cuda(), ids.cuda(), w, codes, scales, row_off, kidx, sp0, spec(0.0),
                              0, False, stream)
    torch.cuda.synchronize()
    assert torch.equal(dense, sp0)

    # thr > 0: 참조는 페어 에너지 ≥ thr² 인 페어만 남긴 x로 계산
    thr = 0.5
    masked = torch.zeros_like(dense)
    gemv_mxfp4_indexed_sparse(x.cuda(), ids.cuda(), w, codes, scales, row_off, kidx, masked, spec(thr),
                              0, False, stream)
    torch.cuda.synchronize()
    xc, ac, cc = x.float(), a.cpu(), c.cpu()
    ref = torch.zeros(M, TOPK, N)
    kidx_c, ro_c = kidx.cpu(), row_off.cpu()
    for m in range(M):
        for j in range(TOPK):
            e = int(ids[m, j]); o0 = int(ro_c[e])
            rows = kidx_c[o0:o0 + k_rows].long()
            xg = xc[m, rows].clone()
            for p in range(k_rows // 2):
                x0, x1 = float(xg[2 * p]), float(xg[2 * p + 1])
                ar = o0 + 2 * p
                en = float(ac[ar]) * x0 * x0 + float(ac[ar + 1]) * x1 * x1 + 2 * float(cc[ar // 2]) * x0 * x1
                if max(en, 0.0) < thr * thr:
                    xg[2 * p] = 0; xg[2 * p + 1] = 0
            ref[m, j] = wref[e][:, rows] @ xg
    torch.testing.assert_close(masked.float().cpu(), ref, rtol=2e-2, atol=2e-2)
    assert not torch.equal(masked, dense)  # 실제로 무언가 마스킹됐다


@cuda_required
@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("m", [8, 300])
def test_grouped_mxfp4_matches_gemv(pair, exact, m):
    from sglang.jit_kernel.prism_gemv_mxfp4 import gemv_mxfp4_indexed
    from sglang.jit_kernel.prism_grouped_mxfp4 import grouped_mxfp4_indexed, grouped_mxfp4_indexed_pinned
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    N, K, k_rows = 144, 320, 96   # kr=96: 64 배수 아님 → 마지막 K 타일 절반 유효 경로
    codes, scales, row_off, kidx, wref = _store(N, K, k_rows, exact, seed=11)
    x, ids = _inputs(m, K, exact, pair, seed=12 + m)
    if m >= 64:
        ids[:, 0] = 3          # expert 3에 pair 몰기 → 타일 여러 개
        ids[ids == 5] = 6      # expert 5는 pair 0개
    stream = torch.cuda.current_stream()
    ref = torch.zeros(m, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv_mxfp4_indexed(x.cuda(), ids.cuda(), codes, scales, row_off, kidx, ref, 0, pair, stream)
    grouping = build_grouping(ids.cuda(), E)
    out = torch.zeros_like(ref)
    grouped_mxfp4_indexed(x.cuda(), grouping, codes, scales, row_off, kidx, out, 0, pair, stream)
    outp = torch.zeros_like(ref)
    grouped_mxfp4_indexed_pinned(x.cuda(), grouping, codes.cpu().pin_memory(), scales.cpu().pin_memory(),
                                 row_off, kidx, outp, 0, pair, stream, 64)
    torch.cuda.synchronize()
    assert torch.equal(out, outp)
    if exact:
        assert torch.equal(out, ref)
    else:
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
        fref = _ref(x, ids, wref, kidx.cpu(), row_off.cpu(), pair)
        torch.testing.assert_close(out.float().cpu(), fref, rtol=2e-2, atol=2e-2)


@cuda_required
def test_grouped_mxfp4_gateup_fused():
    from sglang.jit_kernel.prism_grouped_mxfp4 import grouped_mxfp4_indexed, grouped_mxfp4_indexed_gateup
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    N, K, k_rows, m = 128, 256, 128, 40
    c1, s1, ro1, ki1, _ = _store(N, K, k_rows, True, seed=13)
    c2, s2, ro2, ki2, _ = _store(N, K, k_rows, True, seed=14)
    x, ids = _inputs(m, K, True, False, seed=15)
    stream = torch.cuda.current_stream()
    grouping = build_grouping(ids.cuda(), E)
    ref = torch.zeros(m, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
    grouped_mxfp4_indexed(x.cuda(), grouping, c1, s1, ro1, ki1, ref, 0, False, stream)
    grouped_mxfp4_indexed(x.cuda(), grouping, c2, s2, ro2, ki2, ref, N, False, stream)
    out = torch.zeros_like(ref)
    grouped_mxfp4_indexed_gateup(x.cuda(), grouping, c1, s1, ro1, ki1, c2, s2, ro2, ki2, out, 0, N,
                                 False, stream)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@cuda_required
@pytest.mark.parametrize("m", [8, 200])
def test_grouped_mxfp4_cold_slab_matches_pairrow(m):
    """KT_FP4 로더(kt BufferBInt4KGroupImpl: [n][k/2] nibble 행우선 + fp32 배율)가 pair-row 스토어와
    같은 값을 곱한다 — 정확표현 입력에서 비트일치. slab은 expert 블록(64 B 정렬)을 이어 붙인 u8
    host 텐서(pinned)이고 blk_off는 바이트 오프셋."""
    from types import SimpleNamespace

    from sglang.jit_kernel.prism_grouped_mxfp4 import grouped_mxfp4_cold, grouped_mxfp4_indexed
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    N, K, k_rows = 96, 256, 96          # kr=96: 64 배수 아님
    g = torch.Generator().manual_seed(21)
    cs, ss, kidx, blocks, offs = [], [], [], [], [0]
    for _ in range(E):
        c_ck, s_ck = random_expert_ckpt(N, K, g, exact=True)
        rows = aligned_index(K, k_rows, g)
        c, s = pairrow_store(c_ck, s_ck, rows)
        cs.append(c); ss.append(s); kidx.append(rows)
        # kt 블록: nibble 행 [N, k/2] (선택 페어 열) + fp32 d [N, k/32]
        pairs = (rows[0::2] // 2).long(); blks = (rows[0::32] // 32).long()
        nib = c_ck.view(torch.uint8).index_select(1, pairs).contiguous()          # [N, k/2]
        d = torch.ldexp(torch.ones(N, k_rows // 32), s_ck.index_select(1, blks).int() - 127)
        blk = torch.cat([nib.reshape(-1), d.contiguous().view(torch.uint8).reshape(-1)])
        pad = (-blk.numel()) % 64
        blocks.append(torch.cat([blk, torch.zeros(pad, dtype=torch.uint8)]))
        offs.append(offs[-1] + blocks[-1].numel())
    slab = torch.cat(blocks)
    slab = torch.cat([slab, torch.zeros((-slab.numel()) % 4096, dtype=torch.uint8)]).pin_memory()
    codes = torch.cat(cs).cuda(); scales = torch.cat(ss).cuda()
    row_off = (torch.arange(E + 1, dtype=torch.int32) * k_rows).cuda()
    kidx_t = torch.cat(kidx).to(torch.uint16).cuda()
    cold = SimpleNamespace(slab=slab, blk_off=torch.tensor(offs[:-1], dtype=torch.int64).cuda(),
                           row_off=row_off, k_index=kidx_t, n=N, n_start=0, layout="kt_fp4")
    x, ids = _inputs(m, K, True, False, seed=22 + m)
    stream = torch.cuda.current_stream()
    grouping = build_grouping(ids.cuda(), E)
    ref = torch.zeros(m, TOPK, N, dtype=torch.bfloat16, device="cuda")
    grouped_mxfp4_indexed(x.cuda(), grouping, codes, scales, row_off, kidx_t, ref, 0, False, stream)
    out = torch.zeros_like(ref)
    grouped_mxfp4_cold(x.cuda(), grouping, cold, out, 0, False, stream, 64)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@cuda_required
@pytest.mark.parametrize("m", [8, 200])
def test_grouped_mxfp4_cold_tile_slab_matches_pairrow(m):
    """KT_TILE4 로더(kt GemmKernelTileK2MXFP4::BufferB: fp4 32k×256n 타일 + 전치 E8M0)가 pair-row
    스토어와 같은 값을 곱한다 — 정확표현 입력에서 비트일치."""
    from types import SimpleNamespace

    from mxfp4_ref import tile_block
    from sglang.jit_kernel.prism_grouped_mxfp4 import grouped_mxfp4_cold, grouped_mxfp4_indexed
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    N, K, k_rows = 256, 256, 96          # kr=96: 64 배수 아님 (마지막 GPU 타일 절반 유효)
    g = torch.Generator().manual_seed(31)
    cs, ss, kidx, blocks, offs = [], [], [], [], [0]
    for _ in range(E):
        c_ck, s_ck = random_expert_ckpt(N, K, g, exact=True)
        rows = aligned_index(K, k_rows, g)
        c, s = pairrow_store(c_ck, s_ck, rows)
        cs.append(c); ss.append(s); kidx.append(rows)
        blk = tile_block(c_ck, s_ck, rows)
        pad = (-blk.numel()) % 64
        blocks.append(torch.cat([blk, torch.zeros(pad, dtype=torch.uint8)]))
        offs.append(offs[-1] + blocks[-1].numel())
    slab = torch.cat(blocks)
    slab = torch.cat([slab, torch.zeros((-slab.numel()) % 4096, dtype=torch.uint8)]).pin_memory()
    codes = torch.cat(cs).cuda(); scales = torch.cat(ss).cuda()
    row_off = (torch.arange(E + 1, dtype=torch.int32) * k_rows).cuda()
    kidx_t = torch.cat(kidx).to(torch.uint16).cuda()
    cold = SimpleNamespace(slab=slab, blk_off=torch.tensor(offs[:-1], dtype=torch.int64).cuda(),
                           row_off=row_off, k_index=kidx_t, n=N, n_start=0, layout="kt_tile4")
    x, ids = _inputs(m, K, True, False, seed=32 + m)
    stream = torch.cuda.current_stream()
    grouping = build_grouping(ids.cuda(), E)
    ref = torch.zeros(m, TOPK, N, dtype=torch.bfloat16, device="cuda")
    grouped_mxfp4_indexed(x.cuda(), grouping, codes, scales, row_off, kidx_t, ref, 0, False, stream)
    out = torch.zeros_like(ref)
    grouped_mxfp4_cold(x.cuda(), grouping, cold, out, 0, False, stream, 64)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
