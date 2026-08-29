"""FP8 GPU 커널(worklist GEMV) — 참조 dequant GEMV와의 등가성 (계약 ⑤ 이원화).

- 랜덤 입력: fp32 dequant 참조와 tolerance
- 정확표현 입력(작은 정수 x, 배율 1.0, 코드 {0,±1,±2}): dense ↔ sparse(thr=0) ↔ 융합 비트일치
- pinned(UVA) 쌍둥이는 device 변형과 비트일치
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from fp8_ref import aligned_index, dequant_ckpt, random_expert_ckpt, row_store  # noqa: E402

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

E, TOPK = 8, 4


def _store(N, K, k_rows, exact, seed):
    """E expert의 fp8 스토어 (device) + 참조 dequant W [E, N, K] fp32 + kidx."""
    g = torch.Generator().manual_seed(seed)
    cs, ss, kidx, wref = [], [], [], []
    for _ in range(E):
        c_ck, s_ck = random_expert_ckpt(N, K, g, exact=exact)
        rows = aligned_index(K, k_rows, g)
        c, s = row_store(c_ck, s_ck, rows)
        cs.append(c); ss.append(s); kidx.append(rows)
        wref.append(dequant_ckpt(c_ck, s_ck))  # [N, K]
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
def test_gemv_fp8_matches_reference(pair, exact):
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8_indexed

    N, K, k_rows, M = 256, 512, 256, 3
    codes, scales, row_off, kidx, wref = _store(N, K, k_rows, exact, seed=1)
    x, ids = _inputs(M, K, exact, pair, seed=2)
    out = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv_fp8_indexed(x.cuda(), ids.int().cuda(), codes, scales, row_off, kidx, out, 0, pair,
                     torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = _ref(x, ids, wref, kidx.cpu(), row_off.cpu(), pair)
    if exact:
        assert torch.equal(out.cpu(), ref.to(torch.bfloat16))
    else:
        torch.testing.assert_close(out.float().cpu(), ref, rtol=2e-2, atol=2e-2)


@cuda_required
def test_gemv_fp8_pinned_bitwise_and_offset():
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8_indexed, gemv_fp8_indexed_pinned

    N, K, k_rows, M = 128, 512, 256, 2
    codes, scales, row_off, kidx, _ = _store(N, K, k_rows, False, seed=3)
    x, ids = _inputs(M, K, False, False, seed=4)
    stream = torch.cuda.current_stream()
    out_d = torch.zeros(M, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
    out_p = torch.zeros_like(out_d)
    gemv_fp8_indexed(x.cuda(), ids.cuda(), codes, scales, row_off, kidx, out_d, N, False, stream)
    cp = codes.cpu().pin_memory(); sp = scales.cpu().pin_memory()
    gemv_fp8_indexed_pinned(x.cuda(), ids.cuda(), cp, sp, row_off, kidx, out_p, N, False, stream)
    torch.cuda.synchronize()
    assert torch.equal(out_d, out_p)
    assert torch.all(out_d[:, :, :N] == 0)  # 오프셋 밖은 건드리지 않는다


@cuda_required
@pytest.mark.parametrize("pinned", [False, True])
def test_gemv_fp8_gateup_fused_bitwise(pinned):
    """융합 진입점 4개(= {device, pinned} × {dense, sparse})가 2회 launch와 비트일치."""
    from sglang.jit_kernel.prism_gemv_fp8 import (
        gemv_fp8_indexed, gemv_fp8_indexed_gateup,
        gemv_fp8_indexed_pinned, gemv_fp8_indexed_pinned_gateup,
    )

    N, K, k_rows, M = 128, 256, 128, 1
    c1, s1, ro1, ki1, _ = _store(N, K, k_rows, False, seed=5)
    c2, s2, ro2, ki2, _ = _store(N, K, k_rows, False, seed=6)
    x, ids = _inputs(M, K, False, False, seed=7)
    if pinned:
        c1, s1, c2, s2 = (t.cpu().pin_memory() for t in (c1, s1, c2, s2))
    single = gemv_fp8_indexed_pinned if pinned else gemv_fp8_indexed
    fused = gemv_fp8_indexed_pinned_gateup if pinned else gemv_fp8_indexed_gateup
    stream = torch.cuda.current_stream()
    ref = torch.zeros(M, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
    single(x.cuda(), ids.cuda(), c1, s1, ro1, ki1, ref, 0, False, stream)
    single(x.cuda(), ids.cuda(), c2, s2, ro2, ki2, ref, N, False, stream)
    out = torch.zeros_like(ref)
    fused(x.cuda(), ids.cuda(), c1, s1, ro1, ki1, c2, s2, ro2, ki2, out, 0, N, False, stream)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def _sparse_spec(E_, total, seed, thr_val):
    from sglang.srt.layers.moe.prism.tiers import SparseSpec

    g = torch.Generator().manual_seed(seed)
    a = torch.rand(total, generator=g).cuda()
    c = (torch.rand(total // 2, generator=g) * 0.1).cuda()
    return SparseSpec(a=a, c=c, thr=torch.full((E_, 4), thr_val, device="cuda"),
                      p=0.5, lam=0.0, pmax=0.9, grid=0.1, ng=4, renorm_it=1)


@cuda_required
def test_gemv_fp8_wide_tile_bitwise():
    """타일 heuristic이 갈리는 치수 — pinned/device와 융합/2회 launch가 여전히 비트일치.

    2026-08-29에 들어온 타일 선택(열 타일 kV·행 슬롯 kNY를 치수에서 고른다)의 회귀 방어선.
    **kNY는 ty 트리의 누산 순서를 정하므로 수치 계약의 일부다** — hot(device)/warm(pinned)이나
    융합/단일 launch가 서로 다른 kNY를 고르면 일반 입력에서 여기서 비트가 갈린다.
    위 테스트들은 k_rows가 작아(≤ 512) 넓은 분기(kNY=64, fp8은 kV=8도)를 한 번도 타지 않는다.
    """
    from sglang.jit_kernel.prism_gemv_fp8 import (
        gemv_fp8_indexed, gemv_fp8_indexed_gateup, gemv_fp8_indexed_pinned,
    )

    # N·k_rows를 이만큼 잡아야 fp8의 "SM당 바이트" 문턱(128 KB)을 넘어 kV=8 분기를 탄다:
    # pairs(8) × k_rows(2048) × N(2048) / SM ≈ 197 KB (170 SM) / 254 KB (132 SM).
    N, K, k_rows, M = 2048, 2048, 2048, 2
    c1, s1, ro1, ki1, _ = _store(N, K, k_rows, False, seed=41)
    x, ids = _inputs(M, K, False, False, seed=43)
    xd, idsd, stream = x.cuda(), ids.int().cuda(), torch.cuda.current_stream()

    dev = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    pin = torch.zeros_like(dev)
    gemv_fp8_indexed(xd, idsd, c1, s1, ro1, ki1, dev, 0, False, stream)
    gemv_fp8_indexed_pinned(xd, idsd, c1.cpu().pin_memory(), s1.cpu().pin_memory(),
                            ro1, ki1, pin, 0, False, stream)
    torch.cuda.synchronize()
    assert torch.equal(dev, pin), "pinned가 device와 다른 kNY를 골랐다"

    c2, s2, ro2, ki2, _ = _store(N, K, k_rows, False, seed=42)
    ref = torch.zeros(M, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
    gemv_fp8_indexed(xd, idsd, c1, s1, ro1, ki1, ref, 0, False, stream)
    gemv_fp8_indexed(xd, idsd, c2, s2, ro2, ki2, ref, N, False, stream)
    fused = torch.zeros_like(ref)
    gemv_fp8_indexed_gateup(xd, idsd, c1, s1, ro1, ki1, c2, s2, ro2, ki2, fused, 0, N,
                            False, stream)
    torch.cuda.synchronize()
    assert torch.equal(fused, ref), "융합(grid.z=2)이 단일 launch와 다른 타일을 골랐다"


@cuda_required
def test_gemv_fp8_sparse_thr0_bitwise_and_masked():
    """thr=0(전부 keep) ↔ dense 비트일치; thr>0 ↔ 마스크 적용 참조."""
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8_indexed, gemv_fp8_indexed_sparse

    N, K, k_rows, M = 128, 256, 128, 2
    codes, scales, row_off, kidx, wref = _store(N, K, k_rows, False, seed=8)
    x, ids = _inputs(M, K, False, False, seed=9)
    stream = torch.cuda.current_stream()
    total = E * k_rows
    w = torch.rand(M, TOPK, generator=torch.Generator().manual_seed(11)).cuda()
    sp_dense, sp_masked = _sparse_spec(E, total, 10, 0.0), _sparse_spec(E, total, 10, 0.5)

    dense = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv_fp8_indexed(x.cuda(), ids.cuda(), codes, scales, row_off, kidx, dense, 0, False, stream)
    sp0 = torch.zeros_like(dense)
    gemv_fp8_indexed_sparse(x.cuda(), ids.cuda(), w, codes, scales, row_off, kidx, sp0, sp_dense,
                            0, False, stream)
    torch.cuda.synchronize()
    assert torch.equal(dense, sp0)

    # thr > 0: 참조는 페어 에너지 ≥ thr² 인 페어만 남긴 x로 계산
    thr = 0.5
    masked = torch.zeros_like(dense)
    gemv_fp8_indexed_sparse(x.cuda(), ids.cuda(), w, codes, scales, row_off, kidx, masked, sp_masked,
                            0, False, stream)
    torch.cuda.synchronize()
    xc, ac, cc = x.float(), sp_masked.a.cpu(), sp_masked.c.cpu()
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
def test_gemv_fp8_sparse_gateup_fused_bitwise():
    """sparse 융합(warm의 실제 경로) ↔ sparse 2회 launch 비트일치, device/pinned 양쪽."""
    from sglang.jit_kernel.prism_gemv_fp8 import (
        gemv_fp8_indexed_pinned_sparse, gemv_fp8_indexed_pinned_sparse_gateup,
        gemv_fp8_indexed_sparse, gemv_fp8_indexed_sparse_gateup,
    )

    N, K, k_rows, M = 128, 256, 128, 2
    c1, s1, ro1, ki1, _ = _store(N, K, k_rows, False, seed=12)
    c2, s2, ro2, ki2, _ = _store(N, K, k_rows, False, seed=13)
    x, ids = _inputs(M, K, False, False, seed=14)
    w = torch.rand(M, TOPK, generator=torch.Generator().manual_seed(15)).cuda()
    spg = _sparse_spec(E, E * k_rows, 16, 0.4)
    spu = _sparse_spec(E, E * k_rows, 17, 0.4)
    stream = torch.cuda.current_stream()
    for single, fused, store in ((gemv_fp8_indexed_sparse, gemv_fp8_indexed_sparse_gateup, False),
                                 (gemv_fp8_indexed_pinned_sparse,
                                  gemv_fp8_indexed_pinned_sparse_gateup, True)):
        cc1, ss1, cc2, ss2 = (tuple(t.cpu().pin_memory() for t in (c1, s1, c2, s2)) if store
                              else (c1, s1, c2, s2))
        ref = torch.zeros(M, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
        single(x.cuda(), ids.cuda(), w, cc1, ss1, ro1, ki1, ref, spg, 0, False, stream)
        single(x.cuda(), ids.cuda(), w, cc2, ss2, ro2, ki2, ref, spu, N, False, stream)
        out = torch.zeros_like(ref)
        fused(x.cuda(), ids.cuda(), w, cc1, ss1, ro1, ki1, cc2, ss2, ro2, ki2, out, spg, spu,
              0, N, False, stream)
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


# ─── grouped GEMM (prefill 형태) ───────────────────────────────────────────────
@cuda_required
@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("m", [8, 300])
def test_grouped_fp8_matches_gemv(pair, exact, m):
    """grouped ↔ GEMV: 배율이 2의 거듭제곱이라 `bf16(code·scale)`이 정확 →
    정확표현 입력에서는 비트일치, 랜덤 입력에서는 누산 순서만큼의 tolerance."""
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8_indexed
    from sglang.jit_kernel.prism_grouped_fp8 import grouped_fp8_indexed, grouped_fp8_indexed_pinned
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    N, K, k_rows = 256, 512, 256
    codes, scales, row_off, kidx, wref = _store(N, K, k_rows, exact, seed=21)
    x, ids = _inputs(m, K, exact, pair, seed=22 + m)
    if m >= 64:
        ids[:, 0] = 3          # expert 3에 pair 몰기 → 타일 여러 개
        ids[ids == 5] = 6      # expert 5는 pair 0개
    stream = torch.cuda.current_stream()
    ref = torch.zeros(m, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv_fp8_indexed(x.cuda(), ids.cuda(), codes, scales, row_off, kidx, ref, 0, pair, stream)
    grouping = build_grouping(ids.cuda(), E)
    out = torch.zeros_like(ref)
    grouped_fp8_indexed(x.cuda(), grouping, codes, scales, row_off, kidx, out, 0, pair, stream)
    outp = torch.zeros_like(ref)
    grouped_fp8_indexed_pinned(x.cuda(), grouping, codes.cpu().pin_memory(),
                               scales.cpu().pin_memory(), row_off, kidx, outp, 0, pair, stream, 64)
    torch.cuda.synchronize()
    assert torch.equal(out, outp)
    if exact:
        assert torch.equal(out, ref)
    else:
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
        fref = _ref(x, ids, wref, kidx.cpu(), row_off.cpu(), pair)
        torch.testing.assert_close(out.float().cpu(), fref, rtol=2e-2, atol=2e-2)


@cuda_required
def test_grouped_fp8_gateup_fused():
    from sglang.jit_kernel.prism_grouped_fp8 import grouped_fp8_indexed, grouped_fp8_indexed_gateup
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    N, K, k_rows, m = 128, 256, 128, 40
    c1, s1, ro1, ki1, _ = _store(N, K, k_rows, True, seed=23)
    c2, s2, ro2, ki2, _ = _store(N, K, k_rows, True, seed=24)
    x, ids = _inputs(m, K, True, False, seed=25)
    stream = torch.cuda.current_stream()
    grouping = build_grouping(ids.cuda(), E)
    ref = torch.zeros(m, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
    grouped_fp8_indexed(x.cuda(), grouping, c1, s1, ro1, ki1, ref, 0, False, stream)
    grouped_fp8_indexed(x.cuda(), grouping, c2, s2, ro2, ki2, ref, N, False, stream)
    out = torch.zeros_like(ref)
    grouped_fp8_indexed_gateup(x.cuda(), grouping, c1, s1, ro1, ki1, c2, s2, ro2, ki2, out,
                               0, N, False, stream)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@cuda_required
@pytest.mark.parametrize("exact", [True, False])
def test_grouped_fp8_cold_tile_matches_gemv(exact):
    """KT_TILE8 로더: kt `GemmKernelTileK2FP8B128::BufferB` slab을 제자리 읽어 GEMV와 일치."""
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8_indexed
    from sglang.jit_kernel.prism_grouped_fp8 import grouped_fp8_cold
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    sys.path.insert(0, os.path.dirname(__file__))
    from fp8_ref import tile_block  # noqa: E402

    N, K, k_rows, m = 256, 512, 256, 40
    g = torch.Generator().manual_seed(31)
    blocks, cs, ss, kidx, wref = [], [], [], [], []
    for _ in range(E):
        c_ck, s_ck = random_expert_ckpt(N, K, g, exact=exact)
        rows = aligned_index(K, k_rows, g)
        blocks.append(tile_block(c_ck, s_ck, rows))
        c, s = row_store(c_ck, s_ck, rows)
        cs.append(c); ss.append(s); kidx.append(rows)
        wref.append(dequant_ckpt(c_ck, s_ck))
    slab = torch.cat(blocks).pin_memory()
    blk_off = torch.tensor([0] + list(torch.cumsum(torch.tensor([b.numel() for b in blocks]), 0)[:-1]),
                           dtype=torch.int64).cuda()
    row_off = (torch.arange(E + 1, dtype=torch.int32) * k_rows).cuda()
    kidx_t = torch.cat(kidx).to(torch.uint16).cuda()
    codes, scales = torch.cat(cs).cuda(), torch.cat(ss).cuda()

    x, ids = _inputs(m, K, exact, False, seed=32)
    stream = torch.cuda.current_stream()
    ref = torch.zeros(m, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv_fp8_indexed(x.cuda(), ids.cuda(), codes, scales, row_off, kidx_t, ref, 0, False, stream)

    class _Slab:  # ColdSlab의 최소 형태 (cold_gpu.ColdSlab과 같은 필드 이름)
        pass

    cold = _Slab()
    cold.slab, cold.blk_off, cold.row_off, cold.k_index = slab, blk_off, row_off, kidx_t
    cold.n, cold.n_start, cold.layout = N, 0, "kt_tile8"
    out = torch.zeros_like(ref)
    grouped_fp8_cold(x.cuda(), build_grouping(ids.cuda(), E), cold, out, 0, False, stream)
    torch.cuda.synchronize()
    if exact:
        assert torch.equal(out, ref)
    else:
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
