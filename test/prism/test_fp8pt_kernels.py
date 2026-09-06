"""FP8 **per-tensor 배율** GPU 커널 — worklist GEMV(`gemv_fp8pt_*`)와 grouped(`grouped_fp8pt_*`,
`grouped_fp8_cold` layout kt_tile8pt)의 참조 등가성 (계약 ⑤).

블록판 테스트(test_fp8_kernels.py)와 같은 틀에 세 가지가 더 있다:
- 스토어 k 행 수가 128의 배수가 **아닌** 32의 배수(k_rows=96, 160) — pt에서만 허용되는 기하이고
  grouped의 부분 K 타일(kr % 64 == 32) 경로를 실제로 탄다.
- 블록판 ↔ pt 등가: 블록표를 스칼라로 채운 b128 스토어와 pt 스토어가 exact 픽스처에서 비트일치.
- kt per-tensor cold slab(코드 타일 + 64 B 배율 슬롯) 로더.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from fp8_ref import (  # noqa: E402
    aligned_index_pt, dequant_ckpt_pt, random_expert_ckpt_pt, row_store_pt, tile_block_pt,
)

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

E, TOPK = 8, 4


def _store(N, K, k_rows, exact, seed, pow2_scale=True):
    """E expert의 pt 스토어 (device): codes [E·k_rows, N], scales [E], row_off, kidx, W_ref [E, N, K]."""
    g = torch.Generator().manual_seed(seed)
    cs, ss, kidx, wref = [], [], [], []
    for _ in range(E):
        c_ck, s_ck = random_expert_ckpt_pt(N, K, g, exact=exact, pow2_scale=pow2_scale)
        rows = aligned_index_pt(K, k_rows, g)
        cs.append(row_store_pt(c_ck, rows)); ss.append(s_ck); kidx.append(rows)
        wref.append(dequant_ckpt_pt(c_ck, s_ck))
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


def _ref(x, ids, wref, kidx_cpu, row_off_cpu, pair, with_mag=False):
    """fp32 참조. with_mag=True면 항 크기 합 Σ|x·w|도 돌려준다 (상쇄가 큰 출력의 허용치 기준)."""
    M = ids.shape[0]
    N = wref.shape[1]
    out = torch.zeros(M, TOPK, N, dtype=torch.float32)
    mag = torch.zeros_like(out)
    xf = x.float()
    for m in range(M):
        for j in range(TOPK):
            e = int(ids[m, j])
            rows = kidx_cpu[int(row_off_cpu[e]):int(row_off_cpu[e + 1])].long()
            r = m * TOPK + j if pair else m
            out[m, j] = wref[e][:, rows] @ xf[r, rows]
            if with_mag:
                mag[m, j] = wref[e][:, rows].abs() @ xf[r, rows].abs()
    return (out, mag) if with_mag else out


# ─── worklist GEMV ────────────────────────────────────────────────────────────
@cuda_required
@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("k_rows", [256, 96])  # 96: 128 정렬 아님
def test_gemv_fp8pt_matches_reference(pair, exact, k_rows):
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8pt_indexed

    N, K, M = 256, 512, 3
    codes, scales, row_off, kidx, wref = _store(N, K, k_rows, exact, seed=101)
    x, ids = _inputs(M, K, exact, pair, seed=102)
    out = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv_fp8pt_indexed(x.cuda(), ids.int().cuda(), codes, scales, row_off, kidx, out, 0, pair,
                       torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = _ref(x, ids, wref, kidx.cpu(), row_off.cpu(), pair)
    if exact:
        assert torch.equal(out.cpu(), ref.to(torch.bfloat16))
    else:
        torch.testing.assert_close(out.float().cpu(), ref, rtol=2e-2, atol=2e-2)


@cuda_required
def test_gemv_fp8pt_pinned_bitwise_and_offset():
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8pt_indexed, gemv_fp8pt_indexed_pinned

    N, K, k_rows, M = 128, 512, 160, 2
    codes, scales, row_off, kidx, _ = _store(N, K, k_rows, False, seed=103)
    x, ids = _inputs(M, K, False, False, seed=104)
    stream = torch.cuda.current_stream()
    out_d = torch.zeros(M, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
    out_p = torch.zeros_like(out_d)
    gemv_fp8pt_indexed(x.cuda(), ids.cuda(), codes, scales, row_off, kidx, out_d, N, False, stream)
    gemv_fp8pt_indexed_pinned(x.cuda(), ids.cuda(), codes.cpu().pin_memory(),
                              scales.cpu().pin_memory(), row_off, kidx, out_p, N, False, stream)
    torch.cuda.synchronize()
    assert torch.equal(out_d, out_p)
    assert torch.all(out_d[:, :, :N] == 0)


@cuda_required
@pytest.mark.parametrize("pinned", [False, True])
def test_gemv_fp8pt_gateup_fused_bitwise(pinned):
    from sglang.jit_kernel.prism_gemv_fp8 import (
        gemv_fp8pt_indexed, gemv_fp8pt_indexed_gateup,
        gemv_fp8pt_indexed_pinned, gemv_fp8pt_indexed_pinned_gateup,
    )

    N, K, k_rows, M = 128, 256, 96, 1
    c1, s1, ro1, ki1, _ = _store(N, K, k_rows, False, seed=105)
    c2, s2, ro2, ki2, _ = _store(N, K, k_rows, False, seed=106)
    x, ids = _inputs(M, K, False, False, seed=107)
    if pinned:
        c1, s1, c2, s2 = (t.cpu().pin_memory() for t in (c1, s1, c2, s2))
    single = gemv_fp8pt_indexed_pinned if pinned else gemv_fp8pt_indexed
    fused = gemv_fp8pt_indexed_pinned_gateup if pinned else gemv_fp8pt_indexed_gateup
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
def test_gemv_fp8pt_sparse_thr0_bitwise_and_masked():
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8pt_indexed, gemv_fp8pt_indexed_sparse

    N, K, k_rows, M = 128, 256, 160, 2
    codes, scales, row_off, kidx, wref = _store(N, K, k_rows, False, seed=108)
    x, ids = _inputs(M, K, False, False, seed=109)
    stream = torch.cuda.current_stream()
    total = E * k_rows
    w = torch.rand(M, TOPK, generator=torch.Generator().manual_seed(111)).cuda()
    sp_dense, sp_masked = _sparse_spec(E, total, 110, 0.0), _sparse_spec(E, total, 110, 0.5)

    dense = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv_fp8pt_indexed(x.cuda(), ids.cuda(), codes, scales, row_off, kidx, dense, 0, False, stream)
    sp0 = torch.zeros_like(dense)
    gemv_fp8pt_indexed_sparse(x.cuda(), ids.cuda(), w, codes, scales, row_off, kidx, sp0, sp_dense,
                              0, False, stream)
    torch.cuda.synchronize()
    assert torch.equal(dense, sp0)

    thr = 0.5
    masked = torch.zeros_like(dense)
    gemv_fp8pt_indexed_sparse(x.cuda(), ids.cuda(), w, codes, scales, row_off, kidx, masked, sp_masked,
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
    assert not torch.equal(masked, dense)


@cuda_required
def test_gemv_fp8pt_equals_b128_with_uniform_scales():
    """pt 스토어 ≡ 같은 스칼라를 모든 블록에 채운 b128 스토어 (exact 픽스처 → 비트일치).
    두 판이 같은 가중치를 같은 값으로 읽는다는 교차검증이고, 배율 위치(끝 1회 vs 청크마다)가
    2^n 배율에서 결과를 바꾸지 않음을 함께 확인한다."""
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8_indexed, gemv_fp8pt_indexed

    N, K, k_rows, M = 256, 512, 256, 3  # b128이 받는 128 정렬 기하
    g = torch.Generator().manual_seed(112)
    cs, ss, kidx = [], [], []
    for _ in range(E):
        c_ck, s_ck = random_expert_ckpt_pt(N, K, g, exact=True)
        rows = (torch.randperm(K // 128, generator=g)[: k_rows // 128].sort().values[:, None] * 128
                + torch.arange(128)[None, :]).reshape(-1)
        cs.append(row_store_pt(c_ck, rows)); ss.append(s_ck); kidx.append(rows)
    codes = torch.cat(cs).cuda()
    scales_pt = torch.cat(ss).cuda()
    scales_blk = scales_pt.repeat_interleave(k_rows // 128)[:, None].expand(-1, N // 128).contiguous()
    row_off = (torch.arange(E + 1, dtype=torch.int32) * k_rows).cuda()
    kidx_t = torch.cat(kidx).to(torch.uint16).cuda()
    x, ids = _inputs(M, K, True, False, seed=113)
    stream = torch.cuda.current_stream()
    a = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros_like(a)
    gemv_fp8_indexed(x.cuda(), ids.cuda(), codes, scales_blk, row_off, kidx_t, a, 0, False, stream)
    gemv_fp8pt_indexed(x.cuda(), ids.cuda(), codes, scales_pt, row_off, kidx_t, b, 0, False, stream)
    torch.cuda.synchronize()
    assert torch.equal(a, b)


@cuda_required
def test_gemv_fp8pt_rejects_block_scale_shape():
    """배율 모양 계약: pt 진입점에 블록표를 주면 즉사 (조용히 남의 값을 읽지 않는다)."""
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8pt_indexed

    N, K, k_rows, M = 128, 256, 128, 1
    codes, scales, row_off, kidx, _ = _store(N, K, k_rows, True, seed=114)
    x, ids = _inputs(M, K, True, False, seed=115)
    out = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    bad = torch.ones(E * k_rows // 128, N // 128, device="cuda")
    with pytest.raises(Exception):
        gemv_fp8pt_indexed(x.cuda(), ids.cuda(), codes, bad, row_off, kidx, out, 0, False,
                           torch.cuda.current_stream())
        torch.cuda.synchronize()


@cuda_required
def test_gemv_fp8pt_non_pow2_scale():
    """임의 fp32 배율 — GEMV는 fp32 부분합 × fp32 배율이라 가중치 반올림이 없다. fp32 참조와
    항 크기 합 대비 1e-5 안에서 일치해야 한다 (Mistral 체크포인트 배율은 2^n이 아니다)."""
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8pt_indexed

    N, K, k_rows, M = 256, 512, 160, 3
    codes, scales, row_off, kidx, wref = _store(N, K, k_rows, False, seed=116, pow2_scale=False)
    x, ids = _inputs(M, K, False, False, seed=117)
    out = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv_fp8pt_indexed(x.cuda(), ids.cuda(), codes, scales, row_off, kidx, out, 0, False,
                       torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref, mag = _ref(x, ids, wref, kidx.cpu(), row_off.cpu(), False, with_mag=True)
    # 출력이 bf16이라 |ref|의 2⁻⁸까지는 출력 반올림, 나머지는 누산 순서(fp32) 몫.
    err = (out.float().cpu() - ref).abs()
    assert torch.all(err <= ref.abs() * 2 ** -7 + mag * 1e-5 + 1e-3), float((err / (mag + 1e-9)).max())


# ─── grouped GEMM (prefill 형태) ───────────────────────────────────────────────
@cuda_required
@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("m", [8, 300])
@pytest.mark.parametrize("k_rows", [256, 160])  # 160: kr % 64 == 32 부분 K 타일
def test_grouped_fp8pt_matches_gemv(pair, exact, m, k_rows):
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8pt_indexed
    from sglang.jit_kernel.prism_grouped_fp8 import grouped_fp8pt_indexed, grouped_fp8pt_indexed_pinned
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    N, K = 256, 512
    codes, scales, row_off, kidx, wref = _store(N, K, k_rows, exact, seed=121)
    x, ids = _inputs(m, K, exact, pair, seed=122 + m)
    if m >= 64:
        ids[:, 0] = 3
        ids[ids == 5] = 6
    stream = torch.cuda.current_stream()
    ref = torch.zeros(m, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv_fp8pt_indexed(x.cuda(), ids.cuda(), codes, scales, row_off, kidx, ref, 0, pair, stream)
    grouping = build_grouping(ids.cuda(), E)
    out = torch.zeros_like(ref)
    grouped_fp8pt_indexed(x.cuda(), grouping, codes, scales, row_off, kidx, out, 0, pair, stream)
    outp = torch.zeros_like(ref)
    grouped_fp8pt_indexed_pinned(x.cuda(), grouping, codes.cpu().pin_memory(),
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
def test_grouped_fp8pt_gateup_fused():
    from sglang.jit_kernel.prism_grouped_fp8 import grouped_fp8pt_indexed, grouped_fp8pt_indexed_gateup
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    N, K, k_rows, m = 128, 256, 96, 40
    c1, s1, ro1, ki1, _ = _store(N, K, k_rows, True, seed=123)
    c2, s2, ro2, ki2, _ = _store(N, K, k_rows, True, seed=124)
    x, ids = _inputs(m, K, True, False, seed=125)
    stream = torch.cuda.current_stream()
    grouping = build_grouping(ids.cuda(), E)
    ref = torch.zeros(m, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
    grouped_fp8pt_indexed(x.cuda(), grouping, c1, s1, ro1, ki1, ref, 0, False, stream)
    grouped_fp8pt_indexed(x.cuda(), grouping, c2, s2, ro2, ki2, ref, N, False, stream)
    out = torch.zeros_like(ref)
    grouped_fp8pt_indexed_gateup(x.cuda(), grouping, c1, s1, ro1, ki1, c2, s2, ro2, ki2, out,
                                 0, N, False, stream)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@cuda_required
@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("k_rows", [256, 160])
def test_grouped_fp8pt_cold_tile_matches_gemv(exact, k_rows):
    """KT_TILE8_PT 로더: kt `GemmKernelTileK2FP8PT::BufferB` slab(코드 타일 + 64 B 배율 슬롯)을
    제자리 읽어 GEMV와 일치. k_rows=160은 32 정렬(128 아님)."""
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8pt_indexed
    from sglang.jit_kernel.prism_grouped_fp8 import grouped_fp8_cold
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    N, K, m = 256, 512, 40
    g = torch.Generator().manual_seed(131)
    blocks, cs, ss, kidx = [], [], [], []
    for _ in range(E):
        c_ck, s_ck = random_expert_ckpt_pt(N, K, g, exact=exact)
        rows = aligned_index_pt(K, k_rows, g)
        blocks.append(tile_block_pt(c_ck, s_ck, rows))
        cs.append(row_store_pt(c_ck, rows)); ss.append(s_ck); kidx.append(rows)
    slab = torch.cat(blocks).pin_memory()
    blk_off = torch.tensor([0] + list(torch.cumsum(torch.tensor([b.numel() for b in blocks]), 0)[:-1]),
                           dtype=torch.int64).cuda()
    row_off = (torch.arange(E + 1, dtype=torch.int32) * k_rows).cuda()
    kidx_t = torch.cat(kidx).to(torch.uint16).cuda()
    codes, scales = torch.cat(cs).cuda(), torch.cat(ss).cuda()

    x, ids = _inputs(m, K, exact, False, seed=132)
    stream = torch.cuda.current_stream()
    ref = torch.zeros(m, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv_fp8pt_indexed(x.cuda(), ids.cuda(), codes, scales, row_off, kidx_t, ref, 0, False, stream)

    class _Slab:
        pass

    cold = _Slab()
    cold.slab, cold.blk_off, cold.row_off, cold.k_index = slab, blk_off, row_off, kidx_t
    cold.n, cold.n_start, cold.layout = N, 0, "kt_tile8pt"
    out = torch.zeros_like(ref)
    grouped_fp8_cold(x.cuda(), build_grouping(ids.cuda(), E), cold, out, 0, False, stream)
    torch.cuda.synchronize()
    if exact:
        assert torch.equal(out, ref)
    else:
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


@cuda_required
def test_grouped_fp8pt_non_pow2_scale_within_bf16_weight_rounding():
    """임의 배율에서 grouped는 `bf16(code·scale)`을 텐서코어에 넣으므로(.cuh 수치 계약) GEMV와
    가중치당 상대 2⁻⁹ 반올림만큼 다르다. 항 크기 합 × 2⁻⁸ 안이어야 하고, 그 이상이면 배율 위치가
    틀린 것이다 (틀리면 상대 오차가 배율 크기만큼 난다)."""
    from sglang.jit_kernel.prism_gemv_fp8 import gemv_fp8pt_indexed
    from sglang.jit_kernel.prism_grouped_fp8 import grouped_fp8pt_indexed
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    N, K, k_rows, m = 256, 512, 160, 40
    codes, scales, row_off, kidx, wref = _store(N, K, k_rows, False, seed=141, pow2_scale=False)
    x, ids = _inputs(m, K, False, False, seed=142)
    stream = torch.cuda.current_stream()
    ref = torch.zeros(m, TOPK, N, dtype=torch.bfloat16, device="cuda")
    gemv_fp8pt_indexed(x.cuda(), ids.cuda(), codes, scales, row_off, kidx, ref, 0, False, stream)
    out = torch.zeros_like(ref)
    grouped_fp8pt_indexed(x.cuda(), build_grouping(ids.cuda(), E), codes, scales, row_off, kidx, out,
                          0, False, stream)
    torch.cuda.synchronize()
    fref, mag = _ref(x, ids, wref, kidx.cpu(), row_off.cpu(), False, with_mag=True)
    err = (out.float().cpu() - fref).abs()
    bound = mag * 2 ** -8 + fref.abs() * 2 ** -7 + 1e-3
    assert torch.all(err <= bound), float((err / (mag + 1e-9)).max())
    # 그리고 GEMV와도 같은 한도 안 (둘 다 같은 스칼라를 곱했다는 뜻)
    err2 = (out.float() - ref.float()).abs().cpu()
    assert torch.all(err2 <= bound)
