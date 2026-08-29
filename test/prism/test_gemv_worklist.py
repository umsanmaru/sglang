"""gemv_worklist 커널 단위 테스트 — 계약 ⑤ 이원화(tolerance + 정확표현 비트일치)."""
import pytest
import torch

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _ref(x2d, topk_ids, w, k_offset, x_row_is_pair):
    """fp32 레퍼런스: out[m,j,:] = x_row · W[e]  (누산 전부 fp32)."""
    M, k = topk_ids.shape
    E, k_rows, N = w.shape
    out = torch.empty(M, k, N, dtype=torch.float32)
    xc, wc, ids = x2d.float().cpu(), w.float().cpu(), topk_ids.cpu()
    for m in range(M):
        for j in range(k):
            row = m * k + j if x_row_is_pair else m
            e = int(ids[m, j])
            out[m, j] = xc[row, k_offset:k_offset + k_rows] @ wc[e]
    return out


def _mk(M=2, k=4, E=16, k_rows=64, N=96, Kx=128, exact=False, x_row_is_pair=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    Rx = M * k if x_row_is_pair else M
    if exact:  # 작은 정수 — bf16/fp32 라운딩 무손실 → 비트일치 요구 가능
        x = torch.randint(-4, 5, (Rx, Kx), generator=g).to(torch.bfloat16)
        w = torch.randint(-4, 5, (E, k_rows, N), generator=g).to(torch.bfloat16)
    else:
        x = torch.randn(Rx, Kx, generator=g).to(torch.bfloat16)
        w = (torch.randn(E, k_rows, N, generator=g) * 0.1).to(torch.bfloat16)
    ids = torch.randint(0, E, (M, k), generator=g)  # 중복 expert 자연 발생
    return x, w, ids


@cuda_required
@pytest.mark.parametrize("x_row_is_pair", [False, True])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_gemv_worklist_device_tolerance(x_row_is_pair, ids_dtype):
    from sglang.jit_kernel.prism_gemv import gemv_worklist
    x, w, ids = _mk(x_row_is_pair=x_row_is_pair)
    xd, wd = x.cuda(), w.cuda()
    idsd = ids.to(ids_dtype).cuda()
    out = torch.zeros(2, 4, 96, dtype=torch.bfloat16, device="cuda")
    gemv_worklist(xd, idsd, wd, out, 32, 0, x_row_is_pair, torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = _ref(x, ids, w, 32, x_row_is_pair)
    torch.testing.assert_close(out.float().cpu(), ref, rtol=2e-2, atol=2e-2)


@cuda_required
def test_gemv_worklist_device_exact_ints_bitwise():
    from sglang.jit_kernel.prism_gemv import gemv_worklist
    x, w, ids = _mk(exact=True)
    out = torch.zeros(2, 4, 96, dtype=torch.bfloat16, device="cuda")
    gemv_worklist(x.cuda(), ids.int().cuda(), w.cuda(), out, 0, 0, False,
                  torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = _ref(x, ids, w, 0, False).to(torch.bfloat16)  # 무손실 입력 → 순서 무관 비트일치
    assert torch.equal(out.cpu(), ref)


@cuda_required
def test_gemv_worklist_out_col_offset_slice():
    """gate/up이 [M,k,2I]의 열 반쪽에 각자 쓰는 시나리오."""
    from sglang.jit_kernel.prism_gemv import gemv_worklist
    x, w, ids = _mk(exact=True, N=96)
    I = 96
    gu = torch.zeros(2, 4, 2 * I, dtype=torch.bfloat16, device="cuda")
    gemv_worklist(x.cuda(), ids.int().cuda(), w.cuda(), gu, 0, 0, False,
                  torch.cuda.current_stream())     # gate → [:, :, :I]
    gemv_worklist(x.cuda(), ids.int().cuda(), w.cuda(), gu, 32, I, False,
                  torch.cuda.current_stream())     # up   → [:, :, I:]
    torch.cuda.synchronize()
    assert torch.equal(gu[:, :, :I].cpu(), _ref(x, ids, w, 0, False).to(torch.bfloat16))
    assert torch.equal(gu[:, :, I:].cpu(), _ref(x, ids, w, 32, False).to(torch.bfloat16))


@cuda_required
def test_gemv_worklist_pinned_matches_device():
    """W가 pinned CPU(UVA)여도 device 변형과 비트 동일 — warm 티어 경로."""
    from sglang.jit_kernel.prism_gemv import gemv_worklist, gemv_worklist_pinned
    x, w, ids = _mk(exact=True)
    out_d = torch.zeros(2, 4, 96, dtype=torch.bfloat16, device="cuda")
    out_p = torch.zeros_like(out_d)
    s = torch.cuda.current_stream()
    gemv_worklist(x.cuda(), ids.int().cuda(), w.cuda(), out_d, 0, 0, False, s)
    gemv_worklist_pinned(x.cuda(), ids.int().cuda(), w.pin_memory(), out_p, 0, 0, False, s)
    torch.cuda.synchronize()
    assert torch.equal(out_d.cpu(), out_p.cpu())


@cuda_required
def test_gemv_worklist_pinned_rejects_device_src():
    from sglang.jit_kernel.prism_gemv import gemv_worklist_pinned
    x, w, ids = _mk()
    out = torch.zeros(2, 4, 96, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(Exception):
        gemv_worklist_pinned(x.cuda(), ids.int().cuda(), w.cuda(), out, 0, 0, False,
                             torch.cuda.current_stream())


@cuda_required
def test_gemv_worklist_perf_smoke():
    """35B h375 hot gate 치수(g=8, 768x512)에서 (gather+bmm) 대비 과도한
    회귀가 없는지 — 상한 2배의 러프 가드 (튜닝 회귀 감지용, 마이크로벤치 아님)."""
    import time
    from sglang.jit_kernel.prism_gemv import gemv_worklist
    E, k_rows, N, M, k = 64, 768, 512, 1, 8
    w = torch.randn(E, k_rows, N).to(torch.bfloat16).cuda()
    x = torch.randn(M, 2048).to(torch.bfloat16).cuda()
    ids = torch.randint(0, E, (M, k)).int().cuda()
    out = torch.zeros(M, k, N, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.current_stream()

    def bench(fn, iters=200):
        for _ in range(20):
            fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1e6

    t_wl = bench(lambda: gemv_worklist(x, ids, w, out, 0, 0, False, s))
    sel = ids.view(-1).long()
    def ref():
        wg = w.index_select(0, sel)
        torch.bmm(x[:, :k_rows].unsqueeze(0).expand(k, -1, -1), wg)
    t_ref = bench(ref)
    print(f"worklist {t_wl:.1f}us vs gather+bmm {t_ref:.1f}us")
    assert t_wl < 2.0 * t_ref


# ── 인덱스 변형 (계약 ① 2026-08-25: 밴드 → 가변 per-expert 인덱스) ──────────


def _ref_indexed(x2d, topk_ids, w_flat, row_off, kidx, x_row_is_pair):
    """fp32 레퍼런스: out[m,j,:] = x_row[kidx[o0:o1]] · W_flat[o0:o1]."""
    M, k = topk_ids.shape
    N = w_flat.shape[1]
    out = torch.zeros(M, k, N, dtype=torch.float32)
    xc, wc = x2d.float().cpu(), w_flat.float().cpu()
    off, idx, ids = row_off.cpu().long(), kidx.cpu().long(), topk_ids.cpu()
    for m in range(M):
        for j in range(k):
            row = m * k + j if x_row_is_pair else m
            e = int(ids[m, j])
            o0, o1 = int(off[e]), int(off[e + 1])
            if o1 > o0:
                out[m, j] = xc[row, idx[o0:o1]] @ wc[o0:o1]
    return out


def _mk_indexed(k_per_expert, N=96, Kx=128, M=2, k=4, exact=True, seed=0,
                x_row_is_pair=False, shuffle=True):
    """expert마다 다른 행 수 + (선택) 셔플된 열 인덱스로 flat 스토어를 만든다."""
    g = torch.Generator().manual_seed(seed)
    E = len(k_per_expert)
    Rx = M * k if x_row_is_pair else M
    total = sum(k_per_expert)
    if exact:
        x = torch.randint(-4, 5, (Rx, Kx), generator=g).to(torch.bfloat16)
        w_flat = torch.randint(-4, 5, (total, N), generator=g).to(torch.bfloat16)
    else:
        x = torch.randn(Rx, Kx, generator=g).to(torch.bfloat16)
        w_flat = (torch.randn(total, N, generator=g) * 0.1).to(torch.bfloat16)
    rows = []
    for kr in k_per_expert:
        cols = torch.randperm(Kx, generator=g)[:kr] if shuffle else torch.arange(kr)
        rows.append(cols)
    kidx = torch.cat(rows).to(torch.uint16) if rows else torch.empty(0, dtype=torch.uint16)
    row_off = torch.zeros(E + 1, dtype=torch.int32)
    for e, kr in enumerate(k_per_expert):
        row_off[e + 1] = row_off[e] + kr
    ids = torch.randint(0, E, (M, k), generator=g)
    return x, w_flat, row_off, kidx, ids


@cuda_required
@pytest.mark.parametrize("x_row_is_pair", [False, True])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_indexed_contiguous_matches_band_bitwise(x_row_is_pair, ids_dtype):
    """전환기의 합격 기준: 연속 인덱스는 밴드 경로와 **비트일치**여야 한다.

    같은 원소를 같은 순서로 누산하므로 성립한다 — 이 등호가 있어야 S5~S7이
    밴드 경로를 기준으로 검증될 수 있다.
    """
    from sglang.jit_kernel.prism_gemv import gemv_worklist, gemv_worklist_indexed
    E, kr, N, Kx, M, k, off = 8, 64, 96, 128, 2, 4, 32
    x, w, ids = _mk(M=M, k=k, E=E, k_rows=kr, N=N, Kx=Kx, exact=True,
                    x_row_is_pair=x_row_is_pair)
    xd, idsd = x.cuda(), ids.to(ids_dtype).cuda()
    s = torch.cuda.current_stream()

    band = torch.zeros(M, k, N, dtype=torch.bfloat16, device="cuda")
    gemv_worklist(xd, idsd, w.cuda(), band, off, 0, x_row_is_pair, s)

    w_flat = w.reshape(E * kr, N).contiguous().cuda()
    row_off = (torch.arange(E + 1) * kr).to(torch.int32).cuda()
    kidx = torch.arange(off, off + kr).repeat(E).to(torch.uint16).cuda()
    idx_out = torch.zeros_like(band)
    gemv_worklist_indexed(xd, idsd, w_flat, row_off, kidx, idx_out, 0, x_row_is_pair, s)
    torch.cuda.synchronize()
    assert torch.equal(band.cpu(), idx_out.cpu())


@cuda_required
@pytest.mark.parametrize("x_row_is_pair", [False, True])
def test_indexed_shuffled_matches_reference(x_row_is_pair):
    """셔플 인덱스 — 밴드로는 표현 불가능한 구성. 좌표가 뒤섞이면 여기서 걸린다."""
    from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed
    x, w_flat, row_off, kidx, ids = _mk_indexed(
        [64] * 8, x_row_is_pair=x_row_is_pair)
    out = torch.zeros(2, 4, 96, dtype=torch.bfloat16, device="cuda")
    gemv_worklist_indexed(x.cuda(), ids.int().cuda(), w_flat.cuda(),
                          row_off.cuda(), kidx.cuda(), out, 0, x_row_is_pair,
                          torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = _ref_indexed(x, ids, w_flat, row_off, kidx, x_row_is_pair)
    assert torch.equal(out.cpu(), ref.to(torch.bfloat16))


@cuda_required
def test_indexed_variable_k_per_expert():
    """expert마다 행 수가 다른 구성 — 가변 K가 이 표현의 존재 이유다."""
    from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed
    k_per = [8, 64, 32, 128, 16, 64, 96, 4]
    x, w_flat, row_off, kidx, ids = _mk_indexed(k_per)
    out = torch.zeros(2, 4, 96, dtype=torch.bfloat16, device="cuda")
    gemv_worklist_indexed(x.cuda(), ids.int().cuda(), w_flat.cuda(),
                          row_off.cuda(), kidx.cuda(), out, 0, False,
                          torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = _ref_indexed(x, ids, w_flat, row_off, kidx, False)
    assert torch.equal(out.cpu(), ref.to(torch.bfloat16))


@cuda_required
def test_indexed_zero_rows_writes_zero():
    """k[e] == 0은 특례가 아니다 — 루프가 0회 돌고 0을 쓰는데, 그게 그 티어의
    정확한 부분합이다 ("이 expert는 이 티어에 없음")."""
    from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed
    k_per = [0, 64, 0, 32]
    x, w_flat, row_off, kidx, _ = _mk_indexed(k_per)
    ids = torch.tensor([[0, 1, 2, 3]])  # expert 0, 2가 빈 티어
    out = torch.full((1, 4, 96), 7.0, dtype=torch.bfloat16, device="cuda")
    gemv_worklist_indexed(x[:1].cuda(), ids.int().cuda(), w_flat.cuda(),
                          row_off.cuda(), kidx.cuda(), out, 0, False,
                          torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.equal(out[0, 0].cpu(), torch.zeros(96, dtype=torch.bfloat16))
    assert torch.equal(out[0, 2].cpu(), torch.zeros(96, dtype=torch.bfloat16))
    ref = _ref_indexed(x[:1], ids, w_flat, row_off, kidx, False)
    assert torch.equal(out.cpu(), ref.to(torch.bfloat16))


@cuda_required
def test_indexed_pinned_matches_device():
    """W가 pinned여도 device 변형과 비트 동일 — warm 티어의 제자리 UVA 읽기."""
    from sglang.jit_kernel.prism_gemv import (
        gemv_worklist_indexed, gemv_worklist_indexed_pinned,
    )
    x, w_flat, row_off, kidx, ids = _mk_indexed([32, 64, 96, 16])
    xd, idsd = x.cuda(), ids.int().cuda()
    ro, ki = row_off.cuda(), kidx.cuda()
    s = torch.cuda.current_stream()
    out_d = torch.zeros(2, 4, 96, dtype=torch.bfloat16, device="cuda")
    out_p = torch.zeros_like(out_d)
    gemv_worklist_indexed(xd, idsd, w_flat.cuda(), ro, ki, out_d, 0, False, s)
    gemv_worklist_indexed_pinned(xd, idsd, w_flat.pin_memory(), ro, ki, out_p, 0, False, s)
    torch.cuda.synchronize()
    assert torch.equal(out_d.cpu(), out_p.cpu())


@cuda_required
def test_indexed_out_col_offset_slice():
    """gate/up이 [M,k,2I]의 열 반쪽에 각자 쓰는 시나리오 — 인덱스도 서로 다르다."""
    from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed
    I = 96
    xg, wg, og, kg, ids = _mk_indexed([64] * 8, N=I, seed=1)
    xu, wu, ou, ku, _ = _mk_indexed([48] * 8, N=I, seed=2)
    s = torch.cuda.current_stream()
    gu = torch.zeros(2, 4, 2 * I, dtype=torch.bfloat16, device="cuda")
    gemv_worklist_indexed(xg.cuda(), ids.int().cuda(), wg.cuda(), og.cuda(),
                          kg.cuda(), gu, 0, False, s)
    gemv_worklist_indexed(xg.cuda(), ids.int().cuda(), wu.cuda(), ou.cuda(),
                          ku.cuda(), gu, I, False, s)
    torch.cuda.synchronize()
    assert torch.equal(gu[:, :, :I].cpu(),
                       _ref_indexed(xg, ids, wg, og, kg, False).to(torch.bfloat16))
    assert torch.equal(gu[:, :, I:].cpu(),
                       _ref_indexed(xg, ids, wu, ou, ku, False).to(torch.bfloat16))


@cuda_required
@pytest.mark.parametrize("broken", ["kidx_dtype", "kidx_len", "row_off_host", "row_off_short"])
def test_indexed_guards(broken):
    """오프셋 테이블과 스토어가 어긋나면 조용히 남의 행을 읽는다 — 즉사시킨다."""
    from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed
    x, w_flat, row_off, kidx, ids = _mk_indexed([32, 64])
    xd, idsd = x.cuda(), ids.int().cuda()
    wd, ro, ki = w_flat.cuda(), row_off.cuda(), kidx.cuda()
    if broken == "kidx_dtype":
        ki = kidx.to(torch.int32).cuda()
    elif broken == "kidx_len":
        ki = kidx[:-2].cuda()          # w_flat 행 수와 불일치
    elif broken == "row_off_host":
        ro = row_off                    # device 상주여야 한다
    elif broken == "row_off_short":
        ro = row_off[:1].cuda()
    out = torch.zeros(2, 4, 96, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(Exception):
        gemv_worklist_indexed(xd, idsd, wd, ro, ki, out, 0, False,
                              torch.cuda.current_stream())


@cuda_required
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_gemv_gateup_fusion_all_four_bitwise(pinned, sparse):
    """융합 진입점 네 조합({device, pinned} × {dense, sparse})이 2회 launch와 비트일치.

    warm dense와 hot sparse는 2026-08-29에 채운 조합이다 — 그 전에는 `gemv_gateup`이
    None을 돌려주어 같은 스텝이 조용히 2회 launch로 떨어졌다 (mxfp4/fp8은 네 개 다 있었다).
    """
    from sglang.jit_kernel import prism_gemv as k
    from sglang.srt.layers.moe.prism.formats import BF16
    from sglang.srt.layers.moe.prism.tiers import SparseSpec

    I = 96
    xg, wg, og, kg, ids = _mk_indexed([64] * 8, N=I, exact=False, seed=21)
    _, wu, ou, ku, _ = _mk_indexed([64] * 8, N=I, exact=False, seed=22)
    x, ids_d, s = xg.cuda(), ids.int().cuda(), torch.cuda.current_stream()
    store = (lambda t: t.cpu().pin_memory()) if pinned else (lambda t: t.cuda())
    wgd, wud = store(wg), store(wu)
    ogd, kgd, oud, kud = og.cuda(), kg.cuda(), ou.cuda(), ku.cuda()

    def spec(seed):
        g = torch.Generator().manual_seed(seed)
        total = 8 * 64
        return SparseSpec(a=torch.rand(total, generator=g).cuda(),
                          c=(torch.rand(total // 2, generator=g) * 0.1).cuda(),
                          thr=torch.full((8, 4), 0.3, device="cuda"),
                          p=0.5, lam=0.0, pmax=0.9, grid=0.1, ng=4, renorm_it=1)

    w = torch.rand(2, 4, generator=torch.Generator().manual_seed(23)).cuda()
    sg, su = spec(24), spec(25)
    single = BF16.gemv(pinned=pinned, sparse=sparse)
    fused = BF16.gemv_gateup(pinned=pinned, sparse=sparse)
    assert fused is not None
    ref = torch.zeros(2, 4, 2 * I, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros_like(ref)
    if sparse:
        single(x, ids_d, w, wgd, ogd, kgd, ref, sg, 0, False, s)
        single(x, ids_d, w, wud, oud, kud, ref, su, I, False, s)
        fused(x, ids_d, w, wgd, ogd, kgd, wud, oud, kud, out, sg, su, 0, I, False, s)
    else:
        single(x, ids_d, wgd, ogd, kgd, ref, 0, False, s)
        single(x, ids_d, wud, oud, kud, ref, I, False, s)
        fused(x, ids_d, wgd, ogd, kgd, wud, oud, kud, out, 0, I, False, s)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
