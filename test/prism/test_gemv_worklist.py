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
