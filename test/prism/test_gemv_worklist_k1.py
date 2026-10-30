"""bf16 worklist GEMV의 **score k1(per-k)** 경로 — `*_sparsek1` 진입점 (CUDA 필요).

fp8 판(`test_fp8_kernels.py::test_gemv_fp8_sparsek1_*`)과 같은 불변식을 bf16에서 본다.
per-k 마스킹이 조용히 틀리는 방식이 둘 있고, 각각을 한 테스트가 잡는다:

  1. **마스크가 안 걸린다** (thr을 잘못 읽거나 keep 배열 첨자가 어긋남) → thr을 키워도
     결과가 dense와 같다. `test_sparsek1_masks_what_it_says`가 keep 비율과 값 둘 다 본다.
  2. **엉뚱한 k를 죽인다** (페어 첨자 `r>>1`을 그대로 씀) → 값이 "그럴듯하게 나쁜" 것이
     되어 tolerance 뒤에 숨는다. 그래서 레퍼런스를 **마스킹된 dense**로 두고 **비트일치**를
     요구한다 — 죽은 k는 dense에서 x=0이라 곱이 0이고, 커널은 그 행의 W 로드를 건너뛸
     뿐 누산 순서를 바꾸지 않으므로 두 계산의 항과 순서가 같다.

`thr=0`은 전량 통과이므로 dense와 비트일치해야 한다 (계약 ⑤의 p=0 게이트가 GPU에서
성립하는 자리).
"""

import pytest
import torch

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

M, TOPK, E, N, KX = 3, 2, 4, 96, 256
NG = 8


def _spec(thr: float, per_k: bool = True, a=None):
    """dense 레인과 같은 필드 이름의 spec (커널 래퍼는 덕 타이핑으로 읽는다).

    `a`를 주면 score k1w1 (per-k × 열 노름)이고, 안 주면 순수 k1이다 — 래퍼도 커널도
    같은 규칙으로 가른다 (`a`의 유무).
    """
    from sglang.srt.layers.moe.prism.tiers import SparseSpec

    tab = torch.full((E, NG), float(thr), dtype=torch.float32).cuda()
    # per-k에서 a는 **k1w1일 때만** 채운다 — a의 유무가 곧 점수 선택이다.
    return SparseSpec(a=a, c=None, thr=tab, p=0.5, lam=0.0,
                      pmax=0.9, grid=0.1, ng=NG, renorm_it=1, per_k=per_k)


def _store(seed=0, k_rows=128):
    """expert마다 같은 행 수, 셔플된 열 인덱스의 flat 스토어 + 정확표현 입력."""
    g = torch.Generator().manual_seed(seed)
    total = E * k_rows
    x = torch.randint(-4, 5, (M, KX), generator=g).to(torch.bfloat16)
    w = torch.randint(-4, 5, (total, N), generator=g).to(torch.bfloat16)
    kidx = torch.cat([torch.randperm(KX, generator=g)[:k_rows] for _ in range(E)])
    row_off = torch.arange(0, total + 1, k_rows, dtype=torch.int32)
    ids = torch.randint(0, E, (M, TOPK), generator=g)
    return (x.cuda(), w.cuda(), row_off.cuda(), kidx.to(torch.uint16).cuda(), ids.cuda(),
            torch.rand(M, TOPK, generator=g).cuda())


def _masked_x(x, thr):
    """|x| < thr 인 열을 0으로. bf16 왕복이 무손실이라 항이 정확히 사라진다."""
    xf = x.float()
    return torch.where(xf.abs() >= thr, xf, torch.zeros_like(xf)).to(torch.bfloat16)


@cuda_required
@pytest.mark.parametrize("pinned", [False, True])
def test_sparsek1_thr0_is_dense_bitwise(pinned):
    from sglang.jit_kernel.prism_gemv import (
        gemv_worklist_indexed, gemv_worklist_indexed_pinned,
        gemv_worklist_indexed_pinned_sparse, gemv_worklist_indexed_sparse,
    )

    x, w, row_off, kidx, ids, wts = _store(seed=11)
    if pinned:
        w = w.cpu().pin_memory()
    dense_fn = gemv_worklist_indexed_pinned if pinned else gemv_worklist_indexed
    sparse_fn = gemv_worklist_indexed_pinned_sparse if pinned else gemv_worklist_indexed_sparse
    s = torch.cuda.current_stream()

    dense = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    dense_fn(x, ids, w, row_off, kidx, dense, 0, False, s)
    got = torch.zeros_like(dense)
    sparse_fn(x, ids, wts, w, row_off, kidx, got, _spec(0.0), 0, False, s)
    torch.cuda.synchronize()
    assert torch.equal(dense, got)


@cuda_required
@pytest.mark.parametrize("pinned", [False, True])
def test_sparsek1_masks_what_it_says(pinned):
    """thr>0 ↔ 마스킹된 dense와 **비트일치**, 그리고 실제로 일부만 산다."""
    from sglang.jit_kernel.prism_gemv import (
        gemv_worklist_indexed, gemv_worklist_indexed_pinned,
        gemv_worklist_indexed_pinned_sparse, gemv_worklist_indexed_sparse,
    )

    x, w, row_off, kidx, ids, wts = _store(seed=21)
    w_dense = w
    if pinned:
        w = w.cpu().pin_memory()
    dense_fn = gemv_worklist_indexed_pinned if pinned else gemv_worklist_indexed
    sparse_fn = gemv_worklist_indexed_pinned_sparse if pinned else gemv_worklist_indexed_sparse
    s = torch.cuda.current_stream()

    thr = 2.5                       # 정수 입력 ±4 → |x| ∈ {0..4}, 3·4만 산다
    keep = (x.float().abs() >= thr)
    assert 0 < int(keep.sum()) < keep.numel(), "마스크가 전부/전무면 아무것도 못 잡는다"

    got = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    sparse_fn(x, ids, wts, w, row_off, kidx, got, _spec(thr), 0, False, s)
    ref = torch.zeros_like(got)
    dense_fn(_masked_x(x, thr), ids, w if pinned else w_dense, row_off, kidx, ref, 0, False, s)
    torch.cuda.synchronize()
    assert torch.equal(got, ref)
    # dense와는 달라야 한다 (안 그러면 마스킹이 실제로는 안 걸린 것이다)
    full = torch.zeros_like(got)
    dense_fn(x, ids, w if pinned else w_dense, row_off, kidx, full, 0, False, s)
    torch.cuda.synchronize()
    assert not torch.equal(got, full)


@cuda_required
def test_sparsek1_gateup_fusion_matches_two_launches():
    """warm의 실제 경로(gate|up 융합)가 per-k 2회 launch와 비트일치. thr은 서로 다르게."""
    from sglang.jit_kernel.prism_gemv import (
        gemv_worklist_indexed_pinned_sparse, gemv_worklist_indexed_pinned_sparse_gateup,
        gemv_worklist_indexed_sparse, gemv_worklist_indexed_sparse_gateup,
    )

    xg, wg, rog, kig, ids, wts = _store(seed=31)
    _, wu, rou, kiu, _, _ = _store(seed=32)
    s = torch.cuda.current_stream()
    spg, spu = _spec(1.5), _spec(3.5)
    for single, fused, pin in ((gemv_worklist_indexed_sparse,
                                gemv_worklist_indexed_sparse_gateup, False),
                               (gemv_worklist_indexed_pinned_sparse,
                                gemv_worklist_indexed_pinned_sparse_gateup, True)):
        w1 = wg.cpu().pin_memory() if pin else wg
        w2 = wu.cpu().pin_memory() if pin else wu
        two = torch.zeros(M, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
        single(xg, ids, wts, w1, rog, kig, two, spg, 0, False, s)
        single(xg, ids, wts, w2, rou, kiu, two, spu, N, False, s)
        one = torch.zeros_like(two)
        fused(xg, ids, wts, w1, rog, kig, w2, rou, kiu, one, spg, spu, 0, N, False, s)
        torch.cuda.synchronize()
        assert torch.equal(one, two), f"pinned={pin}"


@cuda_required
def test_sparsek1_and_pair_masks_differ():
    """같은 thr에서 per-k와 페어 마스크는 **다른 마스크**다 — 섞이면 조용히 틀린다."""
    from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed_sparse

    x, w, row_off, kidx, ids, wts = _store(seed=41)
    s = torch.cuda.current_stream()
    a = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros_like(a)
    gemv_worklist_indexed_sparse(x, ids, wts, w, row_off, kidx, a, _spec(2.5, per_k=True),
                                 0, False, s)
    # 페어 판은 a=wn²=1, c=0이므로 점수가 sqrt(x0²+x1²) — 같은 thr에서 더 많이 산다.
    from sglang.srt.layers.moe.prism.tiers import SparseSpec

    total = int(row_off[-1])
    pair = SparseSpec(a=torch.ones(total, device="cuda"),
                      c=torch.zeros(total // 2, device="cuda"),
                      thr=torch.full((E, NG), 2.5, device="cuda"),
                      p=0.5, lam=0.0, pmax=0.9, grid=0.1, ng=NG, renorm_it=1, per_k=False)
    gemv_worklist_indexed_sparse(x, ids, wts, w, row_off, kidx, b, pair, 0, False, s)
    torch.cuda.synchronize()
    assert not torch.equal(a, b)


# ── score k1w1 (per-k × 열 노름) ───────────────────────────────────────────


@cuda_required
@pytest.mark.parametrize("pinned", [False, True])
def test_sparsek1w_matches_weighted_masked_dense(pinned):
    """`|x_k|·wn_k >= thr` ↔ 그 마스크로 x를 0으로 둔 dense와 **비트일치**.

    a(=wn²)는 weight와 같은 오프셋 표를 쓰므로, 인덱스가 어긋나면 엉뚱한 k가 죽고
    비트일치가 깨진다 — 이 테스트가 그 대응을 잡는다.
    """
    from sglang.jit_kernel.prism_gemv import (
        gemv_worklist_indexed, gemv_worklist_indexed_pinned,
        gemv_worklist_indexed_pinned_sparse, gemv_worklist_indexed_sparse,
    )

    x, w, row_off, kidx, ids, wts = _store(seed=51)
    w_dev = w
    if pinned:
        w = w.cpu().pin_memory()
    dense_fn = gemv_worklist_indexed_pinned if pinned else gemv_worklist_indexed
    sparse_fn = gemv_worklist_indexed_pinned_sparse if pinned else gemv_worklist_indexed_sparse
    s = torch.cuda.current_stream()

    g = torch.Generator().manual_seed(52)
    a = (torch.rand(int(row_off[-1]), generator=g) * 3.0 + 0.25).cuda()   # wn², 행 정렬
    e0 = int(ids[0, 0])
    o0, o1 = int(row_off[e0]), int(row_off[e0 + 1])
    rows0 = kidx[o0:o1].to(torch.int64)
    score = x[0, rows0].float().abs() * a[o0:o1].sqrt()
    flat = score.sort().values
    thr = float(0.5 * (flat[flat.numel() // 2] + flat[flat.numel() // 2 + 1]))

    got = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    sparse_fn(x, ids, wts, w, row_off, kidx, got, _spec(thr, a=a), 0, False, s)
    torch.cuda.synchronize()

    # 레퍼런스: (m, j)마다 그 expert의 a로 마스킹한 x를 dense에 태운다. x 행이 토큰이므로
    # expert마다 다른 마스크를 쓰려면 호출을 쪼개야 한다 — 한 슬롯씩 확인한다.
    for m in range(M):
        for j in range(TOPK):
            e = int(ids[m, j])
            oa, ob = int(row_off[e]), int(row_off[e + 1])
            rows = kidx[oa:ob].to(torch.int64)
            keep = x[m, rows].float().abs() * a[oa:ob].sqrt() >= thr
            xm = x[m].float().clone()
            drop = rows[~keep]
            xm[drop] = 0.0
            xm = xm.to(torch.bfloat16).unsqueeze(0).repeat(M, 1).contiguous()
            ref = torch.zeros_like(got)
            dense_fn(xm, ids, w if pinned else w_dev, row_off, kidx, ref, 0, False, s)
            torch.cuda.synchronize()
            assert torch.equal(got[m, j], ref[m, j]), f"m={m} j={j} pinned={pinned}"


@cuda_required
def test_sparsek1w_differs_from_plain_k1():
    """같은 thr에서 wn을 곱하면 다른 k가 죽는다 — 아니면 a가 안 읽힌 것이다."""
    from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed_sparse

    x, w, row_off, kidx, ids, wts = _store(seed=61)
    s = torch.cuda.current_stream()
    g = torch.Generator().manual_seed(62)
    a = (torch.rand(int(row_off[-1]), generator=g) * 3.0 + 0.25).cuda()
    plain = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device="cuda")
    weighted = torch.zeros_like(plain)
    gemv_worklist_indexed_sparse(x, ids, wts, w, row_off, kidx, plain, _spec(2.5), 0, False, s)
    gemv_worklist_indexed_sparse(x, ids, wts, w, row_off, kidx, weighted, _spec(2.5, a=a),
                                 0, False, s)
    torch.cuda.synchronize()
    assert not torch.equal(plain, weighted)
