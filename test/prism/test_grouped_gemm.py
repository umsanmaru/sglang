"""grouped GEMM(prefill 형태) 커널 — worklist GEMV와의 등가성 (계약 ⑤ 이원화).

grouped 커널은 pair를 expert로 묶어 W를 expert당 한 번 읽는다. 읽는 원소는
worklist와 같고 누산 순서만 다르므로, 정확표현 입력(작은 정수)에서는 **비트일치**,
일반 입력에서는 tolerance다. pinned(UVA) 쌍둥이는 device 변형과 비트일치해야 한다.
"""
import pytest
import torch

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

E, K = 16, 4


def _store(k_rows, n_cols, kx, exact, seed):
    g = torch.Generator().manual_seed(seed)
    if exact:
        w = torch.randint(-2, 3, (E * k_rows, n_cols), generator=g).to(torch.bfloat16)
    else:
        w = (torch.randn(E * k_rows, n_cols, generator=g) * 0.05).to(torch.bfloat16)
    row_off = (torch.arange(E + 1, dtype=torch.int32) * k_rows).cuda()
    # 셔플 인덱스 (계약 ⑤ 2026-08-25: 좌표 검출기는 순열 인덱스에서 성립해야 한다)
    kidx = torch.stack([torch.randperm(kx, generator=g)[:k_rows].sort().values
                        for _ in range(E)]).reshape(-1).to(torch.uint16).cuda()
    return w, row_off, kidx


def _inputs(m, kx, exact, pair, seed):
    g = torch.Generator().manual_seed(seed)
    rx = m * K if pair else m
    if exact:
        x = torch.randint(-2, 3, (rx, kx), generator=g).to(torch.bfloat16)
    else:
        x = (torch.randn(rx, kx, generator=g) * 0.1).to(torch.bfloat16)
    # 중복 expert가 필연이 되도록 E보다 훨씬 많은 pair (M·K ≫ E) — 타일이 여러 개인
    # expert(count > 128)와 pair가 하나도 없는 expert가 함께 나오게 한다.
    ids = torch.randint(0, E, (m, K), generator=g)
    if m >= 64:
        ids[:, 0] = 3          # expert 3에 pair 몰기 → count > TILE_M
        ids[ids == 5] = 6      # expert 5는 pair 0개
    return x.cuda(), ids.cuda()


def _run_pair(m, exact, pair, pinned, fused):
    from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed
    from sglang.jit_kernel.prism_grouped import (
        grouped_gemm_indexed, grouped_gemm_indexed_gateup,
        grouped_gemm_indexed_pinned, grouped_gemm_indexed_pinned_gateup,
    )
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    k_rows, n_cols, kx = (32, 64, 96) if pair else (64, 48, 128)
    x, ids = _inputs(m, kx, exact, pair, seed=m)
    w, row_off, kidx = _store(k_rows, n_cols, kx, exact, seed=1)
    w2, row_off2, kidx2 = _store(k_rows, n_cols, kx, exact, seed=2)
    stream = torch.cuda.current_stream()
    w_row = 2 * n_cols if fused else n_cols
    ref = torch.zeros(m, K, w_row, dtype=torch.bfloat16, device="cuda")
    gemv_worklist_indexed(x, ids, w.cuda(), row_off, kidx, ref, 0, pair, stream)
    if fused:
        gemv_worklist_indexed(x, ids, w2.cuda(), row_off2, kidx2, ref, n_cols, pair, stream)
    g = build_grouping(ids, E)
    out = torch.zeros_like(ref)
    wa = w.pin_memory() if pinned else w.cuda()
    wb = w2.pin_memory() if pinned else w2.cuda()
    if fused:
        fn = grouped_gemm_indexed_pinned_gateup if pinned else grouped_gemm_indexed_gateup
        fn(x, g, wa, row_off, kidx, wb, row_off2, kidx2, out, 0, n_cols, pair, stream)
    else:
        fn = grouped_gemm_indexed_pinned if pinned else grouped_gemm_indexed
        fn(x, g, wa, row_off, kidx, out, 0, pair, stream)
    torch.cuda.synchronize()
    return out, ref


@cuda_required
@pytest.mark.parametrize("m", [1, 7, 64, 300])
@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_grouped_matches_worklist_bitwise_on_exact_ints(m, pair, pinned, fused):
    out, ref = _run_pair(m, exact=True, pair=pair, pinned=pinned, fused=fused)
    assert torch.equal(out, ref)


@cuda_required
@pytest.mark.parametrize("m", [5, 300])
@pytest.mark.parametrize("pair", [False, True])
def test_grouped_matches_worklist_tolerance(m, pair):
    out, ref = _run_pair(m, exact=False, pair=pair, pinned=False, fused=True)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-3)


@cuda_required
def test_grouping_tensors():
    """pair_off/tile_off의 정의 — count>TILE_M인 expert는 타일이 여러 개, 0인
    expert는 구간이 비어야 한다. host sync 없이 device에서 만들어진다."""
    from sglang.jit_kernel.prism_grouped import TILE_M
    from sglang.srt.layers.moe.prism.grouping import build_grouping

    ids = torch.randint(0, E, (300, K)).cuda()
    ids[:, 0] = 3
    ids[ids == 5] = 6
    g = build_grouping(ids, E)
    counts = torch.bincount(ids.reshape(-1).cpu(), minlength=E)
    assert g.pair_off.cpu().tolist() == [0] + counts.cumsum(0).tolist()
    tiles = (counts + TILE_M - 1) // TILE_M
    assert g.tile_off.cpu().tolist() == [0] + tiles.cumsum(0).tolist()
    assert int(counts[5]) == 0 and int(counts[3]) > TILE_M
    # 정렬된 pair는 expert 오름차순이고 순열이다
    flat = ids.reshape(-1).cpu()
    sorted_e = flat[g.pair_sorted.cpu().long()]
    assert torch.equal(sorted_e, sorted_e.sort().values)
    assert torch.equal(g.pair_sorted.cpu().sort().values, torch.arange(300 * K, dtype=torch.int32))
