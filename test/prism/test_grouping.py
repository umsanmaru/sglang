import torch
from sglang.srt.layers.moe.prism.grouping import (
    DedupGrouping, select_grouping,
)


def test_select_grouping_by_m():
    assert type(select_grouping(1)).__name__ == "SlotOrderGrouping"
    assert type(select_grouping(2)).__name__ == "DedupGrouping"


def test_slot_order_groups_follow_topk_order():
    ids = torch.tensor([[7, 3, 9, 1]])            # M=1, k=4 (top-k는 무중복)
    groups = select_grouping(1).make_groups(ids, n_slots=4)
    assert groups == [[7, 3, 9, 1]]               # 정렬하지 않는다 — 슬롯 순서 보존


def test_slot_order_equals_dedup_full_layer():
    """m=1에서 두 전략의 gateup+down 결과가 일치 (배치=slot 좌표 등가성)."""
    torch.manual_seed(0)
    m, k, inter, rows_gu, rows_dn, h, n_slots = 1, 4, 6, 5, 3, 8, 4
    topk = torch.tensor([[7, 3, 9, 1]], device="cuda")
    gate_w = torch.randn(16, rows_gu, inter, device="cuda")   # arena 대역 [slot, rows, N]
    up_w = torch.randn(16, rows_gu, inter, device="cuda")
    down_w = torch.randn(16, rows_dn, h, device="cuda")
    hidden_band = torch.randn(m, rows_gu, device="cuda", dtype=torch.float32)
    act_band = torch.randn(m, k, rows_dn, device="cuda", dtype=torch.float32)

    outs = {}
    for name, strat in (("dedup", DedupGrouping()), ("slot", select_grouping(1))):
        warm_gu = torch.zeros(m, k, 2 * inter, device="cuda")
        warm_dn = torch.zeros(m, k, h, device="cuda")
        for gi, group in enumerate(strat.make_groups(topk.cpu(), n_slots)):
            g = len(group)
            wg = torch.stack([gate_w[e] for e in group])      # stage 모사
            wu = torch.stack([up_w[e] for e in group])
            wd = torch.stack([down_w[e] for e in group])
            gate_out = torch.bmm(hidden_band.expand(g, -1).unsqueeze(1), wg.float()).reshape(g, m, inter)
            up_out = torch.bmm(hidden_band.expand(g, -1).unsqueeze(1), wu.float()).reshape(g, m, inter)
            strat.scatter_gateup(warm_gu, topk, group, gi, n_slots, gate_out, up_out, inter)
            warm_dn = strat.down_apply(warm_dn, topk, group, gi, n_slots, act_band, wd)
        outs[name] = (warm_gu.clone(), warm_dn.clone())
    torch.testing.assert_close(outs["slot"][0], outs["dedup"][0])
    torch.testing.assert_close(outs["slot"][1], outs["dedup"][1])  # GEMM 형상이 달라 bitwise 아님 — assert_close 기본 fp32 tol


def test_dedup_make_groups_sorted_unique():
    ids = torch.tensor([[7, 3, 3, 1], [1, 5, 7, 2]])
    groups = DedupGrouping().make_groups(ids, n_slots=4)
    assert groups == [[1, 2, 3, 5], [7]]   # unique는 정렬, n_slots 단위 절단


def test_dedup_scatter_gateup_places_by_topk_coord():
    m, k, inter, g = 2, 2, 3, 2
    topk = torch.tensor([[4, 9], [9, 4]])
    warm_gu = torch.zeros(m, k, 2 * inter)
    gate_out = torch.arange(g * m * inter, dtype=torch.float32).reshape(g, m, inter)
    up_out = gate_out + 100
    DedupGrouping().scatter_gateup(warm_gu, topk, [4, 9], 0, g, gate_out, up_out, inter)
    # expert 4 = slot0: (m=0,j=0)과 (m=1,j=1)에 gate_out[0, m]이 놓인다
    assert torch.equal(warm_gu[0, 0, :inter], gate_out[0, 0])
    assert torch.equal(warm_gu[1, 1, :inter], gate_out[0, 1])
    assert torch.equal(warm_gu[0, 1, inter:], up_out[1, 0])   # expert 9 = slot1
