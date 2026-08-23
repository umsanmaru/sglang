"""M-적응 라우팅 전략 — 그룹 구성과 (m, j) 좌표 복원은 한 몸이다.

M>1(prefill): dedup + broadcast-where 산란. 스테이징 대역폭(expert 재사용)이
  우선이라 distinct expert 단위로 GEMM하고 (m, j)로 되산란한다.
M==1(decode): Task 2의 SlotOrderGrouping — topk 슬롯 순서 그대로 그룹을 만들면
  GEMM 배치 축 == j 좌표가 되어 산란이 소멸한다 (planir expert_moe_warm.cc의
  slot-major [m*k, N, K]와 동형; NVTX 실측에서 산란 dispatch가 층당 1.4ms).
"""
from __future__ import annotations

from typing import Protocol, Sequence

import torch


def expert_groups(unique_ids: Sequence[int], n_slots: int) -> list[list[int]]:
    """expert id 목록을 arena slot 수 단위로 절단 (직렬 그룹 루프)."""
    return [list(unique_ids[i : i + n_slots]) for i in range(0, len(unique_ids), n_slots)]


class GroupingStrategy(Protocol):
    def make_groups(self, ids_cpu: torch.Tensor, n_slots: int) -> list[list[int]]: ...
    def scatter_gateup(self, warm_gu: torch.Tensor, topk_ids: torch.Tensor,
                       group: Sequence[int], gi: int, n_slots: int,
                       gate_out: torch.Tensor, up_out: torch.Tensor, inter: int) -> None: ...
    def down_apply(self, warm_down: torch.Tensor, topk_ids: torch.Tensor,
                   group: Sequence[int], gi: int, n_slots: int,
                   act_band: torch.Tensor, w: torch.Tensor) -> torch.Tensor: ...


class DedupGrouping:
    """M>1: distinct expert 그룹 + sync-free broadcast-where (executor에서 이동)."""

    def make_groups(self, ids_cpu, n_slots):
        return expert_groups(torch.unique(ids_cpu).tolist(), n_slots)

    def scatter_gateup(self, warm_gu, topk_ids, group, gi, n_slots, gate_out, up_out, inter):
        # sync-free 규칙: host 동기화 유발 연산(nonzero, bool(any()), item) 금지.
        for slot, e in enumerate(group):
            mask = (topk_ids == e).unsqueeze(-1)       # [M, k, 1] device bool
            g = gate_out[slot].float().unsqueeze(1)    # [M, 1, inter] → k broadcast
            u = up_out[slot].float().unsqueeze(1)
            warm_gu[:, :, :inter] = torch.where(mask, g, warm_gu[:, :, :inter])
            warm_gu[:, :, inter:] = torch.where(mask, u, warm_gu[:, :, inter:])

    def down_apply(self, warm_down, topk_ids, group, gi, n_slots, act_band, w):
        for slot, e in enumerate(group):
            mask = (topk_ids == e).unsqueeze(-1)          # [M, k, 1]
            contrib = act_band @ w[slot].float()          # [M, k, H] fp32 누산 (계약 ⑤)
            warm_down = torch.where(mask, contrib, warm_down)
        return warm_down


class SlotOrderGrouping:
    """M==1: 그룹 = topk 슬롯 순서 절단 → GEMM 배치 축이 곧 j 좌표. 산란 없음.

    전제: top-k는 토큰 내 무중복이므로 M==1에서 slot==distinct. M>1에는 쓰지
    않는다 (중복 expert가 슬롯마다 재스테이징되어 dedup의 존재 이유를 깬다).
    """

    def make_groups(self, ids_cpu, n_slots):
        return expert_groups(ids_cpu.view(-1).tolist(), n_slots)

    def scatter_gateup(self, warm_gu, topk_ids, group, gi, n_slots, gate_out, up_out, inter):
        j0, g = gi * n_slots, len(group)
        warm_gu[0, j0 : j0 + g, :inter] = gate_out[:g, 0].float()
        warm_gu[0, j0 : j0 + g, inter:] = up_out[:g, 0].float()

    def down_apply(self, warm_down, topk_ids, group, gi, n_slots, act_band, w):
        j0, g = gi * n_slots, len(group)
        # [g,1,rows] @ [g,rows,H] — slot당 루프였던 것을 bmm 1회로 (fp32 누산, 계약 ⑤)
        contrib = torch.bmm(act_band[0, j0 : j0 + g].unsqueeze(1), w[:g].float())
        warm_down[0, j0 : j0 + g] = contrib[:, 0]
        return warm_down


# 두 전략 모두 stateless — 호출(층×2/step)마다 할당하지 않도록 모듈 싱글턴.
_DEDUP = DedupGrouping()
_SLOT_ORDER = SlotOrderGrouping()


def select_grouping(m: int) -> GroupingStrategy:
    return _SLOT_ORDER if m == 1 else _DEDUP
