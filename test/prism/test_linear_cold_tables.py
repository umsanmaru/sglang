"""dense COLD의 sparsity 표 조립 — kt·CUDA 없이 보는 부분 (P1 게이트).

`test_linear_cold.py`는 kt와 CUDA를 요구해 실기 없이는 아무것도 못 본다. 그런데
kt에 주입하는 **표의 모양과 순서**는 순수 torch 계산이고, 여기가 틀리면 마스크
비트가 다른 행을 가리킨 채 서버가 정상으로 보인다 — 그래서 실기 없이 잡을 수 있는
만큼을 여기서 잡는다.

두 가지다:

  1. **score가 표의 집합을 정한다.** k1은 thr 셋만, k2wl2는 점수 재료 9개까지.
     k1에 페어 재료가 섞이면 kt가 그것을 무시하므로 조용하지만, 반대로 k2wl2에서
     재료가 빠지면 `_install_sparsity`가 죽는다 (그건 잡히는 실패다).
  2. **expert 순서 = `group.units` 순서.** thr 곡선도 점수 재료도 그 순서로
     이어 붙어야 하고, 슬롯이 더미면 0으로 채워야 한다 (`_dummy_weight`가 0).
"""

from __future__ import annotations

import copy

import pytest
import torch

from sglang.srt.layers.prism.linear.calib import LinearCalibShard
from sglang.srt.layers.prism.linear.cold_backend import (
    ColdGroup,
    GroupKey,
    KtLinearColdBackend,
    Unit,
)
from sglang.srt.layers.prism.linear.plan import parse_plan
from sglang.srt.layers.prism.linear.weights import LinearColdShard, PartSparsity

H, I, NG, TILE, NODES = 256, 512, 201, 32, 2


def _plan_dict(score="k2wl2", kernel="kt_tile_k2_bf16", p=0.5):
    d = {
        "schema_version": 1,
        "model_id": "test",
        "dims": {"num_layers": 2, "dtype": "bfloat16"},
        "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": kernel},
        "sparsity": {"score": score, "calib": {"path": "/x", "sha256": "0" * 64},
                     "pmax": 0.9, "grid": 0.005, "ng": NG, "renorm_it": 3},
        "projs": {
            "mlp.down_proj": {
                "k": I, "n": H, "bands": [[0, I, "cold"]],
                "cold_shards": [[0, 0, H // 2], [1, H // 2, H]],
                "calib": "d", "p": p, "lambda": 0.0,
            },
        },
    }
    return d


def _backend(score="k2wl2", kernel="kt_tile_k2_bf16"):
    plan = parse_plan(_plan_dict(score, kernel))
    return KtLinearColdBackend(plan, max_tokens=8, num_numa_nodes=NODES)


def _unit(layer, k_pad, real, *, sparse=True, p=0.5, thr_fill=0.25):
    cold = LinearColdShard(
        w_flat=torch.zeros(H, k_pad, dtype=torch.bfloat16),
        k_index=torch.arange(k_pad, dtype=torch.int64).to(torch.uint16),
        real_rows=real, contiguous=True,
    )
    sp = None
    if sparse:
        sp = PartSparsity(
            thr=torch.full((1, NG), thr_fill),
            p=p, lam=0.0,
            cold=LinearCalibShard(wn=torch.full((k_pad,), 2.0),
                                  pair_dot=torch.full((k_pad // 2,), 3.0)),
        )
    return Unit(layer=layer, proj="mlp.down_proj", k=I, n_start=0,
                colds=(cold,), shards=((0, 0, H // 2), (1, H // 2, H)), sp=(sp,))


def _group(units, kernel="kt_tile_k2_bf16"):
    key = GroupKey(entry="down", k=I, n=H, kernel=kernel,
                   shards=units[0].shards, budget=units[0].budget)
    g = ColdGroup(key)
    g.units.extend(units)
    g.freeze()
    return g


# ── score가 표의 집합을 정한다 ─────────────────────────────────────────────


def test_k1_injects_thresholds_only():
    be = _backend(score="k1", kernel="kt_tile_k1_fp8b128")
    g = _group([_unit(0, 64, 64, thr_fill=0.1), _unit(1, 64, 64, thr_fill=0.2)],
               kernel="kt_tile_k1_fp8b128")
    t = be._sparsity_tables(g)
    assert set(t) == {"thr_gate", "thr_up", "thr_down"}
    # 실슬롯(down)은 unit 순서로 이어 붙고, 더미(gate/up)는 0이다.
    assert t["thr_down"].numel() == 2 * NG
    torch.testing.assert_close(t["thr_down"][:NG], torch.full((NG,), 0.1))
    torch.testing.assert_close(t["thr_down"][NG:], torch.full((NG,), 0.2))
    assert float(t["thr_gate"].abs().max()) == 0.0


def test_k2wl2_injects_score_material_in_unit_order():
    be = _backend()
    g = _group([_unit(0, 64, 64), _unit(1, 96, 90)])
    t = be._sparsity_tables(g)
    assert set(t) == {"thr_gate", "thr_up", "thr_down", "gate_wn_sq", "gate_pair_dot",
                      "up_wn_sq", "up_pair_dot", "down_wn_sq", "down_pair_dot"}
    # 실슬롯: Σ k_pad (패딩 포함 — 스토어와 같은 길이여야 마스크가 대응된다)
    assert t["down_wn_sq"].numel() == 64 + 96
    assert t["down_pair_dot"].numel() == (64 + 96) // 2
    torch.testing.assert_close(t["down_wn_sq"][:4], torch.full((4,), 4.0))  # wn² = 2²
    # 더미 슬롯: expert마다 타일 하나 (`_set_dummy_kindex`와 같은 기하)
    assert t["gate_wn_sq"].numel() == 2 * TILE
    assert float(t["gate_wn_sq"].abs().max()) == 0.0


def test_k1w1_injects_thresholds_and_wn_but_no_pair_dot():
    """score k1w1(`|x_k|·wn_k >= thr`)은 wn까지만 필요하다 — pair_dot을 보내면 "이 점수가
    그걸 쓴다"는 잘못된 선언이 된다."""
    be = _backend(score="k1w1", kernel="kt_tile_k1_bf16")
    g = _group([_unit(0, 64, 64), _unit(1, 96, 90)], kernel="kt_tile_k1_bf16")
    t = be._sparsity_tables(g)
    assert set(t) == {"thr_gate", "thr_up", "thr_down",
                      "gate_wn_sq", "up_wn_sq", "down_wn_sq"}
    assert t["down_wn_sq"].numel() == 64 + 96
    torch.testing.assert_close(t["down_wn_sq"][:4], torch.full((4,), 4.0))   # wn² = 2²


def test_per_k_kernel_takes_both_per_k_scores():
    """per-k 커널은 k1과 k1w1 둘 다 받는다 — 마스크 단위는 커널, 점수식은 표가 정한다."""
    for score in ("k1", "k1w1"):
        _backend(score=score, kernel="kt_tile_k1_bf16")          # 안 죽는다
    with pytest.raises(Exception, match=r"받는 score는"):
        _backend(score="k2wl2", kernel="kt_tile_k1_bf16")


def test_non_sparse_unit_contributes_zeros():
    """한 그룹 안에서 마스킹하지 않는 unit은 0 재료를 받는다 — thr=0이면 전량 통과."""
    be = _backend()
    g = _group([_unit(0, 64, 64), _unit(1, 64, 64, sparse=False)])
    t = be._sparsity_tables(g)
    assert float(t["thr_down"][NG:].abs().max()) == 0.0
    assert float(t["down_wn_sq"][64:].abs().max()) == 0.0


# ── 예산은 인스턴스 스칼라라 키의 일부다 ───────────────────────────────────


@pytest.mark.parametrize("a,b,same", [
    ((0.5, 0.0), (0.5, 0.0), True),
    ((0.5, 0.0), (0.3, 0.0), False),
    (None, (0.5, 0.0), False),
])
def test_budget_is_part_of_the_group_key(a, b, same):
    """`cfg.partial.sparsity.p_*`가 인스턴스 스칼라이므로 예산이 다르면 다른 그룹이다.

    안 갈라지면 한 unit의 p가 조용히 다른 unit에도 적용된다 — 두 층이 다른
    sparsity로 돌아야 하는데 같은 값으로 도는 형태의 무증상 오류다.
    """
    def key(budget):
        return GroupKey(entry="down", k=I, n=H, kernel="kt_tile_k2_bf16",
                        shards=((0, 0, H),), budget=(budget,))
    assert (key(a) == key(b)) is same


def test_score_kernel_mismatch_dies():
    plan = parse_plan(_plan_dict(score="k1", kernel="kt_tile_k2_bf16"))
    with pytest.raises(Exception, match=r"받는 score는 \['k2wl2'\]"):
        KtLinearColdBackend(plan, max_tokens=8, num_numa_nodes=NODES)
