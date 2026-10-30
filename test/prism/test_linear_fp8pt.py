"""dense per-tensor FP8 스토어 (`fp8pt`) — 배율이 스칼라 하나인 포맷 (GPU 불필요).

대상: Mistral-Medium-3.5-128B 류 static-tensor FP8 (`weight_block_size: null`,
`weight_scale_inv`가 선형층당 스칼라). blockwise와 코드 바이트는 같고 **배율 기하만**
다르므로, 여기서 잡는 것은 그 축이다:

  1. **배율을 어디서 고르는가.** 융합 모듈(`qkv_proj`)의 배율은 논리 shard마다 하나이므로
     조각 번호로 골라야 한다. 블록 행 슬라이스를 하면 q의 배율이 k에 붙는 식으로
     **조용히 다른 수치**가 된다 (출력이 그럴듯하게 나쁘다).
  2. **K 정렬이 32로 풀린다.** 배율 블록이 없으니 티어를 32행 단위로 자를 수 있다 —
     128 배수가 아닌 밴드를 blockwise는 거절하고 per-tensor는 받아야 한다.
  3. **cold 주입의 배율 계약**: proj당 fp32 `[E]` (슬롯당 스칼라 1개), 더미 슬롯도 값이
     있어야 한다. 빠지면 kt wrapper가 원소 수 검증에서 죽는다.
  4. **커널 짝**: `gemv_worklist_fp8pt`는 `kt_tile_k{1,2}_fp8pt`와만 맞는다. blockwise
     cold 커널과 섞이면 startup에서 죽어야 한다.
"""

from __future__ import annotations

import copy

import pytest
import torch

from sglang.srt.layers.prism.geometry import PlanError, Tier
from sglang.srt.layers.prism.linear.formats import FP8PT, store_format
from sglang.srt.layers.prism.linear.plan import parse_plan, validate_static
from sglang.srt.layers.prism.linear.weights import prepare_linear_weights

K, N = 256, 512          # k % 32, n % 256 (fp8pt 타일 super)
CPU = torch.device("cpu")

# hot [0,64) · warm [64,160) · cold [160,256) — 96행 = 3×32, 128의 배수가 아니다.
PLAN = {
    "schema_version": 1,
    "model_id": "test/fp8pt",
    "dims": {"num_layers": 1, "dtype": "bfloat16"},
    "kernels": {"gpu_warm": "gemv_worklist_fp8pt", "cpu_cold": "kt_tile_k1_fp8pt"},
    "projs": {
        "self_attn.qkv_proj": {
            "k": K, "n": N,
            "parts": [
                {"name": "q", "n": 256, "bands": [[0, 64, "hot"], [64, 160, "warm"],
                                                  [160, K, "cold"]],
                 "cold_shards": [[0, 0, 256]]},
                {"name": "k", "n": 256, "bands": [[0, 64, "hot"], [64, K, "cold"]],
                 "cold_shards": [[0, 0, 256]]},
            ],
        },
    },
}


@pytest.fixture
def weight():
    torch.manual_seed(0)
    return torch.randint(1, 120, (N, K), dtype=torch.uint8).view(torch.float8_e4m3fn)


@pytest.fixture
def scale():
    return torch.tensor([0.5, 2.0], dtype=torch.float32)   # 논리 shard 2개 (q, k)


def _prep(weight, scale, raw=None):
    plan = parse_plan(raw or PLAN)
    validate_static(plan)
    return prepare_linear_weights(0, "self_attn.qkv_proj", weight, plan, scale=scale,
                                  device=CPU, pin_memory=False)


# ── 커널 짝 ────────────────────────────────────────────────────────────────


def test_format_pairs_only_with_pt_cold_kernels():
    for cold in ("kt_tile_k2_fp8pt", "kt_tile_k1_fp8pt"):
        assert store_format("gemv_worklist_fp8pt", cold) is FP8PT
    for cold in ("kt_tile_k2_fp8b128", "kt_tile_k1_fp8b128", "kt_tile_k2_bf16"):
        with pytest.raises(PlanError, match="not compatible"):
            store_format("gemv_worklist_fp8pt", cold)


def test_k_align_is_32_not_128():
    """배율 블록이 없으니 128 배수가 아닌 밴드도 통과한다 (blockwise는 거절한다)."""
    assert FP8PT.k_align == 32
    rows = torch.arange(96, dtype=torch.int64).to(torch.uint16)   # 96 = 3×32
    FP8PT.check_rows(rows, "w")                                   # 안 죽는다


# ── 배율 계약 ──────────────────────────────────────────────────────────────


def test_proj_scale_is_one_per_logical_shard(weight):
    """proj 단위는 `[조각 수]`가 정상이고, blockwise의 2-D 표는 거절해야 한다."""
    FP8PT.check(weight, torch.ones(1, dtype=torch.float32), K, N, "w")
    FP8PT.check(weight, torch.ones(2, dtype=torch.float32), K, N, "w")
    with pytest.raises(PlanError, match="blockwise"):
        FP8PT.check(weight, torch.ones(N // 128, K // 128, dtype=torch.float32), K, N, "w")


def test_part_scale_not_applied_dies(weight):
    """조각 배율을 안 고르고 통째로 넘기면 절단에서 죽는다 (조용히 첫 값을 쓰지 않는다)."""
    rows = torch.arange(64, dtype=torch.int64).to(torch.uint16)
    with pytest.raises(PlanError, match="스칼라 하나여야"):
        FP8PT.gather(weight, torch.ones(2, dtype=torch.float32), rows, True, 0)


def test_missing_scale_dies(weight):
    with pytest.raises(PlanError, match="needs the per-tensor weight_scale"):
        FP8PT.check(weight, None, K, N, "w")


def test_part_scale_picks_by_part_index(scale):
    """융합 모듈의 배율은 조각 번호로 고른다 — 블록 행 슬라이스가 아니다."""
    assert float(FP8PT.part_scale(scale, 0, 0, 256, "w")[0]) == 0.5
    assert float(FP8PT.part_scale(scale, 1, 256, 512, "w")[0]) == 2.0
    with pytest.raises(PlanError, match="part 2"):
        FP8PT.part_scale(scale, 2, 512, 768, "w")


def test_prepared_shards_carry_their_part_scale(weight, scale):
    prepared = _prep(weight, scale)
    for part, want in zip(prepared.parts, (0.5, 2.0)):
        for tier in (Tier.HOT, Tier.WARM, Tier.COLD):
            shard = part.tier(tier)
            if shard is None:
                continue
            assert shard.s_flat is not None, f"{part.name} {tier.value}: 배율이 없다"
            assert shard.s_flat.numel() == 1
            assert float(shard.s_flat[0]) == want


def test_codes_follow_the_tier_rows(weight, scale):
    prepared = _prep(weight, scale)
    q = prepared.part("q")
    codes = weight.view(torch.uint8)[0:256]
    # hot 스토어는 [k_tier, N] K-major
    torch.testing.assert_close(q.hot.w_flat, codes.t()[0:64].contiguous())
    # cold는 [N, k_pad] ckpt 방향 (타일 32 배수 → 96행이면 패딩 없음)
    assert q.cold.real_rows == 96 and q.cold.k_pad == 96
    torch.testing.assert_close(q.cold.w_flat, codes[:, 160:256].contiguous())


# ── cold 주입의 배율 인자 ──────────────────────────────────────────────────


def test_cold_scale_kwargs_are_per_slot_with_dummies(weight, scale):
    from sglang.srt.layers.prism.linear.cold_backend import (
        ColdGroup,
        GroupKey,
        KtLinearColdBackend,
        split_units,
    )

    plan = parse_plan(PLAN)
    be = KtLinearColdBackend(plan, max_tokens=8, num_numa_nodes=1)
    prepared = _prep(weight, scale)
    units = split_units(0, "self_attn.qkv_proj", prepared, (((0, 0, 256),), ((0, 0, 256),)))
    # q와 k는 N이 같고 out3d에서 인접하므로 **gateup unit 하나**로 묶인다 (`split_units`).
    assert len(units) == 1 and units[0].entry == "gateup"
    key = GroupKey(entry="gateup", k=K, n=256, kernel="kt_tile_k1_fp8pt",
                   shards=((0, 0, 256),), budget=(None, None))
    g = ColdGroup(key)
    g.units.extend(units)
    g.freeze()
    kw = be._scale_kwargs(g, g.num_experts)
    assert set(kw) == {"gate_scale", "up_scale", "down_scale"}
    # 실슬롯 gate=q, up=k → 각 조각의 배율. 더미(down)는 1.0.
    torch.testing.assert_close(kw["gate_scale"], torch.tensor([0.5]))
    torch.testing.assert_close(kw["up_scale"], torch.tensor([2.0]))
    torch.testing.assert_close(kw["down_scale"], torch.ones(1))


def test_bf16_store_asks_for_no_scales(weight):
    """배율 없는 포맷은 인자를 만들지 않는다 — 있으면 kt가 '배율을 안 받는다'로 죽는다."""
    from sglang.srt.layers.prism.linear.cold_backend import (
        ColdGroup,
        GroupKey,
        KtLinearColdBackend,
        split_units,
    )

    raw = copy.deepcopy(PLAN)
    raw["kernels"] = {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_tile_k2_bf16"}
    plan = parse_plan(raw)
    torch.manual_seed(1)
    w = torch.randn(N, K, dtype=torch.bfloat16)
    prepared = prepare_linear_weights(0, "self_attn.qkv_proj", w, plan, device=CPU,
                                      pin_memory=False)
    be = KtLinearColdBackend(plan, max_tokens=8, num_numa_nodes=1)
    units = split_units(0, "self_attn.qkv_proj", prepared, (((0, 0, 256),), ((0, 0, 256),)))
    key = GroupKey(entry="gateup", k=K, n=256, kernel="kt_tile_k2_bf16",
                   shards=((0, 0, 256),), budget=(None, None))
    g = ColdGroup(key)
    g.units.extend(units)
    g.freeze()
    assert be._scale_kwargs(g, g.num_experts) == {}
