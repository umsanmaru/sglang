"""dense Stage 2 (weight 절단·배치) 테스트 — CUDA 불필요.

가장 중요한 불변식은 **계약 ⑤**다: 세 티어의 부분합이 원래 행렬곱과 같다.
K축이 합의 축이라는 것이 K-split의 존립 근거이므로, 그것이 깨지면 나머지는
전부 무의미하다. 나머지 테스트는 그 등식이 성립하기 위한 조건들 — 티어가 K를
분할하는가, 각 shard가 실제로 그 행들을 담았는가, cold가 ckpt 방향과 패딩
규약을 지키는가 — 를 각각 잡는다.
"""

import copy

import pytest
import torch

from sglang.srt.layers.prism.geometry import PlanError, Tier
from sglang.srt.layers.prism.linear.plan import parse_plan
from sglang.srt.layers.prism.linear.weights import (
    LinearColdShard,
    prepare_linear_weights,
    tier_rows,
)

K, N = 200, 64
NAME = "self_attn.wq_b"
COLD_TILE = 32  # kt_amx_bf16

# hot [0,64) · warm [64,100) · cold [100,200) — cold 100행은 타일(32) 배수가
# 아니라 패딩 경로를 탄다.
PLAN = {
    "schema_version": 1,
    "model_id": "test/dense",
    "dims": {"num_layers": 2, "dtype": "bfloat16"},
    "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_amx_bf16"},
    "projs": {
        NAME: {
            "k": K,
            "n": N,
            "bands": [[0, 64, "hot"], [64, 100, "warm"], [100, K, "cold"]],
            "cold_shards": [[0, 0, 32], [1, 32, N]],
        }
    },
}


@pytest.fixture
def weight():
    torch.manual_seed(0)
    return torch.randn(N, K, dtype=torch.bfloat16)


def _prep(weight, raw=None, **kw):
    plan = parse_plan(raw or PLAN)
    kw.setdefault("device", torch.device("cpu"))
    kw.setdefault("pin_memory", False)
    return prepare_linear_weights(0, NAME, weight, plan, **kw)


def _mutate(**bands_by_key):
    raw = copy.deepcopy(PLAN)
    for key, value in bands_by_key.items():
        raw["projs"][NAME][key] = value
    return raw


# ── 계약 ⑤: 부분합 = 전체 ───────────────────────────────────────────────────


def test_partials_sum_to_full_matmul(weight):
    """세 티어가 낸 부분합의 fp32 합 == 원래 행렬곱.

    이것이 K-split이 정당한 이유 전부다. 티어 배치를 바꿔도 이 등식은 유지된다
    (아래 test_placement_invariance).
    """
    p = _prep(weight)
    x = torch.randn(8, K, dtype=torch.bfloat16)

    acc = torch.zeros(8, N, dtype=torch.float32)
    for shard in (p.sole.hot, p.sole.warm):
        rows = shard.k_index.to(torch.int64)
        acc += (x[:, rows].float() @ shard.w_flat.float())
    rows = p.sole.cold.k_index[: p.sole.cold.real_rows].to(torch.int64)
    acc += x[:, rows].float() @ p.sole.cold.w_flat[:, : p.sole.cold.real_rows].float().t()

    # tolerance는 절단이 아니라 **재결합** 때문이다 — 부분합은 K를 세 덩이로 나눠
    # 더하고 기준은 한 번에 더한다. 절단의 정확성은 test_placement_invariance가
    # 비트일치로 잡는다.
    torch.testing.assert_close(acc, x.float() @ weight.float().t(), rtol=1e-2, atol=1e-4)


def test_placement_invariance(weight):
    """같은 K를 다르게 나눠도 결과가 같다 (계약 ⑤의 plan 불변성).

    hot/warm/cold 경계를 옮긴 두 plan의 재구성 weight가 비트일치해야 한다.
    """

    def reconstruct(raw):
        p = _prep(weight, raw)
        out = torch.zeros(N, K, dtype=weight.dtype)
        for shard in (p.sole.hot, p.sole.warm):
            if shard is not None:
                out[:, shard.k_index.to(torch.int64)] = shard.w_flat.t()
        if p.sole.cold is not None:
            r = p.sole.cold.real_rows
            out[:, p.sole.cold.k_index[:r].to(torch.int64)] = p.sole.cold.w_flat[:, :r]
        return out

    a = reconstruct(PLAN)
    b = reconstruct(_mutate(bands=[[0, 20, "warm"], [20, 150, "cold"], [150, K, "hot"]]))
    assert torch.equal(a, b) and torch.equal(a, weight)


# ── K 분할 ──────────────────────────────────────────────────────────────────


def test_tiers_partition_k(weight):
    p = _prep(weight)
    rows = []
    for shard in (p.sole.hot, p.sole.warm):
        rows += shard.k_index.to(torch.int64).tolist()
    rows += p.sole.cold.k_index[: p.sole.cold.real_rows].to(torch.int64).tolist()
    assert sorted(rows) == list(range(K))  # 완전 커버 + 무중첩


def test_rows_accessor(weight):
    p = _prep(weight)
    assert (p.rows(Tier.HOT), p.rows(Tier.WARM), p.rows(Tier.COLD)) == (64, 36, 100)
    assert sum(p.rows(t) for t in Tier) == K


def test_absent_tier_is_none_not_empty(weight):
    """'이 티어는 여기 없다'는 길이 0 텐서가 아니라 부재로 표현된다."""
    p = _prep(weight, _mutate(bands=[[0, K, "hot"]], cold_shards=[]))
    assert p.sole.warm is None and p.sole.cold is None
    assert p.sole.hot is not None and p.sole.hot.k_rows == K


# ── shard 내용 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tier_name", ["hot", "warm"])
def test_gpu_shard_is_k_major_and_matches_source(weight, tier_name):
    p = _prep(weight)
    shard = getattr(p.sole, tier_name)
    rows = shard.k_index.to(torch.int64)
    assert shard.w_flat.shape == (rows.numel(), N)  # K-major
    torch.testing.assert_close(shard.w_flat, weight[:, rows].t())


def test_cold_keeps_ckpt_direction(weight):
    p = _prep(weight)
    r = p.sole.cold.real_rows
    rows = p.sole.cold.k_index[:r].to(torch.int64)
    assert p.sole.cold.w_flat.shape[0] == N  # [N, k_pad] — 전치 안 함
    torch.testing.assert_close(p.sole.cold.w_flat[:, :r], weight[:, rows])


def test_cold_pads_to_tile_with_zeros(weight):
    p = _prep(weight)
    assert p.sole.cold.real_rows == 100
    assert p.sole.cold.k_pad == 128 and p.sole.cold.k_pad % COLD_TILE == 0
    assert torch.all(p.sole.cold.w_flat[:, 100:] == 0)
    assert torch.all(p.sole.cold.k_index[100:] == 0)  # 유효한 축 값 (kt가 범위 검증)


def test_cold_no_padding_when_tile_aligned(weight):
    p = _prep(weight, _mutate(bands=[[0, 72, "hot"], [72, K, "cold"]]))
    assert p.sole.cold.real_rows == 128 and p.sole.cold.k_pad == 128


# ── contiguous 판정 (커널이 gather를 건너뛸 수 있는가) ──────────────────────


def test_single_band_tier_is_contiguous(weight):
    p = _prep(weight)
    assert p.sole.hot.contiguous and p.sole.hot.k_start == 0
    assert p.sole.warm.contiguous and p.sole.warm.k_start == 64


def test_split_band_tier_is_not_contiguous(weight):
    """hot이 warm을 사이에 두고 두 조각이면 gather가 필요하다."""
    p = _prep(
        weight,
        _mutate(bands=[[0, 40, "hot"], [40, 80, "cold"], [80, 120, "hot"], [120, K, "warm"]]),
    )
    assert p.sole.hot.k_rows == 80
    assert not p.sole.hot.contiguous and p.sole.hot.k_start is None
    # 내용은 그래도 맞아야 한다
    rows = p.sole.hot.k_index.to(torch.int64)
    torch.testing.assert_close(p.sole.hot.w_flat, weight[:, rows].t())


def test_tier_rows_follows_band_order(weight):
    plan = parse_plan(_mutate(bands=[[0, 40, "hot"], [40, 80, "cold"], [80, 120, "hot"], [120, K, "warm"]]))
    pp = plan.proj(0, NAME).sole
    assert tier_rows(pp, Tier.HOT) == list(range(0, 40)) + list(range(80, 120))


# ── 배치 ────────────────────────────────────────────────────────────────────


def test_hot_goes_to_device_and_index_follows(weight):
    p = _prep(weight)
    assert p.sole.hot.w_flat.device.type == "cpu"      # 이 테스트의 device
    assert p.sole.hot.k_index.device == p.sole.hot.w_flat.device
    assert p.sole.hot.k_index.dtype == torch.uint16


def test_hot_without_device_dies(weight):
    with pytest.raises(PlanError, match="HOT rows but no device"):
        _prep(weight, device=None)


def test_no_device_needed_without_hot(weight):
    p = _prep(weight, _mutate(bands=[[0, 100, "warm"], [100, K, "cold"]]), device=None)
    assert p.sole.hot is None and p.sole.warm is not None and p.sole.cold is not None


# ── 입력 방어 ───────────────────────────────────────────────────────────────


def test_unplanned_proj_dies(weight):
    plan = parse_plan(PLAN)
    with pytest.raises(PlanError, match="not in the plan"):
        prepare_linear_weights(0, "mlp.down_proj", weight, plan, pin_memory=False)


def test_dim_mismatch_dies(weight):
    """다른 모델의 plan을 먹이면 즉사한다 — 안 죽으면 조용히 틀린 슬라이스."""
    with pytest.raises(PlanError, match="다른 모델"):
        _prep(torch.randn(N, K + 2, dtype=torch.bfloat16))


def test_non_2d_weight_dies(weight):
    with pytest.raises(PlanError, match=r"2-D \[N, K\]"):
        _prep(weight.unsqueeze(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
def test_cuda_weight_dies(weight):
    """로더가 파라미터를 CUDA로 옮겨둔 상태로 들어오면 즉사 (cold는 host memcpy)."""
    with pytest.raises(PlanError, match="must be on CPU"):
        _prep(weight.cuda(), device=torch.device("cuda"))


def test_mxfp4_store_unsupported(weight):
    """dense는 bf16/fp8만 — mxfp4 체크포인트가 대상에 없다 (2026-08-31 확인)."""
    raw = copy.deepcopy(PLAN)
    raw["kernels"] = {"gpu_warm": "gemv_worklist_mxfp4", "cpu_cold": "kt_amx_fp4"}
    with pytest.raises(PlanError, match=r"supports \['bf16', 'fp8'\]"):
        _prep(weight, raw)


@pytest.mark.parametrize(
    "kern",
    [
        {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_tile_k2_fp8b128"},  # bf16 ← fp8 cold
        {"gpu_warm": "gemv_worklist_fp8", "cpu_cold": "kt_amx_bf16"},     # fp8 ← bf16 cold
    ],
)
def test_mismatched_kernel_pair_dies(weight, kern):
    """GPU 커널은 스토어 형식을, cold 커널은 slab 레이아웃을 함의한다 (계약 ①).
    둘이 어긋나면 한쪽이 남의 바이트를 자기 형식으로 읽는다."""
    raw = copy.deepcopy(PLAN)
    raw["kernels"] = kern
    with pytest.raises(PlanError, match="not compatible"):
        _prep(weight, raw)


def test_bf16_rejects_scale(weight):
    with pytest.raises(PlanError, match="takes no scales"):
        _prep(weight, scale=torch.ones(1, 1))


def test_unknown_kernel_dies(weight):
    raw = copy.deepcopy(PLAN)
    raw["kernels"]["gpu_warm"] = "no_such_kernel"
    with pytest.raises(Exception, match="unknown gpu_warm"):
        _prep(weight, raw)


# ── full 텐서 소멸 (계약 ③) ─────────────────────────────────────────────────


def test_does_not_alias_the_source(weight):
    """산출물이 입력의 뷰이면 호출자가 원본을 놓아도 메모리가 안 준다."""
    p = _prep(weight)
    base = weight.data_ptr()
    for shard in (p.sole.hot, p.sole.warm, p.sole.cold):
        assert shard.w_flat.data_ptr() != base
        assert shard.w_flat._base is None or shard.w_flat._base.data_ptr() != base


def test_cold_shard_type(weight):
    p = _prep(weight)
    assert isinstance(p.sole.cold, LinearColdShard)
    assert p.sole.tier(Tier.COLD) is p.sole.cold


# ── 실제 배치 (CUDA) ────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
def test_real_placement(weight):
    """hot → VRAM, warm → GPU-local NUMA 노드의 pinned, cold → 평범한 host.

    거처가 티어의 정의 전부이므로(계약 ①) 여기가 틀리면 나머지가 다 맞아도
    prism이 아니다. 그리고 warm이 원격 소켓에 앉는 것은 결과가 정확하고 느리기만
    해서 어떤 테스트도 안 잡는 종류의 오류다 — `alloc_pinned_on_node`가 배치 후
    실제 노드를 검증한다.
    """
    from sglang.srt.layers.prism.numa import gpu_numa_node

    dev = torch.device("cuda:0")
    p = _prep(weight, device=dev, warm_node=gpu_numa_node(dev), pin_memory=True)

    assert p.sole.hot.w_flat.device.type == "cuda"
    assert p.sole.hot.k_index.device == p.sole.hot.w_flat.device
    assert p.sole.warm.w_flat.device.type == "cpu" and p.sole.warm.w_flat.is_pinned()
    assert p.sole.warm.k_index.device.type == "cuda"  # 인덱스는 커널이 읽는다
    assert p.sole.cold.w_flat.device.type == "cpu" and not p.sole.cold.w_flat.is_pinned()

    # 계약 ⑤가 실제 배치에서도 성립한다
    x = torch.randn(4, K, dtype=torch.bfloat16)
    acc = torch.zeros(4, N, dtype=torch.float32)
    for shard in (p.sole.hot, p.sole.warm):
        rows = shard.k_index.cpu().to(torch.int64)
        acc += x[:, rows].float() @ shard.w_flat.cpu().float()
    rows = p.sole.cold.k_index[: p.sole.cold.real_rows].to(torch.int64)
    acc += x[:, rows].float() @ p.sole.cold.w_flat[:, : p.sole.cold.real_rows].float().t()
    torch.testing.assert_close(acc, x.float() @ weight.float().t())


# ═══════════════════════════════════════════════════════════════════════════
# blockwise FP8
# ═══════════════════════════════════════════════════════════════════════════
#
# bf16과 갈리는 계약은 하나다: **K를 128 블록 경계에서만 자를 수 있다.** 배율
# 하나가 원본 128k × 128n 블록을 덮으므로, 티어 경계가 블록을 쪼개면 두 티어가
# 같은 배율을 나눠 갖게 되어 "블록당 배율 1"이 깨진다. 재양자화 없이 체크포인트
# 수치를 보존하는 유일한 선택이다.

BLOCK = 128
K8, N8 = 384, 512  # 둘 다 128의 배수 (부분 배율 블록은 표현 불가)

PLAN8 = {
    "schema_version": 1,
    "model_id": "test/dense-fp8",
    "dims": {"num_layers": 2, "dtype": "bfloat16"},
    "kernels": {"gpu_warm": "gemv_worklist_fp8", "cpu_cold": "kt_tile_k2_fp8b128"},
    "projs": {
        NAME: {
            "k": K8,
            "n": N8,
            "bands": [[0, 128, "hot"], [128, 256, "warm"], [256, K8, "cold"]],
            "cold_shards": [[0, 0, 256], [1, 256, N8]],
        }
    },
}


@pytest.fixture
def fp8_weight():
    torch.manual_seed(1)
    codes = torch.randn(N8, K8, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    # scale_inv는 블록당 하나 — [N/128, K/128]
    scale = torch.rand(N8 // BLOCK, K8 // BLOCK, dtype=torch.float32) + 0.5
    return codes, scale


def _prep8(codes, scale, raw=None, **kw):
    plan = parse_plan(raw or PLAN8)
    kw.setdefault("device", torch.device("cpu"))
    kw.setdefault("pin_memory", False)
    return prepare_linear_weights(0, NAME, codes, plan, scale=scale, **kw)


def _mutate8(**by_key):
    raw = copy.deepcopy(PLAN8)
    for key, value in by_key.items():
        raw["projs"][NAME][key] = value
    return raw


def _dequant(codes_u8_or_fp8, scale, k_major: bool):
    """블록 배율을 원소마다 펼쳐 곱한다. k_major면 codes가 [K, N]이고 scale이 [K/128, N/128]."""
    c = codes_u8_or_fp8
    if c.dtype == torch.uint8:
        c = c.view(torch.float8_e4m3fn)
    s = scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return c.float() * s[: c.shape[0], : c.shape[1]]


def _fp8_tier_weights(p):
    """각 티어의 shard를 역양자화해 (원래 K행 번호, `[k_tier, N]`) 로 돌려준다."""
    out = []
    for shard in (p.sole.hot, p.sole.warm):
        if shard is not None:
            out.append((shard.k_index.to(torch.int64),
                        _dequant(shard.w_flat, shard.s_flat, k_major=True)))
    if p.sole.cold is not None:
        r = p.sole.cold.real_rows
        cold = _dequant(p.sole.cold.w_flat[:, :r], p.sole.cold.s_flat[:, : r // BLOCK], k_major=False)
        out.append((p.sole.cold.k_index[:r].to(torch.int64), cold.t()))
    return out


def test_fp8_reconstruction_is_bit_exact(fp8_weight):
    """계약 ⑤의 본체 — 세 티어를 도로 합치면 원본 역양자화 행렬과 **비트일치**한다.

    행렬곱 비교(아래)는 재결합 오차 때문에 tolerance가 필요하지만, 이 등식은
    tolerance가 필요 없다. 절단이 정확한지는 여기가 증명하고, 아래는 그 위에서
    "합의 축이 맞다"만 본다.
    """
    codes, scale = fp8_weight
    p = _prep8(codes, scale)

    out = torch.zeros(K8, N8, dtype=torch.float32)
    seen = []
    for rows, w in _fp8_tier_weights(p):
        out[rows] = w
        seen += rows.tolist()

    assert sorted(seen) == list(range(K8))          # K를 정확히 분할한다
    assert torch.equal(out, _dequant(codes, scale, k_major=False).t())


def test_fp8_partials_sum_to_full_matmul(fp8_weight):
    """계약 ⑤ — fp8에서도 세 티어 부분합이 원래(역양자화) 행렬곱과 같다.

    tolerance가 필요한 이유는 절단이 아니라 **재결합**이다: 티어 부분합은 K를 세
    덩이로 나눠 더하고 기준은 한 번에 더하므로 fp32 누산 순서가 다르다. 절단 자체의
    정확성은 위 `test_fp8_reconstruction_is_bit_exact`가 비트일치로 잡는다.
    """
    codes, scale = fp8_weight
    p = _prep8(codes, scale)
    x = torch.randn(8, K8, dtype=torch.bfloat16)

    acc = torch.zeros(8, N8, dtype=torch.float32)
    for rows, w in _fp8_tier_weights(p):
        acc += x[:, rows].float() @ w

    full = _dequant(codes, scale, k_major=False)
    torch.testing.assert_close(acc, x.float() @ full.t(), rtol=1e-2, atol=1e-4)


def test_fp8_shard_shapes(fp8_weight):
    codes, scale = fp8_weight
    p = _prep8(codes, scale)
    # GPU 티어: 코드 [k_tier, N] u8 + 배율 [k_tier/128, N/128]
    assert p.sole.hot.w_flat.shape == (128, N8) and p.sole.hot.w_flat.dtype == torch.uint8
    assert p.sole.hot.s_flat.shape == (1, N8 // BLOCK) and p.sole.hot.s_flat.dtype == torch.float32
    assert p.sole.warm.w_flat.shape == (128, N8) and p.sole.warm.s_flat.shape == (1, N8 // BLOCK)
    # cold: ckpt 방향 [N, k_pad] + 배율 [N/128, k_pad/128]
    assert p.sole.cold.w_flat.shape == (N8, 128) and p.sole.cold.w_flat.dtype == torch.uint8
    assert p.sole.cold.s_flat.shape == (N8 // BLOCK, 1)


def test_fp8_cold_never_pads(fp8_weight):
    """cold 타일(128)과 K 정렬(128)이 같으므로 fp8 cold에는 패딩이 없다."""
    codes, scale = fp8_weight
    p = _prep8(codes, scale)
    assert p.sole.cold.real_rows == 128 and p.sole.cold.k_pad == 128


def test_fp8_scale_blocks_follow_the_rows(fp8_weight):
    """gather된 배율이 그 티어가 가져간 K 블록의 배율과 같아야 한다."""
    codes, scale = fp8_weight
    p = _prep8(codes, scale)
    for shard, blk in ((p.sole.hot, 0), (p.sole.warm, 1)):
        torch.testing.assert_close(shard.s_flat, scale[:, blk : blk + 1].t())
    torch.testing.assert_close(p.sole.cold.s_flat, scale[:, 2:3])


def test_fp8_split_block_dies(fp8_weight):
    """128 블록을 쪼개는 밴드는 즉사 — 안 죽으면 두 티어가 배율을 나눠 갖는다."""
    codes, scale = fp8_weight
    raw = _mutate8(bands=[[0, 64, "hot"], [64, 256, "warm"], [256, K8, "cold"]])
    with pytest.raises(PlanError, match="multiple of k_align=128"):
        _prep8(codes, scale, raw)


def test_fp8_non_contiguous_blocks_ok(fp8_weight):
    """블록 단위이기만 하면 티어가 흩어져도 된다 (hot이 두 조각)."""
    codes, scale = fp8_weight
    raw = _mutate8(
        bands=[[0, 128, "hot"], [128, 256, "cold"], [256, K8, "hot"]],
        cold_shards=[[0, 0, 256], [1, 256, N8]],
    )
    p = _prep8(codes, scale, raw)
    assert p.sole.hot.k_rows == 256 and not p.sole.hot.contiguous
    rows = p.sole.hot.k_index.to(torch.int64)
    torch.testing.assert_close(
        _dequant(p.sole.hot.w_flat, p.sole.hot.s_flat, k_major=True),
        _dequant(codes, scale, k_major=False)[:, rows].t(),
    )


def test_fp8_missing_scale_dies(fp8_weight):
    codes, _ = fp8_weight
    with pytest.raises(PlanError, match="needs weight_scale_inv"):
        _prep8(codes, None)


def test_fp8_wrong_scale_shape_dies(fp8_weight):
    codes, scale = fp8_weight
    with pytest.raises(PlanError, match="weight_scale_inv shape"):
        _prep8(codes, scale.t().contiguous())


def test_fp8_unaligned_dims_die(fp8_weight):
    """k/n이 128의 배수가 아니면 부분 배율 블록이 생긴다."""
    _, scale = fp8_weight
    raw = copy.deepcopy(PLAN8)
    raw["projs"][NAME].update(k=320, bands=[[0, 320, "hot"]], cold_shards=[])
    codes = torch.zeros(N8, 320, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    with pytest.raises(PlanError, match="multiples of 128"):
        _prep8(codes, torch.ones(N8 // BLOCK, 3), raw)


def test_fp8_ue8m0_scale_rejected(fp8_weight):
    """mxfp8(ue8m0 배율) 체크포인트는 아직 지원하지 않는다 — 조용히 fp32로 읽으면 오답."""
    codes, scale = fp8_weight
    with pytest.raises(PlanError, match="must be fp32"):
        _prep8(codes, scale.to(torch.uint8))


def test_fp8_bf16_weight_rejected():
    codes = torch.randn(N8, K8, dtype=torch.bfloat16)
    with pytest.raises(PlanError, match="must be float8_e4m3fn"):
        _prep8(codes, torch.ones(N8 // BLOCK, K8 // BLOCK))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
def test_fp8_real_placement(fp8_weight):
    """배율이 코드와 **같은 거처**로 간다 — 갈리면 커널이 host 포인터를 device로 읽는다."""
    from sglang.srt.layers.prism.numa import gpu_numa_node

    codes, scale = fp8_weight
    dev = torch.device("cuda:0")
    p = _prep8(codes, scale, device=dev, warm_node=gpu_numa_node(dev), pin_memory=True)

    assert p.sole.hot.w_flat.device.type == "cuda" and p.sole.hot.s_flat.device.type == "cuda"
    assert p.sole.warm.w_flat.is_pinned() and p.sole.warm.s_flat.is_pinned()
    assert p.sole.cold.w_flat.device.type == "cpu" and p.sole.cold.s_flat.device.type == "cpu"


# ── 혼합 포맷 (한 모델 안에서 proj마다 형식이 다르다) ──────────────────────
#
# DSV4의 `wo_a`는 SGLANG_OPT_FP8_WO_A_GEMM이 꺼져 있으면 quant_config=None +
# params_dtype=bfloat16으로 만들어진다 — 같은 모델의 wq_b/wo_b가 fp8인데 혼자
# bf16이다 (models/deepseek_v4.py:641). 커널 쌍이 model-global이면 그 모델을
# 아예 표현할 수 없으므로, plan은 proj별 덮어쓰기를 갖는다.

MIXED = {
    "schema_version": 1,
    "model_id": "test/mixed",
    "dims": {"num_layers": 2, "dtype": "bfloat16"},
    "kernels": {"gpu_warm": "gemv_worklist_fp8", "cpu_cold": "kt_tile_k2_fp8b128"},
    "projs": {
        "self_attn.wq_b": {
            "k": K8, "n": N8,
            "bands": [[0, 256, "hot"], [256, K8, "cold"]],
            "cold_shards": [[0, 0, 256], [1, 256, N8]],
        },
        "self_attn.wo_a": {   # 혼자 bf16
            "k": K, "n": N,
            "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_amx_bf16"},
            "bands": [[0, 64, "hot"], [64, K, "cold"]],
            "cold_shards": [[0, 0, 32], [1, 32, N]],
        },
    },
}


def test_mixed_format_plan(weight, fp8_weight):
    """같은 plan의 두 proj가 서로 다른 스토어 형식으로 절단된다."""
    from sglang.srt.layers.prism.linear.plan import validate_static

    plan = parse_plan(MIXED)
    validate_static(plan)
    assert plan.proj(0, "self_attn.wq_b").kernels.gpu_warm == "gemv_worklist_fp8"
    assert plan.proj(0, "self_attn.wo_a").kernels.gpu_warm == "gemv_worklist"

    codes, scale = fp8_weight
    q = prepare_linear_weights(0, "self_attn.wq_b", codes, plan, scale=scale,
                               device=torch.device("cpu"), pin_memory=False)
    o = prepare_linear_weights(0, "self_attn.wo_a", weight, plan,
                               device=torch.device("cpu"), pin_memory=False)

    assert q.fmt.name == "fp8" and q.sole.hot.s_flat is not None
    assert o.fmt.name == "bf16" and o.sole.hot.s_flat is None
    # 패딩 타일도 커널 키가 정하므로 proj마다 다르다: fp8 128, bf16 32.
    assert q.sole.cold.real_rows == K8 - 256 and q.sole.cold.k_pad == 128
    assert o.sole.cold.real_rows == K - 64 and o.sole.cold.k_pad == 160  # 136 → 32의 배수


def test_mixed_format_wrong_scale_for_bf16_proj(weight, fp8_weight):
    """bf16 proj에 fp8 배율을 주면 즉사 — 형식이 proj별이라 헷갈리기 쉬운 자리다."""
    plan = parse_plan(MIXED)
    _, scale = fp8_weight
    with pytest.raises(PlanError, match="takes no scales"):
        prepare_linear_weights(0, "self_attn.wo_a", weight, plan, scale=scale,
                               device=torch.device("cpu"), pin_memory=False)


# ═══════════════════════════════════════════════════════════════════════════
# N축 분할 (gate_up_proj)
# ═══════════════════════════════════════════════════════════════════════════
#
# weight는 `[2I, K]` 하나인데 sparsity가 gate/up을 따로 캘리브하므로 로드 시
# N축으로 쪼갠다. 여기서 지켜야 할 것: (1) gate가 앞 절반이어야 한다 — 뒤바뀌면
# SiluAndMul이 up에 silu를 걸어 조용히 다른 모델이 된다, (2) 조각마다 자기
# 밴딩으로 잘려야 한다, (3) 두 조각을 합치면 원본이어야 한다.

I2, H2 = 96, 128

SPLIT = {
    "schema_version": 1,
    "model_id": "test/split",
    "dims": {"num_layers": 2, "dtype": "bfloat16"},
    "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_tile_k2_bf16"},
    "projs": {
        "mlp.gate_up_proj": {
            "k": H2,
            "n": 2 * I2,
            "parts": [
                {"name": "gate", "n": I2, "bands": [[0, 64, "hot"], [64, H2, "cold"]],
                 "cold_shards": [[0, 0, I2]]},
                {"name": "up", "n": I2, "bands": [[0, 32, "warm"], [32, H2, "cold"]],
                 "cold_shards": [[0, 0, I2]]},
            ],
        }
    },
}


@pytest.fixture
def gu_weight():
    torch.manual_seed(2)
    return torch.randn(2 * I2, H2, dtype=torch.bfloat16)


def _prep_split(weight, raw=None, **kw):
    plan = parse_plan(raw or SPLIT)
    kw.setdefault("device", torch.device("cpu"))
    kw.setdefault("pin_memory", False)
    return prepare_linear_weights(0, "mlp.gate_up_proj", weight, plan, **kw)


def test_split_produces_two_parts(gu_weight):
    p = _prep_split(gu_weight)
    assert p.split and [q.name for q in p.parts] == ["gate", "up"]
    assert [(q.n_start, q.n_end) for q in p.parts] == [(0, I2), (I2, 2 * I2)]
    for q in p.parts:
        assert q.n == I2


def test_gate_is_the_first_half(gu_weight):
    """뒤바뀌면 SiluAndMul이 up에 silu를 걸어 조용히 다른 모델이 된다."""
    p = _prep_split(gu_weight)
    gate = p.part("gate")
    rows = gate.hot.k_index.to(torch.int64)
    # gate hot은 원본의 앞 I2행 × 그 K행이어야 한다
    torch.testing.assert_close(gate.hot.w_flat, gu_weight[:I2][:, rows].t())


def test_parts_carry_their_own_banding(gu_weight):
    p = _prep_split(gu_weight)
    gate, up = p.part("gate"), p.part("up")
    assert gate.hot is not None and gate.warm is None      # gate: hot + cold
    assert up.hot is None and up.warm is not None          # up: warm + cold
    assert gate.rows(Tier.HOT) == 64 and up.rows(Tier.WARM) == 32
    assert gate.rows(Tier.COLD) == H2 - 64
    assert up.rows(Tier.COLD) == H2 - 32


def test_split_reconstructs_the_original(gu_weight):
    """계약 ⑤ — 조각·티어를 전부 합치면 원본 [2I, K]가 나온다 (비트일치)."""
    p = _prep_split(gu_weight)
    out = torch.zeros(2 * I2, H2, dtype=gu_weight.dtype)
    for q in p.parts:
        for shard in (q.hot, q.warm):
            if shard is not None:
                out[q.n_start : q.n_end, shard.k_index.to(torch.int64)] = shard.w_flat.t()
        if q.cold is not None:
            r = q.cold.real_rows
            out[q.n_start : q.n_end, q.cold.k_index[:r].to(torch.int64)] = q.cold.w_flat[:, :r]
    assert torch.equal(out, gu_weight)


def test_rows_aggregates_over_parts(gu_weight):
    p = _prep_split(gu_weight)
    assert p.rows(Tier.HOT) == 64          # gate만
    assert p.rows(Tier.WARM) == 32         # up만
    assert p.rows(Tier.COLD) == (H2 - 64) + (H2 - 32)


def test_sole_on_split_dies(gu_weight):
    p = _prep_split(gu_weight)
    with pytest.raises(ValueError, match="split into 2 parts"):
        p.sole


def test_cold_shard_is_part_local(gu_weight):
    """cold 스토어의 N은 조각의 I이지 2I가 아니다 (kt 인스턴스가 조각마다 따로)."""
    p = _prep_split(gu_weight)
    for q in p.parts:
        assert q.cold.n == I2
