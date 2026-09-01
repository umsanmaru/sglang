"""dense executor — GPU 티어 실행과 rejoin (CUDA 필요).

여기서 지키는 것은 계약 ①과 ⑤다:

  ① hot과 warm의 **계산 계약이 완전히 동일하다** — 스토어가 device냐 pinned냐만
    다르므로 같은 입력에 **비트일치** 출력이어야 한다.
  ⑤ K를 어떻게 나누든 결과가 같다 — 경계를 옮긴 두 plan이 비트일치해야 한다.

정확도의 기준선은 **torch 자신의 bf16 matmul**이다. "fp32와 얼마나 가까운가"를
절대값으로 재면 bf16 출력 라운딩(2⁻⁸)에 묻혀 버그와 구분이 안 되므로, 같은
입력에 대한 `x @ w.t()`의 오차와 나란히 놓고 본다.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")

from sglang.srt.layers.prism.geometry import PlanError
from sglang.srt.layers.prism.linear.executor import LinearExecutor
from sglang.srt.layers.prism.linear.plan import parse_plan, validate_static
from sglang.srt.layers.prism.linear.weights import prepare_linear_weights

K, N = 512, 256
DEV = "cuda:0"


def _plan(projs):
    return parse_plan({
        "schema_version": 1, "model_id": "t",
        "dims": {"num_layers": 1, "dtype": "bfloat16"},
        "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_tile_k2_bf16"},
        "projs": projs,
    })


def _fit(projs, weights):
    plan = _plan(projs)
    validate_static(plan)
    dev = torch.device(DEV)
    ex = LinearExecutor(max_tokens=512, device=dev)   # grouped 테스트가 M=256까지 쓴다
    for name, w in weights.items():
        ex.register(0, name, prepare_linear_weights(0, name, w, plan, device=dev,
                                                    pin_memory=True))
    return ex


@pytest.fixture
def w():
    torch.manual_seed(0)
    return torch.randn(N, K, dtype=torch.bfloat16)


BANDS = {
    "all_hot": [[0, K, "hot"]],
    "all_warm": [[0, K, "warm"]],
    "hw": [[0, 128, "hot"], [128, K, "warm"]],
    "wh": [[0, 128, "warm"], [128, K, "hot"]],
    "split3": [[0, 64, "hot"], [64, 192, "warm"], [192, K, "hot"]],
}


@pytest.fixture
def ex(w):
    return _fit({n: {"k": K, "n": N, "bands": b} for n, b in BANDS.items()},
                {n: w for n in BANDS})


def test_accuracy_matches_torch_bf16(ex, w):
    """단일 티어는 torch의 bf16 matmul과 **같은 오차**여야 한다.

    더 나쁘면 절단이 틀렸다는 뜻이고, 더 좋을 수는 없다 (같은 fp32 누산 + 같은
    최종 라운딩).
    """
    x = torch.randn(16, K, dtype=torch.bfloat16, device=DEV)
    ref = x.float() @ w.float().t().to(DEV)
    baseline = (x @ w.t().to(DEV)).float().sub(ref).abs().max().item()

    for name in ("all_hot", "all_warm"):
        err = ex.run(0, name, x).float().sub(ref).abs().max().item()
        assert err == pytest.approx(baseline, rel=1e-6), name


def test_hot_and_warm_are_bit_identical(ex):
    """계약 ①: 거처만 다르고 계산은 같다."""
    x = torch.randn(16, K, dtype=torch.bfloat16, device=DEV)
    assert torch.equal(ex.run(0, "all_hot", x), ex.run(0, "all_warm", x))


def test_placement_invariance(ex):
    """계약 ⑤: 같은 K를 다르게 나눠도 결과가 같다.

    partial의 wire가 bf16이지만 **값 자체는 티어와 무관**하므로 fp32 합도 같다 —
    경계를 옮겨도 비트일치다.
    """
    x = torch.randn(16, K, dtype=torch.bfloat16, device=DEV)
    assert torch.equal(ex.run(0, "hw", x), ex.run(0, "wh", x))


def test_multi_band_tier(ex, w):
    """한 티어가 여러 밴드로 흩어져도(gather 경로) 값이 맞아야 한다."""
    x = torch.randn(8, K, dtype=torch.bfloat16, device=DEV)
    ref = x.float() @ w.float().t().to(DEV)
    err = ex.run(0, "split3", x).float().sub(ref).abs().max().item()
    assert err / ref.abs().max().item() < 1e-2


@pytest.mark.parametrize("m", [1, 2, 7, 16, 64])
def test_shapes_across_m(ex, m):
    x = torch.randn(m, K, dtype=torch.bfloat16, device=DEV)
    out = ex.run(0, "hw", x)
    assert out.shape == (m, N) and out.dtype is torch.bfloat16


def test_max_tokens_guard(ex):
    x = torch.randn(513, K, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError, match="exceeds max_tokens"):
        ex.run(0, "hw", x)


# ── N축 분할 (gate_up_proj) ────────────────────────────────────────────────


I2 = 128
GU = {"mlp.gate_up_proj": {"k": K, "n": 2 * I2, "parts": [
    {"name": "gate", "n": I2, "bands": [[0, 128, "hot"], [128, K, "warm"]]},
    {"name": "up", "n": I2, "bands": [[0, K, "hot"]]},   # 일부러 다른 밴딩
]}}


def test_split_output_layout_is_gate_then_up():
    """`apply`가 돌려주는 `[M, 2I]`에서 gate가 앞이어야 sglang의 SiluAndMul이 맞다.

    뒤바뀌면 up에 silu가 걸려 조용히 다른 모델이 된다.
    """
    torch.manual_seed(1)
    w = torch.randn(2 * I2, K, dtype=torch.bfloat16)
    ex = _fit(GU, {"mlp.gate_up_proj": w})
    x = torch.randn(8, K, dtype=torch.bfloat16, device=DEV)

    out = ex.run(0, "mlp.gate_up_proj", x)
    ref = x.float() @ w.float().t().to(DEV)
    assert out.shape == (8, 2 * I2)
    # 두 절반을 각각 대조한다 — 통째로 보면 뒤바뀜을 놓친다
    for lo, hi, half in ((0, I2, "gate"), (I2, 2 * I2, "up")):
        err = out[:, lo:hi].float().sub(ref[:, lo:hi]).abs().max().item()
        assert err / ref.abs().max().item() < 1e-2, half


def test_split_zeroes_columns_a_tier_does_not_write():
    """gate만 warm이면 warm 버퍼의 up 열은 0이어야 한다 — empty면 쓰레기가 더해진다."""
    torch.manual_seed(1)
    w = torch.randn(2 * I2, K, dtype=torch.bfloat16)
    ex = _fit(GU, {"mlp.gate_up_proj": w})
    x = torch.randn(4, K, dtype=torch.bfloat16, device=DEV)
    ref = x.float() @ w.float().t().to(DEV)
    # 여러 번 돌려도 같은 값 — 초기화 안 된 메모리가 섞이면 호출마다 달라진다
    outs = [ex.run(0, "mlp.gate_up_proj", x) for _ in range(4)]
    assert all(torch.equal(outs[0], o) for o in outs[1:])
    assert outs[0].float().sub(ref).abs().max().item() / ref.abs().max().item() < 1e-2


# ── 방어 ───────────────────────────────────────────────────────────────────


def test_cold_rows_are_rejected(w):
    """cold 백엔드 없이 cold plan을 주면 즉사 — 조용히 그 행을 빼면 값이 틀린다.

    백엔드가 붙은 경로는 `test_linear_cold.py`가 본다.
    """
    with pytest.raises(NotImplementedError, match="cold 백엔드가 없다"):
        _fit({"p": {"k": K, "n": N, "bands": [[0, 128, "hot"], [128, K, "cold"]],
                    "cold_shards": [[0, 0, N]]}}, {"p": w})


def test_unregistered_proj_raises(ex):
    x = torch.randn(4, K, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(KeyError, match="not registered"):
        ex.run(0, "nope", x)


def test_k_mismatch_raises(ex):
    x = torch.randn(4, K + 2, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError, match="K="):
        ex.run(0, "hw", x)


# ═══════════════════════════════════════════════════════════════════════════
# grouped GEMM (prefill 형태)
# ═══════════════════════════════════════════════════════════════════════════
#
# worklist는 pair마다 W를 다시 읽는데 **dense는 E=1이라 중복도가 곧 M**이다
# (MoE는 M·k/E). warm은 그 재읽기가 전부 PCIe라 큰 M에서 치명적이다 — Qwen3.8
# M=2048에서 forward 한 번에 89.7 TB, 약 30분이었다 (2026-09-01, prefill CUDA graph
# 캡처가 안 끝나서 발견). grouped는 W를 한 번만 읽는다.
#
# 둘은 같은 스토어를 읽고 같은 레이아웃에 쓰므로 **정확도가 같아야** 한다. 커널
# 선택이 값을 바꾸면 M 경계에서 출력이 튀는데, 그건 어떤 단위 테스트도 안 잡는다.


BF16_EPS = 2.0**-8          # bf16 상대 정밀도 (8비트 가수)


def _both_paths(ex, name, x):
    """같은 입력을 worklist와 grouped로 각각 계산한다 (임계만 바꿔서)."""
    hi = ex._grouped_min_m
    ex._grouped_min_m = 1 << 30          # worklist 강제
    a = ex.run(0, name, x).clone()
    ex._grouped_min_m = 1               # grouped 강제
    b = ex.run(0, name, x).clone()
    ex._grouped_min_m = hi
    return a, b


@pytest.mark.parametrize("m", [16, 64, 129, 256])
def test_grouped_is_as_accurate_as_worklist(ex, w, m):
    """커널 선택은 호출 형태의 결정이지 계약이 아니다 — **정확도가 같아야** 한다.

    "비트일치"가 아니라 "같은 정확도"인 이유: 티어가 둘 이상이면 partial이 bf16으로
    wire되고(계약 ⑤), 두 커널의 K축 누산 순서가 달라 마지막 라운딩이 갈릴 수 있다.
    실측 0.1~0.4 bf16 ULP다. 단일 티어에서는 실제로 비트일치한다.

    그래서 판정을 **fp32 기준 오차의 동일성**으로 한다 — 한쪽이 더 나쁘면 그건
    라운딩이 아니라 결함이다. TILE_M=128 경계(129)를 포함한다.
    """
    x = torch.randn(m, K, dtype=torch.bfloat16, device=DEV)
    ref = x.float() @ w.float().t().to(DEV)
    scale = ref.abs().max().item()
    for name in ("all_hot", "all_warm", "hw"):
        a, b = _both_paths(ex, name, x)
        ea = (a.float() - ref).abs().max().item()
        eb = (b.float() - ref).abs().max().item()
        assert eb == pytest.approx(ea, rel=1e-3), f"{name} M={m}: grouped {eb} vs worklist {ea}"
        # 두 경로의 차이는 wire 라운딩 규모를 넘지 않는다
        d = (b.float() - a.float()).abs().max().item()
        assert d <= 4 * BF16_EPS * scale, f"{name} M={m}: {d / scale / BF16_EPS:.1f} ULP"


def test_grouped_diff_is_reassociation_only(ex, w):
    """두 경로가 다른 원소는 **극소수**이고 차이는 라운딩 규모다.

    비트일치는 성립하지 않는다: 두 커널의 K 타일링이 달라 fp32 누산 순서가 다르고,
    티어가 둘이면 partial의 bf16 wire 라운딩까지 겹친다. 실측(K=512, N=256, M=64)은
    단일 티어에서 16384개 중 **2개**가 0.35 ULP다 — 결함이면 이 비율이 아니다.
    """
    x = torch.randn(64, K, dtype=torch.bfloat16, device=DEV)
    ref = x.float() @ w.float().t().to(DEV)
    scale = ref.abs().max().item()
    for name in ("all_hot", "all_warm"):
        a, b = _both_paths(ex, name, x)
        diff = (a != b).sum().item()
        assert diff / a.numel() < 1e-3, f"{name}: {diff}/{a.numel()} 원소가 다르다"
        assert (a.float() - b.float()).abs().max().item() <= 2 * BF16_EPS * scale


def test_grouped_accuracy_vs_reference(ex, w):
    """grouped도 torch bf16 matmul과 같은 급이어야 한다."""
    x = torch.randn(128, K, dtype=torch.bfloat16, device=DEV)
    ref = x.float() @ w.float().t().to(DEV)
    baseline = (x @ w.t().to(DEV)).float().sub(ref).abs().max().item()
    hi = ex._grouped_min_m
    ex._grouped_min_m = 1
    for name in ("all_hot", "all_warm"):
        err = ex.run(0, name, x).float().sub(ref).abs().max().item()
        assert err <= baseline * 2, f"{name}: {err} vs baseline {baseline}"
    ex._grouped_min_m = hi


def test_grouped_threshold_switches(ex):
    """임계 아래는 worklist, 위는 grouped — 둘 다 돌아야 한다."""
    assert ex._grouped_min_m == 16
    for m in (15, 16):
        out = ex.run(0, "hw", torch.randn(m, K, dtype=torch.bfloat16, device=DEV))
        assert out.shape == (m, N)


def test_grouped_split_layout(ex):
    """N축 분할에서도 grouped가 조각별 열 오프셋을 지킨다."""
    torch.manual_seed(1)
    w = torch.randn(2 * I2, K, dtype=torch.bfloat16)
    e = _fit(GU, {"mlp.gate_up_proj": w})
    x = torch.randn(64, K, dtype=torch.bfloat16, device=DEV)
    a, b = _both_paths(e, "mlp.gate_up_proj", x)
    torch.testing.assert_close(b, a, rtol=2e-2, atol=1e-2)
    ref = x.float() @ w.float().t().to(DEV)
    for lo, hi_, half in ((0, I2, "gate"), (I2, 2 * I2, "up")):
        assert b[:, lo:hi_].float().sub(ref[:, lo:hi_]).abs().max().item() \
            / ref.abs().max().item() < 1e-2, half


def test_grouping_is_cached_per_m(ex):
    """E=1 grouping은 자명하다 — 스텝마다 정렬하면 그게 곧 오버헤드다."""
    x = torch.randn(32, K, dtype=torch.bfloat16, device=DEV)
    ex.run(0, "hw", x); ex.run(0, "hw", x)
    assert 32 in ex._group_cache
    g = ex._group_cache[32]
    assert int(g.pair_off[1]) == 32 and int(g.tile_off[1]) == 1   # TILE_M=128


def test_grouped_rejects_masking(ex):
    """sparsity는 decode 전용 — grouped와 동시에 오면 두 티어의 마스크가 갈린다."""
    from sglang.srt.layers.prism.linear.tiers import LinearGpuTier

    st = ex._projs[(0, "hw")]
    adapter = next(iter(st.tiers.values()))[0]
    x = torch.randn(32, K, dtype=torch.bfloat16, device=DEV)
    buf = torch.zeros(32, 1, N, dtype=torch.bfloat16, device=DEV)
    ids, ones = ex._degenerate(32, x.device)
    with pytest.raises(ValueError, match="decode 전용"):
        adapter.run(x, ids, ones, buf, masking=True, grouping=ex._grouping(32, x.device))
