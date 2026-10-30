"""MoE `Fp8PtFormat`(per-tensor fp8) — 레지스트리 등록, 파라미터 형태, 절단·재조립, 정렬 계약.

fp8(블록판)과 갈리는 것은 배율 기하뿐이다: `w13_weight_scale [E, 2]`(gate, up), `w2_weight_scale
[E]`, 티어 스토어의 `s_flat`은 proj당 `[E]`. 코드 스토어는 fp8과 같으니 재조립 비트일치는 코드로
검사하고, 배율은 "어느 expert의 스칼라가 어느 슬롯에 붙는가"만 본다.

CUDA 없이 돈다 (weights 로더 경로 검증; hot을 CPU device로 둔다).
"""
import pytest
import torch

from sglang.srt.layers.moe.prism.formats import FORMATS, FP8PT, FullWeights, Fp8PtFormat
from sglang.srt.layers.moe.prism.index import TierIndex
from sglang.srt.layers.moe.prism.plan import PlanError, Proj, parse_plan, validate_static
from sglang.srt.layers.moe.prism.weights import prepare_layer_weights

DIMS = {"hidden_size": 256, "intermediate_size": 128, "num_layers": 1, "num_experts": 4,
        "top_k": 2, "dtype": "bfloat16"}
E, H, I = DIMS["num_experts"], DIMS["hidden_size"], DIMS["intermediate_size"]
PIN = torch.cuda.is_available()


def make_plan(gate_bands=None, up_bands=None, down_bands=None, cpu_cold="kt_tile_k2_fp8pt"):
    def proj_entry(bands, N):
        has_cold = any(t == "cold" for _, _, t in bands)
        return {"bands": bands, "cold_shards": [[0, 0, N // 2], [1, N // 2, N]] if has_cold else []}

    raw = {
        "schema_version": 1, "model_id": "test/fp8pt", "dims": dict(DIMS),
        "kernels": {"gpu_warm": "gemv_worklist_fp8pt", "cpu_cold": cpu_cold},
        "default": {
            "gate": proj_entry(gate_bands or [[0, 64, "warm"], [64, 256, "cold"]], I),
            "up": proj_entry(up_bands or [[0, 64, "warm"], [64, 256, "cold"]], I),
            "down": proj_entry(down_bands or [[0, 64, "warm"], [64, 128, "cold"]], H),
        },
    }
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def fp8_codes(shape, g):
    e = torch.randint(1, 15, shape, generator=g)
    m = torch.randint(0, 8, shape, generator=g)
    s = torch.randint(0, 2, shape, generator=g)
    return ((s << 7) | (e << 3) | m).to(torch.uint8)


def make_weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    w13 = fp8_codes((E, 2 * I, H), g).view(torch.float8_e4m3fn)
    w2 = fp8_codes((E, H, I), g).view(torch.float8_e4m3fn)
    w13_s = (torch.rand(E, 2, generator=g) + 0.5).float()
    w2_s = (torch.rand(E, generator=g) + 0.5).float()
    return w13, w2, w13_s, w2_s


def reassemble_codes(prepared, proj, K, N):
    """세 티어의 코드 행을 인덱스대로 원위치에 → [E, N, K] u8 (test_weights.reassemble의 u8판)."""
    out = torch.full((E, N, K), 255, dtype=torch.int16)  # 255 센티널 (코드 0xFF는 생성하지 않는다)

    def scatter(row_off, k_index, rows_of_expert, real=None):
        for e in range(E):
            o0, o1 = int(row_off[e]), int(row_off[e + 1])
            n = o1 - o0 if real is None else int(real[e])
            if n:
                out[e].index_copy_(1, k_index[o0:o0 + n].to(torch.int64), rows_of_expert(e, o0, n).to(torch.int16))

    for store in (prepared.hot, prepared.warm):
        sh = None if store is None else store.band(proj)
        if sh is None:
            continue
        w = sh.w_flat.cpu()
        scatter(sh.row_off.cpu(), sh.k_index.cpu(), lambda e, o0, n, w=w: w[o0:o0 + n].t())
    cold = prepared.cold.band(proj)
    if cold is not None:
        ro, real = cold.row_off.cpu(), cold.real_rows.cpu()

        def cold_rows(e, o0, n, flat=cold.w_flat, ro=ro):
            kp = int(ro[e + 1]) - int(ro[e])
            return flat[int(ro[e]) * N:int(ro[e + 1]) * N].view(N, kp)[:, :n]
        scatter(ro, cold.k_index.cpu(), cold_rows, real=real)
    assert not torch.any(out == 255), "티어들이 K를 완전히 덮지 못했다"
    return out.to(torch.uint8)


# ── 레지스트리 / 파라미터 ─────────────────────────────────────────────────────
def test_registered_in_moe_formats():
    assert FORMATS["fp8pt"] is FP8PT and isinstance(FP8PT, Fp8PtFormat)
    assert FP8PT.k_align == 32 and FP8PT.has_scales and FP8PT.supports_cold
    assert FP8PT.cold_kernels == ("kt_tile_k2_fp8pt",)
    # 진입점 묶음이 전부 있다 (프로파일러 전용 어댑터에는 gemv 둘만 있었다)
    for name in ("gemv", "gemv_gateup"):
        assert getattr(FP8PT, name)(pinned=True, sparse=True).__name__.startswith("gemv_fp8pt_")
    for name in ("grouped", "grouped_gateup"):
        assert getattr(FP8PT, name)(pinned=False).__name__.startswith("grouped_fp8pt_")
    assert FP8PT.grouped_cold().__name__ == "grouped_fp8_cold"
    assert FP8PT.grouped_cold_gateup().__name__ == "grouped_fp8_cold_gateup"
    from sglang.jit_kernel.prism_grouped_fp8 import COLD_LAYOUTS
    from sglang.srt.layers.prism.kernels import cold_slab_layout

    assert COLD_LAYOUTS[cold_slab_layout("kt_tile_k2_fp8pt")] == 3  # KT_TILE8_PT


def test_create_params_matches_sglang_per_tensor_moe():
    """sglang `Fp8MoEMethod`(per-tensor)와 같은 이름·모양이어야 체크포인트 로더가 채운다."""
    layer = torch.nn.Module()
    FP8PT.create_params(layer, E, H, I, torch.bfloat16, {})
    assert tuple(layer.w13_weight.shape) == (E, 2 * I, H) and layer.w13_weight.dtype == torch.float8_e4m3fn
    assert tuple(layer.w2_weight.shape) == (E, H, I)
    assert tuple(layer.w13_weight_scale.shape) == (E, 2) and layer.w13_weight_scale.dtype == torch.float32
    assert tuple(layer.w2_weight_scale.shape) == (E,)
    assert getattr(layer.w13_weight_scale, "quant_method") == "tensor"
    assert tuple(layer.w13_input_scale.shape) == (E,) and tuple(layer.w2_input_scale.shape) == (E,)
    assert not hasattr(layer, "w13_weight_scale_inv")  # 블록판 이름은 없다

    w13, w2, w13_s, w2_s = make_weights()
    layer.w13_weight.data, layer.w2_weight.data = w13, w2
    layer.w13_weight_scale.data, layer.w2_weight_scale.data = w13_s, w2_s
    full = FP8PT.take_full(layer)
    assert torch.equal(full.w13_scale, w13_s) and torch.equal(full.w2_scale, w2_s)
    FP8PT.release(layer)
    assert layer.w13_weight.numel() == 0 and layer.w13_weight_scale.numel() == 0


def test_proj_source_picks_the_right_scalar():
    w13, w2, w13_s, w2_s = make_weights()
    full = FullWeights(w13=w13, w2=w2, w13_scale=w13_s, w2_scale=w2_s)
    assert torch.equal(FP8PT.proj_source(full, I, Proj.GATE).scales, w13_s[:, 0])
    assert torch.equal(FP8PT.proj_source(full, I, Proj.UP).scales, w13_s[:, 1])
    assert torch.equal(FP8PT.proj_source(full, I, Proj.DOWN).scales, w2_s)
    assert FP8PT.proj_source(full, I, Proj.UP).w.dtype == torch.uint8


def test_check_full_shapes_rejects_block_scales():
    w13, w2, w13_s, w2_s = make_weights()
    plan = make_plan()
    with pytest.raises(PlanError, match="scales"):
        FP8PT.check_full_shapes(FullWeights(w13, w2, torch.ones(E, 2 * I // 128, H // 128), w2_s),
                                plan.dims, "t")
    with pytest.raises(PlanError, match="per-tensor scales"):
        FP8PT.check_full_shapes(FullWeights(w13, w2, None, None), plan.dims, "t")


# ── 정렬 계약 ────────────────────────────────────────────────────────────────
def test_check_index_requires_rows_multiple_of_32_but_any_positions():
    good = TierIndex(row_off=torch.tensor([0, 32, 96, 96, 128], dtype=torch.int32),
                     idx=torch.arange(128, dtype=torch.int64), contiguous=False)
    FP8PT.check_index(good, "t")  # 0행 expert, 임의 위치 허용
    # 위치가 128 블록이 아니어도(예: 홀수 블록 섞임) 통과 — 배율 블록 제약이 없다
    scattered = TierIndex(row_off=torch.tensor([0, 32], dtype=torch.int32),
                          idx=torch.randperm(256)[:32].sort().values, contiguous=False)
    FP8PT.check_index(scattered, "t")
    bad = TierIndex(row_off=torch.tensor([0, 30], dtype=torch.int32),
                    idx=torch.arange(30, dtype=torch.int64), contiguous=False)
    with pytest.raises(PlanError, match="multiple of 32"):
        FP8PT.check_index(bad, "t")


# ── 절단 → 재조립 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("gate_bands", [None, [[0, 96, "warm"], [96, 256, "cold"]]])  # 96: 128 정렬 아님
def test_roundtrip_codes_bitexact_and_scales_per_expert(gate_bands):
    plan = make_plan(gate_bands=gate_bands, up_bands=gate_bands)
    w13, w2, w13_s, w2_s = make_weights(seed=3)
    prepared = prepare_layer_weights(0, w13, w2, plan, w13_scale=w13_s, w2_scale=w2_s,
                                     pin_memory=PIN, cold_tile_rows=32)
    codes13 = w13.view(torch.uint8)
    assert torch.equal(reassemble_codes(prepared, Proj.GATE, H, I), codes13[:, :I])
    assert torch.equal(reassemble_codes(prepared, Proj.UP, H, I), codes13[:, I:])
    assert torch.equal(reassemble_codes(prepared, Proj.DOWN, I, H), w2.view(torch.uint8))
    # 배율: 티어마다 proj당 [E] — 잘리지도 늘어나지도 않는다
    warm_g = prepared.warm.band(Proj.GATE)
    assert torch.equal(warm_g.s_flat.cpu(), w13_s[:, 0])
    assert torch.equal(prepared.warm.band(Proj.UP).s_flat.cpu(), w13_s[:, 1])
    assert torch.equal(prepared.warm.band(Proj.DOWN).s_flat.cpu(), w2_s)
    cold_g = prepared.cold.band(Proj.GATE)
    assert torch.equal(cold_g.s_flat, w13_s[:, 0]) and cold_g.s_flat.dtype == torch.float32
    assert FP8PT.cold_load_kwargs(prepared.cold)["down_scale"] is prepared.cold.band(Proj.DOWN).s_flat


def test_cold_padding_to_32_keeps_scale_vector():
    """cold 행 수가 32 배수가 아니면(66 → 190행) 로더가 32로 올린다 — 배율은 [E] 그대로."""
    plan = make_plan(gate_bands=[[0, 66, "warm"], [66, 256, "cold"]], up_bands=[[0, 66, "warm"], [66, 256, "cold"]])
    w13, w2, w13_s, w2_s = make_weights(seed=4)
    # warm 66행은 32 배수가 아니라 거절돼야 한다 (gemv 청크 계약)
    with pytest.raises(PlanError, match="multiple of 32"):
        prepare_layer_weights(0, w13, w2, plan, w13_scale=w13_s, w2_scale=w2_s, pin_memory=PIN,
                              cold_tile_rows=32)
    plan = make_plan(gate_bands=[[0, 64, "warm"], [64, 254, "cold"], [254, 256, "warm"]],
                     up_bands=[[0, 64, "warm"], [64, 254, "cold"], [254, 256, "warm"]])
    # warm 합 66행(64+2) → 역시 거절. cold 190행은 타일 올림 대상이라 허용될 것이지만 warm이 먼저 막는다.
    with pytest.raises(PlanError, match="multiple of 32"):
        prepare_layer_weights(0, w13, w2, plan, w13_scale=w13_s, w2_scale=w2_s, pin_memory=PIN,
                              cold_tile_rows=32)


def test_every_tier_must_be_32_aligned_so_cold_padding_never_triggers():
    """K가 32 배수(check_full_shapes)이고 티어 셋이 [0, K)를 나누므로, 한 티어의 행 수가 32 배수가
    아니면 다른 티어도 아니다 → 어느 한쪽 check_index가 거절한다. 그래서 fp8pt에서는 cold 타일
    패딩이 실제로 발생하지 않는다 — 정렬된 plan에서 real_rows == 스토어 행 수임을 확인한다."""
    plan = make_plan(gate_bands=[[0, 192, "warm"], [192, 222, "cold"], [222, 256, "warm"]],
                     up_bands=[[0, 192, "warm"], [192, 222, "cold"], [222, 256, "warm"]])
    w13, w2, w13_s, w2_s = make_weights(seed=5)
    with pytest.raises(PlanError, match="multiple of 32"):
        prepare_layer_weights(0, w13, w2, plan, w13_scale=w13_s, w2_scale=w2_s, pin_memory=PIN,
                              cold_tile_rows=32)
    plan = make_plan()
    prepared = prepare_layer_weights(0, w13, w2, plan, w13_scale=w13_s, w2_scale=w2_s, pin_memory=PIN,
                                     cold_tile_rows=32)
    cold = prepared.cold.band(Proj.GATE)
    rows = [int(cold.row_off[e + 1]) - int(cold.row_off[e]) for e in range(E)]
    assert cold.real_rows.tolist() == rows == [192] * E


def test_block_fp8_cold_kernel_rejected_for_pt_store():
    """fp8pt 스토어를 블록판 cold 커널이 소비하면 배율 표 자리에 스칼라를 읽는다 — 포맷의
    `check_cold_kernel`이 거절한다 (plan 파싱은 통과하고 cold 백엔드 startup에서 즉사하는 검사)."""
    with pytest.raises(PlanError, match="cannot consume"):
        FP8PT.check_cold_kernel("kt_tile_k2_fp8b128")
    with pytest.raises(PlanError, match="cannot consume"):
        FORMATS["fp8"].check_cold_kernel("kt_tile_k2_fp8pt")
    FP8PT.check_cold_kernel("kt_tile_k2_fp8pt")
