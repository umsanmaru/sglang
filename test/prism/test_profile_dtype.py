"""프로파일 API의 dtype 축 — 이름 하나가 백엔드 전부를 고르는지.

이 패키지는 "이 치수에서 이 연산이 몇 µs냐"에 답하고 그 답이 Plan의 입력이 된다.
dtype이 붙기 전에는 shape과 expert 수만 고를 수 있었고 스토어는 bf16 고정이었다 —
mxfp4/fp8 plan을 세우려면 그 dtype의 커널로 재야 한다.

여기서 보는 것은 (1) 이름 → (GPU 진입점, cold 커널, 정렬)의 대응, (2) 합성 스토어가
실제 커널이 받아들이는 형태라는 것(측정이 돌고 값이 나온다), (3) 잘못된 조합이 조용히
bf16으로 떨어지지 않고 ValueError로 죽는다는 것.
"""
import pytest
import torch

from sglang.srt.layers.moe.prism.profile import STORES, Shape, store_of

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

SHAPE = Shape(experts=8, topk=4, hidden=1024, inter=512)


def test_store_registry_maps_dtype_to_backend():
    """dtype → (포맷, cold 커널, 정렬). 세 축이 한 이름에 묶여 있어야 한다 (계약 ①)."""
    assert sorted(STORES) == ["bf16", "fp8", "mxfp4"]
    bf16, mx, f8 = (store_of(d) for d in ("bf16", "mxfp4", "fp8"))
    assert (bf16.fmt.name, bf16.cpu_kernel, bf16.k_align) == ("bf16", "kt_tile_k2_bf16", 2)
    assert (mx.fmt.name, mx.cpu_kernel, mx.k_align) == ("mxfp4", "kt_tile_k2_mxfp4", 32)
    assert (f8.fmt.name, f8.cpu_kernel, f8.k_align) == ("fp8", "kt_tile_k2_fp8b128", 128)
    # cold 커널은 포맷이 소비할 수 있는 것만
    for st in (bf16, mx, f8):
        assert st.cpu_kernel in st.cpu_kernels
    with pytest.raises(ValueError, match="unknown store dtype"):
        store_of("int4")


def test_store_shapes_match_the_kernel_contract():
    """합성 스토어의 모양이 커널 계약 그대로인가 (코드 행 수·배율 블록)."""
    E, k, n = 2, 256, 128
    w, = store_of("bf16").gpu_store(E, k, n)
    assert w.shape == (E * k, n) and w.dtype is torch.bfloat16
    codes, scales = store_of("mxfp4").gpu_store(E, k, n)
    assert codes.shape == (E * k // 2, n) and scales.shape == (E * k // 32, n)
    assert codes.dtype is torch.uint8 and scales.dtype is torch.uint8
    codes, scales = store_of("fp8").gpu_store(E, k, n)
    assert codes.shape == (E * k, n) and scales.shape == (E * k // 128, n // 128)
    assert codes.dtype is torch.uint8 and scales.dtype is torch.float32
    # 바이트 회계도 dtype을 따른다 (배율 포함)
    assert store_of("bf16").store_bytes(E, k, n) == E * k * n * 2
    assert store_of("mxfp4").store_bytes(E, k, n) == E * k * n // 2 + E * (k // 32) * n
    assert store_of("fp8").store_bytes(E, k, n) == E * k * n + E * (k // 128) * (n // 128) * 4


def test_alignment_violations_are_rejected():
    """배율 블록을 쪼개는 행 수는 즉사한다 — 조용히 어긋난 배율을 쓰느니."""
    with pytest.raises(ValueError, match="multiple of"):
        store_of("fp8").gpu_store(1, 64, 128)      # 64 < 128 블록
    with pytest.raises(ValueError, match="multiple of"):
        store_of("mxfp4").gpu_store(1, 48, 128)    # 48 % 32 != 0
    with pytest.raises(ValueError, match="multiple of 128"):
        store_of("fp8").gpu_store(1, 128, 64)      # N축 배율 블록


def test_cold_kernel_must_match_dtype():
    """dtype과 cold 커널이 어긋나면 로드 전에 죽는다 (bf16 커널로 fp8 slab을 읽지 않는다)."""
    from sglang.srt.layers.moe.prism.profile.warm_cold import ColdTier

    with pytest.raises(ValueError, match="cannot consume"):
        ColdTier(SHAPE, {}, sparsity=0.5, pattern="random", seed=0, numa_split=0.5,
                 threads=2, kernel_key="kt_amx_bf16", dtype="fp8")


@cuda_required
@pytest.mark.parametrize("dtype", ["bf16", "mxfp4", "fp8"])
def test_hot_dense_gemv_runs_on_each_dtype(dtype):
    """세 dtype 모두 자기 커널로 돌고, 리포트가 어느 커널이었는지 말한다."""
    from sglang.srt.layers.moe.prism.profile import hot_dense_gemv

    r = hot_dense_gemv(SHAPE, hot_frac=0.25, device=0, reps=4, replays=2, dtype=dtype)
    assert r.params["dtype"] == dtype
    assert dtype.replace("bf16", "worklist") in r.params["kernel"]
    assert r.layer_gemv_us > 0
    for res in r.results:
        assert res.dtype == dtype
        assert res.k_rows % store_of(dtype).rows_step() == 0
        assert res.us > 0


@cuda_required
@pytest.mark.parametrize("dtype", ["bf16", "mxfp4", "fp8"])
def test_warm_sparse_gemv_runs_on_each_dtype(dtype):
    """warm(pinned UVA) sparse GEMV도 dtype으로 갈린다 — 실현 keep 비율은 같다."""
    from sglang.srt.layers.moe.prism.profile import warm_sparse_gemv

    r = warm_sparse_gemv(1024, 512, 0.5, device=0, reps=4, replays=2, dtype=dtype)
    assert r.us > 0
    assert abs(r.keep_frac - 0.5) < 0.02
    assert r.dense_bytes == store_of(dtype).store_bytes(1, 1024, 512)


# ─── per-k 마스크 커널 (score k1) ─────────────────────────────────────────
def _has_k1_kernel() -> bool:
    try:
        from kt_kernel import kt_kernel_ext
    except ImportError:
        return False
    return hasattr(kt_kernel_ext.moe, "TileK1FP8B128_MOE")


k1_required = pytest.mark.skipif(not _has_k1_kernel(), reason="kt build without TileK1FP8B128_MOE")
K1 = "kt_tile_k1_fp8b128"


def test_per_k_levels_and_thr_realize_per_expert_sparsity():
    """레벨 x는 서로 다른 bf16 값, thr는 expert별 밴드 안 순서통계 — expert마다 다른 sparsity·
    다른 행 집합이 한 x로 실현되고, 실현 keep은 레벨 동률만큼만 어긋난다."""
    from sglang.srt.layers.moe.prism.profile import PER_K_LEVELS, per_k_levels, per_k_thr

    K = 2048
    x = per_k_levels(K, seed=3)
    assert x.dtype is torch.bfloat16 and x.unique().numel() == PER_K_LEVELS
    assert 0.125 <= float(x.min()) and float(x.max()) < 2.0
    assert torch.equal(x, per_k_levels(K, seed=3))                 # 결정적 (gate/up 공유)
    bands = [torch.arange(0, 1792), torch.randperm(K)[:1536], torch.arange(256, K)]
    thr, keeps, frac = per_k_thr(x, bands, [0.5, 0.75, 0.9])
    assert thr.shape == (3, 201) and torch.equal(thr[:, 0], thr[:, -1])   # grid 무관 상수
    for idx, keep, sp in zip(bands, keeps, (0.5, 0.75, 0.9)):
        assert keep.shape == idx.shape
        assert abs(keep.float().mean().item() - (1 - sp)) <= 1.0 / PER_K_LEVELS + 1e-9
    # keep은 정확히 |x| >= thr (커널과 같은 규칙)
    assert torch.equal(keeps[0], x[bands[0]].float() >= thr[0, 0])
    # 극단: 전부 살림 / 전부 죽임
    thr2, keeps2, _ = per_k_thr(x, [torch.arange(K)] * 2, [0.0, 1.0])
    assert keeps2[0].all() and not keeps2[1].any()
    # block 패턴은 앞 행이 산다
    xb = per_k_levels(K, pattern="block", seed=0)
    _, kb, _ = per_k_thr(xb, [torch.arange(K)], 0.75)
    assert kb[0][: K // 4].all() and not kb[0][K // 4:].any()


def test_k1_kernel_is_known_to_the_profiler():
    """N 정렬표와 fp8 포맷의 cold 커널 목록 둘 다에 있어야 `cpu_kernel=`로 고를 수 있다."""
    from sglang.srt.layers.moe.prism.profile import N_ALIGN, kernel_mask_per_k

    assert N_ALIGN[K1] == N_ALIGN["kt_tile_k2_fp8b128"] == 256
    assert K1 in store_of("fp8").cpu_kernels
    assert kernel_mask_per_k("kt_tile_k2_fp8b128") is False
    assert kernel_mask_per_k("no_such_kernel") is False


@k1_required
def test_cold_cpu_k1_realizes_sparsity_through_x():
    """k1 커널로 cold_cpu가 돌고 마스크가 x·thr로 실현된다 (dense보다 sparse가 빠르다).
    gate/up 인덱스가 달라도(split_index) 각자 thr로 실현되므로 허용된다."""
    from sglang.srt.layers.moe.prism.profile import kernel_mask_per_k, cold_cpu

    assert kernel_mask_per_k(K1) is True
    shape = Shape(experts=8, topk=2, hidden=1024, inter=512)
    kw = dict(cold_frac=1.0, dtype="fp8", cpu_kernel=K1, threads=4, numa_map=[0],
              iters=20, replays=2)
    dense = cold_cpu(shape, sparsity=0.0, **kw)
    sparse = cold_cpu(shape, sparsity=0.75, split_index=True, **kw)
    assert dense.mask_per_k and sparse.mask_per_k
    assert dense.keep_frac == 1.0 and abs(sparse.keep_frac - 0.25) < 0.01
    assert sparse.us < dense.us


@k1_required
@cuda_required
@pytest.mark.parametrize("group", ["gateup", "down"])
def test_warm_cold_k1_matches_x_weighted_reference(group):
    """k1: warm(GPU per-k)·cold(kt per-k)가 같은 레벨 x·thr를 보고, 레퍼런스 Σ x_k·W와 맞는다.
    죽인 행도 x ≠ 0이므로 마스킹이 빠지면 여기서 드러난다."""
    from sglang.srt.layers.moe.prism.profile import WarmColdProfiler

    shape = Shape(experts=8, topk=4, hidden=1024, inter=512)
    with WarmColdProfiler(shape, warm_frac=0.25, cold_frac=0.75, sparsity=0.6, device=0,
                          dtype="fp8", cpu_kernel=K1, threads=4, numa_map=[0]) as p:
        assert p.params["mask_per_k"] is True
        assert p.warm["gate"].spec.per_k and p.warm["gate"].spec.a is None
        rep = p.check(group)
        for k, v in rep.items():
            assert v < 0.02, f"{k}={v}"
        r = p.measure(group, reps=4, replays=2, only=("warm_only", "cold_only"))
        assert abs(r.info["warm_keep_frac"] - 0.4) < 0.01 and abs(r.info["cold_keep_frac"] - 0.4) < 0.01


@k1_required
@cuda_required
def test_full_layer_k1_with_expert_varied_tiers_and_sparsity():
    """expert마다 다른 티어 경계·sparsity도 per-k로 실현된다 (순서통계 thr) — 세 티어 합이 레퍼런스와 맞는다."""
    from sglang.srt.layers.moe.prism.profile import FullLayerProfiler

    with FullLayerProfiler(Shape(experts=16, topk=4, hidden=1024, inter=512),
                           hot_frac=0.25, warm_frac=0.25, sparsity=0.5, sparsity_spread=0.3,
                           hot_spread=0.1, warm_spread=0.05, dtype="fp8", cpu_kernel=K1,
                           device=0, seed=0, threads=4, numa_map=[0]) as p:
        assert p.params["mask_per_k"] is True
        for group in ("gateup", "down"):
            rep = p.check(group)
            errs = [v for k, v in rep.items() if k.endswith("_max_rel_err")]
            assert errs and max(errs) < 0.02, rep
