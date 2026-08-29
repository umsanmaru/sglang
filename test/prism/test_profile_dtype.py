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
