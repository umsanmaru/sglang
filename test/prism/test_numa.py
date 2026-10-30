"""NUMA 탐지 테스트. 값은 머신 의존이므로 범위·일관성만 검증한다."""

import pytest
import torch

from sglang.srt.layers.prism.numa import (
    gpu_numa_node,
    numa_node_count,
    other_numa_node,
)

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def test_node_count_positive():
    assert numa_node_count() >= 1


@cuda_required
def test_gpu_numa_node_in_range():
    count = numa_node_count()
    for i in range(torch.cuda.device_count()):
        node = gpu_numa_node(i)
        assert 0 <= node < count
        # device 표현형 무관하게 동일해야 함
        assert gpu_numa_node(torch.device(f"cuda:{i}")) == node


def test_other_numa_node():
    count = numa_node_count()
    if count == 1:
        assert other_numa_node(0) == 0
    else:
        for node in range(count):
            other = other_numa_node(node)
            assert other != node and 0 <= other < count


# ── 메모리 바인딩 ────────────────────────────────────────────────────────────

from sglang.srt.layers.prism.numa import (  # noqa: E402
    _libnuma,
    _MAXNODE,
    _MPOL_DEFAULT,
    alloc_pinned_on_node,
    bind_memory_to_node,
    check_tensor_on_node,
    numa_binding_available,
    tensor_numa_nodes,
)

binding_required = pytest.mark.skipif(
    not numa_binding_available() or numa_node_count() < 2,
    reason="libnuma + 2개 이상의 NUMA node 필요",
)


def test_bind_none_is_noop():
    with bind_memory_to_node(None):
        t = torch.empty(1024, dtype=torch.uint8)
    assert t.numel() == 1024


@binding_required
def test_bind_restores_previous_policy():
    import ctypes

    lib = _libnuma()

    def current():
        mode, mask = ctypes.c_int(), ctypes.c_ulong()
        assert lib.get_mempolicy(ctypes.byref(mode), ctypes.byref(mask),
                                 _MAXNODE, None, 0) == 0
        return mode.value, mask.value

    before = current()
    with bind_memory_to_node(1):
        inside = current()
        assert inside[0] != _MPOL_DEFAULT or before[0] != _MPOL_DEFAULT
    assert current() == before


@binding_required
@cuda_required
@pytest.mark.parametrize("node", list(range(numa_node_count())))
def test_alloc_pinned_lands_on_requested_node(node):
    t = alloc_pinned_on_node((8 << 20,), torch.uint8, node, "test")
    assert t.is_pinned()
    assert tensor_numa_nodes(t) == {node}


@binding_required
@cuda_required
def test_alloc_pinned_survives_poisoned_host_cache():
    """torch pinned CachingHostAllocator가 **다른 노드에서 할당됐다 반납된
    블록**을 돌려주는 것이 이 함수의 존재 이유다 (2026-08-25 실측, torch 2.9.1).
    바인딩만 걸고 검증을 생략하면 여기서 조용히 원격 소켓에 앉는다."""
    shape = (24 << 20,)
    for poison, want in ((1, 0), (0, 1)):
        with bind_memory_to_node(poison):
            junk = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
        assert tensor_numa_nodes(junk) == {poison}
        del junk  # 캐시로 반납 — 다음 할당이 이 블록을 재사용할 수 있다
        t = alloc_pinned_on_node(shape, torch.uint8, want, "test")
        assert tensor_numa_nodes(t) == {want}
        del t


@binding_required
@cuda_required
def test_check_tensor_on_node_detects_wrong_node():
    with bind_memory_to_node(1):
        t = torch.empty(4 << 20, dtype=torch.uint8, pin_memory=True)
    if tensor_numa_nodes(t) != {1}:
        pytest.skip("바인딩이 캐시에 막힘 — 이 케이스는 다른 테스트가 덮는다")
    with pytest.raises(RuntimeError, match="NUMA node"):
        check_tensor_on_node(t, 0, "test")
    check_tensor_on_node(t, 0, "test", hard=False)  # 경고 모드는 안 죽는다
