"""NUMA 탐지 테스트. 값은 머신 의존이므로 범위·일관성만 검증한다."""

import pytest
import torch

from sglang.srt.layers.moe.prism.numa import (
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
