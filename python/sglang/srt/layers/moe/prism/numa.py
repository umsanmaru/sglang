"""(이행 shim) NUMA 유틸은 `layers/prism/numa.py`로 승격됐다 (2026-08-31).

expert 축을 전혀 언급하지 않는 순수 유틸이라 dense linear 오프로드가 변경 0으로
같이 쓴다. 기존 import 경로(`moe.prism.numa`)를 살려두는 것은 42개 파일을
건드리지 않기 위해서다 — 새 코드는 `sglang.srt.layers.prism.numa`를 직접 쓸 것.
"""

from sglang.srt.layers.prism.numa import *  # noqa: F401,F403
from sglang.srt.layers.prism.numa import (  # noqa: F401
    alloc_pinned_on_node,
    bind_memory_to_node,
    check_tensor_on_node,
    gpu_numa_node,
    numa_binding_available,
    numa_node_count,
    other_numa_node,
    page_numa_node,
    tensor_numa_nodes,
)
