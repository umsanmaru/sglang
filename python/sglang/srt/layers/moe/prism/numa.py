"""NUMA 토폴로지 탐지 — 기계만 제공하고, 배치 *정책*은 호출자 소관.

(pinned store를 GPU 소켓에 두고 cold weight를 반대 소켓에 두는 결정은
TieredWeightLoader/plan의 것이지 이 모듈의 것이 아니다.)

이 머신 실측(2026-08-20): RTX PRO 6000 = node 0, H100 = node 1 —
GPU마다 소켓이 다르므로 gpu_numa_node는 반드시 device 인자를 받는다.

P0 한계: pinned 할당의 NUMA *바인딩*은 아직 미구현 (kt-kernel python이
현재 바인딩 API를 노출하지 않음). 탐지만 정확히 하고, 바인딩은 K-side에
set_memory_to_numa 노출을 추가할 때 연결한다. 그때까지 pinned store는
first-touch 배치에 맡겨진다.
"""

from __future__ import annotations

import glob
import os
from typing import Union

import torch

# GPU를 지칭하는 세 가지 통용 표기 중 아무거나 — torch API 관례에 맞춤:
#   int (device 인덱스, 예: 0) / str ("cuda:0") / torch.device 객체.
# 정규화는 받는 함수 안에서 수행하므로 호출자는 손에 든 형태 그대로 넘긴다.
DeviceLike = Union[int, str, torch.device]


def numa_node_count() -> int:
    nodes = glob.glob("/sys/devices/system/node/node[0-9]*")
    return max(len(nodes), 1)


def gpu_numa_node(device: DeviceLike = 0) -> int:
    """device가 붙은 PCIe root의 NUMA node. 알 수 없으면 0."""
    index = torch.device(device).index if not isinstance(device, int) else device
    if index is None:
        index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    pci = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    path = f"/sys/bus/pci/devices/{pci}/numa_node"
    try:
        node = int(open(path).read().strip())
    except (OSError, ValueError):
        return 0
    # 커널은 토폴로지를 모르면 -1을 보고한다 (단일 소켓 등)
    return node if node >= 0 else 0


def other_numa_node(node: int) -> int:
    """2-소켓 가정의 반대 노드. 노드가 1개뿐이면 자기 자신."""
    count = numa_node_count()
    if count <= 1:
        return node
    return (node + 1) % count
