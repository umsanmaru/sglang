"""NUMA 토폴로지 탐지 — 기계만 제공하고, 배치 *정책*은 호출자 소관.

(pinned store를 GPU 소켓에 두고 cold weight를 반대 소켓에 두는 결정은
TieredWeightLoader/plan의 것이지 이 모듈의 것이 아니다.)

이 머신 실측(2026-08-20): RTX PRO 6000 = node 0, H100 = node 1 —
GPU마다 소켓이 다르므로 gpu_numa_node는 반드시 device 인자를 받는다.

바인딩(bind_memory_to_node)과 사후 검증(check_tensor_on_node)은 아래 절에
있다. kt의 set_memory_to_numa를 경유하지 않고 libnuma를 직접 부르는 이유:
warm은 kt가 존재조차 모르는 티어이므로(계약 ③) kt를 배치 경로에 끼우면
경계가 무너진다.
"""

from __future__ import annotations

import contextlib
import ctypes
import glob
import os
from typing import Optional, Union

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


# ---------------------------------------------------------------------------
# 메모리 바인딩 (libnuma via ctypes)
#
# **pinned 할당은 first-touch가 아니다.** `pin_memory=True`는 cudaHostAlloc이고
# 그 호출 안에서 페이지가 물리적으로 커밋·고정된다 — 배치를 정하는 것은 나중에
# 내용을 채우는 스레드가 아니라 **할당을 호출하는 스레드의 메모리 정책**이다.
# 그리고 고정된 페이지는 마이그레이션 대상에서 제외되므로 사후 mbind/move_pages로
# 옮길 수 없다. 따라서 정책은 반드시 할당 **전에** 걸어야 하고, 그것이
# bind_memory_to_node가 컨텍스트 매니저인 이유다.
#
# 정책은 스레드 스코프다 (set_mempolicy(2)). 진입 시 현재 정책을 읽어 두고
# 이탈 시 복원한다 — MPOL_DEFAULT로 되돌리면 `numactl --interleave=all` 같은
# 프로세스 정책을 조용히 지워버리기 때문이다.
#
# 배치 *정책*(어느 노드에 둘 것인가)은 여전히 호출자 소관이다. 이 모듈은
# 기계(바인딩·조회)만 제공한다.
# ---------------------------------------------------------------------------

_MPOL_DEFAULT = 0
_MPOL_BIND = 2
_MPOL_F_NODE = 1 << 0
_MPOL_F_ADDR = 1 << 1
# nodemask 비트 수. 커널은 BITS_TO_LONGS(maxnode) 만큼 읽으므로 ulong 1개.
_MAXNODE = 8 * ctypes.sizeof(ctypes.c_ulong)

_LIBNUMA_UNSET = object()
_libnuma_cache = _LIBNUMA_UNSET


def _libnuma():
    """libnuma.so.1 핸들 또는 None (없는 환경에서도 탐지 함수들은 살아야 한다).

    get_mempolicy/set_mempolicy는 glibc가 래핑하지 않는 syscall이라 libnuma의
    래퍼를 쓴다 (libnuma가 이 두 심볼을 그대로 export한다).
    """
    global _libnuma_cache
    if _libnuma_cache is _LIBNUMA_UNSET:
        try:
            lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
            lib.get_mempolicy.argtypes = [
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong),
                ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
            ]
            lib.get_mempolicy.restype = ctypes.c_long
            lib.set_mempolicy.argtypes = [
                ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong,
            ]
            lib.set_mempolicy.restype = ctypes.c_long
            _libnuma_cache = lib
        except OSError:
            _libnuma_cache = None
    return _libnuma_cache


def numa_binding_available() -> bool:
    """바인딩 기계가 이 환경에서 동작하는가 (libnuma 존재 + 정책 조회 성공)."""
    lib = _libnuma()
    if lib is None:
        return False
    mode = ctypes.c_int()
    mask = ctypes.c_ulong()
    return lib.get_mempolicy(
        ctypes.byref(mode), ctypes.byref(mask), _MAXNODE, None, 0
    ) == 0


@contextlib.contextmanager
def bind_memory_to_node(node: Optional[int]):
    """블록 안의 **할당**을 node에 바인딩한다 (MPOL_BIND, 스레드 스코프).

    node=None이면 no-op — 호출부가 분기 없이 감쌀 수 있게 한 것이다.
    libnuma가 없으면 조용히 no-op으로 떨어진다: 바인딩 실패를 여기서 죽이지
    않는 이유는 배치 성공 여부를 판정하는 곳이 따로 있기 때문이다
    (check_tensor_on_node — 실제로 어디 떨어졌는지를 보는 쪽이 권위다).
    """
    lib = _libnuma() if node is not None else None
    if lib is None:
        yield
        return
    prev_mode = ctypes.c_int()
    prev_mask = ctypes.c_ulong()
    if lib.get_mempolicy(ctypes.byref(prev_mode), ctypes.byref(prev_mask),
                         _MAXNODE, None, 0) != 0:
        yield
        return
    mask = ctypes.c_ulong(1 << node)
    if lib.set_mempolicy(_MPOL_BIND, ctypes.byref(mask), _MAXNODE) != 0:
        yield
        return
    try:
        yield
    finally:
        if prev_mode.value == _MPOL_DEFAULT:
            # MPOL_DEFAULT는 빈 nodemask를 요구한다 (mask 포인터를 그대로
            # 넘기면 EINVAL).
            lib.set_mempolicy(_MPOL_DEFAULT, None, _MAXNODE)
        else:
            lib.set_mempolicy(prev_mode.value, ctypes.byref(prev_mask), _MAXNODE)


def page_numa_node(addr: int) -> Optional[int]:
    """주소 addr가 속한 **페이지가 실제로 올라간** 노드. 알 수 없으면 None.

    정책(어디 두라고 했는가)이 아니라 결과(어디 떨어졌는가)를 본다 —
    MPOL_F_ADDR|MPOL_F_NODE 조합이 그 의미다.
    """
    lib = _libnuma()
    if lib is None:
        return None
    node = ctypes.c_int()
    rc = lib.get_mempolicy(ctypes.byref(node), None, 0,
                           ctypes.c_void_p(addr), _MPOL_F_NODE | _MPOL_F_ADDR)
    return node.value if rc == 0 else None


def tensor_numa_nodes(tensor: "torch.Tensor", samples: int = 8) -> set:
    """텐서 페이지를 samples개 뽑아 실제 노드 집합을 돌려준다.

    전수 조사를 하지 않는 이유: warm store는 GiB 단위이고 한 번의 할당은
    한 정책 아래 놓이므로, 흩어졌는지를 보는 데는 표본으로 충분하다.
    (huge page면 표본 간격이 페이지보다 작을 수 있으나, 그 경우도 "흩어짐
    없음"을 확인하는 목적에는 무해하다.)
    """
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes == 0:
        return set()
    base = tensor.data_ptr()
    step = max(nbytes // max(samples, 1), 1)
    nodes = set()
    for off in range(0, nbytes, step):
        node = page_numa_node(base + off)
        if node is not None:
            nodes.add(node)
    return nodes


def check_tensor_on_node(tensor: "torch.Tensor", node: Optional[int],
                         where: str, *, hard: bool = True) -> None:
    """텐서가 정말 node에 떨어졌는지 검증. 어긋나면 즉사(기본) 또는 경고.

    이것이 이 파일의 존재 이유다: 원격 소켓 배치는 결과를 바꾸지 않고 느리게만
    만들기 때문에 어떤 테스트도 잡아주지 않는다. dims 불일치와 같은 급의
    silent failure이므로 startup에서 죽는 편이 낫다.
    """
    if node is None or numa_node_count() <= 1:
        return
    nodes = tensor_numa_nodes(tensor)
    if not nodes or nodes == {node}:
        return
    msg = (
        f"{where}: pinned 페이지가 NUMA node {sorted(nodes)}에 있다 "
        f"(기대 node {node}) — 원격 소켓 배치는 UVA 읽기에 소켓 간 링크 홉을 "
        f"추가해 warm 티어의 존립 근거(PCIe 대역폭)를 무너뜨린다"
    )
    if hard:
        raise RuntimeError(msg)
    import logging

    logging.getLogger(__name__).error("[prism] %s", msg)


def _empty_host_cache() -> bool:
    """torch의 pinned 캐싱 호스트 할당자를 비운다. 성공 여부를 반환."""
    fn = getattr(torch._C, "_host_emptyCache", None)
    if fn is None:
        return False
    try:
        fn()
    except Exception:
        return False
    return True


def alloc_pinned_on_node(shape, dtype, node: Optional[int], where: str,
                         *, hard: bool = True) -> "torch.Tensor":
    """node에 상주하는 pinned 텐서를 할당한다.

    **바인딩만으로는 부족하다** (2026-08-25 실측): torch의 pinned
    CachingHostAllocator가 **다른 NUMA 정책 아래 할당됐다가 반납된 블록을 그대로
    돌려준다**. 같은 크기를 node 1 → node 0 순으로 요청하면 두 번째가 node 1
    블록을 재사용해 바인딩이 조용히 무시된다 (torch 2.9.1에서 재현). 캐시는
    크기를 버킷으로 반올림하므로 "크기를 조금 다르게" 같은 회피도 믿을 수 없다.

    그래서 순서가 `바인딩 → 할당 → **검증** → (어긋나면) 캐시 비우고 1회 재시도`
    다. 검증이 부가물이 아니라 이 함수의 본체다.
    """
    if node is None or numa_node_count() <= 1 or not numa_binding_available():
        return torch.empty(shape, dtype=dtype, pin_memory=True)

    tensor = None
    for attempt in range(2):
        with bind_memory_to_node(node):
            tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
        nodes = tensor_numa_nodes(tensor)
        if not nodes or nodes == {node}:
            return tensor
        if attempt == 0:
            # 캐시가 돌려준 남의 노드 블록일 가능성 — 반납하고 캐시를 비운 뒤
            # 다시 시도한다. 비우기가 불가능한 torch 빌드면 재시도 의미가 없다.
            del tensor
            tensor = None
            if not _empty_host_cache():
                break
    if tensor is None:
        tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
    check_tensor_on_node(tensor, node, where, hard=hard)
    return tensor
