"""cold GPU view — kt가 pack한 cold weight를 GPU가 **제자리에서** 읽기 위한 기술자.

cold 티어의 거처는 pageable host(NUMA-local)이고 소비자는 CPU AMX다 (계약 ①).
prefill에서 M이 커지면 그 계산이 CPU 대역/연산에 묶여 층당 수십 ms가 되는데
(35B M=2048: 28 ms/층), 같은 바이트를 GPU가 PCIe로 읽는 비용은 M과 무관한
상수다 (75% cold = 1.15 GiB/층 ≈ 24 ms @ 51 GB/s). 그래서 큰 M에서는 GPU가
cold를 읽는 편이 이기고, 그 경계가 `cold_gpu_min_m`이다 (executor).

**weight는 한 벌이다.** GPU용 사본을 만들지 않는다 — kt의 packed AMX 레이아웃
(`BufferBBF16Impl`: n_block → k_block → n_step → k_step → 32×32 VNNI 타일)을
`cudaHostRegister`로 GPU 주소공간에 매핑하고, grouped GEMM의 COLD 레이아웃 로더가
그 타일을 직접 해석한다 (`prism_grouped.cuh`). 그러려면 kt가 expert 버퍼를
(node, proj)당 **페이지 정렬 slab 하나**에 연속 배치해야 하고(`alloc_cold_slabs`),
`cold_slab_views()`가 그 기술자를 내준다.

수명: 등록은 프로세스 수명 동안 유지한다 (slab은 kt 인스턴스와 함께 산다 — 계약 ③).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sglang.srt.layers.moe.prism.plan import Plan, PlanError, Proj

_PROJ_OF = {0: Proj.GATE, 1: Proj.UP, 2: Proj.DOWN}
_HOST_REGISTER_PORTABLE = 1
_HOST_REGISTER_MAPPED = 2


@dataclass(frozen=True)
class ColdSlab:
    """한 (layer, node, proj)의 packed slab. 커널 인자 전부 (grouped_gemm_cold)."""

    slab: torch.Tensor      # bf16 1-D, CPU — host-registered 메모리의 뷰 (사본 아님)
    blk_off: torch.Tensor   # int64 [E] device — expert 블록 시작 (원소)
    row_off: torch.Tensor   # int32 [E+1] device — 타일 올림된 k 누적
    k_index: torch.Tensor   # uint16 [Σ k_pad] device — 패딩 포함 (패딩은 0을 가리킨다)
    n: int                  # 이 노드의 N 행 수
    n_start: int            # 이 노드의 N shard 시작 (출력 열 오프셋)
    n_block: int
    k_block: int
    node: int
    proj: Proj
    k_max: int = 0          # expert당 최대 K(타일 올림) — W-resident 커널의 smem 크기


@dataclass(frozen=True)
class ColdGpuLayer:
    gate: tuple            # ColdSlab per node
    up: tuple
    down: tuple

    def slabs(self, proj: Proj) -> tuple:
        return {Proj.GATE: self.gate, Proj.UP: self.up, Proj.DOWN: self.down}[proj]


def _register_host(ptr: int, nbytes: int) -> None:
    """cudaHostRegister(Portable|Mapped). 이미 등록된 범위면 그대로 쓴다 — slab이
    프로세스 수명 동안 살고 우리가 유일한 등록자이므로 안전하다."""
    rt = torch.cuda.cudart()
    err = rt.cudaHostRegister(ptr, nbytes, _HOST_REGISTER_PORTABLE | _HOST_REGISTER_MAPPED)
    # torch는 cudaError_t를 그대로 돌려준다 (0 = success, 712 = already registered).
    code = int(err) if not hasattr(err, "value") else int(err.value)
    if code not in (0, 712):
        raise RuntimeError(f"cudaHostRegister(ptr=0x{ptr:x}, bytes={nbytes}) failed: {err}")
    if code == 712:
        torch.cuda.cudart().cudaGetLastError() if hasattr(rt, "cudaGetLastError") else None


def _tensor_view(ptr: int, nbytes: int) -> torch.Tensor:
    """raw host 포인터 → bf16 1-D CPU 텐서 (frombuffer — 공유 메모리, 사본 없음)."""
    n = nbytes // 2
    buf = (ctypes.c_uint16 * n).from_address(ptr)
    return torch.frombuffer(buf, dtype=torch.bfloat16)


def build_cold_gpu_layer(
    views: Sequence[dict], plan: Plan, layer_idx: int, cold_shards: dict,
    device: torch.device, shard_override: Optional[dict] = None,
) -> ColdGpuLayer:
    """kt `cold_slab_views()` 결과 + 로더의 ColdShard(패딩된 row_off/k_index) →
    ColdGpuLayer. 여기서 slab을 host-register하고 기하를 대조한다.

    cold_shards: {Proj: ColdShard} — pack 직전의 텐서 소유자. row_off/k_index만
    쓰고 w_flat은 건드리지 않는다 (그 바이트는 이미 kt slab 안에 있다).
    """
    ep = plan.expert(layer_idx, 0)
    E = plan.dims.num_experts
    out: dict = {Proj.GATE: {}, Proj.UP: {}, Proj.DOWN: {}}
    for v in views:
        proj = _PROJ_OF[int(v["proj"])]
        node = int(v["node"])
        shard = cold_shards[proj]
        where = f"layer {layer_idx} {proj.value} node {node}"
        row_off_cpu = shard.row_off
        k_pad = (row_off_cpu[1:] - row_off_cpu[:-1]).tolist()
        if list(v["k"]) != k_pad:
            raise PlanError(f"{where}: kt packed k {list(v['k'])[:4]}... != loader k_pad {k_pad[:4]}...")
        if shard_override is not None:
            o_node, o_start, o_end = shard_override[proj]
            if node != o_node:
                raise PlanError(f"{where}: unexpected node {node} (override node {o_node})")
            n_start, n_rows = o_start, o_end - o_start
        else:
            n_shard = next(s for s in ep.proj(proj).cold_shards if s.node == node)
            n_start, n_rows = n_shard.n_start, n_shard.n_end - n_shard.n_start
        if int(v["n"]) != n_rows:
            raise PlanError(f"{where}: kt n {v['n']} != shard rows {n_rows}")
        if int(v["n_step"]) != 32 or int(v["k_step"]) != 32:
            raise PlanError(f"{where}: unsupported AMX tile {v['n_step']}x{v['k_step']} (kernel assumes 32x32)")
        expert_off = [int(o) for o in v["expert_off"]]
        if len(expert_off) != E + 1 or any(o % 2 for o in expert_off):
            raise PlanError(f"{where}: bad expert_off table")
        ptr, nbytes = int(v["ptr"]), int(v["bytes"])
        if ptr % 4096 or nbytes % 4096:
            raise PlanError(f"{where}: slab must be page aligned (ptr=0x{ptr:x}, bytes={nbytes})")
        _register_host(ptr, nbytes)
        out[proj][node] = ColdSlab(
            slab=_tensor_view(ptr, nbytes),
            blk_off=torch.tensor([o // 2 for o in expert_off[:-1]], dtype=torch.int64, device=device),
            row_off=row_off_cpu.to(device=device, dtype=torch.int32),
            k_index=shard.k_index.to(device),
            n=int(v["n"]), n_start=n_start,
            n_block=int(v["n_block"]), k_block=int(v["k_block"]),
            node=node, proj=proj, k_max=max(k_pad) if k_pad else 0,
        )
    for proj in Proj:
        nodes = sorted(out[proj])
        want = ([shard_override[proj][0]] if shard_override is not None
                else sorted(s.node for s in ep.proj(proj).cold_shards))
        if nodes != want:
            raise PlanError(f"layer {layer_idx} {proj.value}: kt nodes {nodes} != shards {want}")
    return ColdGpuLayer(
        gate=tuple(out[Proj.GATE][n] for n in sorted(out[Proj.GATE])),
        up=tuple(out[Proj.UP][n] for n in sorted(out[Proj.UP])),
        down=tuple(out[Proj.DOWN][n] for n in sorted(out[Proj.DOWN])),
    )
