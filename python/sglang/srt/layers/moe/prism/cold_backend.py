"""Cold 티어의 실행 백엔드 — executor의 submit_cold/sync_cold가 부르는 창구.

책임 (단일: "Plan 어휘 ↔ kt-kernel 원시 어휘의 번역과 cold 인스턴스 수명"):
- Plan → kt `MOEConfig` 번역 (K 밴드, NUMA N-shard 테이블, dims). kt는
  Plan 타입을 절대 보지 않는다 — 여기가 유일한 번역 지점이다.
- 레이어별 `PartialMoEWrapper` 생성 + `PendingColdTensors` 주입 →
  이후 cold weight의 소유권은 C++ (계약 ③; 이 객체가 ColdHandle 역할)
- sparsity 점수 테이블(wn², pair_dot) 주입. weight와 달리 C++가 **매 step
  원시 포인터로 읽으므로** wrapper가 참조를 계속 들고 있어야 한다 —
  PartialMoEWrapper가 그 역할을 한다 (experts_partial.py 수명 규칙).
- P0 실행 제약의 집행: gate == up (K 밴드·N shard), expert 간 균일 기하.

인터페이스 `ColdBackend`는 구현 교체(테스트 mock, 미래의 다른 CPU 백엔드)를
위한 추상이고, `KtColdBackend`가 kt-kernel 구현이다.

kt_kernel import는 지연 — kt가 없는 환경에서 prism의 다른 모듈(plan 등)을
쓰는 데 지장이 없도록.
"""

from __future__ import annotations

from typing import Optional, Protocol

import torch

from sglang.srt.layers.moe.prism.plan import Plan, PlanError, Proj, Tier
from sglang.srt.layers.moe.prism.weights import PendingColdTensors


class ColdBackend(Protocol):
    """executor가 보는 cold 계산 서비스의 계약 (계약 ④의 submit/sync 뒷단)."""

    def load_layer(self, layer_idx: int, cold: PendingColdTensors) -> None: ...
    def submit_gateup(self, layer_idx: int, qlen_ptr: int, k: int,
                      expert_ids_ptr: int, x_ptr: int, out_ptr: int,
                      cuda_stream: Optional[int] = None,
                      thr_ptr: int = 0) -> None: ...
    def submit_down(self, layer_idx: int, qlen_ptr: int, k: int,
                    expert_ids_ptr: int, act_ptr: int, out_ptr: int,
                    cuda_stream: Optional[int] = None,
                    thr_ptr: int = 0) -> None: ...
    def sync(self, cuda_stream: Optional[int] = None) -> None: ...


def _single_cold_band(pp, where: str):
    bands = [b for b in pp.bands if b.tier is Tier.COLD]
    if len(bands) != 1:
        raise NotImplementedError(f"{where}: P0 cold backend supports exactly one COLD band")
    return bands[0]


def _shard_tables(shards, num_nodes: int, where: str):
    """cold_shards → (offset[node], rows[node]) 테이블. 노드당 정확히 1 shard."""
    by_node = {}
    for s in shards:
        if s.node in by_node:
            raise NotImplementedError(f"{where}: P0 supports one shard per numa node")
        by_node[s.node] = s
    if sorted(by_node.keys()) != list(range(num_nodes)):
        raise PlanError(
            f"{where}: cold_shards nodes {sorted(by_node.keys())} != numa nodes 0..{num_nodes - 1}"
        )
    offsets = [by_node[n].n_start for n in range(num_nodes)]
    rows = [by_node[n].n_end - by_node[n].n_start for n in range(num_nodes)]
    return offsets, rows


class KtColdBackend:
    """kt-kernel 구현. CPUInfer(스레드풀)와 레이어별 wrapper의 소유자."""

    def __init__(self, plan: Plan, *, max_tokens: int, num_numa_nodes: int,
                 cpuinfer=None, cpuinfer_threads: int = 60):
        from kt_kernel import kt_kernel_ext  # 지연 import

        self._ext = kt_kernel_ext
        self._plan = plan
        self._max_tokens = max_tokens
        self._num_nodes = num_numa_nodes
        self.cpuinfer = cpuinfer if cpuinfer is not None else kt_kernel_ext.CPUInfer(cpuinfer_threads)
        self._wrappers: dict[int, object] = {}

    # ── Plan → kt config 번역 (유일한 번역 지점) ─────────────────────────
    def _build_config(self, layer_idx: int):
        plan, dims = self._plan, self._plan.dims
        # P0: expert 간 균일 기하 (weights.py 로더와 같은 요구)
        ep = plan.expert(layer_idx, 0)
        for e in range(1, dims.num_experts):
            other = plan.expert(layer_idx, e)
            if other is not ep and other != ep:
                raise NotImplementedError(
                    f"layer {layer_idx}: P0 cold backend requires uniform expert geometry"
                )
        where = f"layer {layer_idx}"
        # P0 영구 제약 (계약 ①): gate == up — 위반은 조용한 오답이 되므로 즉사
        if ep.gate.bands != ep.up.bands or ep.gate.cold_shards != ep.up.cold_shards:
            raise PlanError(f"{where}: gate and up must share K bands and N shards")

        gu_band = _single_cold_band(ep.gate, f"{where}.gateup")
        dn_band = _single_cold_band(ep.down, f"{where}.down")
        gu_off, gu_rows = _shard_tables(ep.gate.cold_shards, self._num_nodes, f"{where}.gateup")
        dn_off, dn_rows = _shard_tables(ep.down.cold_shards, self._num_nodes, f"{where}.down")

        cfg = self._ext.moe.MOEConfig(
            dims.num_experts, dims.top_k, dims.hidden_size, dims.intermediate_size, 0
        )
        cfg.max_len = self._max_tokens
        cfg.layer_idx = layer_idx
        cfg.partial.enabled = True
        cfg.partial.gateup.offset = gu_band.start
        cfg.partial.gateup.rows = gu_band.end - gu_band.start
        cfg.partial.down.offset = dn_band.start
        cfg.partial.down.rows = dn_band.end - dn_band.start
        cfg.partial.n_total = dims.intermediate_size
        cfg.partial.node_gateup_n_offset = gu_off
        cfg.partial.node_gateup_n_rows = gu_rows
        cfg.partial.node_down_n_offset = dn_off
        cfg.partial.node_down_n_rows = dn_rows
        cfg.pool = self.cpuinfer.backend_
        return cfg

    # ── Stage 2: 주입 (이후 PendingColdTensors는 호출자가 해제) ──────────
    def load_layer(self, layer_idx: int, cold: PendingColdTensors) -> None:
        from kt_kernel.experts_partial import PartialMoEWrapper  # 지연 import

        if layer_idx in self._wrappers:
            raise RuntimeError(f"layer {layer_idx} already loaded")
        cfg = self._build_config(layer_idx)
        # 텐서 ↔ 기하 정합 (계약 ②의 shape 검증은 wrapper가 다시 한 번)
        if cold.gate is None or cold.up is None or cold.down is None:
            raise NotImplementedError("P0 cold backend requires cold bands on all projections")
        if cold.gate.k_offset != cfg.partial.gateup.offset or cold.down.k_offset != cfg.partial.down.offset:
            raise PlanError(f"layer {layer_idx}: PendingColdTensors offsets disagree with plan")

        tables = None
        if self._plan.sparsity is not None:
            for name, band in (("gate", cold.gate), ("up", cold.up), ("down", cold.down)):
                if band.calib is None:
                    raise PlanError(
                        f"layer {layer_idx}: plan has sparsity but cold {name} "
                        f"band carries no calib tables"
                    )
            # wn_sq(=a)는 CalibBand가 정의한다 — GPU 측과 같은 형태를 쓰기
            # 위한 단일 정의점 (wn을 그냥 넘기면 마스크가 조용히 갈린다).
            tables = {
                "gate_wn_sq": cold.gate.calib.wn_sq,
                "gate_pair_dot": cold.gate.calib.pair_dot,
                "up_wn_sq": cold.up.calib.wn_sq,
                "up_pair_dot": cold.up.calib.pair_dot,
                "down_wn_sq": cold.down.calib.wn_sq,
                "down_pair_dot": cold.down.calib.pair_dot,
            }

        kernel_key = self._plan.kernels.cpu_cold
        wrapper = PartialMoEWrapper(cfg, self.cpuinfer, kernel_key=kernel_key)
        wrapper.load_weights_from_tensors(
            cold.gate.weights, cold.up.weights, cold.down.weights,
            sparsity_tables=tables,
        )
        self._wrappers[layer_idx] = wrapper

    def _wrapper(self, layer_idx: int):
        try:
            return self._wrappers[layer_idx]
        except KeyError:
            raise RuntimeError(f"layer {layer_idx} not loaded") from None

    # ── step-time: 포인터 pass-through (staging은 호출자 소유 — 계약 ④) ──
    def submit_gateup(self, layer_idx, qlen_ptr, k, expert_ids_ptr, x_ptr, out_ptr, cuda_stream=None,
                      thr_ptr=0):
        self._wrapper(layer_idx).submit_forward_gateup(
            qlen_ptr, k, expert_ids_ptr, x_ptr, out_ptr, cuda_stream, thr_ptr)

    def submit_down(self, layer_idx, qlen_ptr, k, expert_ids_ptr, act_ptr, out_ptr, cuda_stream=None,
                    thr_ptr=0):
        self._wrapper(layer_idx).submit_forward_down(
            qlen_ptr, k, expert_ids_ptr, act_ptr, out_ptr, cuda_stream, thr_ptr)

    def sync(self, cuda_stream=None):
        if cuda_stream is None:
            self.cpuinfer.sync()
        else:
            self.cpuinfer.sync_with_cuda_stream(cuda_stream)
