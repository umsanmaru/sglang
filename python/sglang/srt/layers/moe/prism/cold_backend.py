"""Cold 티어의 실행 백엔드 — executor의 submit_cold/sync_cold가 부르는 창구.

책임 (단일: "Plan 어휘 ↔ kt-kernel 원시 어휘의 번역과 cold 인스턴스 수명"):
- Plan → kt `MOEConfig` 번역 (K 밴드, NUMA N-shard 테이블, dims). kt는
  Plan 타입을 절대 보지 않는다 — 여기가 유일한 번역 지점이다.
- 레이어별 `PartialMoEWrapper` 생성 + `PendingColdTensors` 주입 →
  이후 cold weight의 소유권은 C++ (계약 ③; 이 객체가 ColdHandle 역할)
- sparsity 주입: 점수 테이블(wn², pair_dot)과 threshold 곡선, 그리고 예산
  스칼라(p, lambda, pmax, grid, ng, renorm_it). weight와 달리 C++가 **매 step
  원시 포인터로 읽으므로** wrapper가 참조를 계속 들고 있어야 한다 —
  PartialMoEWrapper가 그 역할을 한다 (experts_partial.py 수명 규칙).
  sparsity 수식 자체는 kt에 있고 prism은 라우터 가중만 step마다 내려준다.
- P0 실행 제약의 집행: gate == up (N shard), 층 전역 스칼라(N shard 테이블·
  sparsity 예산)의 expert 간 일치. K 기하는 expert별로 갈려도 된다.

인터페이스 `ColdBackend`는 구현 교체(테스트 mock, 미래의 다른 CPU 백엔드)를
위한 추상이고, `KtColdBackend`가 kt-kernel 구현이다.

kt_kernel import는 지연 — kt가 없는 환경에서 prism의 다른 모듈(plan 등)을
쓰는 데 지장이 없도록.
"""

from __future__ import annotations

from typing import Optional, Protocol

import torch

from sglang.srt.layers.moe.prism.plan import Plan, PlanError, Proj
from sglang.srt.layers.moe.prism.weights import PendingColdTensors


class ColdBackend(Protocol):
    """executor가 보는 cold 계산 서비스의 계약 (계약 ④의 submit/sync 뒷단)."""

    def load_layer(self, layer_idx: int, cold: PendingColdTensors,
                   thr=None) -> None: ...
    def submit_gateup(self, layer_idx: int, qlen_ptr: int, k: int,
                      expert_ids_ptr: int, x_ptr: int, out_ptr: int,
                      cuda_stream: Optional[int] = None,
                      weights_ptr: int = 0) -> None: ...
    def submit_down(self, layer_idx: int, qlen_ptr: int, k: int,
                    expert_ids_ptr: int, act_ptr: int, out_ptr: int,
                    cuda_stream: Optional[int] = None,
                    weights_ptr: int = 0) -> None: ...
    def sync(self, cuda_stream: Optional[int] = None) -> None: ...


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
                 cpuinfer=None, cpuinfer_threads: int = 60,
                 gpu_view_device=None, hybrid_mask: bool = False):
        """gpu_view_device: 주면 load_layer가 kt packed slab을 그 device에서 읽을 수
        있게 host-register하고 `gpu_view(layer)`로 기술자를 낸다 (cold_gpu.py).
        None이면 cold는 CPU 전용이다.
        hybrid_mask: kt config에 expert skip 마스크(pinned uint8 [E], C++가 매 호출
        읽는다)를 단다. executor가 호출 직전에 값을 써 "이 expert는 GPU가 계산"을
        알린다 (cold hybrid). 마스크가 1인 expert의 partial 행은 kt가 0으로 채운다."""
        from kt_kernel import kt_kernel_ext  # 지연 import

        self._ext = kt_kernel_ext
        self._plan = plan
        self._max_tokens = max_tokens
        self._num_nodes = num_numa_nodes
        self.cpuinfer = cpuinfer if cpuinfer is not None else kt_kernel_ext.CPUInfer(cpuinfer_threads)
        self._wrappers: dict[int, object] = {}
        self._gpu_view_device = gpu_view_device
        self._gpu_views: dict[int, object] = {}
        self._warm_wrappers: dict[int, object] = {}
        self._warm_views: dict[int, object] = {}
        # 전 레이어 공유 — 층마다 값을 바꿔도 kt는 호출 시점의 값을 읽는다.
        # 노드별 마스크: 소켓마다 GPU 몫이 다르다 (GPU-local 노드만 expert 분할, 원격
        # 노드는 CPU가 전부 — 원격 slab의 UVA 읽기는 UPI를 건너 느리다).
        self.hybrid_masks: Optional[list] = (
            [torch.zeros(plan.dims.num_experts, dtype=torch.uint8, pin_memory=True)
             for _ in range(num_numa_nodes)]
            if hybrid_mask else None
        )

    # ── Plan → kt config 번역 (유일한 번역 지점) ─────────────────────────
    @staticmethod
    def _set_kindex(dst, shard) -> None:
        """ColdShard의 기하를 kt `KIndex`로 옮긴다 — 이 클래스의 본래 책임인
        "Plan 어휘 ↔ kt 원시 어휘의 번역"이 인덱스 시대에 갖는 형태다.

        `real_rows`는 타일 올림 **전**의 행 수다. kt는 이것으로 sparse 마스크의
        tail 비트를 끈다 (패딩 행은 weight가 0이라 dense에는 무해하다).
        """
        dst.row_off = shard.row_off.tolist()
        dst.idx = shard.k_index.to(torch.int32).tolist()
        real = shard.real_rows.tolist()
        # 패딩이 없으면 real_rows를 비워 둔다 — kt가 k(e)와 같다고 읽는다.
        if any(r != shard.row_off[e + 1] - shard.row_off[e] for e, r in enumerate(real)):
            dst.real_rows = real

    def _build_config(self, layer_idx: int, cold: PendingColdTensors,
                      n_shards: Optional[dict] = None):
        """n_shards: {"gateup": (offsets, rows), "down": (offsets, rows)} — 주면 plan의
        cold_shards 대신 이 노드 테이블을 쓴다 (warm-kt: GPU-local 노드에 전량)."""
        plan, dims = self._plan, self._plan.dims
        # 균일을 요구하는 것은 **kt config의 스칼라가 되는 것들뿐**이다: N shard
        # 테이블과 sparsity 예산. K 기하(어느 행이 cold냐)는 expert별로 갈려도
        # 되고 — row_off/idx로 내려간다 (계약 ③: flat + offset이라 균일성 요구가
        # 없다) — 예산을 expert별로 쓰는 것이 이 스키마의 존재 이유다.
        ep = plan.expert(layer_idx, 0)
        for e in range(1, dims.num_experts):
            other = plan.expert(layer_idx, e)
            if other is ep or other == ep:
                continue
            for proj in Proj:
                a, b = ep.proj(proj), other.proj(proj)
                if a.cold_shards != b.cold_shards:
                    raise NotImplementedError(
                        f"layer {layer_idx} {proj.value}: cold_shards differ between "
                        f"experts 0 and {e} — N shard 테이블은 층 전역이다"
                    )
                if (a.sparsity_p, a.sparsity_lambda) != (b.sparsity_p, b.sparsity_lambda):
                    raise NotImplementedError(
                        f"layer {layer_idx} {proj.value}: sparsity budget differs "
                        f"between experts 0 and {e} — (p, lambda)는 kt config의 스칼라다"
                    )
        where = f"layer {layer_idx}"
        # N shard는 여전히 gate/up 공유다 (출력 축 — K 인덱스와 무관하다).
        if ep.gate.cold_shards != ep.up.cold_shards:
            raise PlanError(f"{where}: gate and up must share cold N shards")

        if n_shards is None:
            gu_off, gu_rows = _shard_tables(ep.gate.cold_shards, self._num_nodes, f"{where}.gateup")
            dn_off, dn_rows = _shard_tables(ep.down.cold_shards, self._num_nodes, f"{where}.down")
        else:
            (gu_off, gu_rows), (dn_off, dn_rows) = n_shards["gateup"], n_shards["down"]
        # 커널 키가 요구하는 노드 N shard 정렬 (tile fp4 = 256) — kt init의 runtime_error보다 먼저, 이름으로.
        from sglang.srt.layers.moe.prism.kernels import cold_n_align

        align = cold_n_align(plan.kernels.cpu_cold)
        for name, rows in (("gateup", gu_rows), ("down", dn_rows)):
            if any(int(r) % align for r in rows):
                raise PlanError(f"{where}: {name} node shard rows {list(rows)} must be multiples of {align} "
                                f"for cpu_cold '{plan.kernels.cpu_cold}'")

        cfg = self._ext.moe.MOEConfig(
            dims.num_experts, dims.top_k, dims.hidden_size, dims.intermediate_size, 0
        )
        cfg.max_len = self._max_tokens
        cfg.layer_idx = layer_idx
        cfg.partial.enabled = True
        # K 기하는 인덱스로 간다. gate와 up이 같은 인덱스면 kt가 pack을
        # 공유하고(dual_pack() == false) 이전과 같은 경로를 탄다.
        self._set_kindex(cfg.partial.gate, cold.gate)
        self._set_kindex(cfg.partial.up, cold.up)
        self._set_kindex(cfg.partial.down, cold.down)
        cfg.partial.n_total = dims.intermediate_size
        cfg.partial.node_gateup_n_offset = gu_off
        cfg.partial.node_gateup_n_rows = gu_rows
        cfg.partial.node_down_n_offset = dn_off
        cfg.partial.node_down_n_rows = dn_rows
        spec = plan.sparsity
        if spec is not None:
            # 예산·격자를 config에 굽는다 — step마다 넘기는 값이 아니다.
            sp = cfg.partial.sparsity
            sp.pmax, sp.grid = spec.pmax, spec.grid
            sp.ng, sp.renorm_it = spec.ng, spec.renorm_it
            sp.p_gate, sp.lam_gate = ep.gate.sparsity_p, ep.gate.sparsity_lambda
            sp.p_up, sp.lam_up = ep.up.sparsity_p, ep.up.sparsity_lambda
            sp.p_down, sp.lam_down = ep.down.sparsity_p, ep.down.sparsity_lambda
        cfg.pool = self.cpuinfer.backend_
        return cfg

    # ── Stage 2: 주입 (이후 PendingColdTensors는 호출자가 해제) ──────────
    def load_layer(self, layer_idx: int, cold: PendingColdTensors,
                   thr=None) -> None:
        from kt_kernel.experts_partial import PartialMoEWrapper  # 지연 import

        if layer_idx in self._wrappers:
            raise RuntimeError(f"layer {layer_idx} already loaded")
        cfg = self._build_config(layer_idx, cold)
        # 텐서 ↔ 기하 정합 (계약 ②의 shape 검증은 wrapper가 다시 한 번)
        if cold.gate is None or cold.up is None or cold.down is None:
            raise NotImplementedError("P0 cold backend requires cold rows on all projections")

        tables = None
        if self._plan.sparsity is not None:
            if thr is None:
                raise PlanError(
                    f"layer {layer_idx}: plan has sparsity but no threshold "
                    f"curves were passed (PreparedWeights.thr)"
                )
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
                # threshold 곡선 [E, ng] — 밴드와 무관하므로 절단하지 않는다.
                "thr_gate": thr[Proj.GATE],
                "thr_up": thr[Proj.UP],
                "thr_down": thr[Proj.DOWN],
            }

        kernel_key = self._plan.kernels.cpu_cold
        # 스토어 포맷이 정하는 추가 인자 (mxfp4: bf16 배율 셋). 포맷↔커널 호환은 startup 검증.
        cold.gate.fmt.check_cold_kernel(kernel_key)
        wrapper = PartialMoEWrapper(cfg, self.cpuinfer, kernel_key=kernel_key)
        wrapper.load_weights_from_tensors(
            cold.gate.w_flat, cold.up.w_flat, cold.down.w_flat,
            sparsity_tables=tables, **cold.gate.fmt.cold_load_kwargs(cold),
        )
        self._wrappers[layer_idx] = wrapper
        if self.hybrid_masks is not None:
            for node, mask in enumerate(self.hybrid_masks):
                wrapper.set_node_gpu_experts_mask(node, mask)
        if self._gpu_view_device is not None:
            from sglang.srt.layers.moe.prism.cold_gpu import build_cold_gpu_layer

            views = wrapper.cold_slab_views()
            self._gpu_views[layer_idx] = build_cold_gpu_layer(
                views, self._plan, layer_idx,
                {Proj.GATE: cold.gate, Proj.UP: cold.up, Proj.DOWN: cold.down},
                torch.device(self._gpu_view_device), fmt=cold.gate.fmt,
            )

    def gpu_view(self, layer_idx: int):
        """ColdGpuLayer 또는 None (gpu_view_device 미설정)."""
        return self._gpu_views.get(layer_idx)

    # ── warm-kt: warm 행을 kt 포맷 slab으로 (GPU-local 노드 전량, 다른 노드 0행) ──
    def load_warm_layer(self, layer_idx: int, warm: PendingColdTensors, thr=None,
                        local_node: int = 0):
        """warm 행의 kt 인스턴스. 계산은 (prefill CPU 몫이 있을 때만) 이 인스턴스가 하고,
        저장은 slab을 host-register해 GPU가 packed 레이아웃으로 읽는다 (cold_gpu.py).
        gpu_view_device가 필요하다 — warm의 존재 이유가 GPU 읽기이므로."""
        from kt_kernel.experts_partial import PartialMoEWrapper
        from sglang.srt.layers.moe.prism.cold_gpu import build_cold_gpu_layer

        if self._gpu_view_device is None:
            raise ValueError("warm-kt requires gpu_view_device (GPU reads the slab)")
        if layer_idx in self._warm_wrappers:
            raise RuntimeError(f"warm layer {layer_idx} already loaded")
        dims = self._plan.dims
        n = self._num_nodes
        def table(full):
            offs = [full if i != local_node else 0 for i in range(n)]
            rows = [0 if i != local_node else full for i in range(n)]
            return offs, rows
        n_shards = {"gateup": table(dims.intermediate_size), "down": table(dims.hidden_size)}
        cfg = self._build_config(layer_idx, warm, n_shards=n_shards)
        tables = None
        if self._plan.sparsity is not None:
            if thr is None or any(b.calib is None for b in (warm.gate, warm.up, warm.down)):
                raise PlanError(f"layer {layer_idx}: warm-kt sparse needs calib/thr")
            tables = {
                "gate_wn_sq": warm.gate.calib.wn_sq, "gate_pair_dot": warm.gate.calib.pair_dot,
                "up_wn_sq": warm.up.calib.wn_sq, "up_pair_dot": warm.up.calib.pair_dot,
                "down_wn_sq": warm.down.calib.wn_sq, "down_pair_dot": warm.down.calib.pair_dot,
                "thr_gate": thr[Proj.GATE], "thr_up": thr[Proj.UP], "thr_down": thr[Proj.DOWN],
            }
        wrapper = PartialMoEWrapper(cfg, self.cpuinfer, kernel_key=self._plan.kernels.cpu_cold)
        wrapper.load_weights_from_tensors(warm.gate.w_flat, warm.up.w_flat, warm.down.w_flat,
                                          sparsity_tables=tables)
        self._warm_wrappers[layer_idx] = wrapper
        # plan cold_shards 대신 n_shards 기하로 view를 만든다 — build_cold_gpu_layer가 plan을
        # 참조하므로 임시 shard 객체로 대체한다.
        views = [v for v in wrapper.cold_slab_views() if int(v["node"]) == local_node]
        self._warm_views[layer_idx] = build_cold_gpu_layer(
            views, self._plan, layer_idx,
            {Proj.GATE: warm.gate, Proj.UP: warm.up, Proj.DOWN: warm.down},
            torch.device(self._gpu_view_device),
            shard_override={Proj.GATE: (local_node, 0, dims.intermediate_size),
                            Proj.UP: (local_node, 0, dims.intermediate_size),
                            Proj.DOWN: (local_node, 0, dims.hidden_size)},
        )

    def warm_view(self, layer_idx: int):
        return self._warm_views.get(layer_idx)

    def submit_warm_gateup(self, layer_idx, qlen_ptr, k, expert_ids_ptr, x_ptr, out_ptr,
                           cuda_stream=None, weights_ptr=0):
        self._warm_wrappers[layer_idx].submit_forward_gateup(
            qlen_ptr, k, expert_ids_ptr, x_ptr, out_ptr, cuda_stream, weights_ptr)

    def submit_warm_down(self, layer_idx, qlen_ptr, k, expert_ids_ptr, act_ptr, out_ptr,
                         cuda_stream=None, weights_ptr=0):
        self._warm_wrappers[layer_idx].submit_forward_down(
            qlen_ptr, k, expert_ids_ptr, act_ptr, out_ptr, cuda_stream, weights_ptr)

    def _wrapper(self, layer_idx: int):
        try:
            return self._wrappers[layer_idx]
        except KeyError:
            raise RuntimeError(f"layer {layer_idx} not loaded") from None

    # ── step-time: 포인터 pass-through (staging은 호출자 소유 — 계약 ④) ──
    def submit_gateup(self, layer_idx, qlen_ptr, k, expert_ids_ptr, x_ptr, out_ptr, cuda_stream=None,
                      weights_ptr=0):
        self._wrapper(layer_idx).submit_forward_gateup(
            qlen_ptr, k, expert_ids_ptr, x_ptr, out_ptr, cuda_stream, weights_ptr)

    def submit_down(self, layer_idx, qlen_ptr, k, expert_ids_ptr, act_ptr, out_ptr, cuda_stream=None,
                    weights_ptr=0):
        self._wrapper(layer_idx).submit_forward_down(
            qlen_ptr, k, expert_ids_ptr, act_ptr, out_ptr, cuda_stream, weights_ptr)

    def submit_signal(self, cuda_stream: int, flag_ptr: int, value: int) -> None:
        """stream host node로 kt 큐 끝에 '플래그=value' 태스크를 붙인다 (cold_async)."""
        self.cpuinfer.submit_with_cuda_stream(cuda_stream, self.cpuinfer.signal_task(flag_ptr, value))

    def sync(self, cuda_stream=None):
        if cuda_stream is None:
            self.cpuinfer.sync()
        else:
            self.cpuinfer.sync_with_cuda_stream(cuda_stream)
