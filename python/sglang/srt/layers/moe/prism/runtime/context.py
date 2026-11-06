"""Lazy Prism runtime ownership, independent of SGLang lifecycle integration.

The integration owns the process singleton and supplies live configuration and
capture callbacks. This context owns resources, backend and executor; layers
borrow them. Failed initialization retains the same partial-resource/retry
behavior as the legacy implementation. No new cleanup or caching policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from sglang.srt.layers.moe.prism.activation import Activation
from sglang.srt.layers.moe.prism.plan import Plan, Proj, Tier
from sglang.srt.layers.moe.prism.runtime.config import PrismEnvConfig


@dataclass(frozen=True)
class RuntimeBindings:
    """Resolve integration inputs at their original lazy read boundaries."""

    config: Callable[[], PrismEnvConfig]
    cold_gpu_min_m: Callable
    warm_kt_enabled: Callable
    hybrid_local_node: Callable
    capture_mode_fn: Callable[[], Callable[[], bool]]


class PrismRuntime:
    """프로세스 전역 1벌: plan + resources + cold backend + executor.

    (kt의 SharedStagingBuffer/SharedFullContext와 같은 배치 — apply는
    per-layer 호출이라 cross-layer 상태를 전역이 들어야 한다)
    """

    def __init__(self, plan: Plan, calib=None, *, bindings: RuntimeBindings):
        self._bindings = bindings
        from sglang.srt.layers.moe.prism.kernels import gpu_store_format

        self.plan = plan
        # 스토어 포맷 — plan의 GPU 커널 키가 함의한다 (계약 ①). 파라미터 등록·full 텐서
        # 인출·gather·커널 진입점이 전부 이 객체를 통한다 (formats.py).
        self.fmt = gpu_store_format(plan.kernels.gpu_warm)
        # 모델의 gate/up 활성화 (activation.py: silu[+clamp] | situ). create_moe_runner가 layer의
        # MoeRunnerConfig에서 읽어 채운다 — 전 층 공통 값이라 프로세스 1벌. rejoin#1이 이걸로
        # fp32 합에 활성화를 건다 (DSV4-Flash/GLM: silu limit=10, Qwen: silu, Kimi K3: situ 4/25).
        self.activation: Optional[Activation] = None
        # 모델의 routed_scaling_factor (GLM-5.3 2.5 / DSV4-Flash 1.5). 이것도 create_moe_runner에서만
        # 보인다. 일반 경로에서는 **MoE 러너가** 곱하는데(moe_runner/triton.py:109·164·209,
        # triton_kernels.py:205, flashinfer_cutlass.py:151) prism은 apply를 전유해 러너를 안 쓴다.
        # 모델 쪽 사후 곱셈도 prism에는 걸리지 않는다: deepseek_v2.py의 가드는
        # `not _is_cuda ... or isinstance(KTEPWrapperMethod)`이고(upstream:1021-1027,
        # 포크:810-816) prism은 그 목록에 없으며, maybe_fuse_routed_scale_and_shared_add는
        # 비-fused method에 대해 shared만 더하고 스케일 없이 반환한다
        # (mxfp4_flashinfer_trtllm_moe.py:415-417). 그래서 prism이 직접 곱한다.
        # 정답 규약은 체크포인트 레퍼런스가 정한다: DeepSeek-V4-Flash-0731
        # inference/model.py:588 `weights *= self.route_scale` (Gate.forward 안) —
        # 라우터 가중에 접는 것이고, 곱셈이 선형이므로 expert 가중합 결과에 곱하는 것과 같다.
        self.routed_scaling_factor: Optional[float] = None
        # cold(CPU) 행이 plan 어디에도 없으면 kt 백엔드(스레드풀)를 만들지 않는다 — GPU
        # 스트리밍 전용 plan(mxfp4)에서 idle 워커가 CPU를 점유할 이유가 없다.
        self.has_cold = any(
            plan.expert(l, e).proj(p).has_tier(Tier.COLD)
            for (l, e) in plan.experts for p in Proj
        )
        self.has_warm = any(
            plan.expert(l, e).proj(p).has_tier(Tier.WARM)
            for (l, e) in plan.experts for p in Proj
        )
        if self.has_cold:
            # 스토어 포맷과 cold 커널의 호환 (mxfp4 ↔ kt_amx_fp4 / bf16 ↔ kt_*_bf16) — startup에서 즉사.
            self.fmt.check_cold_kernel(plan.kernels.cpu_cold)
        # sparsity 점수·threshold 테이블 (dense plan이면 None). 프로세스에 1벌 —
        # 레이어마다 다시 열면 382MB 자산을 40번 읽게 된다.
        self.calib = calib
        self.max_tokens = self._bindings.config().max_tokens
        self._resources = None
        self._cold = None
        self._executor = None

    def executor(self, device: torch.device):
        if self._executor is None:
            from sglang.srt.layers.moe.prism.executor import PrismExecutor
            from sglang.srt.layers.moe.prism.kernels import resolve_gpu_kernel
            from sglang.srt.layers.moe.prism.resources import (
                ExecutionResources,
                ResourceSpec,
            )

            resolve_gpu_kernel(self.plan.kernels.gpu_warm)  # 이름 검증
            # 이 포맷의 GEMV/grouped 커널 lazy JIT을 startup으로 앞당긴다 — 첫 호출이 캡처
            # 워밍업이면 컴파일이 캡처 순서에 얽힌다.
            self.fmt.warmup()
            # rejoin Triton 커널도 캡처 워밍업 전에 컴파일한다.
            from sglang.srt.layers.moe.prism.rejoin import warmup as warmup_rejoin

            d = self.plan.dims
            warmup_rejoin(device, d.intermediate_size, d.hidden_size, d.top_k,
                          activation=self.activation)
            spec = ResourceSpec.from_plan(
                self.plan, max_tokens=self.max_tokens, device=device)
            self._resources = ExecutionResources(spec)
            # 조립 지점: env·runner 같은 외부 입력은 전부 여기서 읽어 명시
            # 인자로 주입한다 (executor는 hidden input 없음).
            gmin = self._bindings.config().grouped_min_m_raw
            mask_stats = None
            if self._bindings.config().mask_stats_path:
                from sglang.srt.layers.moe.prism.mask_stats import MaskStats

                mask_stats = MaskStats(self.plan, self.calib, self._bindings.config().mask_stats_path,
                                       every=self._bindings.config().mask_stats_every,
                                       nan_probe=self._bindings.config().mask_stats_nan_probe)
            self._executor = PrismExecutor(
                self.plan, self._resources,
                self.cold() if (self.has_cold or
                                (self.has_warm and self._bindings.warm_kt_enabled(self.fmt))) else None,
                cold_stream=self._bindings.config().cold_stream,
                capture_mode_fn=self._bindings.capture_mode_fn(),
                grouped_min_m=self._bindings.config().parse_grouped_min_m(gmin),
                split_streams=self._bindings.config().split_streams,
                cold_gpu_min_m=self._bindings.cold_gpu_min_m(),
                cold_hybrid_frac=self._bindings.config().cold_hybrid_frac,
                # plan shard 인덱스 중 GPU-local NUMA 노드: NUMA_MAP이 있으면 그 리스트에서 찾고,
                # 없으면 shard 인덱스 = 물리 노드.
                hybrid_local_node=self._bindings.hybrid_local_node(device),
                warm_cpu_min_m=self._bindings.config().warm_cpu_min_m,
                warm_cpu_max_m=self._bindings.config().warm_cpu_max_m(self.fmt.name, PrismExecutor.COLD_GPU_MIN_M),
                cold_async=self._bindings.config().cold_async,
                cold_split=self._bindings.config().cold_split,
                mask_stats=mask_stats,
            )
        return self._executor

    def cold(self):
        if self._cold is None:
            from sglang.srt.layers.moe.prism.cold_backend import KtColdBackend
            from sglang.srt.layers.moe.prism.numa import numa_node_count

            # 기본값 = 물리 코어 − 2 (HT 제외, 메인/tokenizer 여유).
            # 과다구독은 submit/sync 고정비를 폭증시킨다 (실측 2026-08-20:
            # 물리 16코어에 60스레드 → sync 회당 1.85ms, 14스레드 → 0.05ms).
            threads = self._bindings.config().cpuinfer_threads
            numa_map = self._bindings.config().numa_map_raw
            # cold GPU 읽기가 켜져 있으면 slab을 이 device에 host-register한다.
            gpu_view_device = (
                torch.device(torch.cuda.current_device())
                if self._bindings.cold_gpu_min_m() is not None else None
            )
            if numa_map:
                from kt_kernel import kt_kernel_ext

                nodes = self._bindings.config().parse_numa_map(numa_map)
                cfg = kt_kernel_ext.WorkerPoolConfig()
                cfg.subpool_count = len(nodes)
                cfg.subpool_numa_map = nodes
                cfg.subpool_thread_count = [
                    threads // len(nodes) + (1 if i < threads % len(nodes) else 0)
                    for i in range(len(nodes))
                ]
                self._cold = KtColdBackend(
                    self.plan,
                    max_tokens=self.max_tokens,
                    num_numa_nodes=len(nodes),
                    cpuinfer=kt_kernel_ext.CPUInfer(cfg),
                    gpu_view_device=gpu_view_device,
                    hybrid_mask=self._bindings.config().hybrid_mask,
                )
            else:
                self._cold = KtColdBackend(
                    self.plan,
                    max_tokens=self.max_tokens,
                    num_numa_nodes=numa_node_count(),
                    cpuinfer_threads=threads,
                    gpu_view_device=gpu_view_device,
                    hybrid_mask=self._bindings.config().hybrid_mask,
                )
        return self._cold

