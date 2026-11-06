"""Lazy environment configuration for the Prism MoE integration.

Reads stay at their original lifecycle boundaries; constructing this object does
not parse, validate or snapshot the environment. In particular, unused settings
must not fail startup. Preserve the historical bool/empty-string conventions.
No torch, CUDA, kt-kernel or SGLang runtime imports belong in this module.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping

_ENV_PLAN = "SGLANG_PRISM_PLAN"
_ENV_MAX_TOKENS = "SGLANG_PRISM_MAX_TOKENS"
_ENV_CPUINFER = "SGLANG_PRISM_CPUINFER_THREADS"
# 실험 노브: 콤마 구분 NUMA node id 리스트 (예: "1" = node 1 단독).
# 설정 시 subpool을 그 노드들에만 만들고, plan의 shard "node i"는
# i번째 서브풀(= 리스트의 i번째 실제 노드)로 매핑된다. 미설정 = 전 노드.
_ENV_NUMA_MAP = "SGLANG_PRISM_NUMA_MAP"
# eager에서도 cold submit/sync를 stream host node로 보내는 opt-in.
# (graph 경로는 env와 무관하게 항상 stream 통합 — executor 참조.)
_ENV_COLD_STREAM = "SGLANG_PRISM_COLD_STREAM"
# prefill grouped GEMM 임계 M (미설정 = executor 기본값). 벤치·회귀 비교용 노브.
_ENV_GROUPED_MIN_M = "SGLANG_PRISM_GROUPED_MIN_M"
# grouped 경로에서 hot/warm 스트림 분리 (기본 1). "0"으로 끄면 직렬 발행.
_ENV_SPLIT_STREAMS = "SGLANG_PRISM_SPLIT_STREAMS"
# 이 M 이상의 prefill에서 cold를 GPU가 packed slab 제자리 읽기로 계산.
# 미설정 = executor.COLD_GPU_MIN_M(실측 교차점), "0" = 끔 (slab host-register도 안 함
# — 등록은 cold 전량을 pinned로 만들어 프로세스 수명 동안 잠근다).
_ENV_COLD_GPU_MIN_M = "SGLANG_PRISM_COLD_GPU_MIN_M"
# cold hybrid: cold GPU 조건에서 expert의 이 비율만 GPU, 나머지 CPU 동시 계산 (미설정 = 끔).
_ENV_COLD_HYBRID_FRAC = "SGLANG_PRISM_COLD_HYBRID_FRAC"
# warm을 kt 포맷 slab(pinned) 한 벌로 (row-major pinned 대신). GPU는 packed GEMV /
# cold-layout grouped로 읽는다. bf16은 기본 ON; GPU view가 필요하다.
_ENV_WARM_KT = "SGLANG_PRISM_WARM_KT"
# 명시적 실험 override: 이 M 이상 warm CPU. bf16 미설정 기본은 1 < M < 1536 CPU.
_ENV_WARM_CPU_MIN_M = "SGLANG_PRISM_WARM_CPU_MIN_M"
_ENV_WARM_CPU_MAX_M = "SGLANG_PRISM_WARM_CPU_MAX_M"
# cold를 전용 stream + 플래그 wait로 (블로킹 콜백 없음). eager 전용. hybrid와 동시 불가.
_ENV_COLD_ASYNC = "SGLANG_PRISM_COLD_ASYNC"
# cold submit/sync host node를 곁 스트림에 얹는다 (P2). graph decode에서
# sync가 hot/warm 커널 완료를 기다리던 가짜 의존을 끊는다 — 실측 down phase −31%.
_ENV_COLD_SPLIT = "SGLANG_PRISM_COLD_SPLIT"
# 로딩 훅(process_weights_after_loading)의 torch intra-op 스레드 수. model_runner가
# load_model 진입부에서 전역 1로 고정하는데(평범한 로딩의 memcpy 스레드 경합 방지),
# prism의 훅은 memcpy가 아니라 층당 3.4 GB를 훑는 K-슬라이스 repack이라 그 값이
# 그대로 병목이 된다 (실측 1스레드 4.0 s vs 16스레드 0.87 s = 4.7×).
# 미설정 = 물리 코어 수(HT 제외). cold_load 시점의 kt 워커들은 50 ms 뒤 condvar에서
# 잠들므로(worker_pool.cpp) prepare 구간에 코어를 다 써도 경합하지 않는다.
_ENV_LOAD_THREADS = "SGLANG_PRISM_LOAD_THREADS"
# effective sparsity 실측 (mask_stats.py): 출력 json 경로. sparse plan + eager decode에서만 값이 쌓인다.
_ENV_MASK_STATS = "SGLANG_PRISM_MASK_STATS"
_ENV_MASK_STATS_EVERY = "SGLANG_PRISM_MASK_STATS_EVERY"
# NaN 진단: 층 입력/act의 유한성을 prefill 포함해 검사 (층당 GPU 리덕션 + 동기).
_ENV_MASK_STATS_NANPROBE = "SGLANG_PRISM_MASK_STATS_NANPROBE"



class PrismEnvConfig:
    """Typed access to live settings, or an injected mapping for CPU-only tests."""

    def __init__(self, environ: Mapping[str, str] | None = None,
                 cpu_count: Callable[[], int | None] | None = None):
        self._environ = environ
        self._cpu_count = cpu_count

    @property
    def env(self) -> Mapping[str, str]:
        return os.environ if self._environ is None else self._environ

    @property
    def plan_path(self) -> str:
        return self.env[_ENV_PLAN]

    @property
    def enabled(self) -> bool:
        return bool(self.env.get(_ENV_PLAN))

    @property
    def max_tokens(self) -> int:
        return int(self.env.get(_ENV_MAX_TOKENS, "4096"))

    def _cores(self) -> int:
        return (self._cpu_count or os.cpu_count)() or 4

    def load_threads(self) -> int:
        default = max(1, self._cores() // 2)
        return max(1, int(self.env.get(_ENV_LOAD_THREADS, str(default))))

    @property
    def cpuinfer_threads(self) -> int:
        default = max(2, self._cores() // 2 - 2)
        return int(self.env.get(_ENV_CPUINFER, str(default)))

    @property
    def numa_map_raw(self) -> str | None:
        return self.env.get(_ENV_NUMA_MAP)

    @staticmethod
    def parse_numa_map(raw: str) -> list[int]:
        return [int(x) for x in raw.split(",")]

    def hybrid_local_node(self, physical_node: int) -> int:
        raw = self.numa_map_raw
        if raw:
            nodes = self.parse_numa_map(raw)
            return nodes.index(physical_node) if physical_node in nodes else 0
        return physical_node

    @property
    def grouped_min_m_raw(self) -> str | None:
        return self.env.get(_ENV_GROUPED_MIN_M)

    @staticmethod
    def parse_grouped_min_m(raw: str | None) -> int | None:
        return None if raw is None else int(raw)

    def cold_gpu_min_m(self, default: Callable[[str | None], int | None]) -> int | None:
        raw = self.env.get(_ENV_COLD_GPU_MIN_M)
        if raw is None:
            return default(self.grouped_min_m_raw)
        value = int(raw)
        return None if value <= 0 else value

    def warm_kt_enabled(self, format_name: str, cold_gpu_min_m: Callable[[], int | None],
                        *, has_cold: bool = True, all_proj_warm: bool = True) -> bool:
        # Keep evaluation before the explicit override, including short circuiting.
        auto = (format_name == "bf16" and cold_gpu_min_m() is not None
                and has_cold and all_proj_warm)
        return self.env.get(_ENV_WARM_KT, "1" if auto else "0") == "1"

    @property
    def cold_stream(self) -> bool:
        return self.env.get(_ENV_COLD_STREAM) == "1"

    @property
    def split_streams(self) -> bool:
        return self.env.get(_ENV_SPLIT_STREAMS, "1") != "0"

    @property
    def cold_hybrid_frac(self) -> tuple[float, ...] | None:
        raw = self.env.get(_ENV_COLD_HYBRID_FRAC)
        return tuple(float(v) for v in raw.split(",")) if raw else None

    @property
    def hybrid_mask(self) -> bool:
        # Presence/truthiness only: do not eagerly parse a fraction for the backend.
        return bool(self.env.get(_ENV_COLD_HYBRID_FRAC))

    @property
    def warm_cpu_min_m(self) -> int | None:
        raw = self.env.get(_ENV_WARM_CPU_MIN_M)
        return int(raw) if raw else None

    def warm_cpu_max_m(self, format_name: str, default: int) -> int | None:
        if format_name == "bf16" and not self.env.get(_ENV_WARM_CPU_MIN_M):
            return int(self.env.get(_ENV_WARM_CPU_MAX_M, default))
        return None

    @property
    def cold_async(self) -> bool:
        return self.env.get(_ENV_COLD_ASYNC) == "1"

    @property
    def cold_split(self) -> bool:
        return self.env.get(_ENV_COLD_SPLIT) == "1"

    @property
    def mask_stats_path(self) -> str | None:
        return self.env.get(_ENV_MASK_STATS)

    @property
    def mask_stats_every(self) -> int:
        return int(self.env.get(_ENV_MASK_STATS_EVERY, "16"))

    @property
    def mask_stats_nan_probe(self) -> bool:
        return self.env.get(_ENV_MASK_STATS_NANPROBE) == "1"
