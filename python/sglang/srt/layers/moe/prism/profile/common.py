"""프로파일 API의 공통 부품 — shape 어휘, 타이머, sparsity 합성, 리포트 헬퍼.

이 패키지는 **Plan을 읽지 않는다.** 프로파일러는 "이 치수에서 이 연산이 몇
µs냐"만 묻고 그 답이 Plan을 만드는 입력이 되므로, 역방향 의존을 만들면 안 된다
(그래서 `plan.py`/`weights.py`를 import하지 않고 스토어 형태만 흉내낸다).

**라이브러리 규약**: 여기서는 `SystemExit`을 던지지 않는다. 잘못된 입력은
`ValueError`다 — 남의 프로그램에 심었을 때 프로세스를 죽이지 않기 위해서고,
CLI 껍데기가 그것을 잡아 `SystemExit`으로 바꾼다.

sparsity 합성: 커널(GPU/CPU 양쪽)이 스스로 threshold를 계산하고 페어 점수와
비교하므로 (계약 ①의 k2wl2), 원하는 마스크를 얻으려면 그 입력을 거꾸로 짜야 한다:

    imp²[j] = a[2j]·x0² + a[2j+1]·x1² + 2·c[j]·x0·x1
    keep[j] = imp[j] >= thr[e, round(s/grid)]

x ≡ 1, c ≡ 0, a[2j] = a[2j+1] ∈ {1, 0}, thr 곡선을 상수로 채우면 imp²[j] ∈
{2, 0}이고 thr² = 0.25이므로 keep[j] = (a[2j] == 1)이 된다. 곡선이 상수라 격자
인덱스(→ s → 라우터 가중)가 무엇이든 조회값이 같다 — 즉 **라우터 가중과 무관하게**
마스크가 결정되고, GPU와 CPU가 같은 마스크를 본다.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import socket
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch

# kt packed 저장의 K축 타일 (kernels.cold_pack_tile_rows의 값). 티어 행 수를 이
# 배수로 잡으면 로더의 타일 올림(real_rows)이 필요 없어 패딩 없는 구성만 다룬다.
K_STEP = 32
PAIR_GROUP = 2

# sparsity 합성 상수 (위 docstring의 역산). thr 곡선이 상수이므로 예산
# 스칼라(p/lam/pmax/grid/ng/renorm_it)는 마스크에 영향을 주지 않지만, kt와
# GPU 커널 양쪽이 유효 범위를 검증하므로 실 plan과 같은 값을 쓴다.
THR_CONST = 0.5
NG, GRID, PMAX, RENORM_IT = 201, 0.005, 0.9, 3
SPARSITY_P, SPARSITY_LAM = 0.5, 4.0

PROJS = ("gate", "up", "down")


# ─── shape 어휘 ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Shape:
    """MoE 한 레이어의 치수. proj가 K축과 N을 결정한다."""

    experts: int
    topk: int
    hidden: int
    inter: int

    def __post_init__(self) -> None:
        if self.topk > 16:
            raise ValueError(f"top_k <= 16 (커널의 per-thread slot 예산), got {self.topk}")
        for name in ("experts", "topk", "hidden", "inter"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def k_axis(self, proj: str) -> int:
        """이 proj의 contraction 축 전체 길이 (티어가 여기서 행을 나눠 갖는다)."""
        self._check(proj)
        return self.inter if proj == "down" else self.hidden

    def n_cols(self, proj: str) -> int:
        self._check(proj)
        return self.hidden if proj == "down" else self.inter

    def x_row_is_pair(self, proj: str) -> bool:
        """down의 x는 expert별 act라 행이 pair (m, j)다 (executor와 같은 규약)."""
        self._check(proj)
        return proj == "down"

    @staticmethod
    def _check(proj: str) -> None:
        if proj not in PROJS:
            raise ValueError(f"unknown proj {proj!r} (expected one of {PROJS})")

    def replace(self, **kw) -> "Shape":
        """치수 하나만 바꾼 사본 — E 스윕처럼 한 인자만 훑을 때 쓴다."""
        return Shape(**{**self.as_dict(), **kw})

    def as_dict(self) -> dict:
        return {"experts": self.experts, "topk": self.topk,
                "hidden": self.hidden, "inter": self.inter}


def split_rows(k_axis: int, frac: float, *, step: int = K_STEP) -> int:
    """K축에서 비율 `frac`에 해당하는 행 수를 `step` 배수로 만든다.

    0과 1은 정확히 보존한다 (frac=0 → 0행, frac=1 → 전체). 그 사이에서는
    반올림하되 최소 한 타일은 준다 — "티어가 있는데 행이 0"은 프로파일 입력으로
    의미가 없다.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {frac}")
    if frac == 0.0:
        return 0
    if frac == 1.0:
        return k_axis
    rows = int(round(k_axis * frac / step)) * step
    return max(step, min(k_axis, rows))


def tier_index(k_axis: int, k_rows: int, *, skip: int = 0,
               shuffle: bool = False, seed: int = 0) -> torch.Tensor:
    """이 티어가 소유하는 K축 행 번호 [k_rows] int32.

    실제 plan의 티어 멤버십은 중요도 순으로 뽑힌 **흩어진 행**이므로 (계약 ①의
    가변 per-expert 인덱스), 고정 시드 순열에서 `skip` 이후 `k_rows`개를 취한다.
    저장 순서는 오름차순 — 로더가 그렇게 굽고, 그 순서가 gather 지역성을
    결정한다. `shuffle=True`는 정렬하지 않은 최악 경우다.
    """
    if k_rows > k_axis - skip:
        raise ValueError(f"tier rows {k_rows} + skip {skip} exceed axis {k_axis}")
    g = torch.Generator().manual_seed(seed)
    rows = torch.randperm(k_axis, generator=g)[skip: skip + k_rows]
    if not shuffle:
        rows = rows.sort().values
    return rows.to(torch.int32).contiguous()


# ─── sparsity 합성 ─────────────────────────────────────────────────────────
def sparse_tables(experts: int, k_rows: int, sparsity: float, *,
                  pattern: str = "random", seed: int = 0,
                  ng: int = NG, thr: float = THR_CONST):
    """요청 sparsity를 정확히 실현하는 (a, c, thr_tab, 실현 keep 비율).

    a: fp32 [E·k_rows] — wn². c: fp32 [E·k_rows/2] — 0. thr_tab: fp32 [E, ng].
    모두 weight 스토어와 같은 오프셋(expert 블록 이어붙인 flat)이다.

    pattern:
      random — 페어를 시드 고정 랜덤으로 죽인다 (실제 마스크의 산포에 가깝다).
      block  — 앞쪽 페어만 살린다 (kt의 16-페어 워드 스킵이 최대로 먹는 최선 경우).
    """
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    if k_rows % PAIR_GROUP:
        raise ValueError(f"tier rows must be even (pair group), got {k_rows}")
    npairs = k_rows // PAIR_GROUP
    keep_n = int(round(npairs * (1.0 - sparsity)))
    a = torch.zeros(experts, k_rows, dtype=torch.float32)
    for e in range(experts):
        if pattern == "block":
            sel = torch.arange(keep_n)
        elif pattern == "random":
            g = torch.Generator().manual_seed(seed + e)
            sel = torch.randperm(npairs, generator=g)[:keep_n]
        else:
            raise ValueError(f"unknown mask pattern {pattern!r} (random|block)")
        pair = torch.zeros(npairs, dtype=torch.float32)
        pair[sel] = 1.0
        a[e] = pair.repeat_interleave(PAIR_GROUP)
    return (
        a.reshape(-1).contiguous(),
        torch.zeros(experts * npairs, dtype=torch.float32),
        torch.full((experts, ng), thr, dtype=torch.float32).contiguous(),
        keep_n / npairs if npairs else 1.0,
    )


# ─── 타이머 ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Timing:
    """iteration당 µs의 표본 요약. median을 대표값으로 쓴다 — 공유 머신의
    간헐적 간섭이 mean을 오염시키므로."""

    us: float
    min_us: float
    max_us: float
    p90_us: float
    replays: int

    @classmethod
    def of(cls, samples: Sequence[float]) -> "Timing":
        if not samples:
            raise ValueError("no samples")
        s = sorted(samples)
        return cls(
            us=round(statistics.median(s), 3),
            min_us=round(s[0], 3),
            max_us=round(s[-1], 3),
            p90_us=round(s[min(len(s) - 1, int(0.9 * len(s)))], 3),
            replays=len(s),
        )

    def as_dict(self) -> dict:
        return {"us": self.us, "min_us": self.min_us, "max_us": self.max_us,
                "p90_us": self.p90_us, "replays": self.replays}


@dataclass(frozen=True)
class SparseGemv:
    """[k_rows, n_cols] weight 하나의 **sparse** GEMV 결과.

    dense와 달리 "몇 바이트를 읽었나"가 두 개다 — 마스킹 전(dense_bytes)과 실제로
    읽은 양(kept_bytes). GB/s를 dense 바이트로 나누면 대역폭이 과대평가되고 kept로
    나누면 실효값이 나오므로 둘 다 노출한다 (한쪽만 두면 반드시 오독된다).
    """

    where: str            # "warm" (pinned/UVA) | "cold" (CPU/kt)
    k_rows: int
    n_cols: int
    sparsity: float       # 요청값
    keep_frac: float      # 실현값 (합성이 정확하므로 요청과 거의 같다)
    dense_bytes: int
    timing: Timing

    @property
    def us(self) -> float:
        return self.timing.us

    @property
    def kept_bytes(self) -> int:
        return int(self.dense_bytes * self.keep_frac)

    @property
    def gbps(self) -> float:
        """실효 대역폭 — 실제로 읽은 바이트 기준."""
        return gbps(self.kept_bytes, self.timing.us)

    @property
    def gbps_dense(self) -> float:
        """마스킹 전 바이트 기준. dense 구성과 직접 비교할 때만 의미가 있다."""
        return gbps(self.dense_bytes, self.timing.us)

    def as_dict(self) -> dict:
        d = dict(self.timing.as_dict())
        d.update(where=self.where, k_rows=self.k_rows, n_cols=self.n_cols,
                 sparsity=self.sparsity, keep_frac=round(self.keep_frac, 4),
                 dense_bytes=self.dense_bytes, kept_bytes=self.kept_bytes,
                 gbps=self.gbps, gbps_dense=self.gbps_dense)
        return d


@contextlib.contextmanager
def nvtx(name: str):
    """NVTX 구간 — nsys 타임라인에서 어느 변형/어느 단계인지 구분하는 유일한 표시.

    표시가 없으면 리포트의 변형들이 이름 없이 섞여서, 어느 커널이 어느 측정에
    속하는지 순서로 추측해야 한다. CUDA가 없으면(cold 단독 경로) no-op이다.
    """
    if not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def capture(launch: Callable[[int], None], reps: int, *, warmup: int = 3,
            error_mode: str = "thread_local") -> "torch.cuda.CUDAGraph":
    """`launch(i)`를 i=0..reps-1로 한 그래프에 캡처한다.

    error_mode는 thread_local이 기본이다: 캡처 중 cold의 host node(kt의
    cudaLaunchHostFunc)와 그 뒤의 host 측 할당이 global 모드에서 불필요하게
    잡히는 것을 피한다 — sglang의 CudaGraphRunner도 같은 이유로 이 모드다.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(warmup):
            launch(i % reps)  # reps < warmup이면 iteration 자원을 돌려 쓴다
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode=error_mode):
        for i in range(reps):
            launch(i)
    return graph


def replay_timing(graph: "torch.cuda.CUDAGraph", reps: int, *,
                  replays: int = 20, warmup: int = 3) -> Timing:
    """graph replay 시간 / reps = launch당 µs."""
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    per = []
    for _ in range(replays):
        beg = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        beg.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        per.append(beg.elapsed_time(end) * 1e3 / reps)
    return Timing.of(per)


def graph_timing(launch: Callable[[int], None], reps: int, *,
                 replays: int = 20, error_mode: str = "thread_local") -> Timing:
    """커널 `reps`개를 그래프 하나에 담아 replay/reps를 잰다.

    launch당 고정비(커널 launch ~2 µs)를 graph가 지우므로 남는 것이 커널 자체의
    시간이다. **같은 launch를 reps번 반복하면 L2-hot 시간**이 나오므로, 호출자가
    `launch(i)`의 i로 iteration마다 다른 expert를 태워야 한다 (실측 차이 30%).
    """
    graph = capture(launch, reps, error_mode=error_mode)
    try:
        return replay_timing(graph, reps, replays=replays)
    finally:
        # 그래프를 살려두면 캡처한 pool이 다음 캡처의 할당과 겹친다.
        del graph
        torch.cuda.synchronize()


def host_timing(step: Callable[[int], None], reps: int, *, replays: int = 20,
                sync_cuda: bool = False, warmup_rounds: int = 1) -> Timing:
    """host 루프 타이머 — CUDA graph에 담을 수 없는 경로(cold 단독, eager 교차
    검증)용. sync_cuda면 라운드 끝에서 GPU까지 기다린다 (겹침 측정)."""
    for _ in range(warmup_rounds):
        for i in range(reps):
            step(i)
    if sync_cuda:
        torch.cuda.synchronize()
    per = []
    for _ in range(replays):
        t0 = time.perf_counter()
        for i in range(reps):
            step(i)
        if sync_cuda:
            torch.cuda.synchronize()
        per.append((time.perf_counter() - t0) / reps * 1e6)
    return Timing.of(per)


# ─── 리포트 ────────────────────────────────────────────────────────────────
def env_stamp(device: Optional[torch.device] = None) -> dict:
    """숫자를 나중에 해석할 수 있게 하는 최소 정보. 공유 머신이라 GPU 점유량도
    같이 남긴다 (남이 쓰는 중이면 절대값이 흔들린다)."""
    out = {
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available() and device is not None:
        idx = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        out["gpu"] = {
            "index": idx, "name": props.name,
            "sm": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
            "total_mem_gb": round(props.total_memory / 1e9, 1),
            "mem_used_gb": round((total - free) / 1e9, 1),
        }
    keep = ("CUDA_VISIBLE_DEVICES", "SGLANG_PRISM_CPUINFER_THREADS",
            "SGLANG_PRISM_NUMA_MAP", "OMP_NUM_THREADS")
    out["env"] = {k: os.environ[k] for k in keep if k in os.environ}
    return out


def numa_nodes() -> int:
    """NUMA 노드 수. `numa.py`를 쓰지 않는 이유는 이 모듈이 CUDA 없이도 import
    되어야 하기 때문이다 (cold 단독 경로)."""
    try:
        return len([d for d in os.listdir("/sys/devices/system/node")
                    if d.startswith("node") and d[4:].isdigit()]) or 1
    except OSError:
        return 1


def default_cpuinfer_threads() -> int:
    """method.py와 같은 관례: 물리 코어 − 2. 과다구독은 submit/sync 고정비를
    폭증시킨다 (실측: 물리 16코어에 60스레드 → sync 회당 1.85 ms)."""
    env = os.environ.get("SGLANG_PRISM_CPUINFER_THREADS")
    if env:
        return int(env)
    return max(2, (os.cpu_count() or 4) // 2 - 2)


def gbps(nbytes: float, micros: float) -> float:
    return round(nbytes / (micros * 1e-6) / 1e9, 1)


def select_device(index) -> torch.device:
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    dev = torch.device(index if isinstance(index, str) else f"cuda:{int(index)}")
    if (dev.index or 0) >= torch.cuda.device_count():
        raise ValueError(f"device {dev} >= {torch.cuda.device_count()} devices")
    torch.cuda.set_device(dev)
    return dev


def emit(payload: dict, out: Optional[str] = None, *, quiet: bool = False) -> str:
    """JSON 직렬화 + (선택) 파일 기록. CLI 껍데기가 쓰는 헬퍼."""
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if not quiet:
        print(text)
    if out:
        Path(out).write_text(text + "\n")
        if not quiet:
            print(f"\n-> {out}", flush=True)
    return text
