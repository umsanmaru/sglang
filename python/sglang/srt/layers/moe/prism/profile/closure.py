"""closure 검사 — 티어별 launch 시간을 **cost model이 합성하는 그 식대로** 합쳤을 때 한
레이어를 통째로 잰 값이 재현되는가.

컴파일러(별도 repo)의 하드웨어 DB는 티어마다 **세 진입점만** 따로 불러 만든다
(`prism-paper/profiler/scripts/prism_tier_sweep.py`):

  hot   hot_dense_gemv(shape, hot_frac=f, ...)               .us("gateup") / .us("down")
  warm  WarmColdProfiler(...).measure(g, only=("warm_only",))  .us("warm_only")
  cold  ColdCpuProfiler(..., proj=g).measure(...)             .us

그리고 phase(gateup, down)마다 GPU 갈래 = hot + warm (같은 스트림이라 더한다),
region = max(GPU 갈래, CPU cold) + handshake 보정으로 합성한다. 이 모듈은 **그 세 호출을
그대로** 불러 같은 식으로 합성하고(`compose`), 같은 점을 `FullLayerProfiler`로 한 번에
잰 `combined`/`layer`와 나란히 놓는다. `planner_model`(접힌 helper 커널, down cold에
`proj="down"` 누락)은 컴파일러가 쓰는 값이 아니므로 모델 쪽으로 쓰지 않는다.

오차 규약: `err = predicted / measured − 1`. **양수 = 모델이 비관적**(실측보다 비싸게
본다), 음수 = 낙관적. 통과는 `|err| <= tolerance`를 기준 변형(`reference`, 기본
`layer`)에 대해 본다.

행 수는 패키지 자신의 반올림을 따른다 — `split_rows`(step = `store_of(dtype).rows_step()`)로
hot/warm 행을 정하고 cold = 축 − hot − warm이다. 이것이 `FullLayerProfiler`가 `cold_frac=None`
으로 보는 행 수(`build_splits`)와 같은지 측정 **전에** 대조하고, 각 티어 호출이 실제로 만든
행 수도 측정 후 다시 대조한다. 맞출 수 없으면 `ValueError`다.

**측정 순서가 중요하다**: 티어 쪽(hot → warm → cold)을 전부 재고 닫은 다음에
full-layer 인스턴스를 만든다. CPUInfer 풀 두 벌이 동시에 살면 서로의 cold 시간을
왜곡한다 (`full_layer.compare`와 같은 이유). warm 프로파일러도 cold 인스턴스(= 풀)를
굽으므로 warm과 cold도 겹치지 않게 차례로 닫는다.
"""

from __future__ import annotations

import gc
import math
import os
import subprocess
from importlib import import_module
from pathlib import Path
from typing import Mapping, Optional, Sequence

from sglang.srt.layers.moe.prism.profile.common import (
    PROJS,
    Shape,
    default_cpuinfer_threads,
    env_stamp,
    split_rows,
    store_of,
)
from sglang.srt.layers.moe.prism.profile.full_layer import GROUPS, build_splits

_P = "sglang.srt.layers.moe.prism.profile."

MODELS = ("max", "handshake", "api")
REFERENCES = ("combined", "layer")
# full-layer가 내는 phase 전체 값들 — split은 cold를 곁 스트림으로 뺀 변형 (P2).
MEASURED = ("combined", "layer", "combined_split", "layer_split")
ENV_FLAGS = ("SGLANG_PRISM_WARM_KT", "SGLANG_PRISM_COLD_SPLIT",
             "SGLANG_PRISM_COLD_GPU_MIN_M")

# ─── 컴파일러 상수 (CompilerMetadata 기본값, µs로 환산) ─────────────────────
# prism-paper/cost_model/types.py 의 CompilerMetadata:
#   cold_handshake_ns = 2_500, tier_join_ns = 5_250, tier_contention = 1.0
#   pcie_bw = 5.553e10, pcie_latency_ns = 16_115, cpu_io_charging = "split_only"
#   gpu_mem_bw = 1.867e12, gpu_kernel_launch_ns = 4_864, cuda_graph = False
#   precision_bytes = {fp16/bf16: 2, fp8: 1, mxfp4: 0.5}
COLD_HANDSHAKE_US = 2.5
TIER_JOIN_US = 5.25
TIER_CONTENTION = 1.0
API_META = {
    "pcie_bw": 5.553e10,
    "pcie_latency_us": 16.115,
    "cpu_io_charging": "split_only",
    "gpu_mem_bw": 1.867e12,
    "gpu_kernel_launch_us": 4.864,
    "cuda_graph": False,
    # api.py:156 — merge 출력 폭은 4096으로 박혀 있다 (base_shapes를 보지 않는다).
    "merge_out_n": 4096,
}
ELEM_BYTES = {"bf16": 2.0, "fp8": 1.0, "fp8pt": 1.0, "mxfp4": 0.5}

# full-layer 측정이 **빠뜨리는 것** — full_layer.py를 보고 확인한 것만 적는다.
NOTES = (
    "full-layer는 입력 staging(x·router 가중·down 입력 act)을 측정 graph **밖에서** 한 번 "
    "채운다 (`FullLayerProfiler.measure`의 st.fill_x / st.fill_act / st.fill_topk_w) — "
    "x·router 가중·activation의 D2H 복사는 combined/layer에 들어 있지 않다. iteration 안의 "
    "D2H는 expert_ids(fill_expert_ids)뿐이다.",
    "cold를 포함한 변형(cold_graph, combined*, layer*)은 flush 인자와 무관하게 L2 flush 없이 "
    "잰다 (`measure`의 VARIANTS 표) — 그 안의 hot은 L2 상주분만큼 낙관적일 수 있다. "
    "hot_only/warm_only/gpu_only만 flush 차분이다.",
    "layer는 combined + cold partial H2D + rejoin이다. gateup의 출력 activation이 down "
    "입력으로 가는 경로는 없다 — down 입력은 합성 act를 graph 밖에서 채운 것이다.",
    "티어 쪽 cold(ColdCpuProfiler)는 kt 동기 forward_*_partial이라 async submit/sync 왕복과 "
    "graph host node 디스패치가 빠져 있다 — handshake 모델의 상수가 바로 그 몫을 메우려는 항이다.",
    "티어 쪽 hot(hot_dense_gemv)은 L2 flush 없이 expert 회전만 한다; full-layer의 hot_only는 "
    "flush 차분이다.",
)


# ─── 합성 (순수 함수) ───────────────────────────────────────────────────────
def _check_us(name: str, v: float) -> float:
    v = float(v)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be a finite value >= 0, got {v}")
    return v


def compose(hot_us: float, warm_us: float, cold_us: float, *,
            model: str = "handshake",
            cold_handshake_us: float = COLD_HANDSHAKE_US,
            tier_join_us: float = TIER_JOIN_US,
            tier_contention: float = TIER_CONTENTION,
            input_transfer_us: float = 0.0, cold_partial_us: float = 0.0,
            merge_us: float = 0.0) -> float:
    """launch 단위 티어 시간 → phase 하나의 예측 µs. 티어가 없으면 그 시간을 0으로 준다.

    모델 (전부 prism-paper/cost_model을 그대로 옮긴 것이고, 상수는 인자다):

      max        max(hot + warm, cold)
                 — compute_3tier.py:319-325 (hot·warm은 같은 스트림이라 더한다)와
                   :383-385 (한쪽 갈래만 있을 때의 max).
      handshake  cold > 0 이고 hot + warm > 0 이면
                     max(hot + warm, cold + cold_handshake) × tier_contention + tier_join
                 아니면 max(hot + warm, cold)
                 — compute_3tier.py:350-385 (`region_compute_ns`; 주석의
                   "combined = max(warm, cold + 2.5 us) + 5.25 us"). 상수 기본값은
                   types.py CompilerMetadata (2_500 / 5_250 ns, contention 1.0).
                   tier_combo DB 조회 갈래(:358-375)는 DB가 있어야 하므로 옮기지 않는다.
      api        input_transfer + max(hot + warm, cold + cold_partial) + merge
                 — api.py:92-109가 3-tier 결과를 **region_compute_ns가 아니라**
                   gpu_branch/cpu_branch 두 의사-segment로 compose_latency(transfer.py,
                   PIPELINE: max(path) + merge)에 넘기고, :180에서 input_transfer를 앞에
                   직렬로 더한다. cold가 없으면 cold 쪽 경로와 input 전송은 없다
                   (estimate_input_transfer_ns는 CPU를 건드리는 plan에만 과금).
                   즉 api 층은 handshake 보정을 쓰지 않는다. 기본 전송값은 `api_overheads`.

    `cold_scaled_ns`(스레드 비 보정)와 `graph_corrected_ns`는 CompilerMetadata 기본값
    (cold_threads=0, cuda_graph=False)에서 항등이라 여기서는 생략한다.
    """
    hot = _check_us("hot_us", hot_us)
    warm = _check_us("warm_us", warm_us)
    cold = _check_us("cold_us", cold_us)
    gpu = hot + warm
    if model == "max":
        return max(gpu, cold)
    if model == "handshake":
        hs = _check_us("cold_handshake_us", cold_handshake_us)
        join = _check_us("tier_join_us", tier_join_us)
        cont = _check_us("tier_contention", tier_contention)
        if cold > 0 and gpu > 0:
            return max(gpu, cold + hs) * cont + join
        return max(gpu, cold)
    if model == "api":
        inp = _check_us("input_transfer_us", input_transfer_us)
        part = _check_us("cold_partial_us", cold_partial_us)
        merge = _check_us("merge_us", merge_us)
        if cold > 0:
            return inp + max(gpu, cold + part) + merge
        return gpu + merge
    raise ValueError(f"unknown model {model!r} (known: {MODELS})")


def api_overheads(shape: Shape, group: str, dtype: str, *, m: int = 1,
                  n_segments: int = 3, meta: Optional[Mapping] = None) -> dict:
    """api 층이 phase 하나에 더하는 전송·merge 항 (µs) — CompilerMetadata 기본값으로.

      input_transfer  transfer.py estimate_input_transfer_ns: M·K·b / pcie_bw + pcie_latency
                      (K = 그 region의 입력 폭 = 이 group의 K축)
      cold_partial    api.py:126-136 + transfer.py estimate_transfer_ns:
                      M·N_out·b / pcie_bw + pcie_latency (N_out = group 출력 폭; gateup은
                      gate+up 융합이라 2·inter — sweep의 NCOLS와 같다)
      merge           api.py:156-158 + transfer.py estimate_merge_ns:
                      M·4096·b·(n_segments+1) / gpu_mem_bw + launch (cuda_graph면 launch 0)

    `b`는 컴파일러처럼 **weight dtype**의 바이트다 (elem_bytes_of(plan)) — activation이
    bf16이어도 그렇다. 컴파일러의 습관을 그대로 옮긴 것이지 옳다는 주장이 아니다.
    pcie_latency는 cpu_io_charging="amortized"면 0이다 (region_pcie_latency_ns).
    """
    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r} (gateup|down)")
    mt = dict(API_META)
    if meta:
        unknown = set(meta) - set(API_META)
        if unknown:
            raise ValueError(f"unknown api meta keys {sorted(unknown)} (known: {sorted(API_META)})")
        mt.update(meta)
    name = store_of(dtype).name
    b = ELEM_BYTES[name]
    k = shape.hidden if group == "gateup" else shape.inter
    n_out = 2 * shape.inter if group == "gateup" else shape.hidden
    lat = 0.0 if mt["cpu_io_charging"] == "amortized" else float(mt["pcie_latency_us"])
    bw = float(mt["pcie_bw"])
    launch = 0.0 if mt["cuda_graph"] else float(mt["gpu_kernel_launch_us"])
    merge_bytes = m * mt["merge_out_n"] * b * (n_segments + 1)
    return {
        "input_transfer_us": round(m * k * b / bw * 1e6 + lat, 4),
        "cold_partial_us": round(m * n_out * b / bw * 1e6 + lat, 4),
        "merge_us": round(merge_bytes / float(mt["gpu_mem_bw"]) * 1e6 + launch, 4),
        "elem_bytes": b, "k": k, "n_out": n_out, "n_segments": n_segments,
    }


def closure_error(predicted_us: float, measured_us: Optional[float]) -> Optional[float]:
    """`predicted / measured − 1`. 양수 = 모델이 비관적. 실측이 없거나 0 이하면 None."""
    if measured_us is None or not measured_us > 0:
        return None
    return round(predicted_us / measured_us - 1.0, 4)


# ─── 행 계획 ───────────────────────────────────────────────────────────────
def _check_frac(name: str, f: float) -> float:
    f = float(f)
    if not math.isfinite(f) or not 0.0 <= f <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {f}")
    return f


def plan_rows(shape: Shape, *, hot_frac: float, warm_frac: float,
              dtype: str = "bf16", groups: Sequence[str] = tuple(GROUPS)) -> dict:
    """이 점에서 세 티어가 볼 행 수 — 패키지 자신의 반올림으로.

    group마다 (K축 = gateup: hidden, down: inter):
      hot  = split_rows(axis, hot_frac, step)     (hot_dense_gemv와 같은 식)
      warm = split_rows(axis, warm_frac, step)    (make_split와 같은 식)
      cold = axis − hot − warm                    (FullLayerProfiler, cold_frac=None)

    그리고 그 값이 `build_splits`(= FullLayerProfiler가 실제로 쓰는 분할)와 같은지
    본다. cold 티어 호출에 넘길 비율은 `cold / axis`이고, 그 비율을 다시 `split_rows`에
    넣었을 때 같은 행이 나오는지까지 확인한다 — 셋 중 하나라도 어긋나면 ValueError다.
    요청 cold 비율(1 − hot − warm)이 0보다 큰데 반올림 후 cold가 0이 되는 것도 거부한다
    (모델이 다른 점을 재게 된다).
    """
    hot_frac = _check_frac("hot_frac", hot_frac)
    warm_frac = _check_frac("warm_frac", warm_frac)
    if hot_frac + warm_frac > 1.0 + 1e-9:
        raise ValueError(f"hot_frac + warm_frac = {hot_frac + warm_frac} > 1")
    for g in groups:
        if g not in GROUPS:
            raise ValueError(f"unknown group {g!r} (gateup|down)")
    store = store_of(dtype)
    step = store.rows_step()
    cold_req = max(0.0, 1.0 - hot_frac - warm_frac)
    out = {"dtype": store.name, "step": step, "hot_frac": hot_frac,
           "warm_frac": warm_frac, "cold_frac_requested": round(cold_req, 6),
           "groups": {}}
    per_axis = {}
    for g in GROUPS:                         # 두 축 다 본다 — 프로파일러가 전 proj를 굽는다
        axis = shape.k_axis(GROUPS[g][0])
        h = split_rows(axis, hot_frac, step=step)
        w = split_rows(axis, warm_frac, step=step)
        c = axis - h - w
        if c < 0:
            raise ValueError(
                f"{g}: hot {h} + warm {w} rows exceed axis {axis} after rounding to "
                f"step {step} — 행을 맞출 수 없다 (frac을 줄여라)")
        if cold_req > 1e-9 and c == 0:
            raise ValueError(
                f"{g}: cold_frac {cold_req:.4f} requested but rounding (step {step}) left "
                f"0 cold rows on axis {axis} (hot {h} + warm {w}) — 행을 맞출 수 없다")
        cold_frac = c / axis
        if c and split_rows(axis, cold_frac, step=step) != c:
            raise ValueError(f"{g}: cold_frac {cold_frac} does not reproduce {c} rows "
                             f"on axis {axis} (step {step})")
        per_axis[g] = {"k_axis": axis, "hot": h, "warm": w, "cold": c,
                       "cold_frac": cold_frac}
    # FullLayerProfiler가 보는 분할과 대조 (spread=0이면 expert마다 같은 값이어야 한다).
    splits = build_splits(shape, hot_frac=hot_frac, warm_frac=warm_frac,
                          cold_frac=None, dtype=store)
    for g, r in per_axis.items():
        for proj in GROUPS[g]:
            sp = splits[proj]
            for tier in ("hot", "warm", "cold"):
                if set(sp.rows(tier)) != {r[tier]}:
                    raise ValueError(
                        f"{proj}: build_splits gives {tier} rows {sorted(set(sp.rows(tier)))}, "
                        f"tier calls would see {r[tier]} — 행을 맞출 수 없다")
    out["groups"] = {g: per_axis[g] for g in groups}
    out["all_axes"] = per_axis
    return out


def _warm_profiler_cold_frac(rows: dict) -> tuple:
    """WarmColdProfiler에 줄 cold_frac 하나.

    이 프로파일러는 한 비율로 **세 proj 전부**의 cold 스토어를 굽는다 (`make_split`).
    warm_only 시간은 cold 행과 무관하지만, 가능하면 실현 cold 행을 두 축에서 그대로
    재현하는 비율을 고른다. 그런 비율이 없거나(축마다 반올림이 달라서) cold가 0이면
    sweep과 같은 한 타일(`step / inter`, prism_tier_sweep.py:171)로 둔다.
    반환 (cold_frac, 출처 문자열)."""
    step = rows["step"]
    axes = rows["all_axes"]
    if all(a["cold"] > 0 for a in axes.values()):
        for a in axes.values():
            f = a["cold_frac"]
            if all(split_rows(b["k_axis"], f, step=step) == b["cold"] for b in axes.values()):
                return f, "realized"
    inter = axes["down"]["k_axis"]
    f = step / inter
    for g, a in axes.items():
        kc = split_rows(a["k_axis"], f, step=step)
        if a["warm"] + kc > a["k_axis"]:
            raise ValueError(
                f"{g}: warm {a['warm']} + one cold tile {kc} > axis {a['k_axis']} — "
                f"WarmColdProfiler가 warm을 이만큼 잴 수 없다 (warm_frac을 줄여라)")
    return f, "one_tile"


# ─── 측정 ──────────────────────────────────────────────────────────────────
def _mod(name: str):
    """측정 진입점은 모듈 속성으로 찾는다 (테스트가 monkeypatch로 갈아끼울 수 있게).
    패키지가 함수 `cold_cpu`를 재수출하므로 `from ... import`가 아니라 import_module이다."""
    return import_module(_P + name)


def _release() -> None:
    """닫은 프로파일러의 CPUInfer 풀이 다음 생성 전에 확실히 사라지게."""
    gc.collect()


def measure_tiers(shape: Shape, *, hot_frac: float, warm_frac: float,
                  sparsity: float, dtype: str = "bf16",
                  cpu_kernel: Optional[str] = None, device=0,
                  threads: Optional[int] = None,
                  groups: Sequence[str] = tuple(GROUPS),
                  reps: int = 200, replays: int = 40,
                  cold_iters: int = 100, cold_replays: int = 8,
                  numa_split: float = 0.5, rows: Optional[dict] = None) -> dict:
    """세 티어를 컴파일러 sweep과 **같은 세 호출**로 따로 잰다 (hot → warm → cold, 차례로).

    hot_frac == 0 / warm_frac == 0 / cold 행 0인 티어는 부르지 않고 0 µs로 둔다 (빈 티어로
    프로파일러를 만들지 않는다). 각 호출이 실제로 만든 행 수를 `plan_rows`와 대조해
    어긋나면 ValueError다.
    """
    rows = rows or plan_rows(shape, hot_frac=hot_frac, warm_frac=warm_frac,
                             dtype=dtype, groups=groups)
    store = store_of(dtype)
    cpu_kernel = cpu_kernel or store.cpu_kernel
    threads = threads or default_cpuinfer_threads()
    out = {g: {"hot_us": 0.0, "warm_us": 0.0, "cold_us": 0.0, "realized": {}}
           for g in groups}
    calls = []

    # ① hot — GPU만 (CPU 풀 없음). sweep과 같은 인자 (projs 기본 = gateup, down).
    if hot_frac > 0:
        r = _mod("hot").hot_dense_gemv(shape, hot_frac=hot_frac, device=device,
                                       dtype=dtype, reps=reps, replays=replays)
        calls.append("hot_dense_gemv")
        for g in groups:
            got = r[g].k_rows
            want = rows["groups"][g]["hot"]
            if got != want:
                raise ValueError(f"{g}: hot_dense_gemv measured {got} rows, expected {want}")
            out[g]["hot_us"] = float(r.us(g))
            out[g]["realized"]["hot"] = got
        del r
    else:
        for g in groups:
            out[g]["realized"]["hot"] = 0

    # ② warm — WarmColdProfiler는 cold 인스턴스(CPU 풀)도 굽는다. 닫고 나서 ③.
    warm_cold_frac = None
    if warm_frac > 0:
        warm_cold_frac, src = _warm_profiler_cold_frac(rows)
        with _mod("warm_cold").WarmColdProfiler(
                shape, warm_frac=warm_frac, cold_frac=warm_cold_frac,
                sparsity=sparsity, numa_split=numa_split, device=device,
                dtype=dtype, threads=threads, cpu_kernel=cpu_kernel) as p:
            calls.append("WarmColdProfiler")
            for g in groups:
                want = rows["groups"][g]["warm"]
                for proj in GROUPS[g]:
                    got = p.splits[proj].k_warm
                    if got != want:
                        raise ValueError(f"{proj}: WarmColdProfiler warm rows {got}, "
                                         f"expected {want}")
                res = p.measure(g, reps=reps, replays=replays, only=("warm_only",))
                out[g]["warm_us"] = float(res.us("warm_only"))
                out[g]["realized"]["warm"] = want
            del p
        _release()
        warm_info = {"cold_frac": warm_cold_frac, "cold_frac_source": src}
    else:
        warm_info = None
        for g in groups:
            out[g]["realized"]["warm"] = 0

    # ③ cold — group마다 인스턴스 하나 (proj=group), 하나씩 닫는다.
    for g in groups:
        c_rows = rows["groups"][g]["cold"]
        out[g]["realized"]["cold"] = c_rows
        if c_rows == 0:
            continue
        cf = rows["groups"][g]["cold_frac"]
        with _mod("cold_cpu").ColdCpuProfiler(
                shape, cold_frac=cf, sparsity=sparsity, proj=g,
                numa_split=numa_split, threads=threads, dtype=dtype,
                cpu_kernel=cpu_kernel) as c:
            calls.append(f"ColdCpuProfiler[{g}]")
            for proj in GROUPS[g]:
                got = c.k_cold[proj]
                if got != c_rows:
                    raise ValueError(f"{proj}: ColdCpuProfiler cold rows {got}, "
                                     f"expected {c_rows}")
            rep = c.measure(iters=cold_iters, replays=cold_replays)
            out[g]["cold_us"] = float(rep.us)
            out[g]["cold_keep_frac"] = round(float(rep.keep_frac), 5)
            del c
        _release()

    return {"groups": out, "calls": calls, "threads": threads,
            "cpu_kernel": cpu_kernel, "warm_profiler": warm_info,
            "params": {"reps": reps, "replays": replays, "cold_iters": cold_iters,
                       "cold_replays": cold_replays, "numa_split": numa_split}}


def measure_full_layer(shape: Shape, *, hot_frac: float, warm_frac: float,
                       sparsity: float, dtype: str = "bf16",
                       cpu_kernel: Optional[str] = None, device=0,
                       threads: Optional[int] = None,
                       groups: Sequence[str] = tuple(GROUPS),
                       reps: int = 50, replays: int = 10, rounds: int = 3,
                       flush: bool = True, numa_split: float = 0.5,
                       sparsity_spread: float = 0.0, seed: int = 0,
                       rows: Optional[dict] = None) -> dict:
    """같은 점을 `FullLayerProfiler`로 한 번에 잰다 (`cold_frac=None`).

    `sparsity_spread` 기본값은 0이다 (FullLayerProfiler 기본 0.3과 다르다) — 티어 호출은
    전 expert 균일 sparsity라, 그것과 같은 점을 재야 closure가 된다.
    반환 group별 `measured`(combined/layer와 split 변형), 전체 변형 µs, info.
    """
    rows = rows or plan_rows(shape, hot_frac=hot_frac, warm_frac=warm_frac,
                             dtype=dtype, groups=groups)
    store = store_of(dtype)
    cpu_kernel = cpu_kernel or store.cpu_kernel
    threads = threads or default_cpuinfer_threads()
    out = {}
    with _mod("full_layer").FullLayerProfiler(
            shape, hot_frac=hot_frac, warm_frac=warm_frac, cold_frac=None,
            sparsity=sparsity, sparsity_spread=sparsity_spread, dtype=dtype,
            device=device, numa_split=numa_split, cpu_kernel=cpu_kernel,
            threads=threads, seed=seed) as p:
        for g in groups:
            want = rows["groups"][g]
            for proj in GROUPS[g]:
                sp = p.splits[proj]
                for tier in ("hot", "warm", "cold"):
                    if set(sp.rows(tier)) != {want[tier]}:
                        raise ValueError(
                            f"{proj}: FullLayerProfiler {tier} rows {sorted(set(sp.rows(tier)))}, "
                            f"tier calls saw {want[tier]}")
        for g in groups:
            rep = p.measure(g, reps=reps, replays=replays, flush=flush, rounds=rounds)
            timings = {k: float(t.us) for k, t in rep.timings.items()}
            out[g] = {"measured": {k: timings[k] for k in MEASURED if k in timings},
                      "timings_us": timings, "info": rep.info}
        params = dict(getattr(p, "params", {}) or {})
        del p
    _release()
    return {"groups": out, "params": params,
            "run": {"reps": reps, "replays": replays, "rounds": rounds, "flush": flush,
                    "sparsity_spread": sparsity_spread, "seed": seed}}


# ─── 출처 · 런타임 경로 ─────────────────────────────────────────────────────
def provenance(*, threads: int, cpu_kernel: str, dtype: str, device=None) -> dict:
    """숫자를 나중에 해석하려면 어느 트리·어느 kt 빌드·어느 env에서 나왔는지가 필요하다."""
    here = Path(__file__).resolve().parent
    commit = dirty = None
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10,
                                check=True).stdout.strip() or None
        st = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "-uno"],
                            capture_output=True, text=True, timeout=10, check=True).stdout
        dirty = bool(st.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import importlib.util

        spec = importlib.util.find_spec("kt_kernel")
        kt_path = spec.origin if spec is not None else None
    except (ImportError, ValueError):
        kt_path = None
    return {
        "env": env_stamp(device),
        "source_commit": commit, "source_dirty": dirty, "source_dir": str(here),
        "kt_kernel_file": kt_path,
        "env_flags": {k: os.environ.get(k) for k in ENV_FLAGS},
        "threads": threads, "cpu_kernel": cpu_kernel, "dtype": store_of(dtype).name,
    }


def runtime_path(dtype: str, *, has_cold: bool, all_proj_warm: bool,
                 environ: Optional[Mapping[str, str]] = None) -> dict:
    """serving이 이 점에서 **프로파일한 것과 다른 warm 커널**을 탈지 런타임 설정에 묻는다.

    규칙을 다시 짜지 않고 method.py(`_cold_gpu_min_m`, `_warm_kt_enabled`)와 같은 방식으로
    `PrismEnvConfig.warm_kt_enabled`와 포맷의 `default_cold_gpu_min_m`을 부른다. warm-kt가
    켜지면 warm은 kt packed slab(`tiers.PackedWarmGateUp`/`PackedWarmTier`)으로 읽히고,
    여기서 잰 row-major pinned sparse GEMV와 다른 커널이다.
    """
    from sglang.srt.layers.moe.prism.runtime.config import PrismEnvConfig

    cfg = PrismEnvConfig(environ)
    store = store_of(dtype)
    if not store.gpu or store.gpu_adapter is not None:
        return {"applicable": False, "dtype": store.name,
                "reason": f"{store.name}는 MoE 포맷(formats.FORMATS)이 없다 — MoE 런타임 경로 없음",
                "warm_kt": False}
    from sglang.srt.layers.moe.prism.executor import PrismExecutor
    from sglang.srt.layers.moe.prism.formats import FORMATS

    fmt = FORMATS[store.name]

    def default(grouped_min_m):
        return fmt.default_cold_gpu_min_m(
            PrismExecutor.COLD_GPU_MIN_M,
            PrismExecutor.GROUPED_MIN_M if grouped_min_m is None
            else cfg.parse_grouped_min_m(grouped_min_m))

    def cold_gpu_min_m():
        return cfg.cold_gpu_min_m(default)

    warm_kt = cfg.warm_kt_enabled(fmt.name, cold_gpu_min_m, has_cold=has_cold,
                                  all_proj_warm=all_proj_warm)
    return {
        "applicable": True, "dtype": store.name, "format": fmt.name,
        "has_cold": has_cold, "all_proj_warm": all_proj_warm,
        "cold_gpu_min_m": cold_gpu_min_m(),
        "warm_kt": bool(warm_kt),
        "cold_split": cfg.cold_split,
        "profiled_warm": "row-major pinned sparse GEMV (fmt.gemv(pinned=True, sparse=True))",
        "serving_warm": ("kt packed slab (tiers.PackedWarmGateUp / PackedWarmTier)" if warm_kt
                         else "row-major pinned sparse GEMV (same as profiled)"),
    }


def _runtime_warnings(rt: dict, reference: str) -> list:
    w = []
    if rt.get("warm_kt"):
        w.append("runtime_path: serving은 warm-kt 경로(SGLANG_PRISM_WARM_KT)로 warm을 kt packed "
                 "slab에서 읽는다 — 여기서 잰 row-major pinned sparse GEMV와 다른 커널이므로 "
                 "closure가 통과해도 serving의 warm 시간을 보증하지 않는다")
    if rt.get("cold_split") and reference in ("combined", "layer"):
        w.append("runtime_path: SGLANG_PRISM_COLD_SPLIT=1 — serving graph decode는 cold를 곁 "
                 f"스트림으로 돌린다; 기준 '{reference}'보다 '{reference}_split'이 대응한다")
    return w


# ─── closure ───────────────────────────────────────────────────────────────
def _check_gate(model: str, reference: str, tolerance: float) -> None:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r} (known: {MODELS})")
    if reference not in REFERENCES:
        raise ValueError(f"unknown reference {reference!r} (known: {REFERENCES})")
    if not (isinstance(tolerance, (int, float)) and math.isfinite(tolerance) and tolerance > 0):
        raise ValueError(f"tolerance must be > 0, got {tolerance}")


def closure(shape: Shape, *, hot_frac: float, warm_frac: float, sparsity: float,
            dtype: str = "bf16", cpu_kernel: Optional[str] = None, device=0,
            threads: Optional[int] = None, groups: Sequence[str] = tuple(GROUPS),
            tolerance: float = 0.10, reference: str = "layer",
            model: str = "handshake",
            cold_handshake_us: float = COLD_HANDSHAKE_US,
            tier_join_us: float = TIER_JOIN_US,
            tier_contention: float = TIER_CONTENTION,
            api_meta: Optional[Mapping] = None,
            tier_reps: int = 200, tier_replays: int = 40,
            cold_iters: int = 100, cold_replays: int = 8,
            full_reps: int = 50, full_replays: int = 10, rounds: int = 3,
            flush: bool = True, numa_split: float = 0.5,
            sparsity_spread: float = 0.0, seed: int = 0) -> dict:
    """한 점의 closure 리포트 (JSON 그대로).

    group마다: 티어 µs, 모델별 예측, full-layer 실측(combined/layer와 split 변형), 모델별
    `err = predicted / measured − 1` (**양수 = 모델 비관적**, 음수 = 낙관적), 그리고
    `model`의 `reference` 대비 `|err| <= tolerance` 판정. 최상위 `pass`는 전 group 통과다.

    입력 검증(모델·기준·허용오차·비율·행 정렬)은 하드웨어를 건드리기 **전에** 끝난다.
    """
    _check_gate(model, reference, tolerance)
    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    groups = tuple(groups)
    rows = plan_rows(shape, hot_frac=hot_frac, warm_frac=warm_frac, dtype=dtype,
                     groups=groups)
    store = store_of(dtype)
    cpu_kernel = cpu_kernel or store.cpu_kernel
    threads = threads or default_cpuinfer_threads()
    overheads = {g: api_overheads(shape, g, store.name, meta=api_meta,
                                  n_segments=sum(rows["all_axes"][g][t] > 0
                                                 for t in ("hot", "warm", "cold")))
                 for g in groups}
    has_cold = any(a["cold"] > 0 for a in rows["all_axes"].values())
    all_proj_warm = all(a["warm"] > 0 for a in rows["all_axes"].values())
    rt = runtime_path(store.name, has_cold=has_cold, all_proj_warm=all_proj_warm)

    # 순서: 티어 쪽을 전부 재고 닫은 뒤 full-layer (CPUInfer 풀이 동시에 둘 살지 않게).
    tiers = measure_tiers(shape, hot_frac=hot_frac, warm_frac=warm_frac,
                          sparsity=sparsity, dtype=dtype, cpu_kernel=cpu_kernel,
                          device=device, threads=threads, groups=groups,
                          reps=tier_reps, replays=tier_replays,
                          cold_iters=cold_iters, cold_replays=cold_replays,
                          numa_split=numa_split, rows=rows)
    full = measure_full_layer(shape, hot_frac=hot_frac, warm_frac=warm_frac,
                              sparsity=sparsity, dtype=dtype, cpu_kernel=cpu_kernel,
                              device=device, threads=threads, groups=groups,
                              reps=full_reps, replays=full_replays, rounds=rounds,
                              flush=flush, numa_split=numa_split,
                              sparsity_spread=sparsity_spread, seed=seed, rows=rows)

    consts = {"cold_handshake_us": cold_handshake_us, "tier_join_us": tier_join_us,
              "tier_contention": tier_contention}
    warnings = _runtime_warnings(rt, reference)
    out_groups = {}
    ok = True
    for g in groups:
        t = tiers["groups"][g]
        hot, warm, cold = t["hot_us"], t["warm_us"], t["cold_us"]
        ov = overheads[g]
        pred = {
            "max": compose(hot, warm, cold, model="max"),
            "handshake": compose(hot, warm, cold, model="handshake", **consts),
            "api": compose(hot, warm, cold, model="api",
                           input_transfer_us=ov["input_transfer_us"],
                           cold_partial_us=ov["cold_partial_us"],
                           merge_us=ov["merge_us"]),
        }
        pred = {k: round(v, 3) for k, v in pred.items()}
        meas = full["groups"][g]["measured"]
        err = {m: {k: closure_error(pred[m], v) for k, v in meas.items()} for m in MODELS}
        e = err[model].get(reference)
        passed = e is not None and abs(e) <= tolerance
        if e is None:
            warnings.append(f"{g}: reference '{reference}' missing or <= 0 — fail")
        ok = ok and passed
        out_groups[g] = {
            "rows": {k: rows["groups"][g][k] for k in ("k_axis", "hot", "warm", "cold")},
            "realized_rows": t["realized"],
            "tiers_us": {"hot": hot, "warm": warm, "cold": cold,
                         "gpu_branch": round(hot + warm, 3)},
            "api_overheads_us": ov,
            "predicted_us": pred,
            "measured_us": meas,
            "err": err,
            "verdict": {"model": model, "reference": reference, "err": e,
                        "tolerance": tolerance, "pass": passed},
            "full_layer_us": full["groups"][g]["timings_us"],
            "full_layer_info": full["groups"][g]["info"],
        }
        if "cold_keep_frac" in t:
            out_groups[g]["cold_keep_frac"] = t["cold_keep_frac"]

    errs = [abs(v["verdict"]["err"]) for v in out_groups.values()
            if v["verdict"]["err"] is not None]
    return {
        "bench": "closure",
        "shape": shape.as_dict(),
        "point": {"hot_frac": float(hot_frac), "warm_frac": float(warm_frac),
                  "cold_frac": rows["cold_frac_requested"], "sparsity": float(sparsity),
                  "dtype": store.name, "cpu_kernel": cpu_kernel, "threads": threads,
                  "device": device if isinstance(device, (int, str)) else str(device),
                  "numa_split": numa_split,
                  "sparsity_spread": sparsity_spread},
        "sign": "err = predicted / measured - 1; positive = model pessimistic",
        "model": model, "reference": reference, "tolerance": tolerance,
        "model_consts": consts,
        "rows": {k: rows[k] for k in ("step", "cold_frac_requested")} | {
            "axes": rows["all_axes"]},
        "groups": out_groups,
        "pass": ok,
        "worst_abs_err": round(max(errs), 4) if errs else None,
        "tier_calls": tiers["calls"],
        "tier_params": tiers["params"] | {"warm_profiler": tiers["warm_profiler"]},
        "full_layer_params": full["params"],
        "full_layer_run": full["run"],
        "runtime_path": rt,
        "warnings": warnings,
        "notes": list(NOTES),
        "provenance": provenance(threads=threads, cpu_kernel=cpu_kernel,
                                 dtype=store.name, device=device),
    }


def _as_point(pt) -> dict:
    if isinstance(pt, Mapping):
        d = dict(pt)
    else:
        vals = tuple(pt)
        if len(vals) != 3:
            raise ValueError(f"point must be (hot_frac, warm_frac, sparsity), got {pt!r}")
        d = dict(zip(("hot_frac", "warm_frac", "sparsity"), vals))
    missing = {"hot_frac", "warm_frac", "sparsity"} - set(d)
    if missing:
        raise ValueError(f"point {pt!r} missing {sorted(missing)}")
    return d


def closure_sweep(shape: Shape, points: Sequence, **kw) -> dict:
    """점 목록을 차례로 closure한다. 리포트 전부 + 요약 (최악 오차, 실패 수).

    점마다 `(hot_frac, warm_frac, sparsity)` 또는 같은 키의 dict. 한 점의 ValueError/
    RuntimeError는 스윕을 죽이지 않고 `error`로 남기며 실패로 센다 (dense_sweep과 같은
    규약). 모델·기준·허용오차가 틀린 것은 점의 문제가 아니므로 시작 전에 ValueError다.
    """
    _check_gate(kw.get("model", "handshake"), kw.get("reference", "layer"),
                kw.get("tolerance", 0.10))
    pts = [_as_point(p) for p in points]
    if not pts:
        raise ValueError("no points")
    reports, worst, worst_at, failing, errors = [], None, None, 0, 0
    for i, p in enumerate(pts):
        try:
            rep = closure(shape, **p, **kw)
        except (ValueError, RuntimeError) as e:
            reports.append({"point": p, "error": f"{type(e).__name__}: {e}"})
            failing += 1
            errors += 1
            continue
        reports.append(rep)
        if not rep["pass"]:
            failing += 1
        for g, v in rep["groups"].items():
            e = v["verdict"]["err"]
            if e is not None and (worst is None or abs(e) > abs(worst)):
                worst, worst_at = e, {"index": i, "group": g, **p}
    return {
        "bench": "closure_sweep",
        "shape": shape.as_dict(),
        "reports": reports,
        "summary": {"points": len(pts), "failing": failing, "errors": errors,
                    "worst_err": worst, "worst_at": worst_at,
                    "model": kw.get("model", "handshake"),
                    "reference": kw.get("reference", "layer"),
                    "tolerance": kw.get("tolerance", 0.10),
                    "pass": failing == 0},
    }


# ─── 구성 요소 합성 (executor 의존 구조) ────────────────────────────────────
EXEC_TARGETS = ("combined", "layer", "runtime")


def compose_exec(c: Mapping[str, float], *, group: str, target: str = "layer",
                 masking: bool = True, contention: bool = False,
                 gpu_branch: str = "sum", tail: bool = False) -> dict:
    """구성 요소 µs → phase 하나의 예측 µs와 항별 분해 (순수 함수).

    executor.py의 graph decode 경로(HOST_BLOCK dispatch, cold host node가 main 스트림)를
    그대로 따라 적는다 (줄 번호는 이 worktree의 executor.py):

      · graph 경로는 `use_cold_stream=True`, `stream_arg` = 현재(main) 스트림
        (:392-405). cold_split이 꺼져 있으면 dispatch는 HOST_BLOCK (:443-446).
      · phase 한 번 = `_cpu_partials_begin` → `_run_{gateup,down}` → `_cpu_partials_end`
        → rejoin (:570-585 gateup, :588-599 down).
      · begin (:670-708): 입력 D2H를 main에 non_blocking으로 — gateup은 fill_x +
        fill_expert_ids (:188-190) + masking이면 fill_topk_w (:692-694), down은 fill_act
        하나 (:192-194; expert_ids는 gateup에서 이미 내려갔다). 그 다음 kt submit host
        node (:696-701) — CPU 작업은 이 노드가 실행될 때 시작한다.
      · GPU 티어 (:872-903): decode는 grouping이 None이라 곁 스트림이 없고(:851)
        순서는 HOT → WARM, 둘 다 main (:857-861).
      · end (:710-724): sync host node가 main에 있다 — 스트림 순서상 hot·warm 커널이
        끝나야 실행되고, 실행되면 CPU 완료까지 블록한다. 그 뒤 partial H2D (:717-723).
      · rejoin 커널 (:584 / :599)이 main에서 마지막.

    그래서 phase는 직렬 사슬 하나와 그 안의 겹침 구간 하나다:

        T = D2H_in + max(H + G, C) + H2D + rejoin

      G  = GPU 갈래 = hot + warm (같은 스트림이라 더한다; gpu_branch="seq"면 연속 측정값)
      C  = cold_eff = (graph 안 submit→sync, GPU 일 없음) − d2h_ids
           — CPU kt 계산 + submit/sync host node 왕복 전부 (GPU가 짧을 때 임계)
      H  = handshake_gpu = GPU가 길 때 submit·sync 노드가 GPU 갈래 위에 얹는 고정비
           (cold_sleep 곡선에서 L ≥ C 구간의 f(L) − L − d2h_ids)
      contention="graph"면 C에 `warm_interference`(graph 안에서 submit~sync 사이에 진짜 warm
           커널을 둔 것 − 같은 길이 sleep을 둔 것, 라운드 짝 중앙값)를 더한다 — 간섭을
           실행 구조 그대로 잰 값이고 host 루프 비율을 창에 얹는 근사가 필요 없다.
      contention=True면 warm이 도는 **구간에서만** CPU 진행 속도를 1/s로 늦춘다
           (s = cold_host_pcie / cold_host — warm의 pinned 읽기가 CPU DRAM 대역폭을 나눠
           쓰는 몫, 별도 측정값). 구간은 CPU 시작 기준 [hot, hot + warm) — HOT → WARM
           순서(:857-861)에서 warm이 hot 뒤에 시작하기 때문이다 (`cold_under_traffic`).
      tail=True면 H2D + rejoin 대신 합류 꼬리 실측값을 쓴다 — sync host node 뒤 첫 GPU
           노드의 전이 비용까지 담은 값. 임계 갈래에 맞춰 고른다: CPU 임계면
           `join_tail` = cold_graph_tail − cold_graph, GPU 임계면 `join_tail_gpu` =
           (긴 sleep + 꼬리) − (긴 sleep) (없으면 join_tail).
           tail=False면 단독 측정 h2d_partial + rejoin의 합 (GPU 노드끼리 이어 잰 값).

    target:
      combined  FullLayerProfiler `combined`와 같은 범위 — D2H_in = d2h_ids만
                (full-layer는 x·act·router 가중을 graph 밖에서 채운다), H2D·rejoin 없음
      layer     combined + H2D + rejoin (FullLayerProfiler `layer`)
      runtime   executor 그대로 — D2H_in = gateup: d2h_x + d2h_ids (+ d2h_topk_w),
                down: d2h_act
    """
    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r} (gateup|down)")
    if target not in EXEC_TARGETS:
        raise ValueError(f"unknown target {target!r} (known: {EXEC_TARGETS})")
    if gpu_branch not in ("sum", "seq"):
        raise ValueError(f"gpu_branch must be sum|seq, got {gpu_branch!r}")

    def get(name: str, default: Optional[float] = None) -> float:
        if name not in c or c[name] is None:
            if default is None:
                raise ValueError(f"component {name!r} missing (needed for {group}/{target})")
            return default
        return float(c[name])          # 차분 값은 음수일 수 있다 — 그대로 둔다

    hot, warm = get("hot", 0.0), get("warm", 0.0)
    G = get("gpu_seq") if gpu_branch == "seq" else hot + warm
    H = get("handshake_gpu")
    C = get("cold_eff")
    scale = 1.0
    if contention and warm > 0:
        base = get("cold_host")
        if base > 0:
            scale = get("cold_host_pcie") / base
    if contention == "graph":
        # in-graph 간섭 실측 (cold_warm_graph − cold_wsleep_graph, 라운드 짝 중앙값)
        C_used = C + max(0.0, get("warm_interference", 0.0)) if warm > 0 else C
    elif contention:
        C_used = cold_under_traffic(C, scale, start=hot, length=warm)
    else:
        C_used = C
    if target == "runtime":
        if group == "gateup":
            d2h = {"d2h_x": get("d2h_x"), "d2h_ids": get("d2h_ids")}
            if masking:
                d2h["d2h_topk_w"] = get("d2h_topk_w")
        else:
            d2h = {"d2h_act": get("d2h_act")}
    else:
        d2h = {"d2h_ids": get("d2h_ids")}
    gpu_path = H + G
    overlap = max(gpu_path, C_used)
    terms = {**d2h, "overlap": round(overlap, 3)}
    if target in ("layer", "runtime"):
        if tail:
            # 임계 갈래에 맞는 꼬리: GPU가 임계면 join_tail_gpu(있으면), 아니면 join_tail.
            if gpu_path >= C_used and c.get("join_tail_gpu") is not None:
                terms["join_tail_gpu"] = get("join_tail_gpu")
            else:
                terms["join_tail"] = get("join_tail")
        else:
            terms["h2d_partial"] = get("h2d_partial")
            terms["rejoin"] = get("rejoin")
    total = sum(terms.values())
    return {
        "us": round(total, 3), "terms": terms,
        "overlap_detail": {"gpu_path": round(gpu_path, 3), "handshake_gpu": H, "gpu": round(G, 3),
                           "hot": hot, "warm": warm, "cold_eff": C,
                           "contention_scale": round(scale, 4), "cold_used": round(C_used, 3),
                           "critical": "gpu" if gpu_path >= C_used else "cpu",
                           "slack": round(abs(gpu_path - C_used), 3)},
        "target": target, "group": group, "contention": contention, "gpu_branch": gpu_branch,
        "tail": tail,
    }


def cold_under_traffic(cold_us: float, slowdown: float, *, start: float,
                       length: float) -> float:
    """CPU 작업량 `cold_us`(단독 µs)가 [start, start+length) 구간에서만 1/slowdown 속도로
    진행할 때의 완료 시각 (순수 함수). slowdown ≤ 1이면 그대로 둔다 (가속은 모형 밖)."""
    if slowdown <= 1.0 or length <= 0 or cold_us <= start:
        return cold_us
    done = start                          # [0, start) 동안 진행량 = start
    in_traffic = length / slowdown        # 구간 전체에서의 진행량
    if cold_us - done <= in_traffic:
        return start + (cold_us - done) * slowdown
    return start + length + (cold_us - done - in_traffic)


def component_values(comp: Mapping) -> dict:
    """measure_components의 group 항목 → compose_exec 입력 {name: µs}."""
    t = {k: v["us"] for k, v in comp["components"].items()}
    t["handshake_gpu"] = comp["handshake"]["handshake_gpu_us"]
    t["cold_eff"] = comp["handshake"]["cold_eff_us"]
    for name, v in (comp.get("tails") or {}).items():
        t[name] = v["us"]
    if "join_tail" not in t and "cold_graph_tail" in t and "cold_graph" in t:
        t["join_tail"] = round(t["cold_graph_tail"] - t["cold_graph"], 3)
    return t


def exec_variants(values: Mapping[str, float], group: str, *, masking: bool = True) -> dict:
    """비교에 쓰는 합성 변형들 — 이름 → compose_exec 결과."""
    out = {}
    for target in EXEC_TARGETS:
        for cont in (False, True, "graph"):
            if cont is True and "cold_host_pcie" not in values:
                continue
            if cont == "graph" and "warm_interference" not in values:
                continue
            for tail in ((False, True) if target != "combined" else (False,)):
                if tail and "join_tail" not in values:
                    continue
                ctag = {False: "", True: "+pcie", "graph": "+gint"}[cont]
                name = f"{target}{ctag}{'+tail' if tail else ''}"
                out[name] = compose_exec(values, group=group, target=target, masking=masking,
                                         contention=cont, tail=tail)
    if "warm_packed" in values:
        # 런타임 warm-kt 경로: warm 항만 packed slab 커널 시간으로 바꾼다.
        pv = dict(values, warm=values["warm_packed"])
        for cont in (False, True, "graph"):
            if cont is True and "cold_host_pcie" not in values:
                continue
            if cont == "graph" and "warm_interference" not in values:
                continue
            for tail in (False, True):
                if tail and "join_tail" not in values:
                    continue
                ctag = {False: "", True: "+pcie", "graph": "+gint"}[cont]
                name = f"runtime+packed{ctag}{'+tail' if tail else ''}"
                r = compose_exec(pv, group=group, target="runtime", masking=masking,
                                 contention=cont, tail=tail)
                r["warm_path"] = "packed"
                out[name] = r
    return out


def residuals(pred: Mapping[str, Mapping], measured: Mapping[str, float]) -> dict:
    """target 이름이 같은 실측과 짝지어 err = pred/meas − 1 (양수 = 비관적)."""
    out = {}
    for name, p in pred.items():
        target = p["target"]
        m = measured.get(target)
        if m is None:
            continue
        out[name] = {"pred": p["us"], "meas": m, "err": closure_error(p["us"], m),
                     "resid_us": round(p["us"] - m, 3)}
    return out


# ─── 기준 그리드 채점 ───────────────────────────────────────────────────────
SCORE_PHASES = ("gateup", "down", "layer")


def point_key(d: Mapping) -> tuple:
    """채점 키 (dtype, hot_frac, warm_frac, sparsity, phase) — 비율은 4자리로 맞춘다.
    phase는 gateup | down | layer (층 전체)."""
    g = d.get("group", d.get("phase"))
    if g not in SCORE_PHASES:
        raise ValueError(f"point {dict(d)!r}: group/phase must be one of {SCORE_PHASES}")
    return (store_of(d["dtype"]).name, round(float(d["hot_frac"]), 4),
            round(float(d["warm_frac"]), 4), round(float(d["sparsity"]), 4), g)


def score(predictions: Sequence[Mapping], reference: Sequence[Mapping], *,
          held_out: Optional[Sequence[tuple]] = None) -> dict:
    """예측 점들을 기준 점들과 짝지어 채점한다 (순수 함수).

    predictions: {dtype, hot_frac, warm_frac, sparsity, group, us}
    reference:   {dtype, hot_frac, warm_frac, sparsity, group|phase, us, aa_spread?}
                 (`aa_spread` = 기준 자신의 A/A 상대 폭, 예: 0.03 = 3%)
    held_out: 튜닝에 쓰지 않은 점들의 키 — 주면 tuned/held_out을 따로 센다.

    점마다 err = pred/ref − 1. 요약: 중앙·최대 |err|, ±5%/±10% 안 비율, 기준 A/A 폭의
    중앙값과 최댓값 (달성 가능한 하한의 근거).
    """
    ref = {}
    for r in reference:
        ref[point_key(r)] = r
    rows = []
    for p in predictions:
        k = point_key(p)
        r = ref.get(k)
        if r is None:
            continue
        e = closure_error(float(p["us"]), float(r["us"]))
        aa_vals = r.get("aa_values_us") or []
        in_band = (min(aa_vals) <= float(p["us"]) <= max(aa_vals)) if len(aa_vals) > 1 else None
        rows.append({"key": k, "pred": float(p["us"]), "ref": float(r["us"]), "err": e,
                     "aa_spread": r.get("aa_spread"), "in_aa_band": in_band})
    held = set(tuple(h) for h in (held_out or ()))

    def summary(sel) -> dict:
        errs = [abs(x["err"]) for x in sel if x["err"] is not None]
        aa = [float(x["aa_spread"]) for x in sel if x.get("aa_spread") is not None]
        if not errs:
            return {"n": 0}
        s = sorted(errs)
        return {"n": len(s), "median_abs_err": round(s[len(s) // 2] if len(s) % 2
                                                      else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2, 4),
                "max_abs_err": round(s[-1], 4),
                "within_5": round(sum(e <= 0.05 for e in s) / len(s), 4),
                "within_10": round(sum(e <= 0.10 for e in s) / len(s), 4),
                "aa_median": round(sorted(aa)[len(aa) // 2], 4) if aa else None,
                "aa_max": round(max(aa), 4) if aa else None,
                "target_met": (sum(e <= 0.10 for e in s) / len(s) >= 0.9
                               and sorted(s)[len(s) // 2] <= 0.05),
                "in_aa_band": sum(1 for x in sel if x.get("in_aa_band")),
                "with_aa_band": sum(1 for x in sel if x.get("in_aa_band") is not None),
                "signed_median_err": round(sorted(x["err"] for x in sel
                                                  if x["err"] is not None)[len(errs) // 2], 4)}

    out = {"points": rows, "all": summary(rows),
           "unmatched_predictions": len(predictions) - len(rows)}
    if held:
        out["tuned"] = summary([x for x in rows if x["key"] not in held])
        out["held_out"] = summary([x for x in rows if x["key"] in held])
    out["by_phase"] = {}
    for ph in SCORE_PHASES:
        sel = [x for x in rows if x["key"][-1] == ph]
        if not sel:
            continue
        out["by_phase"][ph] = {"all": summary(sel)}
        if held:
            out["by_phase"][ph]["tuned"] = summary([x for x in sel if x["key"] not in held])
            out["by_phase"][ph]["held_out"] = summary([x for x in sel if x["key"] in held])
    return out


def compose_layer(pred: Mapping[str, Mapping], chain_us: float) -> dict:
    """층 전체 예측 = gateup 예측 + down 예측 + 경계 항 (같은 변형 이름끼리, 순수 함수).

    run_layer(executor.py:570-601)는 두 phase를 한 스트림에 잇는다: rejoin#1(:584) 직후
    down의 입력 D2H·submit host node(:588-589). phase 예측은 각 phase를 따로 잰 그래프와
    같은 범위이므로, 이어 붙일 때 생기는 차이는 `measure_chain`의 chain(= AB − A − B)으로
    따로 잰 값을 더한다. 음수일 수 있다 (경계에서 겹침이 생기면).
    """
    if "gateup" not in pred or "down" not in pred:
        raise ValueError("compose_layer needs both gateup and down predictions")
    out = {}
    for name, g in pred["gateup"].items():
        d = pred["down"].get(name)
        if d is None or not name.startswith("runtime"):
            continue
        out[name] = {"us": round(g["us"] + d["us"] + float(chain_us), 3),
                     "terms": {"gateup": g["us"], "down": d["us"], "chain": float(chain_us)}}
    return out


def compose_v3(c: Mapping[str, float], *, group: str, masking: bool = True,
               warm_key: str = "warm") -> dict:
    """진단 변형 v3 (`runtime+ginter+tail`) — v2(`runtime+pcie+tail`)의 host-루프 기반 간섭
    근사를 **phase 사슬 안에서 잰 상호 간섭** `gpu_interference`로 바꾼다 (순수 함수).

        T = runtime 사슬(입력 D2H + max(H + G, C) + join_tail)  +  gpu_interference

    gpu_interference = seq_full − seq_sleep (라운드 짝 중앙값): executor 순서의 phase 사슬에서
    GPU 갈래를 진짜 hot·warm으로 둔 것과 같은 길이 sleep으로 둔 것의 차. CPU 임계면 kt가
    GPU 트래픽에 늦어진 몫, GPU 임계면 warm이 kt 트래픽에 늦어진 몫이다 (어느 쪽이든 max의
    결과에 더해진다). 음수는 0으로 자른다.
    """
    v = dict(c)
    if warm_key != "warm":
        v["warm"] = c[warm_key]
    base = compose_exec(v, group=group, target="runtime", masking=masking, contention=False,
                        tail=True)
    gi = max(0.0, float(c.get("gpu_interference", 0.0) or 0.0))
    out = dict(base)
    out["terms"] = dict(base["terms"], gpu_interference=round(gi, 3))
    out["us"] = round(base["us"] + gi, 3)
    out["variant"] = "runtime+ginter+tail"
    return out
