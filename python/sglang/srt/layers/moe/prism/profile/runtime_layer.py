"""runtime-exact 레이어 프로파일 — **서빙 executor 자체**를 CUDA graph로 잰 decode 시간.

다른 프로파일러(①②④, closure)는 커널을 손으로 다시 엮는다. 이 모듈은 엮지 않는다:
합성 plan·calib 자산·가중치를 파일/파라미터로 만들어 놓고, 서빙이 쓰는 **조립 코드를
그대로** 부른다 —

    SGLANG_PRISM_PLAN=<plan.json>  →  method._get_runtime()   (plan·calib 로드 + validate_static)
    PrismMoEMethod(stub, l).create_weights(...)                (포맷이 파라미터를 만든다)
    <합성 값 채우기>                                            (weight_loader 자리)
    PrismMoEMethod.process_weights_after_loading(layer)        (prepare → kt load → warm-kt → register)
    runtime.executor(device).run_layer(l, x, ids, w, activation=runtime.activation)   (apply와 같은 호출)

그래서 warm-kt on/off, gateup 융합, cold dispatch, kt max_len(=SGLANG_PRISM_MAX_TOKENS 기본
4096), CPUInfer 스레드·NUMA 같은 **결정은 전부 런타임의 기본값**이 내리고, 여기서는 그
결정을 읽어 리포트에 적기만 한다 (`decisions`).

타이밍은 sglang 그래프 러너와 같은 형태다: `n_layers`개의 서로 다른 층을 `passes`번 도는
forward를 **한 그래프**에 담고(thread_local 캡처, side stream 워밍업 2회), replay/(passes·n_layers)
= 층당 µs. pass마다(그리고 층마다) expert id가 다르다. phase 그래프(gateup만 / down만)는
executor의 phase 메서드(`_plan_flow`·`_cpu_partials_begin/end`·`_run_gateup/_run_down`·rejoin)를
run_layer와 같은 순서로 부른 것이라 행동이 바뀌지 않는다 — 단, down만의 그래프는 gateup이
staging에 내려 두던 expert_ids·topk_w D2H를 자기가 해야 하므로 그 몫(`down_prep`)을 따로
재서 뺀 값(`down_net`)도 낸다.

**한 프로세스에 한 구성** 원칙: kt CPUInfer를 한 프로세스에서 두 번 띄우면 스레드풀이
겹친다 (test_graph_decode의 기록 참조). A/A는 새 프로세스로 다시 구성해서 잰다
(bench_runtime_layer.py `--aa`).

**고정된 측정 방법 (reference, 2026-11-18 00시 UTC 확정)**:

  · n_layers=8, passes=8, pass·층마다 expert id 회전, L2 flush 없음, replays 200, 대표값은
    `layer`(층당 µs 중앙값; 서빙이 도는 것이 이 그래프다). gateup/down은 보조 지표.
  · **프로세스를 GPU-local이면서 kt 풀이 안 쓰는 코어에 묶는다**: `taskset -c 7,23 python ...`
    (이 머신: kt 워커가 CPU 0–6·8–14에 스스로 pin하고 busy-spin한다; 7/23은 node 0의 남는
    코어와 그 HT 형제). 묶지 않으면 메인 스레드·CUDA host-callback 스레드·cuda-EvtHandlr가
    kt 코어의 놀고 있는 HT 형제(schedutil에서 800 MHz)로 떠돌아, cold submit/sync host
    node의 handshake가 프로세스마다·시간마다 달라진다 (unpinned 약 11 µs 산포, 층 A/A 2–15%;
    같은 구성 안에서도 빠른/느린 모드가 0.2–2 s 단위로 바뀐다 — `timing_dict`의 slow_frac).
    묶으면 handshake가 1.5 µs 안에서 반복되고, 이 bench가 fresh 구성 간 phase 0.0–0.8%,
    whole layer 0.8–4.2%로 반복된다 (component 측 실측). node만 묶거나(membind/cpunodebind)
    양 노드의 빈 코어를 허용하는 것으로는 안정되지 않는다.
  · 리포트의 `host_state`가 스레드 배치(이름·허용 CPU·마지막 CPU)와 큰 매핑의 NUMA 배치를 남긴다.

라이브러리 규약: 잘못된 입력은 ValueError (SystemExit 금지).
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch

from sglang.srt.layers.moe.prism.profile.common import (
    PROJS,
    Shape,
    Timing,
    env_stamp,
    split_rows,
)

# dtype → plan의 GPU 커널 키 (gen_uniform_plan.py의 선택지와 같다). cold 커널은 profile
# Store의 기본값(`store_of(dtype).cpu_kernel`)을 쓴다 — FullLayerProfiler와 같은 커널.
GPU_KERNEL = {"bf16": "gemv_worklist", "fp8": "gemv_worklist_fp8",
              "mxfp4": "gemv_worklist_mxfp4"}
DEFAULT_CPU_KERNEL = {"bf16": "kt_tile_k2_bf16", "fp8": "kt_tile_k2_fp8b128",
                      "mxfp4": "kt_tile_k2_mxfp4"}
K_ALIGN = {"bf16": 2, "fp8": 128, "mxfp4": 32}
# 행 수 반올림 단위 — profile Store.rows_step()과 같은 값 (FullLayerProfiler와 같은 행 수)
ROW_STEP = {"bf16": 32, "fp8": 128, "mxfp4": 32}

# sparsity 합성 (k2wl2): e_j = a[2j]x0² + a[2j+1]x1² (+ 2c x0x1, c ≡ 0), a = wn².
# 살릴 페어는 wn = WN_KEEP, 죽일 페어는 wn = 0, thr 곡선은 grid 전체 상수 THR.
# → keep_j ⇔ (wn_j > 0) ∧ (x0² + x1²) ≥ (THR/WN_KEEP)² = 1e-16 — 실제 활성(hidden·act)에서
# x가 두 채널 모두 정확히 0인 경우만 예외다. 그래서 마스크가 라우터 가중·입력과 무관하게
# 테이블로 정해지고, 그래도 **실제 마스크 경로(커널의 thr 조회·비교)**를 탄다.
WN_KEEP = 1.0e4
THR = 1.0e-4
NG, GRID, PMAX, RENORM_IT = 201, 0.005, 0.9, 3
SP_P, SP_LAM = 0.5, 4.0
PAIR = 2
INDEX_MODES = ("scattered", "contiguous", "scattered_shared")


# ─── 점 ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RuntimePoint:
    """한 측정점. `index_mode`는 plan의 티어 멤버십 모양:

    scattered  — expert·층·proj마다 독립인 순열로 `chunk`행 단위 밴드를 흩는다
                 (컴파일 plan처럼 gate/up 인덱스가 서로 다르다 → kt dual pack).
    contiguous — gen_uniform_plan.py처럼 [hot | warm | cold] 연속 밴드, gate == up.
    scattered_shared — scattered와 같되 up이 gate의 배치를 그대로 쓴다 (gate/up 독립성만
                 바꾸는 대조군 — kt 단일 pack 경로).

    `tier_keep`은 균일 `sparsity` 대신 **티어별 keep**을 준다:
    {"gate": {"warm": kw, "cold": kc}, "up": {...}, "down": {...}} — 마스크 티어(warm·cold)
    밴드 안에서 expert마다 정확히 round(npairs·k)개 페어를 살린다. hot 밴드는 균일 경로와
    같이 전부 산다 (hot은 dense라 시간과 무관하다). `sparsity`와 함께 줄 수 없다
    (둘 다 주면 어느 쪽이 마스크를 정하는지 모호하다). 주면 plan은 sparse(schema 2)다.
    """

    shape: Shape
    hot_frac: float
    warm_frac: float
    sparsity: float = 0.0          # 0 = dense plan (schema 1)
    dtype: str = "bf16"
    cpu_kernel: Optional[str] = None
    n_layers: int = 4
    index_mode: str = "scattered"
    chunk: Optional[int] = None    # scattered 밴드 단위 (None = max(ROW_STEP, 32))
    sparsity_spread: float = 0.0
    numa_split: float = 0.5
    seed: int = 0
    tier_keep: Optional[dict] = field(default=None, hash=False)

    def __post_init__(self) -> None:
        if self.dtype not in GPU_KERNEL:
            raise ValueError(f"dtype must be one of {sorted(GPU_KERNEL)}, got {self.dtype!r}")
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1), got {self.sparsity}")
        if self.tier_keep is not None:
            if self.sparsity > 0 or self.sparsity_spread > 0:
                raise ValueError("tier_keep and sparsity/sparsity_spread are mutually exclusive")
            object.__setattr__(self, "tier_keep", normalize_tier_keep(self.tier_keep))
        if self.hot_frac < 0 or self.warm_frac < 0 or self.hot_frac + self.warm_frac > 1.0:
            raise ValueError(f"bad fractions hot {self.hot_frac} warm {self.warm_frac}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.index_mode not in INDEX_MODES:
            raise ValueError(f"index_mode must be one of {INDEX_MODES}, got {self.index_mode!r}")

    @property
    def kernel(self) -> str:
        return self.cpu_kernel or DEFAULT_CPU_KERNEL[self.dtype]

    @property
    def step(self) -> int:
        return ROW_STEP[self.dtype]

    @property
    def sparse(self) -> bool:
        """마스크가 있는 plan(schema 2)인가 — 균일 sparsity > 0 또는 티어별 keep."""
        return self.sparsity > 0 or self.tier_keep is not None

    def key(self) -> str:
        s = self.shape
        if self.tier_keep is not None:
            tk = "/".join(f"{p[0]}{v['warm']:g}-{v['cold']:g}" for p, v in self.tier_keep.items())
            mask = f"tk{tk}"
        else:
            mask = f"sp{self.sparsity:g}"
        return (f"{self.dtype}/E{s.experts}k{s.topk}H{s.hidden}I{s.inter}"
                f"/h{self.hot_frac:g}w{self.warm_frac:g}/{mask}")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = self.shape.as_dict()
        d["cpu_kernel"] = self.kernel
        return d


MASK_TIERS = ("warm", "cold")


def normalize_tier_keep(tk) -> dict:
    """티어별 keep 입력 검증 → {proj: {"warm": float, "cold": float}} (PROJS 순서).
    proj 셋이 모두 있어야 하고 값은 [0, 1]이다. 잘못되면 ValueError."""
    if not isinstance(tk, dict):
        raise ValueError(f"tier_keep must be a dict of proj -> {{warm, cold}}, got {type(tk).__name__}")
    if set(tk) != set(PROJS):
        raise ValueError(f"tier_keep needs exactly projections {list(PROJS)}, got {sorted(tk)}")
    out = {}
    for p in PROJS:
        v = tk[p]
        if not isinstance(v, dict) or set(v) != set(MASK_TIERS):
            raise ValueError(f"tier_keep[{p!r}] needs keys {list(MASK_TIERS)}, got {v!r}")
        out[p] = {}
        for t in MASK_TIERS:
            x = float(v[t])
            if not 0.0 <= x <= 1.0:
                raise ValueError(f"tier_keep[{p!r}][{t!r}] must be in [0, 1], got {x}")
            out[p][t] = x
    return out


# ─── plan 합성 ──────────────────────────────────────────────────────────────
def tier_rows(k_axis: int, hot_frac: float, warm_frac: float, step: int) -> tuple:
    """(hot, warm, cold) 행 수 — FullLayerProfiler(`split_rows`)와 같은 반올림."""
    hot = split_rows(k_axis, hot_frac, step=step)
    warm = split_rows(k_axis, warm_frac, step=step)
    if hot + warm > k_axis:
        raise ValueError(f"hot {hot} + warm {warm} > axis {k_axis} after rounding to {step}")
    return hot, warm, k_axis - hot - warm


def node_shards(n: int, nodes: int, align: int, split: float = 0.5) -> list:
    """cold N축 NUMA shard [[node, start, end], ...] — 노드 0 몫을 `split`에 가장 가까운
    `align` 배수로 (FullLayerProfiler ColdTier의 numa_split 실현과 같은 규칙)."""
    if nodes <= 1:
        return [[0, 0, n]]
    if nodes != 2:
        raise ValueError(f"only 1 or 2 NUMA nodes supported, got {nodes}")
    first = int(round(n * split / align)) * align
    first = max(align, min(n - align, first))
    if first <= 0 or first >= n or n % 32:
        raise ValueError(f"cannot split N={n} into 2 shards aligned to {align}")
    return [[0, 0, first], [1, first, n]]


def _bands_from_labels(labels: Sequence[str], unit: int) -> list:
    """행 묶음 라벨 열 → 인접 같은 티어를 합친 [start, end, tier] 밴드."""
    out = []
    for i, t in enumerate(labels):
        s, e = i * unit, (i + 1) * unit
        if out and out[-1][2] == t and out[-1][1] == s:
            out[-1][1] = e
        else:
            out.append([s, e, t])
    return out


def proj_labels(k_axis: int, rows: tuple, *, mode: str, chunk: int,
                gen: Optional[torch.Generator]) -> list:
    """K축을 `chunk`행 묶음으로 보고 묶음마다 hot/warm/cold 라벨. 행 수가 chunk 배수여야 한다."""
    hot, warm, cold = rows
    if k_axis % chunk or hot % chunk or warm % chunk:
        raise ValueError(f"tier rows {rows} / axis {k_axis} not multiples of chunk {chunk}")
    n = k_axis // chunk
    lab = ["hot"] * (hot // chunk) + ["warm"] * (warm // chunk) + ["cold"] * (cold // chunk)
    if mode in ("scattered", "scattered_shared"):
        perm = torch.randperm(n, generator=gen).tolist()
        out = [None] * n
        for i, p in enumerate(perm):
            out[p] = lab[i]
        lab = out
    return lab


def build_plan_raw(point: RuntimePoint, *, nodes: int) -> tuple:
    """(plan dict, labels, info) — labels[(layer, expert, proj)] = 묶음 라벨 리스트 (calib 합성이
    쓴다). sparse 점이면 밴드에 p/λ만 달고, `sparsity` 블록은 자산이 생긴 뒤
    `attach_sparsity`가 단다 (sha256이 자산 내용에서 나오므로)."""
    from sglang.srt.layers.moe.prism.kernels import cold_n_align

    s = point.shape
    chunk = point.chunk or max(point.step, 32)
    if chunk % K_ALIGN[point.dtype]:
        raise ValueError(f"chunk {chunk} not a multiple of {point.dtype} k_align {K_ALIGN[point.dtype]}")
    align = cold_n_align(point.kernel)
    rows = {p: tier_rows(s.k_axis(p), point.hot_frac, point.warm_frac, point.step) for p in PROJS}
    for p in PROJS:
        if any(r % chunk for r in rows[p]):
            # 반올림 단위가 chunk보다 작으면 chunk를 줄인다 (행 수는 바꾸지 않는다)
            g = chunk
            while any(r % g for r in rows[p]) or s.k_axis(p) % g:
                g //= 2
            if g < K_ALIGN[point.dtype] or g < PAIR:
                raise ValueError(f"{p}: tier rows {rows[p]} cannot be banded at alignment "
                                 f"{K_ALIGN[point.dtype]}")
            chunk = g
    has_cold = {p: rows[p][2] > 0 for p in PROJS}
    shards = {p: (node_shards(s.n_cols(p), nodes, align, point.numa_split) if has_cold[p] else [])
              for p in PROJS}
    sparse = point.sparse

    def entry(p, labels):
        e = {"bands": _bands_from_labels(labels, chunk), "cold_shards": shards[p]}
        if sparse:
            e["p"], e["lambda"] = SP_P, SP_LAM
        return e

    labels = {}
    gen = torch.Generator().manual_seed(point.seed + 1000)
    overrides = []
    for layer in range(point.n_layers):
        for ex in range(s.experts):
            ent = {}
            for p in PROJS:
                lab = proj_labels(s.k_axis(p), rows[p], mode=point.index_mode, chunk=chunk, gen=gen)
                if point.index_mode in ("contiguous", "scattered_shared") and p == "up":
                    lab = labels[(layer, ex, "gate")]
                labels[(layer, ex, p)] = lab
                ent[p] = entry(p, lab)
            if point.index_mode != "contiguous":
                overrides.append({"layer": layer, "expert": ex, **ent})
    base = {p: entry(p, labels[(0, 0, p)]) for p in PROJS}
    raw = {
        "schema_version": 2 if sparse else 1,
        "model_id": "profile/runtime_layer",
        "dims": {"hidden_size": s.hidden, "intermediate_size": s.inter,
                 "num_layers": point.n_layers, "num_experts": s.experts,
                 "top_k": s.topk, "dtype": "bfloat16"},
        "kernels": {"gpu_warm": GPU_KERNEL[point.dtype], "cpu_cold": point.kernel},
        "default": base,
    }
    if overrides:
        raw["overrides"] = overrides
    info = {"chunk": chunk, "rows": {p: list(rows[p]) for p in PROJS},
            "realized_frac": {p: {t: round(r / s.k_axis(p), 4)
                                  for t, r in zip(("hot", "warm", "cold"), rows[p])}
                              for p in PROJS},
            "cold_shards": shards, "cold_n_align": align}
    return raw, labels, info


def attach_sparsity(raw: dict, point: RuntimePoint, calib_path: str, calib_sha: str) -> dict:
    """plan에 k2wl2 sparsity 블록을 단다 (자산 경로 + 내용 sha256 — 서빙과 같은 무결성 대조)."""
    from sglang.srt.layers.moe.prism.kernels import cold_sparsity_score

    score = cold_sparsity_score(point.kernel)
    if score != "k2wl2":
        raise ValueError(f"runtime_layer synthesizes k2wl2 tables only; {point.kernel} uses {score}")
    raw["sparsity"] = {"score": "k2wl2", "calib": {"path": calib_path, "sha256": calib_sha},
                       "pmax": PMAX, "grid": GRID, "ng": NG, "renorm_it": RENORM_IT}
    return raw


def build_calib_blob(point: RuntimePoint, labels: dict, chunk: int) -> tuple:
    """k2wl2 자산 (gatedyn_calib.pt 키) + 테이블이 정하는 keep 비율.

    마스크 티어(warm·cold)의 밴드 안에서 expert마다 **정확히** round(npairs·keep)개 페어를
    살린다 (proj·층·expert마다 독립 랜덤). keep은 균일 경로면 1−s_e(expert별 spread),
    `point.tier_keep`이 있으면 그 proj·티어의 값이다 (전 expert 공통). hot 밴드의 wn은
    전부 WN_KEEP (hot은 dense라 무관).

    반환 keep은 쓴 테이블에서 센 값이다 (구성상 실현값 — 측정이 아니다; 런타임 측정은
    `RuntimeLayerProfiler.realized_keep`).
    """
    from sglang.srt.layers.moe.prism.profile.common import spread_values

    s = point.shape
    L, E = point.n_layers, s.experts
    tkeep = point.tier_keep
    sps = (None if tkeep is not None else
           spread_values(point.sparsity, E, spread=point.sparsity_spread, seed=point.seed + 44))
    key = {"gate": ("wn_g", "cg", "tg2l"), "up": ("wn_u", "cu", "tu2l"), "down": ("wn_d", "cd", "td2l")}
    blob = {}
    kept = {p: {"warm": [0, 0], "cold": [0, 0]} for p in PROJS}
    g = torch.Generator().manual_seed(point.seed + 2000)
    for p in PROJS:
        K = s.k_axis(p)
        wn = torch.zeros(L, E, K, dtype=torch.float32)
        for layer in range(L):
            for e in range(E):
                lab = labels[(layer, e, p)]
                rows = torch.arange(K).reshape(-1, chunk)
                for tier in ("hot", "warm", "cold"):
                    sel = [i for i, t in enumerate(lab) if t == tier]
                    if not sel:
                        continue
                    r = rows[sel].reshape(-1)
                    if tier == "hot":
                        wn[layer, e, r] = WN_KEEP
                        continue
                    pairs = r.reshape(-1, PAIR)[:, 0]          # 페어 첫 행
                    npairs = pairs.numel()
                    want = tkeep[p][tier] if tkeep is not None else 1.0 - sps[e]
                    keep_n = int(round(npairs * want))
                    pick = pairs[torch.randperm(npairs, generator=g)[:keep_n]]
                    wn[layer, e, pick] = WN_KEEP
                    wn[layer, e, pick + 1] = WN_KEEP
                    kept[p][tier][0] += keep_n
                    kept[p][tier][1] += npairs
        wk, ck, tk = key[p]
        blob[wk] = wn
        blob[ck] = torch.zeros(L, E, K // PAIR, dtype=torch.float32)
        blob[tk] = torch.full((L, E, NG), THR, dtype=torch.float32)
    keep = {p: {t: (round(k / n, 6) if n else None) for t, (k, n) in v.items()}
            for p, v in kept.items()}
    tot_k = sum(k for v in kept.values() for k, _ in v.values())
    tot_n = sum(n for v in kept.values() for _, n in v.values())
    keep["masked_all"] = round(tot_k / tot_n, 6) if tot_n else None
    return blob, keep


# ─── 합성 가중치 (weight_loader 자리) ──────────────────────────────────────
class _GpuMethodStub:
    """PrismMoEMethod가 감싸는 GPU method 자리. prism은 apply를 전유하므로 실제로 불리는
    것은 속성 조회(load_up_proj_weight_first, 클래스 이름 → mxfp4 명명)뿐이다."""

    load_up_proj_weight_first = False


def fill_synthetic(layer: torch.nn.Module, dtype: str, seed: int) -> None:
    """포맷이 만든 파라미터를 합성 값으로 채운다. 배율 파라미터는 create_params의 기본값
    (1.0 / E8M0 127)을 그대로 둔다 — dummy 로더와 같은 상태."""
    g = torch.Generator().manual_seed(seed)
    for name in ("w13_weight", "w2_weight"):
        p = getattr(layer, name).data
        if dtype == "bf16":
            p.copy_((torch.randn(p.shape, generator=g) / 10.0).to(torch.bfloat16))
        elif dtype == "fp8":
            p.copy_((torch.randn(p.shape, generator=g) / 4.0).to(p.dtype))
        else:   # mxfp4: nibble 코드 2개/바이트 — e2m1에는 NaN 코드가 없다
            p.copy_(torch.randint(-128, 128, p.shape, generator=g, dtype=torch.int16).to(p.dtype))


# ─── 측정 보조 ──────────────────────────────────────────────────────────────
class L2Flush:
    """L2보다 큰 버퍼를 0으로 쓰기 (full_layer.L2Flush와 같은 방식)."""

    def __init__(self, device, factor: float = 1.25):
        l2 = torch.cuda.get_device_properties(device).L2_cache_size
        self.buf = torch.empty(max(1, int(factor * l2) // 4), dtype=torch.float32, device=device)
        self.l2_mb = round(l2 / 2 ** 20, 1)
        self.mb = round(self.buf.numel() * 4 / 2 ** 20, 1)

    def __call__(self) -> None:
        self.buf.zero_()


def _sglang_capture_ctx():
    """sglang 러너의 capture-mode 플래그 (runner_utils.capture_mode). 없으면 no-op."""
    try:
        from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
        return model_capture_mode()
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def capture_graph(body: Callable[[], None], *, warmup: int = 2) -> torch.cuda.CUDAGraph:
    """side stream 워밍업 `warmup`회 → thread_local 캡처 (CudaGraphRunner와 같은 절차).
    둘 다 sglang capture-mode 구간 안에서 — 서빙에서 워밍업이 보는 것과 같은 신호."""
    with _sglang_capture_ctx():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            body()
    torch.cuda.synchronize()
    return graph


def replay_us(graph: torch.cuda.CUDAGraph, units: int, replays: int, warmup: int = 3) -> list:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(replays):
        b = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        b.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        out.append(b.elapsed_time(e) * 1e3 / units)
    return out


# 느린 모드 판정선: 최솟값 대비 이 비율을 넘는 replay를 "slow"로 센다. 같은 구성 안에서 모드가
# 수십~수백 replay 단위로 바뀐다 (진단: 빠른 ~260 / 느린 ~290 µs, fp8 4층) — 버리지 않고 비율로 남긴다.
SLOW_RATIO = 1.06


def timing_dict(samples: Sequence[float]) -> dict:
    t = Timing.of(samples)
    lo = min(samples)
    fast = [x for x in samples if x <= SLOW_RATIO * lo]
    slow = [x for x in samples if x > SLOW_RATIO * lo]
    return {"us": t.us, "min_us": t.min_us, "p90_us": t.p90_us, "max_us": t.max_us,
            "replays": t.replays, "samples_us": [round(x, 2) for x in samples],
            "slow_frac": round(len(slow) / len(samples), 3),
            "fast_median_us": round(statistics.median(fast), 3) if fast else None,
            "slow_median_us": round(statistics.median(slow), 3) if slow else None}


def diff_timing(work: Sequence[float], base: Sequence[float]) -> dict:
    """flush 차분: 중앙값끼리 뺀다 (min/p90도 같은 방식, 대표값은 중앙값의 차)."""
    a, b = Timing.of(work), Timing.of(base)
    return {"us": round(a.us - b.us, 3), "min_us": round(a.min_us - b.min_us, 3),
            "p90_us": round(a.p90_us - b.p90_us, 3), "max_us": round(a.max_us - b.max_us, 3),
            "replays": a.replays, "raw_us": a.us, "flush_us": b.us}


def source_commit(path: Optional[str] = None) -> dict:
    root = path or str(Path(__file__).resolve().parents[6])
    out = {"root": root}
    for k, cmd in (("commit", ["git", "-C", root, "rev-parse", "HEAD"]),
                   ("dirty", ["git", "-C", root, "status", "--short"])):
        try:
            out[k] = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as err:  # git 없음 등 — 기록만
            out[k] = f"unavailable: {err}"
    return out


ENV_KEYS = ("SGLANG_PRISM_PLAN", "SGLANG_PRISM_MAX_TOKENS", "SGLANG_PRISM_CPUINFER_THREADS",
            "SGLANG_PRISM_NUMA_MAP", "SGLANG_PRISM_COLD_STREAM", "SGLANG_PRISM_COLD_SPLIT",
            "SGLANG_PRISM_WARM_KT", "SGLANG_PRISM_COLD_GPU_MIN_M", "SGLANG_PRISM_GROUPED_MIN_M",
            "SGLANG_PRISM_SPLIT_STREAMS", "SGLANG_PRISM_COLD_ASYNC", "SGLANG_PRISM_COLD_HYBRID_FRAC",
            "SGLANG_PRISM_WARM_CPU_MIN_M", "SGLANG_PRISM_WARM_CPU_MAX_M", "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS")


def host_state() -> dict:
    """이 프로세스의 스레드 배치(이름·허용 CPU·마지막 CPU)와 큰 매핑의 NUMA 배치 스냅샷.
    모드(빠른/느린) 판정과 나란히 놓기 위한 기록이다 — 실패해도 측정은 계속한다."""
    out = {"threads": {}, "numa_mb_by_node": {}, "big_maps": []}
    try:
        for t in os.listdir("/proc/self/task"):
            try:
                nm = open(f"/proc/self/task/{t}/comm").read().strip()
                st = open(f"/proc/self/task/{t}/status").read()
                aff = [l.split(":")[1].strip() for l in st.splitlines() if l.startswith("Cpus_allowed_list")][0]
                cpu = int(open(f"/proc/self/task/{t}/stat").read().split(")")[-1].split()[36])
                key = nm if nm.startswith(("numa_", "cuda")) else "other"
                out["threads"].setdefault(key, []).append({"aff": aff, "cpu": cpu})
            except Exception:
                continue
        tot = {}
        for line in open("/proc/self/numa_maps"):
            parts = line.split()
            kb = 2048 if "kernelpagesize_kB=2048" in line else 4
            nodes = {int(p[1:].split("=")[0]): int(p.split("=")[1]) for p in parts
                     if p.startswith("N") and "=" in p}
            mb = sum(nodes.values()) * kb / 1024
            for n, v in nodes.items():
                tot[n] = tot.get(n, 0) + v * kb / 1024
            if mb > 100:
                out["big_maps"].append({"mb": round(mb), "policy": parts[1],
                                        "by_node_mb": {n: round(v * kb / 1024) for n, v in nodes.items()}})
        out["numa_mb_by_node"] = {n: round(v) for n, v in sorted(tot.items())}
        out["big_maps"].sort(key=lambda m: -m["mb"])
        out["big_maps"] = out["big_maps"][:12]
    except Exception as err:
        out["error"] = str(err)
    return out


# ─── 프로파일러 ─────────────────────────────────────────────────────────────
class RuntimeLayerProfiler:
    """서빙 조립 경로로 `point.n_layers`층을 구성하고 executor의 decode를 잰다.

    프로세스당 한 인스턴스 (kt CPUInfer 한 벌). `env`는 구성 전에 os.environ에 얹을
    런타임 env 노브 (예: {"SGLANG_PRISM_MAX_TOKENS": "1"}) — 기본은 아무것도 얹지 않아
    서빙 기본값이 결정한다. 얹은 env(와 SGLANG_PRISM_PLAN)는 `close()`가 이전 값으로
    되돌린다 (없던 키는 지운다).

    `workdir`(합성 plan·calib 파일 자리) 기본값 None은 새 임시 디렉터리이고 `close()`가
    통째로 지운다. 경로를 주면 그 아래에 만든 파일만 지운다 (디렉터리는 남긴다).
    """

    def __init__(self, point: RuntimePoint, *, device=0, workdir: Optional[str] = None,
                 env: Optional[dict] = None, verbose: bool = True):
        # close()가 부분 구성 상태에서도 돌 수 있게 정리 대상부터 둔다
        self._files: list = []
        self._tmpdir: Optional[str] = None
        self._saved_env: dict = {}
        self.ex = None
        try:
            self._build(point, device=device, workdir=workdir, env=env, verbose=verbose)
        except BaseException:
            self.close()
            raise

    def _set_env(self, key: str, value: str) -> None:
        """os.environ에 얹되 처음 건드리는 키의 이전 값(없으면 None)을 기억한다."""
        if key not in self._saved_env:
            self._saved_env[key] = os.environ.get(key)
        os.environ[key] = str(value)

    def _build(self, point: RuntimePoint, *, device, workdir: Optional[str],
               env: Optional[dict], verbose: bool) -> None:
        import tempfile

        from sglang.srt.layers.moe.prism import method
        from sglang.srt.layers.moe.prism.numa import numa_node_count

        self.point = point
        self.verbose = verbose
        self.device = torch.device(f"cuda:{int(device)}") if not isinstance(device, torch.device) else device
        torch.cuda.set_device(self.device)
        for k, v in (env or {}).items():
            self._set_env(k, v)
        self.env_overrides = dict(env or {})
        t0 = time.perf_counter()
        if workdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="prism_runtime_layer_")
            work = Path(self._tmpdir)
        else:
            work = Path(workdir)
            work.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha1(json.dumps(point.as_dict(), sort_keys=True).encode()).hexdigest()[:10]
        nodes = numa_node_count()
        calib_path = None
        self.calib_keep = None
        raw, labels, self.plan_info = build_plan_raw(point, nodes=nodes)
        if point.sparse:
            blob, self.calib_keep = build_calib_blob(point, labels, self.plan_info["chunk"])
            calib_file = work / f"calib_{tag}_{os.getpid()}.pt"
            self._files.append(calib_file)
            torch.save(blob, calib_file)
            del blob
            calib_path = str(calib_file)
            attach_sparsity(raw, point, calib_path,
                            hashlib.sha256(calib_file.read_bytes()).hexdigest())
        del labels
        plan_file = work / f"plan_{tag}_{os.getpid()}.json"
        self._files.insert(0, plan_file)
        plan_file.write_text(json.dumps(raw))
        self._set_env("SGLANG_PRISM_PLAN", str(plan_file))
        # 서빙과 같은 단일 진입점: plan·calib 로드 + validate_static + PrismRuntime
        method._RUNTIME = None
        self.runtime = method._get_runtime()
        t_plan = time.perf_counter()
        s = point.shape
        self.layers = []
        for l in range(point.n_layers):
            mod = torch.nn.Module()
            pm = method.PrismMoEMethod(_GpuMethodStub(), l)
            pm.create_weights(mod, s.experts, s.hidden, s.inter, torch.bfloat16)
            fill_synthetic(mod, point.dtype, seed=point.seed * 1000 + l)
            pm.process_weights_after_loading(mod)
            self.layers.append(pm)
            if verbose:
                print(f"[runtime_layer] layer {l} registered ({time.perf_counter() - t0:.1f}s)", flush=True)
        self.ex = self.runtime.executor(self.device)
        self.activation = self.runtime.activation
        self.build_s = {"plan_s": round(t_plan - t0, 2), "total_s": round(time.perf_counter() - t0, 2)}
        # 결정 기록: run_layer 안의 _plan_flow 결과를 가로채 둔다 (반환값 그대로 — 행동 불변).
        self._flows = []
        orig = self.ex._plan_flow

        def recording(*a, **kw):
            f = orig(*a, **kw)
            if self._record:
                self._flows.append(f)
            return f
        self._record = False
        self.ex._plan_flow = recording
        self.flush = L2Flush(self.device)
        self._inputs_cache = {}

    def close(self) -> None:
        """합성 파일·임시 디렉터리 삭제, 얹은 env 복원. 여러 번 불러도 된다."""
        import shutil

        for f in self._files:
            for g in (Path(f), Path(f).with_suffix(".maskstats.json")):
                try:
                    g.unlink()
                except OSError:
                    pass
        self._files = []
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._saved_env = {}
        self.ex = None
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── 입력 ─────────────────────────────────────────────────────────────
    def inputs(self, passes: int, n_layers: int, rotate: bool = True, seed: int = 1234) -> tuple:
        """pass·층마다 hidden [1, H] bf16, topk_ids [1, k] int64, topk_weights [1, k] fp32 (합 1).
        rotate=False면 모든 pass·층이 같은 expert id·가중을 쓴다."""
        key = (passes, n_layers, rotate, seed)
        if key in self._inputs_cache:
            return self._inputs_cache[key]
        s, dev = self.point.shape, self.device
        g = torch.Generator().manual_seed(seed)
        X, IDS, W = [], [], []
        fixed_ids = torch.randperm(s.experts, generator=g)[: s.topk].reshape(1, -1)
        fixed_w = torch.softmax(torch.randn(1, s.topk, generator=g), -1)
        for _ in range(passes):
            X.append([(torch.randn(1, s.hidden, generator=g)).to(torch.bfloat16).to(dev)
                      for _ in range(n_layers)])
            if rotate:
                IDS.append([torch.randperm(s.experts, generator=g)[: s.topk].reshape(1, -1).to(dev)
                            for _ in range(n_layers)])
                W.append([torch.softmax(torch.randn(1, s.topk, generator=g), -1).to(dev)
                          for _ in range(n_layers)])
            else:
                IDS.append([fixed_ids.to(dev) for _ in range(n_layers)])
                W.append([fixed_w.to(dev) for _ in range(n_layers)])
        out = (X, IDS, W)
        self._inputs_cache[key] = out
        return out

    # ── phase 본체 (run_layer와 같은 호출 순서) ───────────────────────────
    def _gateup(self, l, x, ids, w, parts_on=("cpu", "gpu")):
        from sglang.srt.layers.moe.prism.executor import GATEUP
        from sglang.srt.layers.moe.prism.rejoin import rejoin_gateup

        ex = self.ex
        d = ex._plan.dims
        flow = ex._plan_flow(l, x.shape[0], ids.shape[1], ids, w, x.device)
        ex._observe(l, GATEUP, x, ids, flow)
        cpu, gpu = "cpu" in parts_on, "gpu" in parts_on
        pend = ex._cpu_partials_begin(l, GATEUP, x, ids, flow) if cpu else None
        parts = ex._run_gateup(ex._tiers[l], x, ids, flow.topk_w32, flow.m, flow.k,
                               d.intermediate_size, flow.masking, grouping=flow.grouping,
                               cold_gpu=flow.cold_on_gpu, cold_grouping=flow.cold_grouping_gateup,
                               cold_partial=flow.hybrid, skip_warm=flow.skip_gpu_warm) if gpu else []
        c, wm = ex._cpu_partials_end(GATEUP, pend, flow, x.device) if cpu else (None, None)
        return rejoin_gateup(parts + [c, wm], d.intermediate_size, None, activation=self.activation)

    def _down_prep(self, ids, w):
        """gateup이 staging에 내려 두던 것 (expert_ids, masking이면 topk_w) — down만의 그래프용."""
        st = self.ex._res.staging
        st.fill_expert_ids(ids, non_blocking=True)
        if self.ex._sparse:
            st.fill_topk_w(w if w.dtype is torch.float32 else w.float(), non_blocking=True)

    def _down(self, l, act, ids, w, parts_on=("cpu", "gpu")):
        from sglang.srt.layers.moe.prism.executor import DOWN
        from sglang.srt.layers.moe.prism.rejoin import rejoin_down

        ex = self.ex
        d = ex._plan.dims
        flow = ex._plan_flow(l, act.shape[0], ids.shape[1], ids, w, act.device)
        ex._observe(l, DOWN, act, ids, flow)
        cpu, gpu = "cpu" in parts_on, "gpu" in parts_on
        pend = ex._cpu_partials_begin(l, DOWN, act, ids, flow) if cpu else None
        parts = ex._run_down(ex._tiers[l], act, ids, flow.topk_w32, flow.m, flow.k,
                             d.intermediate_size, d.hidden_size, flow.masking,
                             grouping=flow.grouping, cold_gpu=flow.cold_on_gpu,
                             cold_grouping=flow.cold_grouping_down, cold_partial=flow.hybrid,
                             skip_warm=flow.skip_gpu_warm) if gpu else []
        c, wm = ex._cpu_partials_end(DOWN, pend, flow, act.device) if cpu else (None, None)
        return rejoin_down(parts + [c, wm], flow.topk_w32)

    def _layer(self, l, x, ids, w):
        return self.ex.run_layer(l, x, ids, w, activation=self.activation)

    # ── 측정 ─────────────────────────────────────────────────────────────
    def measure(self, *, passes: int = 16, replays: int = 20, n_layers: Optional[int] = None,
                rotate: bool = True, flush: bool = False, phases: bool = True,
                eager: bool = True, eager_rounds: int = 10, diag: bool = False) -> dict:
        """층당 µs. `flush`면 층마다 앞에 L2 flush를 넣고 같은 수의 flush만 담은 그래프를
        빼서 차분한다. `diag`면 phase마다 CPU 몫만(`*_cpu_only`: 같은 executor 경로에서 GPU
        티어 발행만 뺀 것)과 GPU 티어만(`*_gpu_only`)도 잰다 — 겹침 진단용이며 서빙 경로가 아니다."""
        n = n_layers or self.point.n_layers
        if n > self.point.n_layers:
            raise ValueError(f"n_layers {n} > constructed {self.point.n_layers}")
        X, IDS, W = self.inputs(passes, n, rotate)
        units = passes * n
        out = {"passes": passes, "replays": replays, "n_layers": n, "rotate": rotate,
               "flush": flush, "units": units}

        def run_graph(fn) -> list:
            g = capture_graph(fn)
            try:
                return replay_us(g, units, replays)
            finally:
                del g
                torch.cuda.synchronize()

        base = None
        if flush:
            def flush_only():
                for _ in range(units):
                    self.flush()
            base = run_graph(flush_only)
            out["flush_only"] = timing_dict(base)

        def wrap(per_layer):
            def body():
                for p in range(passes):
                    for l in range(n):
                        if flush:
                            self.flush()
                        per_layer(p, l)
            return body

        def finish(samples):
            return diff_timing(samples, base) if flush else timing_dict(samples)

        self._flows.clear()
        self._record = True
        out["layer"] = finish(run_graph(wrap(lambda p, l: self._layer(l, X[p][l], IDS[p][l], W[p][l]))))
        self._record = False
        out["decisions_graph"] = self._flow_summary()
        if phases:
            out["gateup"] = finish(run_graph(wrap(
                lambda p, l: self._gateup(l, X[p][l], IDS[p][l], W[p][l]))))
            # down 입력 = 실제 rejoin#1 출력 (eager로 미리 계산)
            ACT = [[self._gateup(l, X[p][l], IDS[p][l], W[p][l]).clone() for l in range(n)]
                   for p in range(passes)]
            torch.cuda.synchronize()

            def down_full(p, l):
                self._down_prep(IDS[p][l], W[p][l])
                self._down(l, ACT[p][l], IDS[p][l], W[p][l])
            out["down"] = finish(run_graph(wrap(down_full)))
            out["down_prep"] = finish(run_graph(wrap(
                lambda p, l: self._down_prep(IDS[p][l], W[p][l]))))
            out["down_net_us"] = round(out["down"]["us"] - out["down_prep"]["us"], 3)
            if diag:
                for nm, po in (("cpu_only", ("cpu",)), ("gpu_only", ("gpu",))):
                    out[f"gateup_{nm}"] = finish(run_graph(wrap(
                        lambda p, l, po=po: self._gateup(l, X[p][l], IDS[p][l], W[p][l], po))))

                    def dn(p, l, po=po):
                        self._down_prep(IDS[p][l], W[p][l])
                        self._down(l, ACT[p][l], IDS[p][l], W[p][l], po)
                    out[f"down_{nm}"] = finish(run_graph(wrap(dn)))
                for ph in ("gateup", "down"):
                    full = out[ph]["us"]
                    c, g = out[f"{ph}_cpu_only"]["us"], out[f"{ph}_gpu_only"]["us"]
                    out[f"{ph}_overlap"] = {"full_minus_cpu_only_us": round(full - c, 3),
                                            "full_minus_max_us": round(full - max(c, g), 3),
                                            "serial_sum_minus_full_us": round(c + g - full, 3)}
            out["phase_sum_us"] = round(out["gateup"]["us"] + out["down_net_us"], 3)
            out["phase_sum_minus_layer_us"] = round(out["phase_sum_us"] - out["layer"]["us"], 3)
        if eager:
            out["eager"] = self.eager(passes=passes, n_layers=n, rotate=rotate, rounds=eager_rounds)
        return out

    def eager(self, *, passes: int = 16, n_layers: Optional[int] = None, rotate: bool = True,
              rounds: int = 10) -> dict:
        """graph 없이 같은 forward (서빙의 --disable-cuda-graph 경로) — host 루프, 라운드 끝 sync."""
        n = n_layers or self.point.n_layers
        X, IDS, W = self.inputs(passes, n, rotate)

        def one():
            for p in range(passes):
                for l in range(n):
                    self._layer(l, X[p][l], IDS[p][l], W[p][l])
        one()
        torch.cuda.synchronize()
        per = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            one()
            torch.cuda.synchronize()
            per.append((time.perf_counter() - t0) * 1e6 / (passes * n))
        self._flows.clear()
        self._record = True
        self._layer(0, X[0][0], IDS[0][0], W[0][0])
        self._record = False
        d = timing_dict(per)
        d["decisions_eager"] = self._flow_summary()
        return d

    def _flow_summary(self) -> Optional[dict]:
        if not self._flows:
            return None
        f = self._flows[-1]
        return {"graph_flow": f.graph_flow, "use_cold_stream": f.use_cold_stream,
                "cpu_dispatch": f.cpu_dispatch.value, "cpu_cold": f.cpu_cold,
                "cpu_warm": f.cpu_warm, "skip_gpu_warm": f.skip_gpu_warm,
                "cold_on_gpu": f.cold_on_gpu, "hybrid": f.hybrid, "masking": f.masking,
                "grouped": f.grouping is not None, "n_flows_recorded": len(self._flows)}

    def realized_keep(self, passes: int = 8) -> Optional[dict]:
        """런타임 mask_stats로 실제 활성에서의 keep 비율 (eager decode; 측정 전용으로 잠시 붙인다).
        mask_stats가 행을 plan 티어 라벨로 세므로 proj×티어(`by_proj_tier`)가 측정값으로 나온다."""
        if not self.point.sparse:
            return None
        from sglang.srt.layers.moe.prism.mask_stats import MaskStats

        ms = MaskStats(self.runtime.plan, self.runtime.calib,
                       str(Path(self._files[0]).with_suffix(".maskstats.json")), every=10 ** 9)
        for l in range(self.point.n_layers):
            ms.register_layer(l)
        saved = self.ex._mask_stats
        self.ex._mask_stats = ms
        try:
            X, IDS, W = self.inputs(passes, self.point.n_layers, True, seed=99)
            for p in range(passes):
                for l in range(self.point.n_layers):
                    self._layer(l, X[p][l], IDS[p][l], W[p][l])
            torch.cuda.synchronize()
            rep = ms.report()
        finally:
            self.ex._mask_stats = saved
            ms.disabled = True
            # 리포트는 위에서 받았다 — 종료 시 덤프(atexit)는 지워진 workdir에 쓰려 하므로 뗀다
            import atexit
            atexit.unregister(ms._atexit)
        tot = rep["total"]
        return {"masked_all": tot["all"]["masked"]["keep_frac"],
                "warm": tot["all"]["warm"]["keep_frac"], "cold": tot["all"]["cold"]["keep_frac"],
                "by_proj": {p: tot[p]["masked"]["keep_frac"] for p in PROJS},
                "by_proj_tier": {p: {t: tot[p][t]["keep_frac"] for t in MASK_TIERS} for p in PROJS},
                "decode_tokens": rep["decode_tokens"], "nan_slots": rep["nan_slots"],
                "disabled": rep["disabled"]}

    # ── 결정·출처 ─────────────────────────────────────────────────────────
    def decisions(self) -> dict:
        from sglang.srt.layers.moe.prism import method
        from sglang.srt.layers.moe.prism.plan import Tier

        rt = self.runtime
        cfg = method._CONFIG
        tiers = self.ex._tiers[0]
        cold = rt._cold
        d = {
            "store_format": rt.fmt.name,
            "gpu_kernel": rt.plan.kernels.gpu_warm, "cpu_kernel": rt.plan.kernels.cpu_cold,
            "gateup_impl": {t.value: type(v).__name__ for t, v in tiers.gateup.items()},
            "down_impl": {t.value: type(v).__name__ for t, v in tiers.down.items()},
            "warm_kt": bool(self.ex._layer_warm_kt.get(0, False)),
            "cold_gpu_view": bool(self.ex._layer_cold_gpu.get(0, False)),
            "cold_gpu_min_m": self.ex._cold_gpu_min_m,
            "warm_cpu_max_m": self.ex._warm_cpu_max_m,
            "grouped_min_m": self.ex._grouped_min_m,
            "cold_split": self.ex._cold_split, "cold_stream_eager": self.ex._cold_stream,
            "cold_async": self.ex._cold_async, "split_streams": self.ex._split_streams,
            "max_tokens": rt.max_tokens,
            "kt_max_len": getattr(cold, "_max_tokens", None) if cold is not None else None,
            "cpuinfer_threads": cfg.cpuinfer_threads if cold is not None else None,
            "numa_map": cfg.numa_map_raw,
            "numa_nodes": __import__("sglang.srt.layers.moe.prism.numa", fromlist=["x"]).numa_node_count(),
            "sparse": self.ex._sparse,
            "capture_mode_fn_in_capture": None,
        }
        with _sglang_capture_ctx():
            try:
                d["capture_mode_fn_in_capture"] = bool(self.ex._capture_mode_fn())
            except Exception as err:
                d["capture_mode_fn_in_capture"] = f"error: {err}"
        return d

    def provenance(self) -> dict:
        try:
            aff = sorted(os.sched_getaffinity(0))
        except Exception:
            aff = None
        try:
            import kt_kernel
            kt_path = kt_kernel.__file__
        except Exception as err:
            kt_path = f"unavailable: {err}"
        return {"env_stamp": env_stamp(self.device), "source": source_commit(),
                "kt_kernel": kt_path, "env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
                "env_overrides": self.env_overrides, "torch_threads": torch.get_num_threads(),
                "process_affinity": aff,
                "pinned_reference_condition": aff is not None and len(aff) < (os.cpu_count() or 0),
                "l2_mb": self.flush.l2_mb, "flush_mb": self.flush.mb, "build": self.build_s}

    def report(self, conditions: Sequence[dict] = ({},), *, passes: int = 16,
               replays: int = 20, keep_probe: bool = True) -> dict:
        """`conditions`의 각 dict를 measure(**cond)에 넘겨 잰다 (첫 항목이 기본 조건)."""
        res = []
        for c in conditions:
            kw = dict(passes=passes, replays=replays)
            kw.update(c)
            r = self.measure(**kw)
            r["condition"] = dict(c)
            res.append(r)
            if self.verbose:
                print(f"[runtime_layer] {c} layer {r['layer']['us']} "
                      f"gateup {r.get('gateup', {}).get('us')} down_net {r.get('down_net_us')}",
                      flush=True)
        return {
            "bench": "runtime_layer", "key": self.point.key(), "point": self.point.as_dict(),
            "plan_info": self.plan_info, "calib_keep": self.calib_keep,
            "realized_keep": self.realized_keep() if keep_probe else None,
            "host_state": host_state(),
            "decisions": self.decisions(), "results": res, "provenance": self.provenance(),
        }


def aa_summary(values: Sequence[float]) -> dict:
    """fresh 구성 반복값 → 상대 spread ((max−min)/median)와 표준편차/평균."""
    v = [float(x) for x in values]
    med = statistics.median(v)
    return {"values": v, "median": round(med, 3),
            "spread_rel": round((max(v) - min(v)) / med, 4) if med else None,
            "cv": round(statistics.pstdev(v) / statistics.mean(v), 4) if len(v) > 1 else None}
