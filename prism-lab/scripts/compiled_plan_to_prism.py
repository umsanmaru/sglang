#!/usr/bin/env python
"""hetero-dl-compiler의 plan.json.gz -> prism schema v2 plan (DSV4-Flash / GLM-5.3-Flash / Qwen3.8-Flash-Next).

컴파일러 출력은 region(= layer×expert×proj)마다 segment별 tile_list다.
tier는 segment의 device/kernel/W residency로 갈린다:

    gpu + W LOCAL            -> hot     (DenseGemm)
    gpu + W PINNED           -> warm    (DenseGemmStaged)
    cpu                      -> cold    (SparseKtTileK2 / SparseKtAmx)

모델은 plan.json.gz의 graph_ref.graph_id로 자동 판별한다 (--model로 강제 가능):

    deepseek_v4_flash   -> dsv4f    mxfp4  K-정렬 32   region L#.ffn_expert#.{gate,up,down}_proj   L0..42
    glm_5_3_flash       -> glm53f   fp8    K-정렬 128  region L#.mlp_expert#.…                     L3..44 (0..2는 dense)
    qwen3_8_flash_next  -> qwen38fn bf16   K-정렬 2    region L#.mlp_expert#.…                     L0..47

prism plan은 **전역 layer 인덱스**로 num_layers 전체를 덮어야 하므로(plan.validate_static),
컴파일러 plan에 없는 층(GLM의 dense 0~2)은 "default"(전량 hot placeholder)로 채운다.
그 층엔 prism MoE method가 붙지 않아 읽히지 않는다.

티어 단위(--block)는 기본이 포맷의 K-정렬이다(mxfp4 32 / fp8 128 / bf16 2). 컴파일러 타일이
그 배수면(2026-09-04 빌드: mxfp4 blk32, fp8 blk128, bf16 blk2) 타일 런을 그대로 밴드로 낸다 —
일치율 100%. 타일이 더 잘면(예전 DSV4 22gb plan의 2행 타일) region별 **티어 예산(행 수)을
블록 배수로 보존**하고 그 개수만큼의 블록을 타일 구성이 가장 잘 맞는 곳에 배정한다
(scipy LSA, 없으면 hot→warm 탐욕). 단순 다수결은 cold 속 점점이 박힌 warm을 삼켜 warm
예산이 사라지므로 쓰지 않는다. sparsity 마스크는 행의 절대 인덱스로 calib을 조회하므로
어느 행이 어느 티어에 있든 수치는 같다.

gate/up "cold 몫 없음"은 짝이어야 한다(kt가 두 proj를 한 단계에서 융합) — 엇갈리면 남은
쪽의 cold를 hot으로 올려 맞춘다(행은 버리지 않는다).

다른 모드:
  --coalesce           예산만 보존하고 밴드를 hot/warm/cold 셋으로 합친다.
  --uniform-per-layer  층 안 expert들을 같은 기하로 평준화 (P0 시절 요구).
  --emit-runs          컴파일러 타일 런을 정렬 검증 없이 그대로 (대조용).
  --warm-to-cold FRAC  warm 행의 FRAC을 cold로 (hot 불변, 다른 것 유지). 예: 0.5 → warm 절반, 나머지 cold.
  --whole-expert       베이스라인: 컴파일 plan과 hot **총량(바이트)** 을 맞추되 expert 단위 whole-or-nothing으로,
                       앞 layer·앞 expert부터 통째로 hot에 올린다. 남은 expert는 전부 cold (warm 없음, 기본);
                       --whole-warm keep이면 warm 총량도 같은 방식으로 이어서 채운다 (mode 이름 wholehw).

사용:
  compiled_plan_to_prism.py plans/glm53f/compiled/*.plan.json.gz            # 여러 개, 기본 출력 경로
  compiled_plan_to_prism.py <plan.json.gz> -o <out.json> [--block N] [--p 0.5] [--lambda ...] [--dense]
  기본 출력: plans/<lane>/<stem>_<mode>_<sp50|dense>.json  (stem = 파일명에서 .plan.json.gz 제거)

실행 env: numpy 필수, scipy 권장 (prism-e2e / prism-glm 둘 다 있음).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re
import sys
import time

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, "assets")
EXPERT_RE = re.compile(r"^L(\d+)\.(?:ffn|mlp)_expert(\d+)\.(gate|up|down)_proj$")
TIERS = ("hot", "warm", "cold")
TIER_ID = {t: i for i, t in enumerate(TIERS)}
PROJS = ("gate", "up", "down")

# prism 쪽 상수 (layers/prism/geometry.py, layers/prism/kernels.py, moe/prism/formats.py 와 동일해야 한다)
ROW_GROUP = 2
COL_GROUP = 32
K_ALIGN = {"bf16": 2, "mxfp4": 32, "fp8": 128}
COLD_N_ALIGN = {"kt_amx_bf16": 32, "kt_tile_k2_bf16": 32, "kt_amx_fp4": 32,
                "kt_tile_k2_mxfp4": 256, "kt_tile_k2_fp8b128": 256}
BYTES_PER_ELEM = {"bf16": 2.0, "mxfp4": 0.5 + 1.0 / 32, "fp8": 1.0 + 4.0 / (128 * 128)}

# graph_id -> 프리셋. lambda는 각 레인의 손 plan(plans/<lane>/*_sp50*.json)과 같은 값.
PRESETS = {
    "dsv4f": dict(
        graph_id="deepseek_v4_flash", model_id="DeepSeek-V4-Flash-0731", store="mxfp4",
        hidden_size=4096, intermediate_size=2048, num_layers=43, num_experts=256, top_k=6,
        gpu_kernel="gemv_worklist_mxfp4", cpu_kernel="kt_tile_k2_mxfp4",
        calib=os.path.join(ASSETS, "dsv4_flash_0731.pt"),
        calib_sha256="a7f803af460ce83171d394ba48cfa0f52061f6892c17059467be1a63e62561d0",
        lam=1.186396982627626,
    ),
    "glm53f": dict(
        graph_id="glm_5_3_flash", model_id="GLM-5.3-Flash", store="fp8",
        hidden_size=4096, intermediate_size=2048, num_layers=45, num_experts=288, top_k=8,
        gpu_kernel="gemv_worklist_fp8", cpu_kernel="kt_tile_k2_fp8b128",
        calib=os.path.join(ASSETS, "glm53_flash_calib.pt"),
        calib_sha256="aebdcfb0581de79480618ad2924c3c6b2dc5ab12235d83b2f402c4a557dc6204",
        lam=1.420281129520285,
    ),
    "qwen38fn": dict(
        graph_id="qwen3_8_flash_next", model_id="Qwen3.8-Flash-Next", store="bf16",
        hidden_size=2560, intermediate_size=640, num_layers=48, num_experts=512, top_k=10,
        gpu_kernel="gemv_worklist", cpu_kernel="kt_tile_k2_bf16",
        calib=os.path.join(ASSETS, "qwen38_next_calib.pt"),
        calib_sha256="4145ef9962ddc822d843a426208e8eb9b8f5c2c54e387bf382b4a3666ff2cbcd",
        lam=2.1387610087674003,
    ),
}
GRAPH_TO_LANE = {p["graph_id"]: lane for lane, p in PRESETS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 컴파일러 plan 읽기
# ─────────────────────────────────────────────────────────────────────────────
def read_graph_id(path: str) -> str:
    """헤더의 graph_ref.graph_id — 파일은 압축 해제 시 수 GB라 앞부분만 읽는다."""
    with gzip.open(path, "rt") as f:
        head = f.read(4096)
    m = re.search(r'"graph_id":\s*"([^"]+)"', head)
    if not m:
        raise SystemExit(f"{path}: 헤더에서 graph_ref.graph_id를 찾지 못했다")
    return m.group(1)


def iter_expert_regions(path: str):
    """plans 섹션을 expert region 단위로 스트리밍한다 (json.load는 메모리 수십 GB)."""
    buf, cur, depth, in_plans = None, None, 0, False
    with gzip.open(path, "rt") as f:
        for line in f:
            if not in_plans:
                if line.startswith('  "plans": {'):
                    in_plans = True
                continue
            if buf is None:
                m = re.match(r'    "([^"]+)": \{', line)
                if m and EXPERT_RE.match(m.group(1)):
                    cur, buf, depth = m.group(1), ["{"], 1
                elif line.startswith("  }"):
                    return
                continue
            buf.append(line)
            depth += line.count("{") + line.count("[") - line.count("}") - line.count("]")
            if depth > 0:
                continue
            yield cur, json.loads("".join(buf).rstrip().rstrip(","))
            buf = None


def tier_of(seg: str, region: dict) -> str:
    dev = region["device"][seg]
    impl = next(iter(dev["kernels"].values()))["impl"]
    if dev["device"].startswith("cpu"):
        return "cold"
    mem = None
    for name, r in region["residency"].items():
        if not name.split(".")[-1].startswith("W"):
            continue
        mem = (r["residency"] if r["mode"] == "whole" else r["parts"][seg])["location"]["memory"]
    return "warm" if (mem == "PINNED" or impl == "DenseGemmStaged") else "hot"


def runs_of(region: dict, name: str, K: int) -> tuple[list[tuple[int, int, str]], int]:
    """컴파일러 타일 런 [(start, end, tier)] — 절대 K 좌표, 오름차순, 빈틈 없음. 반환 (runs, tile_size)."""
    part = region["partition"]
    if part["mode"] == "whole":
        return [(0, K, tier_of("_whole", region))], K
    dist = None
    for var, d in part["input_dist"].items():
        if var.split(".")[-1].startswith("W"):
            dist = d
    if dist is None or dist["axis"] != "K":
        raise SystemExit(f"{name}: W의 K축 분할을 찾지 못했다")
    tile = None
    tiles: list[str | None] = []
    for seg, m in dist["mapping"].items():
        if m["type"] != "tile_list":
            raise SystemExit(f"{name}: 예상 밖 mapping {m['type']}")
        if tile is None:
            tile = int(m["tile_size"])
            if K % tile:
                raise SystemExit(f"{name}: K={K}가 tile_size={tile}의 배수가 아니다")
            tiles = [None] * (K // tile)
        elif int(m["tile_size"]) != tile:
            raise SystemExit(f"{name}: segment마다 tile_size가 다르다")
        t = tier_of(seg, region)
        for i in m["indices"]:
            tiles[i] = t
    if any(x is None for x in tiles):
        raise SystemExit(f"{name}: 커버되지 않은 타일")
    return _runs_from_labels(tiles, tile), tile


def _runs_from_labels(labels, unit: int) -> list[tuple[int, int, str]]:
    runs: list[list] = []
    for i, t in enumerate(labels):
        if runs and runs[-1][2] == t:
            runs[-1][1] = (i + 1) * unit
        else:
            runs.append([i * unit, (i + 1) * unit, t])
    return [tuple(r) for r in runs]


def _merge_adjacent(runs):
    out: list[list] = []
    for s, e, t in runs:
        if out and out[-1][2] == t and out[-1][1] == s:
            out[-1][1] = e
        else:
            out.append([s, e, t])
    return [tuple(r) for r in out]


# ─────────────────────────────────────────────────────────────────────────────
# 블록 양자화 / gate-up 짝 맞춤
# ─────────────────────────────────────────────────────────────────────────────
def quantize_blocks(runs, K: int, B: int) -> tuple[list[tuple[int, int, str]], int]:
    """타일 런 → B행 블록 라벨. 티어별 행 수는 B 배수로 반올림해 보존하고(최대잉여법),
    블록은 그 티어의 타일을 가장 많이 담은 곳에 배정한다. 반환 (runs, 일치 행 수)."""
    nb = K // B
    cnt = np.zeros((nb, len(TIERS)), dtype=np.int64)  # 블록 × 티어: 컴파일러 타일 행 수
    for s, e, t in runs:
        ti = TIER_ID[t]
        b0, b1 = s // B, (e - 1) // B
        if b0 == b1:
            cnt[b0, ti] += e - s
            continue
        cnt[b0, ti] += (b0 + 1) * B - s
        cnt[b1, ti] += e - b1 * B
        if b1 > b0 + 1:
            cnt[b0 + 1:b1, ti] += B
    rows = cnt.sum(0)
    assert rows.sum() == K, (rows, K)
    exact = rows / B
    n_blk = np.floor(exact).astype(np.int64)
    short = nb - int(n_blk.sum())
    if short:
        # 잉여가 큰 티어부터; 동률이면 hot > warm > cold (GPU 예산은 깎지 않는 쪽으로)
        order = sorted(range(len(TIERS)), key=lambda i: (-(exact[i] - n_blk[i]), i))
        for i in order[:short]:
            n_blk[i] += 1
    assert n_blk.sum() == nb

    labels = np.empty(nb, dtype=np.int64)
    if linear_sum_assignment is not None:
        slots = np.repeat(np.arange(len(TIERS)), n_blk)          # [nb] 슬롯 → 티어
        cost = -cnt[:, slots]                                    # [nb, nb]
        r, c = linear_sum_assignment(cost)
        labels[r] = slots[c]
    else:
        free = np.ones(nb, dtype=bool)
        for ti in range(len(TIERS)):                              # hot → warm → cold
            n = int(n_blk[ti])
            if n == 0:
                continue
            cand = np.flatnonzero(free)
            pick = cand[np.lexsort((cand, -cnt[cand, ti]))[:n]]
            labels[pick] = ti
            free[pick] = False
    agree = int(cnt[np.arange(nb), labels].sum())
    return _runs_from_labels([TIERS[i] for i in labels], B), agree


def demote_warm(got: dict, layers, num_experts: int, frac: float, unit: int) -> tuple[int, int, int]:
    """warm 행의 `frac`을 cold로 내린다 (hot 불변). region마다 warm 블록(unit 행)을 K 순서로 세어
    frac 비율만큼 **하나 건너 하나**를 cold로 바꾼다 (frac=0.5면 짝수 번째는 warm 유지, 홀수 번째는 cold).
    gate/up은 kt 융합 때문에 cold 유무가 짝이어야 하므로, 내린 뒤 한쪽만 cold가 생기는 expert는
    gate/up 둘 다 내리지 않는다(warm 유지). 반환 (내린 행 수, 내린 region 수, 짝 때문에 건너뛴 expert 수)."""
    moved_rows = moved_regions = skipped = 0

    def demoted(runs):
        blocks = [[b0, b0 + unit, t] for s_, e_, t in runs for b0 in range(s_, e_, unit)]
        warm = [i for i, b in enumerate(blocks) if b[2] == "warm"]
        n_demote = int(round(len(warm) * frac))
        if n_demote == 0:
            return None, 0
        # 균등 간격으로 n_demote개 선택 (frac=0.5 → 홀수 번째)
        pick = {warm[int((j + 0.5) * len(warm) / n_demote)] for j in range(n_demote)} if frac < 1 \
            else set(warm)
        for i in pick:
            blocks[i][2] = "cold"
        return _merge_adjacent([tuple(b) for b in blocks]), len(pick) * unit

    for layer in layers:
        for expert in range(num_experts):
            new = {}
            for proj in PROJS:
                new[proj] = demoted(got[(layer, expert, proj)])
            gu_cold = {}
            for proj in ("gate", "up"):
                runs = new[proj][0] or got[(layer, expert, proj)]
                gu_cold[proj] = any(t == "cold" for _, _, t in runs)
            if gu_cold["gate"] != gu_cold["up"]:
                skipped += 1
                new["gate"] = new["up"] = (None, 0)
            for proj in PROJS:
                runs, rows = new[proj]
                if runs is not None:
                    got[(layer, expert, proj)] = runs
                    moved_rows += rows
                    moved_regions += 1
    return moved_rows, moved_regions, skipped


def promote_cold(got: dict, layers, num_experts: int, frac: float, unit: int) -> tuple[int, int, int]:
    """cold 행의 `frac`을 warm으로 올린다 (hot 불변) — demote_warm의 역방향. region마다 cold 블록(unit 행)을
    K 순서로 세어 frac 비율만큼 균등 간격으로 골라 warm으로 바꾼다. gate/up은 kt 융합 때문에 cold 유무가
    짝이어야 하므로, 올린 뒤 한쪽만 cold 0행이 되는 expert는 gate/up 둘 다 올리지 않는다(cold 유지).
    반환 (올린 행 수, 올린 region 수, 짝 때문에 건너뛴 expert 수)."""
    moved_rows = moved_regions = skipped = 0

    def promoted(runs):
        blocks = [[b0, b0 + unit, t] for s_, e_, t in runs for b0 in range(s_, e_, unit)]
        cold = [i for i, b in enumerate(blocks) if b[2] == "cold"]
        n_promote = int(round(len(cold) * frac))
        if n_promote == 0:
            return None, 0
        pick = {cold[int((j + 0.5) * len(cold) / n_promote)] for j in range(n_promote)} if frac < 1 \
            else set(cold)
        for i in pick:
            blocks[i][2] = "warm"
        return _merge_adjacent([tuple(b) for b in blocks]), len(pick) * unit

    for layer in layers:
        for expert in range(num_experts):
            new = {}
            for proj in PROJS:
                new[proj] = promoted(got[(layer, expert, proj)])
            gu_cold = {}
            for proj in ("gate", "up"):
                runs = new[proj][0] or got[(layer, expert, proj)]
                gu_cold[proj] = any(t == "cold" for _, _, t in runs)
            if gu_cold["gate"] != gu_cold["up"]:
                skipped += 1
                new["gate"] = new["up"] = (None, 0)
            for proj in PROJS:
                runs, rows = new[proj]
                if runs is not None:
                    got[(layer, expert, proj)] = runs
                    moved_rows += rows
                    moved_regions += 1
    return moved_rows, moved_regions, skipped


def tier_elems(got: dict, N: dict) -> collections.Counter:
    """티어별 원소 수(행×N) — 요약 비율과 같은 단위."""
    c = collections.Counter()
    for (_, _, proj), runs in got.items():
        for s_, e_, t_ in runs:
            c[t_] += (e_ - s_) * N[proj]
    return c


def reconcile_gateup(got: dict, layers, num_experts: int) -> list:
    """gate/up의 "cold 몫 없음"은 짝이어야 한다 — kt가 두 proj를 한 단계에서 융합하므로
    한쪽만 0행이면 그 단계를 건너뛸 수도, 반쪽만 계산할 수도 없다(kt moe_base가 거부).
    엇갈림은 **행을 버리지 않고** 남은 쪽의 cold를 hot으로 올려 해소한다."""
    fixed = []
    for layer in layers:
        for expert in range(num_experts):
            cold = {}
            for proj in ("gate", "up"):
                rs = got.get((layer, expert, proj))
                if rs is None:
                    continue
                cold[proj] = sum(e - s for s, e, t in rs if t == "cold")
            if len(cold) != 2 or (cold["gate"] == 0) == (cold["up"] == 0):
                continue
            odd = "gate" if cold["gate"] else "up"
            got[(layer, expert, odd)] = _merge_adjacent(
                [(s, e, "hot" if t == "cold" else t) for s, e, t in got[(layer, expert, odd)]]
            )
            fixed.append((layer, expert, odd, cold[odd]))
    return fixed


# ─────────────────────────────────────────────────────────────────────────────
# 변환
# ─────────────────────────────────────────────────────────────────────────────
def default_out_path(src: str, lane: str, mode: str, dense: bool, p: float) -> str:
    stem = os.path.basename(src)
    for suf in (".plan.json.gz", ".json.gz", ".gz"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    tag = "dense" if dense else f"sp{int(round(p * 100)):02d}"
    return os.path.join(REPO, "plans", lane, f"{stem}_{mode}_{tag}.json")


def convert(src: str, out: str | None, a: argparse.Namespace) -> str:
    t0 = time.time()
    graph_id = read_graph_id(src)
    lane = a.model or GRAPH_TO_LANE.get(graph_id)
    if lane is None:
        raise SystemExit(f"{src}: graph_id '{graph_id}'에 대응하는 프리셋이 없다 "
                         f"(알려진: {sorted(GRAPH_TO_LANE)}); --model로 지정")
    P = PRESETS[lane]
    if a.model and P["graph_id"] != graph_id:
        print(f"  [warn] --model {lane}(graph {P['graph_id']}) 이지만 파일의 graph_id는 {graph_id}")

    store = P["store"]
    gpu_kernel = a.gpu_kernel or P["gpu_kernel"]
    cpu_kernel = a.cpu_kernel or P["cpu_kernel"]
    k_align = K_ALIGN[store]
    n_align = max(COL_GROUP, COLD_N_ALIGN.get(cpu_kernel, COL_GROUP))
    H, I = P["hidden_size"], P["intermediate_size"]
    num_layers, num_experts = P["num_layers"], P["num_experts"]
    K = {"gate": H, "up": H, "down": I}
    N = {"gate": I, "up": I, "down": H}
    lam = P["lam"] if a.lam is None else a.lam

    block_mode = not (a.coalesce or a.emit_runs or a.uniform_per_layer or a.whole_expert)
    B = a.block or k_align
    if B % k_align and not a.emit_runs:
        raise SystemExit(f"--block {B}가 {store} K-정렬 {k_align}의 배수가 아니다")
    if block_mode:
        for proj, k in K.items():
            if k % B:
                raise SystemExit(f"{proj}: K={k}가 --block {B}의 배수가 아니다")
        if linear_sum_assignment is None:
            print("  [warn] scipy 없음 — 블록 배정은 hot→warm 탐욕으로")
    for proj in PROJS:
        step = N[proj] // a.numa_nodes
        if N[proj] % a.numa_nodes or step % n_align:
            raise SystemExit(f"{proj}: N={N[proj]}를 {a.numa_nodes} 노드로 나눈 shard {step}가 "
                             f"{cpu_kernel}의 N 정렬 {n_align}에 안 맞는다")

    mode_name = ("block%d" % B if block_mode else
                 "runs" if a.emit_runs else "uniform" if a.uniform_per_layer else
                 ("wholehw" if a.whole_warm == "keep" else "whole") if a.whole_expert else "coalesced")
    if a.warm_to_cold:
        mode_name += f"_w{int(round((1 - a.warm_to_cold) * 100)):02d}"   # 남는 warm 비율
    if a.warm_target is not None:
        mode_name += f"_w{int(round(a.warm_target * 100)):02d}"          # 목표 warm 비율(전체 대비)
    elif a.cold_to_warm:
        mode_name += f"_c2w{int(round(a.cold_to_warm * 100)):02d}"       # 올린 cold 비율
    out = out or default_out_path(src, lane, mode_name, a.dense, a.p)
    print(f"== {src}\n   graph_id={graph_id} lane={lane} store={store} k_align={k_align} "
          f"block={B} mode={mode_name} -> {out}")

    got: dict[tuple[int, int, str], list] = {}
    agree_rows = collections.Counter()   # 블록 티어 == 컴파일러 타일 티어인 행 수 (proj별)
    tile_sizes = collections.Counter()
    n_regions = 0
    for name, region in iter_expert_regions(src):
        L, E, proj = EXPERT_RE.match(name).groups()
        L, E = int(L), int(E)
        if L >= num_layers or E >= num_experts:
            raise SystemExit(f"{name}: 프리셋 범위(num_layers={num_layers}, num_experts={num_experts}) 밖")
        runs, tile = runs_of(region, name, K[proj])
        tile_sizes[tile] += 1
        if block_mode:
            if tile % B == 0:
                agree = K[proj]                       # 이미 블록 정렬 — 그대로
            else:
                runs, agree = quantize_blocks(runs, K[proj], B)
            agree_rows[proj] += agree
        elif not a.emit_runs and any(s % k_align or e % k_align for s, e, _ in runs):
            # coalesce/uniform은 어차피 합치므로 정렬은 아래에서 다시 맞춘다
            pass
        got[(L, E, proj)] = runs
        n_regions += 1
        if n_regions % 20000 == 0:
            print(f"   … {n_regions} regions ({time.time() - t0:.0f}s)", flush=True)
    if not got:
        raise SystemExit(f"{src}: expert region이 하나도 없다 (이름 패턴 {EXPERT_RE.pattern})")
    print(f"   regions={n_regions} tile_size={dict(tile_sizes)} ({time.time() - t0:.0f}s)")

    moe_layers = sorted({l for l, _, _ in got})
    missing = [
        (l, e, p) for l in moe_layers for e in range(num_experts) for p in PROJS
        if (l, e, p) not in got
    ]
    if missing:
        raise SystemExit(f"컴파일러 plan에 없는 region {len(missing)}개 (예: {missing[:3]})")
    dense_layers = [l for l in range(num_layers) if l not in set(moe_layers)]
    if dense_layers:
        print(f"   MoE 층 {moe_layers[0]}..{moe_layers[-1]} ({len(moe_layers)}개); "
              f"컴파일러 plan에 없는 층 {dense_layers} -> default(전량 hot placeholder)")

    if a.warm_to_cold:
        unit = B if block_mode else k_align
        rows_, regs_, skip_ = demote_warm(got, moe_layers, num_experts, a.warm_to_cold, unit)
        print(f"   [warm->cold {a.warm_to_cold:g}] {regs_} region에서 {rows_} 행(K) 내림; "
              f"gate/up 짝 때문에 건너뛴 expert {skip_}개")

    # --warm-target / --cold-to-warm: cold 블록을 warm으로 올린다 (hot 불변). warm-target은 전체 원소 대비
    # 목표 warm 비율에서 region별 균일 승격 비율을 계산한다 (총량 기준 근사; 요약에서 실제 비율 확인).
    c2w = a.cold_to_warm
    if a.warm_target is not None:
        el = tier_elems(got, N)
        tot_el = sum(el.values())
        need = a.warm_target * tot_el - el["warm"]
        if need < 0:
            raise SystemExit(f"--warm-target {a.warm_target:g}: 현재 warm {el['warm'] / tot_el:.1%}가 이미 목표 이상 "
                             f"— --warm-to-cold를 쓸 것")
        if need > el["cold"]:
            raise SystemExit(f"--warm-target {a.warm_target:g}: cold {el['cold'] / tot_el:.1%} 전부 올려도 못 닿는다")
        c2w = need / el["cold"]
        print(f"   [warm-target {a.warm_target:g}] 현재 hot {el['hot'] / tot_el:.2%} / warm {el['warm'] / tot_el:.2%} / "
              f"cold {el['cold'] / tot_el:.2%} -> cold의 {c2w:.4f}를 warm으로")
    if c2w:
        unit = B if block_mode else k_align
        rows_, regs_, skip_ = promote_cold(got, moe_layers, num_experts, c2w, unit)
        el = tier_elems(got, N)
        tot_el = sum(el.values())
        print(f"   [cold->warm {c2w:.4f}] {regs_} region에서 {rows_} 행(K) 올림; gate/up 짝 때문에 건너뛴 expert {skip_}개"
              f" -> hot {el['hot'] / tot_el:.2%} / warm {el['warm'] / tot_el:.2%} / cold {el['cold'] / tot_el:.2%}")

    # 양자화 **뒤에** 맞춘다 — 반올림이 새 엇갈림(한쪽 cold 16행 → 0블록)을 만들 수 있다.
    for layer, expert, odd, rows in reconcile_gateup(got, moe_layers, num_experts):
        print(f"   [reconcile] L{layer} e{expert} {odd}: cold {rows}행 -> hot "
              f"(gate/up 짝 맞춤; 반대쪽은 cold 0행)")

    # --uniform-per-layer: (layer, proj)마다 expert 평균으로 기하를 하나로 만든다.
    uniform: dict[tuple[int, str], tuple[int, int]] = {}
    if a.uniform_per_layer:
        al = max(k_align, a.k_align or 0)
        for layer in moe_layers:
            for proj in PROJS:
                k = K[proj]
                acc = collections.Counter()
                for expert in range(num_experts):
                    for s_, e_, t_ in got[(layer, expert, proj)]:
                        acc[t_] += e_ - s_
                h = int(round(acc["hot"] / num_experts / al)) * al
                w = int(round(acc["warm"] / num_experts / al)) * al
                h = max(0, min(h, k - al))
                w = max(0, min(w, k - al - h))
                uniform[(layer, proj)] = (h, w)

    # --whole-expert: 티어별 총량(행×N)을 expert 단위(3·H·I)로 환산해 앞 layer·앞 expert부터 통째로 배정
    whole: dict[tuple[int, int], str] = {}
    if a.whole_expert:
        units = collections.Counter()
        for (l, e, proj), runs in got.items():
            for s_, e_, t_ in runs:
                units[t_] += (e_ - s_) * N[proj]
        per_expert = sum(K[p_] * N[p_] for p_ in PROJS)          # = 3·H·I
        n_hot = int(round(units["hot"] / per_expert))
        n_warm = int(round(units["warm"] / per_expert)) if a.whole_warm == "keep" else 0
        # 런타임은 층의 cold 유무를 **expert 0**으로 판정한다(method.py / executor.py register_layer:
        # plan.expert(layer, 0)). 경계 층(hot·cold가 섞인 층)에서 expert 0이 hot-only면 그 층의 cold
        # expert가 로드되지 않으므로, 경계 층에서는 hot을 **뒤쪽 expert 인덱스**에 몰아 expert 0을 cold로 둔다.
        order = [(l, e) for l in moe_layers for e in range(num_experts)]
        for i, key in enumerate(order):
            whole[key] = "hot" if i < n_hot else "warm" if i < n_hot + n_warm else "cold"
        for l in moe_layers:
            tiers = [whole[(l, e)] for e in range(num_experts)]
            if len(set(tiers)) > 1:
                # 층 안에서 cold를 앞으로, warm, hot을 뒤로 (개수는 그대로)
                rank = {"cold": 0, "warm": 1, "hot": 2}
                for e, t in enumerate(sorted(tiers, key=lambda t: rank[t])):
                    whole[(l, e)] = t
        def _span(n0, n1):
            if n1 <= n0:
                return "없음"
            (l0, e0), (l1, e1) = order[n0], order[n1 - 1]
            return f"L{l0} e{e0} .. L{l1} e{e1}"
        print(f"   [whole-expert] expert당 {per_expert * BYTES_PER_ELEM[store] / 2**20:.1f} MiB; "
              f"hot {units['hot'] / per_expert:.1f} -> {n_hot}개 ({_span(0, n_hot)}), "
              f"warm {units['warm'] / per_expert:.1f} -> {n_warm}개 ({_span(n_hot, n_hot + n_warm)}), "
              f"cold {len(order) - n_hot - n_warm}개")

    def coalesced_bands(runs, k):
        c = collections.Counter()
        for s, e, t in runs:
            c[t] += e - s
        # 정렬 단위로 반올림해 예산 보존 (최대잉여법)
        exact = {t: c[t] / k_align for t in TIERS}
        n = {t: int(exact[t]) for t in TIERS}
        short = k // k_align - sum(n.values())
        for t in sorted(TIERS, key=lambda t: (-(exact[t] - n[t]), TIER_ID[t]))[:short]:
            n[t] += 1
        bands, cur = [], 0
        for tier in TIERS:
            if n[tier]:
                bands.append([cur, cur + n[tier] * k_align, tier])
                cur += n[tier] * k_align
        assert cur == k, (cur, k)
        return bands

    def proj_entry(bands, proj):
        entry = {"bands": bands}
        if any(b[2] == "cold" for b in bands):
            step = N[proj] // a.numa_nodes
            entry["cold_shards"] = [[i, i * step, (i + 1) * step] for i in range(a.numa_nodes)]
        if not a.dense:
            entry["p"], entry["lambda"] = a.p, lam
        return entry

    def bands_of(runs, proj, layer, expert_=None):
        k = K[proj]
        if block_mode or a.emit_runs:
            return [[s, e, t] for s, e, t in runs]
        if a.whole_expert:
            return [[0, k, whole[(layer, expert_)]]]
        if a.uniform_per_layer:
            h, w = uniform[(layer, proj)]
            bands, cur = [], 0
            for size, tier in ((h, "hot"), (w, "warm"), (k - h - w, "cold")):
                if size:
                    bands.append([cur, cur + size, tier])
                    cur += size
            return bands
        return coalesced_bands(runs, k)

    # 런타임은 층의 cold 유무를 **expert 0**으로 판정한다 (method.py:406 load_layer 호출 여부,
    # executor.py:210 register_layer, cold_gpu.py:100). expert 0에 cold가 없으면 그 층의 다른 expert
    # cold 행은 **조용히 계산에서 빠진다** (dsv4 c22gb 09-04 e2e: L0/22/24/25 `cold=False cold_load 0.0s`).
    # 컴파일러가 expert 0을 전량 GPU로 두는 층이 실제로 있으므로(GLM exp10 L38, DSV4 exp16 L0/22/24/25),
    # 기본으로 **cold expert가 있는 proj마다** expert 0의 마지막 non-cold 블록 하나(k_align 행)를 cold로 내려 맞춘다
    # (warm 우선, 없으면 hot). 층당 3블록 — 예산 변화는 무시할 수준. --no-fix-expert0 로 끈다.
    if not a.whole_expert:
        for layer in moe_layers:
            moved = []
            for proj in PROJS:
                # PER PROJECTION, not per layer. The runtime builds a cold slab for
                # EACH of gate/up/down independently and reads its (node, n_start,
                # n_end) geometry out of expert 0's shard list for THAT projection
                # (cold_gpu.py:118, `next(s for s in ep.proj(proj).cold_shards ...)`).
                # The first version of this fix asked "does expert 0 have cold in ANY
                # projection", which is not the question: DSV4 exp16gb L18 has cold in
                # expert 0's gate and up but NOT its down, while other experts in L18
                # do have cold down rows -- so kt allocated a down slab, the plan held
                # no down shard for expert 0, and startup died on a bare StopIteration
                # out of that next() (2026-09-04 validation run, see
                # prism-eval/runs/logs/prism-dsv4-c16-blk32.server.log). The symmetry
                # the runtime needs is one projection at a time.
                e0 = any(t == "cold" for _, _, t in got[(layer, 0, proj)])
                n_cold = sum(1 for e in range(num_experts)
                             if any(t == "cold" for _, _, t in got[(layer, e, proj)]))
                if e0 or not n_cold:
                    continue
                if a.no_fix_expert0:
                    print(f"   [warn] L{layer} {proj}: cold expert {n_cold}개인데 expert 0은 cold 0행 "
                          f"— 런타임(expert 0 기준 판정)이 이 층의 cold를 로드하지 않는다 (--no-fix-expert0)")
                    continue
                runs = got[(layer, 0, proj)]
                # 마지막 블록의 티어를 cold로 (warm 블록이 있으면 그중 마지막, 없으면 마지막 hot 블록)
                blocks = []
                for s_, e_, t_ in runs:
                    for b0 in range(s_, e_, k_align):
                        blocks.append([b0, b0 + k_align, t_])
                cand = [i for i, b in enumerate(blocks) if b[2] == "warm"] or \
                       [i for i, b in enumerate(blocks) if b[2] == "hot"]
                i = cand[-1]
                moved.append(f"{proj}[{blocks[i][0]}:{blocks[i][1]}] {blocks[i][2]}->cold")
                blocks[i][2] = "cold"
                got[(layer, 0, proj)] = _merge_adjacent([tuple(b) for b in blocks])
            if moved:
                print(f"   [fix-expert0] L{layer}: expert 0에 cold 0행인 proj → " + ", ".join(moved))

    overrides = []
    for layer in moe_layers:
        for expert in range(num_experts):
            ov = {"layer": layer, "expert": expert}
            for proj in PROJS:
                ov[proj] = proj_entry(bands_of(got[(layer, expert, proj)], proj, layer, expert), proj)
            overrides.append(ov)

    plan = {
        "schema_version": 2,
        "model_id": P["model_id"],
        "dims": {
            "hidden_size": H, "intermediate_size": I,
            "num_layers": num_layers, "num_experts": num_experts,
            "top_k": P["top_k"], "dtype": "bfloat16",
        },
        "kernels": {"gpu_warm": gpu_kernel, "cpu_cold": cpu_kernel},
    }
    if dense_layers:
        # prism MoE method가 붙지 않는 층(GLM dense 0~2)용 placeholder — 읽히지 않는다.
        plan["default"] = {
            proj: proj_entry([[0, K[proj], "hot"]], proj) for proj in PROJS
        }
    plan["overrides"] = overrides
    plan["provenance"] = {
        "source": os.path.relpath(src, REPO) if src.startswith(REPO) else src,
        "graph_id": graph_id, "converter": os.path.basename(__file__),
        "mode": mode_name, "block": B, "compiler_tile": sorted(tile_sizes),
        "moe_layers": [moe_layers[0], moe_layers[-1]], "placeholder_layers": dense_layers,
    }
    if a.warm_to_cold:
        plan["provenance"]["warm_to_cold"] = a.warm_to_cold
    if c2w:
        plan["provenance"]["cold_to_warm"] = c2w
        if a.warm_target is not None:
            plan["provenance"]["warm_target"] = a.warm_target
    if a.whole_expert:
        plan["provenance"]["whole_expert"] = {
            "hot_experts": sum(t == "hot" for t in whole.values()),
            "warm_experts": sum(t == "warm" for t in whole.values()),
            "order": "layer asc, expert asc; 경계 층은 hot을 뒤쪽 expert로", "warm": a.whole_warm}
    if not a.dense:
        plan["sparsity"] = {
            "score": "k2wl2",
            "calib": {"path": P["calib"] if a.calib is None else a.calib,
                      "sha256": P["calib_sha256"] if a.calib_sha256 is None else a.calib_sha256},
            "pmax": 0.9, "grid": 0.005, "ng": 201, "renorm_it": 3,
        }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(plan, f, separators=(",", ":"))

    # 요약 (MoE 층만; placeholder 층은 제외)
    rows = collections.Counter()
    nbands = 0
    for ov in overrides:
        for proj in PROJS:
            nbands += len(ov[proj]["bands"])
            for s, en, t in ov[proj]["bands"]:
                rows[t] += (en - s) * N[proj]
    bpe = BYTES_PER_ELEM[store]
    tot = sum(rows.values())
    print(f"   -> {out} ({os.path.getsize(out) / 2**20:.1f} MB, {time.time() - t0:.0f}s)\n"
          f"   overrides={len(overrides)} bands={nbands} ({nbands / (3 * len(overrides)):.1f}/region)")
    for t in TIERS:
        print(f"   {t:5s} {rows[t] * bpe / 2**30:7.2f} GiB  ({rows[t] / tot:5.1%})")
    if block_mode:
        for proj in PROJS:
            tot_p = K[proj] * len(moe_layers) * num_experts
            print(f"   {proj:4s} 블록 티어 == 컴파일러 타일 티어: {agree_rows[proj] / tot_p:6.1%} 행")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("srcs", nargs="+", help="컴파일러 plan.json.gz (여러 개 가능)")
    ap.add_argument("-o", "--out", help="출력 json (src가 하나일 때만). 기본 plans/<lane>/<stem>_<mode>_<tag>.json")
    ap.add_argument("--model", choices=sorted(PRESETS), help="프리셋 강제 (기본: graph_id로 자동)")
    ap.add_argument("--gpu-kernel", help="kernels.gpu_warm 덮어쓰기 (기본 프리셋)")
    ap.add_argument("--cpu-kernel", help="kernels.cpu_cold 덮어쓰기 (기본 프리셋)")
    ap.add_argument("--numa-nodes", type=int, default=2)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--lam", "--lambda", dest="lam", type=float, default=None,
                    help="sparsity lambda (기본: 프리셋 = 레인 손 plan 값)")
    ap.add_argument("--calib", default=None, help="calib 경로 덮어쓰기 (기본 프리셋)")
    ap.add_argument("--calib-sha256", default=None)
    ap.add_argument("--dense", action="store_true", help="sparsity 블록 없이 (p/lambda 생략)")
    ap.add_argument("--block", type=int, default=None,
                    help="기본 모드의 티어 단위 행 수 (기본: 포맷 K-정렬 = mxfp4 32 / fp8 128 / bf16 2)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--coalesce", action="store_true",
                      help="예산만 보존하고 밴드를 hot/warm/cold 셋으로 합친다")
    mode.add_argument("--emit-runs", action="store_true",
                      help="컴파일러 타일 런을 그대로 밴드로 (정렬 검증 없음 — 대조용)")
    mode.add_argument("--whole-expert", action="store_true",
                      help="베이스라인: hot 총량을 맞춰 앞 layer·앞 expert부터 expert 통째로 hot (whole-or-nothing)")
    ap.add_argument("--whole-warm", choices=("keep", "cold"), default="cold",
                    help="--whole-expert에서 남은 expert: cold=전부 cold(기본), keep=warm 총량도 통째로 이어서 채움")
    mode.add_argument("--uniform-per-layer", action="store_true",
                      help="층 안의 expert들을 같은 기하로 평준화. 층별 티어 예산은 보존하고 expert별 선택만 잃는다.")
    ap.add_argument("--warm-to-cold", type=float, default=None, metavar="FRAC",
                    help="warm 행의 FRAC(예: 0.5)을 cold로 내린다 (hot 불변; region마다 warm 블록을 균등 간격으로 골라; "
                         "gate/up cold 짝은 유지). 출력 mode 이름에 _wNN(남는 warm %%)이 붙는다")
    ap.add_argument("--cold-to-warm", type=float, default=None, metavar="FRAC",
                    help="cold 행의 FRAC을 warm으로 올린다 (hot 불변; --warm-to-cold의 역방향; region마다 cold 블록을 "
                         "균등 간격으로 골라; gate/up cold 짝은 유지). 출력 mode 이름에 _c2wNN이 붙는다")
    ap.add_argument("--warm-target", type=float, default=None, metavar="FRAC",
                    help="전체 원소 대비 warm 목표 비율(예: 0.29). 현재 warm과의 차이만큼 cold를 warm으로 올린다 "
                         "(hot 불변). 출력 mode 이름에 _wNN(목표 warm %%)이 붙는다")
    ap.add_argument("--no-fix-expert0", action="store_true",
                    help="expert 0에 cold가 없는 층에 cold 블록을 넣는 보정을 끈다 (런타임의 expert-0 판정 그대로 노출)")
    ap.add_argument("--k-align", type=int, default=None, help="평준화 시 밴드 경계 정렬 (기본: 포맷 K-정렬)")
    a = ap.parse_args()
    if a.out and len(a.srcs) != 1:
        ap.error("-o/--out은 src가 하나일 때만")
    if a.warm_target is not None and a.cold_to_warm is not None:
        ap.error("--warm-target과 --cold-to-warm은 함께 쓸 수 없다")
    if a.warm_to_cold and (a.warm_target is not None or a.cold_to_warm is not None):
        ap.error("--warm-to-cold와 cold->warm 옵션은 함께 쓸 수 없다")
    outs = [convert(src, a.out, a) for src in a.srcs]
    print("\n".join(["== 생성:"] + outs))


if __name__ == "__main__":
    main()
