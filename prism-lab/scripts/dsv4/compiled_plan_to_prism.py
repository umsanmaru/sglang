#!/usr/bin/env python
"""hetero-dl-compiler의 plan.json.gz -> prism schema v2 plan.

컴파일러 출력은 region(= layer×expert×proj)마다 segment별 tile_list다.
tier는 segment의 device/kernel/W residency로 갈린다:

    gpu + W LOCAL            -> hot     (DenseGemm)
    gpu + W PINNED           -> warm    (DenseGemmStaged)
    cpu                      -> cold    (SparseKtTileK2)

**옮기는 것은 티어 예산(행 수)이지 타일 정체가 아니다.** 컴파일러는 타일을
자기 importance 순으로 흩어 배치해서 region당 평균 598개의 런이 나오는데
(22GB plan 실측), prism에서 티어는 "어디서 계산하나"일 뿐이고 sparsity
마스크는 행의 **절대 인덱스**로 calib을 조회하므로(index.from_bands가 절대
인덱스를 유지) 어느 행이 어느 티어에 있든 수치는 같다. 흩어진 런을 그대로
옮기면 밴드가 1,975만 개가 되고 gather만 잘게 쪼개진다 — 그래서 예산은
보존하고 밴드는 hot/warm/cold 순서로 합친다. 런을 그대로 보고 싶으면
--emit-runs를 쓴다(파일 ~600 MB, 검증/대조용).

사용:
  compiled_plan_to_prism.py <plan.json.gz> <out.json> [--p 0.5] [--lambda 1.186396982627626]
                            [--calib PATH --calib-sha256 HEX | --dense] [--emit-runs]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re

EXPERT_RE = re.compile(r"^L(\d+)\.ffn_expert(\d+)\.(gate|up|down)_proj$")
DEFAULT_CALIB = "/home/um3maru/prism-sglang/assets/dsv4_flash_0731.pt"
DEFAULT_SHA = "a7f803af460ce83171d394ba48cfa0f52061f6892c17059467be1a63e62561d0"
TILE = 2  # 컴파일러 tile_size == prism ROW_GROUP/PAIR_GROUP


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


def runs_of(region: dict, name: str, K: int) -> list[tuple[int, int, str]]:
    part = region["partition"]
    if part["mode"] == "whole":
        return [(0, K, tier_of("_whole", region))]
    dist = None
    for var, d in part["input_dist"].items():
        if var.split(".")[-1].startswith("W"):
            dist = d
    if dist is None or dist["axis"] != "K":
        raise SystemExit(f"{name}: W의 K축 분할을 찾지 못했다")
    tiles: list[str | None] = [None] * (K // TILE)
    for seg, m in dist["mapping"].items():
        if m["type"] != "tile_list" or m["tile_size"] != TILE:
            raise SystemExit(f"{name}: 예상 밖 mapping {m['type']}/{m.get('tile_size')}")
        t = tier_of(seg, region)
        for i in m["indices"]:
            tiles[i] = t
    if any(x is None for x in tiles):
        raise SystemExit(f"{name}: 커버되지 않은 타일")
    runs: list[list] = []
    for i, t in enumerate(tiles):
        if runs and runs[-1][2] == t:
            runs[-1][1] = (i + 1) * TILE
        else:
            runs.append([i * TILE, (i + 1) * TILE, t])
    return [tuple(r) for r in runs]


def iter_expert_regions(path: str):
    """plans 섹션을 region 단위로 스트리밍한다 (파일은 압축 해제 시 수 GB)."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--model-id", default="DeepSeek-V4-Flash-0731")
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--intermediate-size", type=int, default=2048)
    ap.add_argument("--num-layers", type=int, default=43)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--numa-nodes", type=int, default=2)
    ap.add_argument("--gpu-kernel", default="gemv_worklist_mxfp4")
    ap.add_argument("--cpu-kernel", default="kt_tile_k2_mxfp4")
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--lam", "--lambda", dest="lam", type=float, default=1.186396982627626)
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--calib-sha256", default=DEFAULT_SHA)
    ap.add_argument("--dense", action="store_true", help="sparsity 블록 없이 (p/lambda 생략)")
    ap.add_argument("--emit-runs", action="store_true",
                    help="합치지 않고 컴파일러의 런을 그대로 밴드로 (대조용, 파일 거대)")
    ap.add_argument("--uniform-per-layer", action="store_true",
                    help="층 안의 256 expert를 같은 기하로 평준화 (P0 cold 백엔드 요구). "
                         "층별 티어 예산은 보존하고 expert별 선택만 잃는다.")
    ap.add_argument("--k-align", type=int, default=32,
                    help="평준화 시 밴드 경계 정렬 (mxfp4 pack 단위)")
    a = ap.parse_args()

    K = {"gate": a.hidden_size, "up": a.hidden_size, "down": a.intermediate_size}
    N = {"gate": a.intermediate_size, "up": a.intermediate_size, "down": a.hidden_size}

    got: dict[tuple[int, int, str], list] = {}
    for name, region in iter_expert_regions(a.src):
        L, E, proj = EXPERT_RE.match(name).groups()
        got[(int(L), int(E), proj)] = runs_of(region, name, K[proj])

    # gate/up의 "cold 몫 없음"은 짝이어야 한다 — kt가 두 proj를 한 단계에서
    # 융합하므로 한쪽만 0행이면 그 단계를 건너뛸 수도, 반쪽만 계산할 수도 없다
    # (kt validate_kindex가 거부한다). 컴파일러 탐색이 드물게 만드는 이 엇갈림은
    # **행을 버리지 않고** 남은 쪽의 cold를 hot으로 올려 해소한다: 행은 그대로
    # 있고 계산 위치만 GPU로 간다 (그 expert의 반대쪽 proj도 이미 전량 GPU다).
    reconciled = []
    for layer in range(a.num_layers):
        for expert in range(a.num_experts):
            cold = {}
            for proj in ("gate", "up"):
                rs = got.get((layer, expert, proj))
                if rs is None:
                    continue
                cold[proj] = sum(e - s for s, e, t in rs if t == "cold")
            if len(cold) != 2 or (cold["gate"] == 0) == (cold["up"] == 0):
                continue
            odd = "gate" if cold["gate"] else "up"
            got[(layer, expert, odd)] = [
                (s, e, "hot" if t == "cold" else t) for s, e, t in got[(layer, expert, odd)]
            ]
            reconciled.append((layer, expert, odd, cold[odd]))
    for layer, expert, odd, rows in reconciled:
        print(f"  [reconcile] L{layer} e{expert} {odd}: cold {rows}행 -> hot "
              f"(gate/up 짝 맞춤; 반대쪽은 cold 0행)")

    missing = [
        (l, e, p)
        for l in range(a.num_layers) for e in range(a.num_experts) for p in ("gate", "up", "down")
        if (l, e, p) not in got
    ]
    if missing:
        raise SystemExit(f"컴파일러 plan에 없는 region {len(missing)}개 (예: {missing[:3]})")

    # --uniform-per-layer: (layer, proj)마다 expert 평균으로 기하를 하나로 만든다.
    # 컴파일러는 expert 단위로 전량 hot/전량 cold를 고르는데(중앙값 hot=0, 일부는
    # K 전체가 hot), prism P0 cold 백엔드는 층 안에서 균일 기하만 받는다
    # (cold_backend._build_config). 층 예산은 보존되고 expert별 선택만 사라진다.
    uniform: dict[tuple[int, str], tuple[int, int]] = {}
    if a.uniform_per_layer:
        al = a.k_align
        for layer in range(a.num_layers):
            for proj in ("gate", "up", "down"):
                k = K[proj]
                acc = collections.Counter()
                for expert in range(a.num_experts):
                    for s_, e_, t_ in got[(layer, expert, proj)]:
                        acc[t_] += e_ - s_
                h = int(round(acc["hot"] / a.num_experts / al)) * al
                w = int(round(acc["warm"] / a.num_experts / al)) * al
                h = max(0, min(h, k - al))
                w = max(0, min(w, k - al - h))
                uniform[(layer, proj)] = (h, w)

    def proj_entry(runs, proj, layer=None):
        k, n = K[proj], N[proj]
        if a.emit_runs:
            bands = [[s, e, t] for s, e, t in runs]
        elif a.uniform_per_layer:
            h, w = uniform[(layer, proj)]
            bands, cur = [], 0
            for size, tier in ((h, "hot"), (w, "warm"), (k - h - w, "cold")):
                if size:
                    bands.append([cur, cur + size, tier])
                    cur += size
        else:
            c = collections.Counter()
            for s, e, t in runs:
                c[t] += e - s
            bands, cur = [], 0
            for tier in ("hot", "warm", "cold"):
                if c[tier]:
                    bands.append([cur, cur + c[tier], tier])
                    cur += c[tier]
            assert cur == k, (cur, k)
        entry = {"bands": bands}
        if any(b[2] == "cold" for b in bands):
            step = n // a.numa_nodes
            entry["cold_shards"] = [[i, i * step, (i + 1) * step] for i in range(a.numa_nodes)]
        if not a.dense:
            entry["p"], entry["lambda"] = a.p, a.lam
        return entry

    overrides = []
    for layer in range(a.num_layers):
        for expert in range(a.num_experts):
            ov = {"layer": layer, "expert": expert}
            for proj in ("gate", "up", "down"):
                ov[proj] = proj_entry(got[(layer, expert, proj)], proj, layer)
            overrides.append(ov)

    plan = {
        "schema_version": 2,
        "model_id": a.model_id,
        "dims": {
            "hidden_size": a.hidden_size, "intermediate_size": a.intermediate_size,
            "num_layers": a.num_layers, "num_experts": a.num_experts,
            "top_k": a.top_k, "dtype": "bfloat16",
        },
        "kernels": {"gpu_warm": a.gpu_kernel, "cpu_cold": a.cpu_kernel},
        "overrides": overrides,
        "provenance": {"source": a.src, "converter": "compiled_plan_to_prism.py",
                       "mode": "runs" if a.emit_runs else "coalesced"},
    }
    if not a.dense:
        plan["sparsity"] = {
            "score": "k2wl2",
            "calib": {"path": a.calib, "sha256": a.calib_sha256},
            "pmax": 0.9, "grid": 0.005, "ng": 201, "renorm_it": 3,
        }
    with open(a.out, "w") as f:
        json.dump(plan, f, separators=(",", ":"))

    rows = collections.Counter()
    for ov in overrides:
        for proj in ("gate", "up", "down"):
            for s, en, t in ov[proj]["bands"]:
                rows[t] += (en - s) * N[proj]
    bpp = 0.5 + 1 / 32  # mxfp4 + e8m0 scale
    print(f"{a.out}: overrides={len(overrides)}")
    for t in ("hot", "warm", "cold"):
        print(f"  {t:5s} {rows[t] * bpp / 2**30:7.2f} GiB  ({rows[t] / sum(rows.values()):5.1%})")


if __name__ == "__main__":
    main()
