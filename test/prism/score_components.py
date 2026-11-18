#!/usr/bin/env python
"""구성 요소 합성 예측을 기준 그리드에 채점한다 — CLI (GPU 불필요).

    python test/prism/score_components.py --reference /tmp/um3maru_ref/reference_grid.json \
        --variant runtime+pcie+tail /tmp/um3maru_comp/grid_*.json \
        --held-out-every 2 --out /tmp/um3maru_comp/score.json

입력:
  예측  bench_components.py 출력 JSON 여러 개. 점마다 group별 `predicted[variant]`를 쓴다.
        bf16에서 `runtime+packed...` 변형이 있으면 `--variant`의 runtime을 그것으로 바꾼다
        (런타임 warm-kt 경로; `--no-packed-swap`으로 끈다).
  기준  {"points": [...]} 또는 [...] — 점마다 dtype, hot_frac, warm_frac, sparsity,
        group|phase, us|median_us, (aa_spread | aa_spread_rel | aa).

held-out: 기준 그리드 점에 `held_out`이 있으면 그것을 쓴다. 아니면 키 정렬 순서로
`--held-out-every`번째마다 하나를 튜닝에서 빼거나 `--tuned-keys FILE`(튜닝에 쓴 키 목록)의
나머지를 held-out으로 둔다.
"""

from __future__ import annotations

import argparse
import json

from sglang.srt.layers.moe.prism.profile.closure import point_key, score
from sglang.srt.layers.moe.prism.profile.common import emit


def load_reference(path: str) -> list:
    """평평한 점 목록, 또는 runtime_layer 기준 그리드(schema prism.runtime_layer.reference/v1:
    점마다 `phases.{gateup,down}.median_us`/`aa_spread_rel`, `held_out`)를 group별 점으로 편다."""
    raw = json.load(open(path))
    pts = raw.get("points", raw) if isinstance(raw, dict) else raw
    flat = []
    for p in pts:
        if "phases" in p:
            for g in ("gateup", "down", "layer"):
                ph = p["phases"].get(g)
                if ph is not None:
                    flat.append({k: v for k, v in p.items() if k != "phases"}
                                | {"group": g, "median_us": ph.get("median_us"),
                                   "aa_spread_rel": ph.get("aa_spread_rel"),
                                   "aa_values_us": ph.get("aa_values_us")})
        else:
            flat.append(p)
    out = []
    for p in flat:
        us = p.get("us", p.get("median_us"))
        if us is None:
            continue
        aa = p.get("aa_spread", p.get("aa_spread_rel", p.get("aa")))
        out.append({**p, "group": p.get("group", p.get("phase")), "us": float(us),
                    "aa_spread": None if aa is None else float(aa)})
    return out


def pooled_handshake(docs) -> float:
    """H(handshake_gpu)의 풀 중앙값 — 모든 파일·phase의 값에서.

    H는 점(dtype·분할·sparsity)보다 **프로세스**에 따라 흔들린다 (같은 점 fresh 프로세스
    32–53 µs, 한 프로세스 안 sleep 점끼리는 ±7 µs). 점마다 자기 프로세스의 H를 쓰면 그
    흔들림이 GPU 임계 phase 예측에 그대로 실린다. 풀 중앙값은 점과 무관한 상수 하나라
    튜닝 점에 맞춘 값이 아니다 (기준 그리드를 보지 않고 우리 측정만으로 정해진다)."""
    hs = sorted(d["groups"][g]["values_us"]["handshake_gpu"] for d in docs for g in d["groups"])
    if not hs:
        raise ValueError("no handshake values to pool")
    n = len(hs)
    return hs[n // 2] if n % 2 else (hs[n // 2 - 1] + hs[n // 2]) / 2


def repool(docs, h_us: float) -> None:
    """docs의 예측을 H = h_us로 다시 합성한다 (제자리)."""
    from sglang.srt.layers.moe.prism.profile.closure import compose_layer, exec_variants

    for d in docs:
        for g, v in d["groups"].items():
            v["values_us"] = dict(v["values_us"], handshake_gpu=h_us)
            v["predicted"] = exec_variants(v["values_us"], g)
        if d.get("chain"):
            d["layer_predicted"] = compose_layer(
                {g: d["groups"][g]["predicted"] for g in d["groups"]}, d["chain"]["chain_us"])


def load_predictions(paths, variant: str, packed_swap: bool, pool_h: bool = False) -> list:
    docs = [(path, json.load(open(path))) for path in paths]
    if pool_h:
        repool([d for _, d in docs], pooled_handshake([d for _, d in docs]))
    out = []
    for path, d in docs:
        pt = d["point"]
        for g, v in d["groups"].items():
            name = variant
            if (packed_swap and variant.startswith("runtime")
                    and variant.replace("runtime", "runtime+packed", 1) in v["predicted"]):
                name = variant.replace("runtime", "runtime+packed", 1)
            pred = v["predicted"].get(name)
            if pred is None:
                raise ValueError(f"{path} {g}: variant {name!r} missing "
                                 f"(have {sorted(v['predicted'])})")
            out.append({**pt, "group": g, "us": pred["us"], "variant": name, "file": path})
        lay = d.get("layer_predicted") or {}
        lname = variant
        if (packed_swap and variant.replace("runtime", "runtime+packed", 1) in lay):
            lname = variant.replace("runtime", "runtime+packed", 1)
        if lname in lay:
            out.append({**pt, "group": "layer", "us": lay[lname]["us"], "variant": lname,
                        "file": path})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions", nargs="+")
    p.add_argument("--reference", required=True)
    p.add_argument("--variant", default="runtime+pcie+tail")
    p.add_argument("--no-packed-swap", action="store_true")
    p.add_argument("--pool-handshake", action="store_true",
                   help="H를 점별 값 대신 전체 파일의 풀 중앙값으로 (pooled_handshake 참조)")
    p.add_argument("--held-out-every", type=int, default=0)
    p.add_argument("--tuned-keys", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    preds = load_predictions(args.predictions, args.variant, not args.no_packed_swap,
                             pool_h=args.pool_handshake)
    ref = load_reference(args.reference)
    keys = sorted({point_key(x) for x in preds})
    held = []
    ref_held = [point_key(r) for r in ref if r.get("held_out")]
    if ref_held and not (args.tuned_keys or args.held_out_every):
        held = sorted(set(ref_held))       # 기준 그리드가 정한 held-out을 그대로 쓴다
    elif args.tuned_keys:
        tuned = {tuple(k) for k in json.load(open(args.tuned_keys))}
        held = [k for k in keys if k not in tuned]
    elif args.held_out_every > 0:
        held = keys[::args.held_out_every]
    res = score(preds, ref, held_out=held)
    res["variant"] = args.variant
    res["pool_handshake"] = args.pool_handshake
    res["held_out_keys"] = [list(k) for k in held]
    emit(res, args.out)


if __name__ == "__main__":
    main()
