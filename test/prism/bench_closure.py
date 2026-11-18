#!/usr/bin/env python
"""closure 검사 — 티어별 launch 시간(컴파일러 sweep과 같은 세 호출)을 cost model 식으로
합친 예측이 full-layer 한 번 측정을 재현하는가 — CLI.

구현은 `sglang.srt.layers.moe.prism.profile.closure`에 있다. 이 파일은 argparse
껍데기고, 같은 검사를 자기 프로그램에 심으려면 API를 직접 부르면 된다:

    from sglang.srt.layers.moe.prism.profile.common import Shape
    from sglang.srt.layers.moe.prism.profile.closure import closure, closure_sweep

    closure(Shape(128, 8, 2048, 768), hot_frac=0.25, warm_frac=0.125,
            sparsity=0.5, dtype="fp8")["groups"]["down"]["verdict"]

오차 규약: err = predicted / measured − 1 (양수 = 모델 비관적). 한 점이라도
|err| > tolerance (기준 변형 `--reference`, 모델 `--model`)면 종료 코드 1이다.

    # 한 점
    python test/prism/bench_closure.py --device 0 --dtype fp8 --point 0.25,0.125,0.5

    # 여러 점, 기준을 combined로
    python test/prism/bench_closure.py --device 0 --dtype fp8 \\
        --point 0.25,0.125,0.5 --point 0.375,0.125,0.9 --reference combined --out /tmp/c.json
"""

from __future__ import annotations

import argparse
import sys

from sglang.srt.layers.moe.prism.profile.closure import (
    MODELS,
    REFERENCES,
    closure_sweep,
)
from sglang.srt.layers.moe.prism.profile.common import Shape, emit


def _point(text: str) -> tuple:
    try:
        h, w, s = (float(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--point는 hot_frac,warm_frac,sparsity 세 수다 (got {text!r})") from None
    return (h, w, s)


def _fmt(v, nd: int = 1) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def table(sweep: dict) -> str:
    """점 × group 한 줄씩: 티어 µs, 예측(선택 모델), 실측, 오차, 판정."""
    s = sweep["summary"]
    model, ref = s["model"], s["reference"]
    head = (f"{'hot':>5} {'warm':>5} {'sp':>4} {'group':>6} | {'t_hot':>6} {'t_warm':>6} "
            f"{'t_cold':>7} | {'pred':>7} {'comb':>7} {'layer':>7} | "
            f"{'err_c':>6} {'err_l':>6} | ok")
    lines = [f"model={model} reference={ref} tolerance={s['tolerance']}", head,
             "-" * len(head)]
    for rep in sweep["reports"]:
        p = rep.get("point", {})
        tag = f"{p.get('hot_frac', 0):>5.3f} {p.get('warm_frac', 0):>5.3f} {p.get('sparsity', 0):>4.2f}"
        if "error" in rep:
            lines.append(f"{tag} {'-':>6} | ERROR {rep['error']}")
            continue
        for g, v in rep["groups"].items():
            t, m, e = v["tiers_us"], v["measured_us"], v["err"][model]
            lines.append(
                f"{tag} {g:>6} | {_fmt(t['hot']):>6} {_fmt(t['warm']):>6} {_fmt(t['cold']):>7} | "
                f"{_fmt(v['predicted_us'][model]):>7} {_fmt(m.get('combined')):>7} "
                f"{_fmt(m.get('layer')):>7} | {_fmt(e.get('combined'), 3):>6} "
                f"{_fmt(e.get('layer'), 3):>6} | {'PASS' if v['verdict']['pass'] else 'FAIL'}")
    lines.append(f"points {s['points']}  failing {s['failing']}  errors {s['errors']}  "
                 f"worst_err {_fmt(s['worst_err'], 4)}")
    for rep in sweep["reports"]:
        for w in rep.get("warnings", ()):
            lines.append(f"warning: {w}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experts", type=int, default=128, help="E (num_experts)")
    p.add_argument("--topk", type=int, default=8, help="top_k (<= 16)")
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=768)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "mxfp4", "fp8", "fp8pt"))
    p.add_argument("--cpu-kernel", default=None,
                   help="kt 커널 키 (기본: dtype의 것)")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--threads", type=int, default=None,
                   help="CPUInfer 스레드 (기본: default_cpuinfer_threads) — 세 CPU 측정 공통")
    p.add_argument("--point", type=_point, action="append", default=None,
                   metavar="HOT,WARM,SP",
                   help="hot_frac,warm_frac,sparsity (반복 가능; 기본 0.25,0.125,0.5)")
    p.add_argument("--groups", default="gateup,down")
    p.add_argument("--tolerance", type=float, default=0.10)
    p.add_argument("--reference", default="layer", choices=REFERENCES)
    p.add_argument("--model", default="handshake", choices=MODELS)
    p.add_argument("--handshake-us", type=float, default=None,
                   help="handshake 모델의 cold_handshake (기본 2.5 µs, 컴파일러 기본값)")
    p.add_argument("--join-us", type=float, default=None,
                   help="handshake 모델의 tier_join (기본 5.25 µs)")
    p.add_argument("--contention", type=float, default=None,
                   help="handshake 모델의 tier_contention (기본 1.0)")
    p.add_argument("--tier-reps", type=int, default=200)
    p.add_argument("--tier-replays", type=int, default=40)
    p.add_argument("--cold-iters", type=int, default=100)
    p.add_argument("--cold-replays", type=int, default=8)
    p.add_argument("--full-reps", type=int, default=50)
    p.add_argument("--full-replays", type=int, default=10)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--no-flush", action="store_true",
                   help="full-layer의 hot/warm 단독 변형 L2 flush를 끈다")
    p.add_argument("--numa-split", type=float, default=0.5)
    p.add_argument("--sparsity-spread", type=float, default=0.0,
                   help="full-layer의 expert별 sparsity 편차 (기본 0 = 티어 호출과 같은 균일)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="JSON 리포트 경로")
    a = p.parse_args()

    consts = {k: v for k, v in (("cold_handshake_us", a.handshake_us),
                                ("tier_join_us", a.join_us),
                                ("tier_contention", a.contention)) if v is not None}
    try:
        shape = Shape(experts=a.experts, topk=a.topk, hidden=a.hidden, inter=a.inter)
        sweep = closure_sweep(
            shape, a.point or [(0.25, 0.125, 0.5)],
            dtype=a.dtype, cpu_kernel=a.cpu_kernel, device=a.device, threads=a.threads,
            groups=[s.strip() for s in a.groups.split(",") if s.strip()],
            tolerance=a.tolerance, reference=a.reference, model=a.model,
            tier_reps=a.tier_reps, tier_replays=a.tier_replays,
            cold_iters=a.cold_iters, cold_replays=a.cold_replays,
            full_reps=a.full_reps, full_replays=a.full_replays, rounds=a.rounds,
            flush=not a.no_flush, numa_split=a.numa_split,
            sparsity_spread=a.sparsity_spread, seed=a.seed, **consts)
    except ValueError as e:
        raise SystemExit(str(e))
    emit(sweep, a.out, quiet=bool(a.out))
    print(table(sweep), flush=True)
    if a.out:
        print(f"\n-> {a.out}", flush=True)
    if not sweep["summary"]["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
