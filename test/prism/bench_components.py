#!/usr/bin/env python
"""구성 요소 프로파일 + executor 의존 구조 합성 — CLI.

구현은 `profile/components.py`(측정)와 `profile/closure.py`의 `compose_exec`(합성)다.
이 파일은 argparse 껍데기다:

    # 스모크 점 (fp8)
    python test/prism/bench_components.py --device 0 --dtype fp8 \
        --hot-frac 0.25 --warm-frac 0.125 --sparsity 0.5 --out /tmp/x.json

    # 조건 요인까지 (회전 끔, GPU L2 flush 차분, cold 사이 L3 훑기)
    python test/prism/bench_components.py --dtype bf16 --factors no_rotate,flush,cpu_flush

    # kt 조건 (cold 인스턴스를 다시 굽는다 — full-layer 기준에도 함께 걸린다)
    python test/prism/bench_components.py --dtype fp8 --max-len 256 --independent-gateup

비교 대상은 `--reference full_layer`(기본: 같은 인스턴스의 FullLayerProfiler
combined/layer)이고, `--reference none`이면 조각만 잰다.
"""

from __future__ import annotations

import argparse
import time

from sglang.srt.layers.moe.prism.profile.closure import (
    component_values,
    exec_variants,
    compose_layer,
    compose_v3,
    residuals,
    runtime_path,
)
from sglang.srt.layers.moe.prism.profile.common import Shape, emit
from sglang.srt.layers.moe.prism.profile.components import (
    ComponentProfiler,
    env_info,
    full_layer_reference,
    measure_chain,
    measure_components,
)


def run(args) -> dict:
    shape = Shape(experts=args.experts, topk=args.topk, hidden=args.hidden, inter=args.inter)
    groups = tuple(g for g in args.groups.split(",") if g)
    factors = tuple(f for f in args.factors.split(",") if f)
    t0 = time.time()
    with ComponentProfiler(shape, hot_frac=args.hot_frac, warm_frac=args.warm_frac,
                           sparsity=args.sparsity, dtype=args.dtype, device=args.device,
                           threads=args.threads, numa_split=args.numa_split,
                           max_len=args.max_len,
                           independent_gateup=args.independent_gateup,
                           n_layers=args.n_layers) as prof:
        packed_err = None
        if args.dtype == "bf16" and not args.no_packed:
            try:
                prof.build_packed_warm()
            except (ValueError, RuntimeError) as e:     # 못 구우면 보고하고 row-major만
                packed_err = f"{type(e).__name__}: {e}"
        build_s = round(time.time() - t0, 1)
        ref_fn = None
        if args.reference == "full_layer":
            def ref_fn(pr, g):
                return full_layer_reference(pr, g, reps=args.reps, replays=args.replays)
        comps = measure_components(prof, groups=groups, reps=args.reps, replays=args.replays,
                                   rounds=args.rounds, factors=factors, reference=ref_fn)
        refs = {g: comps[g].pop("reference") for g in groups if "reference" in comps[g]}
        chain = None
        if set(groups) == {"gateup", "down"} and not args.no_chain:
            chain = measure_chain(prof, reps=args.reps, replays=args.replays, rounds=args.rounds)
        info = env_info(prof)
        point = dict(prof.point)
        conditions = dict(prof.conditions)
    out_groups = {}
    for g in groups:
        vals = component_values(comps[g])
        pred = exec_variants(vals, g)
        if "gpu_interference" in vals:            # 진단 v3 (closure.compose_v3)
            pred["runtime+ginter+tail"] = compose_v3(vals, group=g)
            if "warm_packed" in vals:
                pred["runtime+packed+ginter+tail"] = compose_v3(vals, group=g, warm_key="warm_packed")
        meas = refs.get(g, {}).get("us", {})
        out_groups[g] = {**comps[g], "values_us": vals, "predicted": pred,
                         "reference": refs.get(g),
                         "residuals": residuals(pred, meas)}
    try:
        rt = runtime_path(args.dtype, has_cold=True, all_proj_warm=args.warm_frac > 0)
    except Exception as e:                      # 런타임 설정 조회 실패는 보고만 한다
        rt = {"error": f"{type(e).__name__}: {e}"}
    layer = (compose_layer({g: out_groups[g]["predicted"] for g in groups}, chain["chain_us"])
             if chain is not None else None)
    return {"bench": "components", "shape": shape.as_dict(), "point": point,
            "chain": chain, "layer_predicted": layer,
            "conditions": conditions, "factors": list(factors),
            "runtime_path": rt,
            "warm_path_measured": ("row_major_pinned+packed" if args.dtype == "bf16"
                                   and not args.no_packed and packed_err is None
                                   else "row_major_pinned"),
            "packed_error": packed_err,
            # 런타임이 warm-kt인데 packed를 못 쟀으면 runtime 합성의 warm이 다른 커널이다.
            "warm_path_mismatch": bool(rt.get("warm_kt")) and (
                args.dtype != "bf16" or args.no_packed or packed_err is not None),
            "groups": out_groups, "info": info, "build_s": build_s,
            "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=768)
    p.add_argument("--hot-frac", type=float, default=0.25)
    p.add_argument("--warm-frac", type=float, default=0.125)
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--dtype", default="fp8", choices=("bf16", "mxfp4", "fp8"))
    p.add_argument("--groups", default="gateup,down")
    p.add_argument("--device", default=0)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--numa-split", type=float, default=0.5)
    p.add_argument("--max-len", type=int, default=1)
    p.add_argument("--independent-gateup", action="store_true")
    p.add_argument("--n-layers", type=int, default=1,
                   help="층 수 — iteration마다 층을 돌아가며 (캐시 조건; 런타임 기준의 n_layers와 맞춘다)")
    p.add_argument("--no-chain", action="store_true", help="phase 경계 항(chain)을 재지 않는다")
    p.add_argument("--no-packed", action="store_true", help="bf16 warm-kt packed slab을 굽지 않는다")
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--replays", type=int, default=10)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--factors", default="", help="no_rotate,flush,cpu_flush")
    p.add_argument("--reference", default="full_layer", choices=("full_layer", "none"))
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    emit(run(args), args.out, quiet=args.quiet)


if __name__ == "__main__":
    main()
