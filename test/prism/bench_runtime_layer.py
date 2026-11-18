#!/usr/bin/env python
"""runtime-exact 레이어 decode 시간 — CLI.

구현은 `sglang.srt.layers.moe.prism.profile.runtime_layer`에 있다. 서빙 조립 경로
(method._get_runtime → PrismMoEMethod.create_weights/process_weights_after_loading →
executor.run_layer)를 합성 plan·가중치로 그대로 태우고, n_layers개 층을 한 CUDA graph에
담아 층당 µs를 잰다. 한 번 실행 = 한 구성(kt CPUInfer 한 벌)이므로 A/A는 이 CLI를 새
프로세스로 다시 부르면 된다.

    # 기본 조건 한 점
    python test/prism/bench_runtime_layer.py --dtype fp8 --hot-frac 0.25 --warm-frac 0.125 \
        --sparsity 0.5 --n-layers 4 --out /tmp/x.json

    # 캐시 상태 요인 (n_layers × rotation × flush)
    python test/prism/bench_runtime_layer.py --dtype fp8 --n-layers 8 --factorial

    # 고정 reference 조건: GPU-local이면서 kt 풀이 쓰지 않는 코어에 프로세스를 묶는다
    # (이 머신: kt 워커 = CPU 0–6·8–14 busy-spin → 7과 그 HT 형제 23). 묶지 않으면 host-callback
    # 스레드가 떠돌아 handshake가 프로세스·시간마다 달라진다 (층 A/A 2–15% → 묶으면 0.8–4.2%).
    taskset -c 7,23 python test/prism/bench_runtime_layer.py --dtype fp8 --n-layers 8 --passes 8 \
        --replays 200 --out /tmp/x.json

    # 런타임 env 노브 하나 바꾸기 (예: kt max_len)
    python test/prism/bench_runtime_layer.py --dtype fp8 --env SGLANG_PRISM_MAX_TOKENS=1
"""

from __future__ import annotations

import argparse
import json

from sglang.srt.layers.moe.prism.profile.common import Shape, emit
from sglang.srt.layers.moe.prism.profile.runtime_layer import RuntimeLayerProfiler, RuntimePoint


def factorial_conditions(n_max: int) -> list:
    """n_layers ∈ {1,2,4,8}∩[1,n_max] × rotation × flush. 첫 항목이 기본 조건 (n_max, 회전, flush 없음)."""
    ns = [n for n in (1, 2, 4, 8) if n <= n_max]
    conds = [{"n_layers": n_max, "rotate": True, "flush": False}]
    for n in ns:
        for rot in (True, False):
            for fl in (False, True):
                c = {"n_layers": n, "rotate": rot, "flush": fl}
                if c not in conds:
                    conds.append(c)
    # 요인 스윕에서는 phase·eager를 기본 조건에만 잰다 (시간 절약)
    return [dict(c, phases=(i == 0), eager=(i == 0)) for i, c in enumerate(conds)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=768)
    p.add_argument("--hot-frac", type=float, default=0.25)
    p.add_argument("--warm-frac", type=float, default=0.125)
    p.add_argument("--sparsity", type=float, default=0.5, help="0 = dense plan")
    p.add_argument("--sparsity-spread", type=float, default=0.0)
    p.add_argument("--tier-keep", default=None, metavar="JSON",
                   help='티어별 keep (주면 --sparsity는 무시된다): '
                        '\'{"gate":{"warm":0.6,"cold":0.4},"up":{...},"down":{...}}\'')
    p.add_argument("--dtype", default="fp8", choices=("bf16", "fp8", "mxfp4"))
    p.add_argument("--cpu-kernel", default=None)
    p.add_argument("--n-layers", type=int, default=4, help="구성할 층 수 (서로 다른 가중치)")
    p.add_argument("--index-mode", default="scattered", choices=("scattered", "contiguous", "scattered_shared"))
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--numa-split", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--passes", type=int, default=16)
    p.add_argument("--replays", type=int, default=20,
                   help="reference 조건은 200 (모드 전환을 덮도록), n-layers 8, passes 8, "
                        "taskset -c 7,23 (docstring 참조)")
    p.add_argument("--no-phases", action="store_true")
    p.add_argument("--no-eager", action="store_true")
    p.add_argument("--flush", action="store_true", help="기본 조건에 층마다 L2 flush (차분)")
    p.add_argument("--no-rotate", action="store_true")
    p.add_argument("--factorial", action="store_true",
                   help="n_layers {1,2,4,8} × rotation × flush 요인 스윕")
    p.add_argument("--diag", action="store_true",
                   help="phase별 CPU 몫만 / GPU 티어만 그래프도 잰다 (겹침 진단)")
    p.add_argument("--no-keep-probe", action="store_true", help="mask_stats keep 실측 생략")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                   help="구성 전에 얹을 런타임 env (여러 번)")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--workdir", default=None,
                   help="합성 plan·calib 자리 (기본: 새 임시 디렉터리, 끝나면 지운다)")
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    env = {}
    for kv in a.env:
        if "=" not in kv:
            raise SystemExit(f"--env expects KEY=VAL, got {kv!r}")
        k, v = kv.split("=", 1)
        env[k] = v
    tier_keep = None
    if a.tier_keep:
        try:
            tier_keep = json.loads(a.tier_keep)
        except json.JSONDecodeError as err:
            raise SystemExit(f"--tier-keep is not JSON: {err}") from None
    try:
        point = RuntimePoint(
            shape=Shape(a.experts, a.topk, a.hidden, a.inter), hot_frac=a.hot_frac,
            warm_frac=a.warm_frac, sparsity=0.0 if tier_keep is not None else a.sparsity, dtype=a.dtype, cpu_kernel=a.cpu_kernel,
            n_layers=a.n_layers, index_mode=a.index_mode, chunk=a.chunk,
            sparsity_spread=a.sparsity_spread, numa_split=a.numa_split, seed=a.seed,
            tier_keep=tier_keep)
    except ValueError as err:
        raise SystemExit(str(err)) from None
    if a.factorial:
        conds = factorial_conditions(a.n_layers)
    else:
        conds = [{"n_layers": a.n_layers, "rotate": not a.no_rotate, "flush": a.flush,
                  "phases": not a.no_phases, "eager": not a.no_eager,
                  "diag": a.diag}]
    with RuntimeLayerProfiler(point, device=a.device, workdir=a.workdir, env=env,
                              verbose=not a.quiet) as prof:
        rep = prof.report(conds, passes=a.passes, replays=a.replays,
                          keep_probe=not a.no_keep_probe)
    emit(rep, a.out, quiet=a.quiet)


if __name__ == "__main__":
    main()
