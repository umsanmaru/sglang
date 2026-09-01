#!/usr/bin/env python
"""full-layer decode 시간 (프로파일 ④) + planner 근사와의 대조 — CLI.

구현은 `sglang.srt.layers.moe.prism.profile.full_layer`에 있다. 이 파일은 argparse
껍데기고, 같은 측정을 자기 프로그램에 심으려면 API를 직접 부르면 된다:

    from sglang.srt.layers.moe.prism.profile import Shape, FullLayerProfiler, compare

    with FullLayerProfiler(Shape(128, 8, 2048, 768), hot_frac=0.375,
                           warm_frac=0.125, sparsity=0.5, dtype="fp8") as p:
        p.measure("gateup").us("combined")

    compare(Shape(128, 8, 2048, 768), anchors=(1, 8), dtype="fp8")   # 모델 대조

프로파일 ①②와 갈리는 점: 세 티어가 **한 iteration 안에서 같이** 돌고, 티어 경계와
sparsity가 **expert마다 다르고**, hot의 L2 상주를 flush로 지운다.

    # 실구성 한 점
    python test/prism/bench_full_layer.py --device 0 --dtype fp8 \
        --hot-frac 0.375 --warm-frac 0.125 --sparsity 0.5 --sparsity-spread 0.3

    # planner 근사와 대조 (접힌 앵커 vs 실제 top_k 앵커)
    python test/prism/bench_full_layer.py --device 0 --dtype fp8 --compare --anchors 1,8

    # 캐시 상주가 얼마였는지 (flush 없이)
    python test/prism/bench_full_layer.py --device 0 --dtype fp8 --no-flush
"""

from __future__ import annotations

import argparse

from sglang.srt.layers.moe.prism.profile import (
    FullLayerProfiler,
    Shape,
    compare,
    emit,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experts", type=int, default=128, help="E (num_experts)")
    p.add_argument("--topk", type=int, default=8, help="top_k (<= 16)")
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=768)
    p.add_argument("--hot-frac", type=float, default=0.375)
    p.add_argument("--warm-frac", type=float, default=0.125)
    p.add_argument("--cold-frac", type=float, default=None,
                   help="기본값(None)은 hot·warm이 안 가진 행 전부가 cold")
    p.add_argument("--sparsity", type=float, default=0.5,
                   help="expert 평균 sparsity (정확히 이 값이 실현된다)")
    p.add_argument("--sparsity-spread", type=float, default=0.3,
                   help="expert별 편차 — 0.5±0.3이면 [0.2, 0.8]")
    p.add_argument("--hot-spread", type=float, default=0.0,
                   help="hot 비율의 expert별 편차 (행 수는 정렬 배수로 잘린다)")
    p.add_argument("--warm-spread", type=float, default=0.0)
    p.add_argument("--groups", default="gateup,down")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "mxfp4", "fp8"))
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--replays", type=int, default=10)
    p.add_argument("--rounds", type=int, default=3,
                   help="변형 목록 전체를 이만큼 반복하고 변형별 중앙값을 쓴다 "
                        "(공유 머신의 드리프트 대비)")
    p.add_argument("--no-flush", action="store_true",
                   help="hot의 L2 flush를 끈다 (켠 값과의 차가 캐시 상주분)")
    p.add_argument("--flush-mb", type=int, default=None,
                   help="flush 버퍼 크기 (기본 1.25 × L2)")
    p.add_argument("--numa-split", type=float, default=0.5)
    p.add_argument("--numa-map", default=None, help="예: 0,1")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--cpu-kernel", default=None)
    p.add_argument("--shuffle-index", action="store_true")
    p.add_argument("--compare", action="store_true",
                   help="planner 선형 근사와 나란히 찍는다")
    p.add_argument("--anchors", default="1",
                   help="--compare의 앵커 top_k 목록 (예: 1,8)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out", help="JSON 리포트 경로")
    a = p.parse_args()

    shape = Shape(experts=a.experts, topk=a.topk, hidden=a.hidden, inter=a.inter)
    groups = [s.strip() for s in a.groups.split(",") if s.strip()]
    kw = dict(hot_frac=a.hot_frac, warm_frac=a.warm_frac, cold_frac=a.cold_frac,
              sparsity=a.sparsity, sparsity_spread=a.sparsity_spread,
              hot_spread=a.hot_spread, warm_spread=a.warm_spread,
              dtype=a.dtype, device=a.device, numa_split=a.numa_split,
              cpu_kernel=a.cpu_kernel, threads=a.threads,
              numa_map=([int(x) for x in a.numa_map.split(",")] if a.numa_map else None),
              shuffle_index=a.shuffle_index, flush_mb=a.flush_mb, seed=a.seed)
    try:
        if a.compare:
            report = compare(
                shape, groups=groups,
                anchors=[int(x) for x in a.anchors.split(",")],
                reps=a.reps, replays=a.replays, flush=not a.no_flush,
                rounds=a.rounds, **kw)
        else:
            with FullLayerProfiler(shape, **kw) as prof:
                report = prof.report(groups, reps=a.reps, replays=a.replays,
                                     flush=not a.no_flush, rounds=a.rounds)
    except ValueError as e:
        raise SystemExit(str(e))
    emit(report, a.out)


if __name__ == "__main__":
    main()
