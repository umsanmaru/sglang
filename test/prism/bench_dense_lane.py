#!/usr/bin/env python
"""dense 레인 weight `[k, n]` 하나의 티어별 실행시간 — 프로파일 ⑤ CLI.

구현은 `sglang.srt.layers.moe.prism.profile.dense_lane`에 있다. 이 파일은 argparse
껍데기고, 같은 측정을 자기 프로그램에 심으려면 API를 직접 부르면 된다:

    from sglang.srt.layers.moe.prism.profile import DenseShape, dense_hot, dense_warm, dense_cold
    s = DenseShape(k=5120, n=8192, slots=8)
    dense_hot(s, dtype="fp8", kernel_type="pt", device=0).us
    dense_warm(s, 0.9, device=0).us
    dense_cold(s, 0.9).us

dense 레인은 expert가 없고 **슬롯**(같은 모양의 weight 개수 — 실모델이면 그 선형층을
가진 층 수)만 있다. `--slots`가 1이면 매 iteration이 같은 weight를 읽어 캐시에 남고
실제보다 최대 30% 빠르게 나오므로, 실제 개수를 주거나 최소한 `store_mb`가 GPU L2를
넘게 잡는다.

    # decode hot (마스킹 없음 — sparsity 인자를 받지 않는다)
    python test/prism/bench_dense_lane.py --tier hot --k 5120 --n 8192 --slots 8 --device 0

    # decode warm/cold를 sparsity별로
    python test/prism/bench_dense_lane.py --tier warm --k 5120 --n 8192 --sparsity 0.9 --device 0
    python test/prism/bench_dense_lane.py --tier cold --k 5120 --n 8192 --sparsity 0.9 --numa-map 0,1

    # fp8 커널 변종 (k2 기본 / k1 per-k 마스크 / pt per-tensor 배율)
    python test/prism/bench_dense_lane.py --tier cold --k 5120 --n 8192 \
        --dtype fp8 --kernel-type pt --sparsity none --m 64

    # 고를 수 있는 백엔드 조합
    python test/prism/bench_dense_lane.py --list-backends
"""

import argparse
import json
import sys

from sglang.srt.layers.moe.prism.profile import (
    KERNEL_TYPES,
    DenseShape,
    dense_backends,
    dense_cold,
    dense_hot,
    dense_sweep,
    dense_warm,
)
from sglang.srt.layers.moe.prism.profile.common import emit


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-backends", action="store_true",
                   help="고를 수 있는 (dtype, kernel_type) 조합을 찍고 끝낸다")
    p.add_argument("--tier", default="cold", choices=("hot", "warm", "cold", "all"),
                   help="all이면 셋을 한 번에 (dense_sweep)")
    p.add_argument("--k", type=int, default=5120, help="weight 행 수 (입력 차원)")
    p.add_argument("--n", type=int, default=8192, help="weight 열 수 (출력 차원)")
    p.add_argument("--slots", type=int, default=8,
                   help="같은 모양의 weight 개수 = 회전 풀 (1은 캐시 착시로 낙관적)")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "mxfp4", "fp8", "fp8pt"))
    p.add_argument("--kernel-type", default=None, choices=tuple(KERNEL_TYPES),
                   help="fp8 전용 커널 변종. k2=블록 배율(기본), k1=per-k 마스크, "
                        "pt=per-tensor 배율(K 정렬 32로 완화)")
    p.add_argument("--sparsity", default="0.9",
                   help="warm/cold에서 죽이는 비율. 'none' = 마스크 없는 kt dense 경로 "
                        "(cold 전용). --tier all이면 쉼표로 여러 개")
    p.add_argument("--m", type=int, default=1,
                   help="한 호출의 토큰 수. decode는 1. cold에서 >1은 --sparsity none 필수")
    p.add_argument("--threads", type=int, default=None, help="CPUInfer 스레드 (cold)")
    p.add_argument("--numa-map", default="", help="쉼표 구분 NUMA 노드 (예: 0 / 0,1)")
    p.add_argument("--numa-split", type=float, default=0.5)
    p.add_argument("--mask-pattern", default="random", choices=("random", "block"))
    p.add_argument("--reps", type=int, default=100, help="graph 하나에 담을 launch (hot/warm)")
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--iters", type=int, default=100, help="host 루프 반복 (cold)")
    p.add_argument("--vec", type=int, default=0, choices=(0, 1, 4, 8),
                   help="W 로드 폭 (bf16 hot 전용)")
    p.add_argument("--warm-node", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=0)
    p.add_argument("--out", help="JSON 리포트 경로")
    a = p.parse_args()

    if a.list_backends:
        print(json.dumps(dense_backends(), indent=2, ensure_ascii=False))
        return

    def _sp(text: str):
        return None if text.strip().lower() in ("none", "dense") else float(text)

    shape = DenseShape(k=a.k, n=a.n, slots=a.slots)
    numa_map = [int(x) for x in a.numa_map.split(",") if x.strip()] or None
    common = dict(dtype=a.dtype, kernel_type=a.kernel_type)
    try:
        if a.tier == "all":
            payload = dense_sweep(
                [shape], [_sp(s) for s in a.sparsity.split(",")], device=a.device,
                hot_m=a.m if a.m else 1, cold_m=1, seed=a.seed, reps=a.reps,
                replays=a.replays, iters=a.iters, threads=a.threads,
                numa_map=numa_map, numa_split=a.numa_split,
                mask_pattern=a.mask_pattern, warm_node=a.warm_node,
                out=a.out, quiet=False, **common)
            return
        if a.tier == "hot":
            r = dense_hot(shape, device=a.device, m=a.m, reps=a.reps,
                          replays=a.replays, vec=a.vec, seed=a.seed, **common)
        elif a.tier == "warm":
            r = dense_warm(shape, _sp(a.sparsity), device=a.device, reps=a.reps,
                           replays=a.replays, mask_pattern=a.mask_pattern,
                           warm_node=a.warm_node, seed=a.seed, **common)
        else:
            r = dense_cold(shape, _sp(a.sparsity), m=a.m, threads=a.threads,
                           numa_map=numa_map, numa_split=a.numa_split,
                           iters=a.iters, replays=max(a.replays // 2, 1),
                           mask_pattern=a.mask_pattern, seed=a.seed, **common)
    except ValueError as e:
        raise SystemExit(str(e))
    emit(r.as_dict(), a.out)


if __name__ == "__main__":
    sys.exit(main())
