#!/usr/bin/env python
"""GPU dense GEMV(hot 티어)의 실행시간 — 프로파일 ① CLI.

구현은 `sglang.srt.layers.moe.prism.profile`에 있다. 이 파일은 argparse 껍데기고,
같은 측정을 자기 프로그램에 심으려면 API를 직접 부르면 된다:

    from sglang.srt.layers.moe.prism.profile import Shape, hot_dense_gemv
    r = hot_dense_gemv(Shape(128, 8, 2048, 768), hot_frac=0.375, device=1)
    r.us("gate")            # 17.78
    r.layer_gemv_us         # 47.1

    # 35B(H=2048, I=768, E=128, k=8) 전체 K가 hot일 때
    python test/prism/bench_gpu_dense_gemv.py --device 1

    # h375 (K의 37.5%만 hot), bs=1
    python test/prism/bench_gpu_dense_gemv.py --device 1 --hot-frac 0.375

    # 치수만 직접 주는 경우 (weight shape = [k, n])
    python test/prism/bench_gpu_dense_gemv.py --device 1 --k 768 --n 512

    # dtype으로 백엔드를 고른다 (bf16 / mxfp4 / fp8) — 커널·스토어·정렬이 함께 바뀐다
    python test/prism/bench_gpu_dense_gemv.py --device 1 --dtype fp8
"""

from __future__ import annotations

import argparse

from sglang.srt.layers.moe.prism.profile import Shape, emit, hot_dense_gemv


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experts", type=int, default=128, help="E (num_experts)")
    p.add_argument("--topk", type=int, default=8, help="top_k (<= 16)")
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=768)
    p.add_argument("--hot-frac", type=float, default=1.0,
                   help="K축에서 hot이 소유하는 비율 (기본 1.0 = 전체)")
    p.add_argument("--projs", default="gateup,down",
                   help="측정할 단위 (기본 gateup,down = executor가 실제로 발행하는 "
                        "형태). 비교용으로 gateup_2launch, 또는 proj별 gate/up/down")
    p.add_argument("--k", type=int, help="raw 모드: weight 행 수 (proj 치수 대신)")
    p.add_argument("--n", type=int, help="raw 모드: weight 열 수")
    p.add_argument("--m", type=int, default=1, help="토큰 수 (decode=1)")
    p.add_argument("--vec", type=int, default=0, choices=(0, 1, 4, 8),
                   help="W 로드 폭 (0=자동, bf16 커널 전용)")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "mxfp4", "fp8"),
                   help="스토어 dtype = 커널 백엔드 (bf16 / mxfp4 g32 / fp8 e4m3 b128). "
                        "K 행 정렬도 따라 바뀐다: 2 / 32 / 128")
    p.add_argument("--reps", type=int, default=100,
                   help="그래프 하나에 담을 launch 수")
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--shuffle-index", action="store_true",
                   help="티어 인덱스를 정렬하지 않는다 (gather 최악 경우)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out", help="JSON 리포트 경로")
    a = p.parse_args()

    try:
        report = hot_dense_gemv(
            Shape(experts=a.experts, topk=a.topk, hidden=a.hidden, inter=a.inter),
            hot_frac=a.hot_frac,
            projs=[s.strip() for s in a.projs.split(",") if s.strip()],
            k_rows=a.k, n_cols=a.n, m=a.m, vec=a.vec, reps=a.reps,
            replays=a.replays, device=a.device,
            shuffle_index=a.shuffle_index, seed=a.seed, dtype=a.dtype)
    except ValueError as e:
        raise SystemExit(str(e))
    emit(report.as_dict(), a.out)


if __name__ == "__main__":
    main()
