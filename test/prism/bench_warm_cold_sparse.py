#!/usr/bin/env python
"""warm(GPU) + cold(CPU) sparse GEMV의 실행시간 — 프로파일 ② CLI.

구현은 `sglang.srt.layers.moe.prism.profile`에 있다. 이 파일은 argparse 껍데기고,
같은 측정을 자기 프로그램에 심으려면 API를 직접 부르면 된다 — 스토어 로딩이
비싸므로(cold 1 GB 패킹) 한 번 만들어 여러 번 질의하는 형태가 맞다:

    from sglang.srt.layers.moe.prism.profile import Shape, WarmColdProfiler
    with WarmColdProfiler(Shape(128, 8, 2048, 768),
                          warm_frac=0.125, sparsity=0.9, device=1) as p:
        g = p.measure("gateup")
        g.us("cold_only"), g.us("combined")
        p.check("gateup")

네 개의 숫자를 낸다 (전부 iteration당 µs):

  warm_only      — 커널 reps개를 CUDA graph에 캡처 → replay/reps
  cold_only      — host 루프 reps회 (submit+sync). CPU 단독 시간
  combined       — graph 하나에 reps × (ids D2H → cold submit → warm launch →
                   cold sync). kt 호출이 host node로 캡처되는 실제 graph decode
                   경로 그대로이고, 이 값이 **겹침 이후의 벽시계**다
  combined_eager — 같은 것을 eager로 (host 루프 throughput)
                   + `combined_eager_latency`는 iteration마다 GPU까지 대기

`--with-staging`을 주면 combined에 x/router-가중 D2H와 partial out H2D까지 포함한
변형이 하나 더 붙는다 — cold가 실제로 지불하는 왕복 비용이다.

sparsity는 **정확히 실현된다**: 점수 재료(a, c)와 threshold 곡선을 역산해 심으므로
GPU와 CPU가 같은 페어 집합을 살린다. `--check`는 그 마스크로 계산한 레퍼런스와
두 커널의 출력을 대조한다 — 마스킹이 조용히 사라져도 성능만 달라지므로 이 대조가
유일한 검출기다.

sparse는 decode 전용이라 M=1 고정이다 (executor의 masking 조건).

**cold 커널의 N 정렬**: `kt_tile_k2_bf16`은 노드마다 받는 N shard가 256(N_BLOCK)의
배수여야 한다 (`gemv_slab`이 그 stride를 전제하고, Release 빌드는 assert가 없어
어기면 조용히 남의 메모리를 읽는다). `--numa-split`은 그 블록 단위로 반올림되고
실현값이 리포트의 `node_tables`에 찍힌다.

    # 35B, warm 12.5% / cold 나머지, 페어 90% 스킵, NUMA 반반
    python test/prism/bench_warm_cold_sparse.py --device 1 \
        --warm-frac 0.125 --sparsity 0.9 --numa-split 0.5

    # 자원 소요만 확인
    python test/prism/bench_warm_cold_sparse.py --dry-run --warm-frac 0.125
"""

from __future__ import annotations

import argparse

from sglang.srt.layers.moe.prism.profile import (
    PROJS,
    VARIANTS,
    Shape,
    WarmColdProfiler,
    emit,
    footprint,
    make_split,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8, help="top_k (<= 16)")
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=768)
    p.add_argument("--sparsity", type=float, default=0.9,
                   help="죽이는 페어의 비율 (0=dense, 0.9=90%% 스킵)")
    p.add_argument("--warm-frac", type=float, default=0.125,
                   help="K축에서 warm이 소유하는 비율")
    p.add_argument("--cold-frac", type=float, default=None,
                   help="cold 비율 (기본 1-warm_frac; 합이 1보다 작으면 차이가 hot)")
    p.add_argument("--numa-map", default="",
                   help="쉼표로 구분한 NUMA 노드 목록 (예: 0 / 1 / 0,1). "
                        "실모델의 SGLANG_PRISM_NUMA_MAP과 같은 의미")
    p.add_argument("--numa-split", type=float, default=0.5,
                   help="cold N축에서 node 0의 몫 (커널 N_BLOCK 단위로 반올림)")
    p.add_argument("--groups", default="gateup,down")
    p.add_argument("--only", default=None,
                   help=f"이 변형만 측정 (쉼표 구분: {','.join(VARIANTS)})")
    p.add_argument("--mask-pattern", default="random", choices=("random", "block"))
    p.add_argument("--dtype", default="bf16", choices=("bf16", "mxfp4", "fp8"),
                   help="스토어 dtype = warm GPU 커널 + cold kt 백엔드를 함께 고른다")
    p.add_argument("--cpu-kernel", default=None,
                   choices=("kt_tile_k2_bf16", "kt_amx_bf16", "kt_amx_fp4",
                            "kt_tile_k2_mxfp4", "kt_tile_k2_fp8b128"),
                   help="기본값은 dtype이 정한다")
    p.add_argument("--threads", type=int, default=None,
                   help="CPUInfer 스레드 (기본 cpu_count//2-2, method.py와 같은 관례)")
    p.add_argument("--dense", action="store_true",
                   help="마스킹 없이 (sparse 커널 대신 dense 커널) — 절감 기준선")
    p.add_argument("--with-staging", action="store_true",
                   help="combined에 x/가중 D2H + partial H2D 포함 변형 추가")
    p.add_argument("--no-fused", action="store_true",
                   help="gateup을 융합하지 않고 2회 launch로 측정 (비교용). 기본은 "
                        "executor와 같은 융합 경로다")
    p.add_argument("--check", action="store_true",
                   help="합성 마스크 레퍼런스와 두 커널 출력을 대조")
    p.add_argument("--shuffle-index", action="store_true")
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warm-node", type=int, default=None,
                   help="warm pinned 스토어를 둘 NUMA 노드 (기본 GPU 로컬 노드)")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="자원 소요만 출력")
    p.add_argument("--out")
    a = p.parse_args()

    shape = Shape(experts=a.experts, topk=a.topk, hidden=a.hidden, inter=a.inter)
    cold_frac = 1.0 - a.warm_frac if a.cold_frac is None else a.cold_frac
    try:
        if a.dry_run:
            splits = {proj: make_split(shape, proj, a.warm_frac, cold_frac,
                                       shuffle=a.shuffle_index, seed=a.seed,
                                       dtype=a.dtype)
                      for proj in PROJS}
            emit({"bench": "warm_cold_sparse", "shape": shape.as_dict(),
                  "dtype": a.dtype,
                  "splits": {k: v.as_dict() for k, v in splits.items()},
                  "footprint": footprint(shape, splits, a.dtype)}, a.out)
            return

        with WarmColdProfiler(
            shape, warm_frac=a.warm_frac, cold_frac=a.cold_frac,
            sparsity=a.sparsity, numa_split=a.numa_split,
            mask_pattern=a.mask_pattern, cpu_kernel=a.cpu_kernel, dtype=a.dtype,
            threads=a.threads, device=a.device, masking=not a.dense,
            warm_node=a.warm_node, shuffle_index=a.shuffle_index, seed=a.seed,
            numa_map=[int(x) for x in a.numa_map.split(',') if x.strip()] or None,
        ) as prof:
            payload = prof.report(
                [g.strip() for g in a.groups.split(",") if g.strip()],
                reps=a.reps, replays=a.replays, with_staging=a.with_staging,
                only=([s.strip() for s in a.only.split(",")] if a.only else None),
                do_check=a.check, fused=not a.no_fused)
    except ValueError as e:
        raise SystemExit(str(e))
    emit(payload, a.out)


if __name__ == "__main__":
    main()
