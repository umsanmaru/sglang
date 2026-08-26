#!/usr/bin/env python
"""cold 티어만, GPU 없이 — 프로파일 ③ CLI.

구현은 `sglang.srt.layers.moe.prism.profile`에 있다. 이 파일은 argparse 껍데기고,
같은 측정을 자기 프로그램에 심으려면 API를 직접 부르면 된다:

    from sglang.srt.layers.moe.prism.profile import Shape, cold_cpu, cold_cpu_sweep
    cold_cpu(Shape(128, 8, 2048, 768), sparsity=0.9).us      # 103.5
    cold_cpu_sweep(shape, experts=[8, 32, 64, 128])          # E 판별자

CUDA를 전혀 건드리지 않으므로 (a) GPU가 남에게 점유돼 있어도 돌고, (b) `perf`로
심볼·캐시 이벤트를 바로 볼 수 있다. 재는 것은 kt의 **동기** 진입점
`forward_{gateup,down}_partial` 한 번이다.

원인 규명용 개입 플래그 넷 (2026-08-26에 이걸로 cold 비용의 77%가 `dual_pack()`의
인덱스 벡터 비교임을 갈랐다 — kt `eb780a4`에서 수정):

  --sweep-experts 8,16,32,64,128   A 고정, E만 훑는다. 중첩 루프면 E에 **선형**,
                                   메모리면 캐시 경계에서 **무릎**.
  --band                           밴드 퇴화형 주입 → kt의 `dense_rows`가
                                   zero-copy로 떨어져 gather가 사라진다.
  --fixed-ids                      매 스텝 같은 expert 집합 (재사용 거리 판별자).
  --split-index                    gate != up 인덱스 → `dual_pack()`의 벡터 비교가
                                   조기 종료한다. 일이 **늘어나는데도** 총시간이
                                   줄면 비교 자체가 비용이었다는 확증.

`--sparsity` 기본값은 0.9다. 1.0은 W 바이트를 0으로 만들어 준비 작업만 남기는
**진단용** 값이라 기본값으로 두지 않는다 (모르고 쓰면 cold 비용이 아닌 숫자를
cold 비용으로 읽게 된다).

perf 예시 (paranoid <= 2 필요):

    perf stat -e cycles,instructions,cache-misses,dTLB-load-misses \
        python test/prism/bench_cold_cpu.py --experts 128 --iters 300
    perf record -g python test/prism/bench_cold_cpu.py --experts 128 --iters 300
"""

from __future__ import annotations

import argparse

from sglang.srt.layers.moe.prism.profile import Shape, cold_cpu_sweep, emit


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8, help="top_k (<= 16)")
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=768)
    p.add_argument("--cold-frac", type=float, default=0.875)
    p.add_argument("--sparsity", type=float, default=0.9,
                   help="죽이는 페어의 비율. 1.0 = 준비 작업만 (진단용)")
    p.add_argument("--proj", default="gateup", choices=("gateup", "down"))
    p.add_argument("--band", action="store_true",
                   help="밴드 퇴화형 주입 (kt의 gather zero-copy 경로)")
    p.add_argument("--fixed-ids", action="store_true",
                   help="매 스텝 같은 expert 집합 (재사용 거리 판별자)")
    p.add_argument("--split-index", action="store_true",
                   help="gate != up 인덱스 (dual_pack의 벡터 비교를 조기 종료시킴)")
    p.add_argument("--sweep-experts", default=None,
                   help="쉼표로 구분한 E 목록 — A 고정, E만 훑는다")
    p.add_argument("--mask-pattern", default="random", choices=("random", "block"))
    p.add_argument("--cpu-kernel", default="kt_tile_k2_bf16",
                   choices=("kt_tile_k2_bf16", "kt_amx_bf16"))
    p.add_argument("--numa-map", default="",
                   help="쉼표로 구분한 NUMA 노드 목록 (예: 0 / 1 / 0,1). "
                        "실모델의 SGLANG_PRISM_NUMA_MAP과 같은 의미. 비우면 "
                        "머신 전 노드에 스레드를 분배한다")
    p.add_argument("--numa-split", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--replays", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    a = p.parse_args()

    shape = Shape(experts=a.experts, topk=a.topk, hidden=a.hidden, inter=a.inter)
    experts = ([int(x) for x in a.sweep_experts.split(",")] if a.sweep_experts
               else [a.experts])
    numa_map = [int(x) for x in a.numa_map.split(",") if x.strip()] or None
    try:
        payload = cold_cpu_sweep(
            shape, experts, iters=a.iters, replays=a.replays,
            fixed_ids=a.fixed_ids, cold_frac=a.cold_frac, sparsity=a.sparsity,
            proj=a.proj, band=a.band, split_index=a.split_index,
            mask_pattern=a.mask_pattern, numa_split=a.numa_split,
            threads=a.threads, cpu_kernel=a.cpu_kernel, seed=a.seed,
            numa_map=numa_map)
    except ValueError as e:
        raise SystemExit(str(e))
    emit(payload, a.out)


if __name__ == "__main__":
    main()
