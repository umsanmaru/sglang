#!/usr/bin/env python
"""cold 티어만, GPU 없이 — E×A 항의 원인 규명용 하네스.

`bench_warm_cold_sparse.py`의 cold 절반만 떼어낸 것이다. CUDA를 전혀 건드리지
않으므로 (a) GPU가 남에게 점유돼 있어도 돌고, (b) `perf`로 심볼·캐시 이벤트를
바로 볼 수 있다. 재는 것은 kt의 동기 진입점 `forward_gateup_partial` /
`forward_down_partial` 한 번이고, 그것이 곧 prism의 cold 호출이다.

2026-08-26 실측에서 cold 비용이 `expert_num × activated_expert`에 비례하는
것이 나왔는데(E=128·A=8에서 총 217.8 µs 중 168 µs), decode 경로에는 그 모양의
루프가 없다 — 전부 O(A)거나 O(E) 1회다. 그래서 남은 가설은 "활성 expert마다
expert 번호로 흩어진 구조를 만지고, 그 배열이 E에 비례해 커져 캐시에서 떨어져
나간다"이고, 이 스크립트는 그 가설의 판별자 둘을 돌린다:

  --sweep-experts 8,16,32,64,128   A 고정, E만 훑는다. 중첩 루프면 E에
                                   **선형**, 메모리면 캐시 경계에서 **무릎**.
  --band                           cold 인덱스를 밴드 퇴화형으로 주입한다
                                   (row_off/idx 비우고 offset/rows만) → kt의
                                   `dense_rows`가 zero-copy로 떨어져 gather가
                                   사라진다. 차감이 gather 몫이다.

perf 예시 (paranoid <= 2 필요):

    perf stat -e cycles,instructions,cache-misses,dTLB-load-misses \
        python test/prism/bench_cold_cpu.py --experts 128 --iters 300
    perf record -g python test/prism/bench_cold_cpu.py --experts 128 --iters 300
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

from bench_common import (
    GRID,
    K_STEP,
    NG,
    PMAX,
    RENORM_IT,
    SPARSITY_LAM,
    SPARSITY_P,
    add_shape_args,
    emit,
    shape_from_args,
    sparse_tables,
    split_rows,
    tier_index,
)

# tile_k2의 gemv_slab이 요구하는 노드 N 정렬 (N_BLOCK). bench_warm_cold_sparse와
# 같은 값이지만 여기서 다시 정의하는 이유: 이 스크립트는 CUDA를 import하는
# 모듈을 아예 건드리지 않는다 (perf 프로파일에 CUDA 초기화가 섞이지 않게).
_N_ALIGN = {"kt_tile_k2_bf16": 256, "kt_amx_bf16": 32}


def node_table(n_total: int, frac: float, nodes: int, align: int):
    if n_total % align:
        raise SystemExit(f"N={n_total} not a multiple of {align}")
    blocks = n_total // align
    if blocks < nodes:
        raise SystemExit(f"N={n_total}: {blocks} blocks < {nodes} nodes")
    if nodes == 1:
        rows = [n_total]
    elif nodes == 2:
        b0 = min(blocks - 1, max(1, int(round(blocks * frac))))
        rows = [b0 * align, n_total - b0 * align]
    else:
        base, rem = divmod(blocks, nodes)
        rows = [(base + (1 if i < rem else 0)) * align for i in range(nodes)]
    off, acc = [], 0
    for r in rows:
        off.append(acc)
        acc += r
    return off, rows


def numa_nodes() -> int:
    try:
        return len([d for d in os.listdir("/sys/devices/system/node")
                    if d.startswith("node") and d[4:].isdigit()])
    except OSError:
        return 1


def build_cold(shape, k_cold, *, sparsity, pattern, seed, numa_split, threads,
               kernel_key, band, nodes, split_index, numa_map=None):
    """kt partial 인스턴스 하나 + 주입 텐서. cold_backend가 Plan에서 굽는 것과
    같은 config를 CLI에서 굽는다."""
    from kt_kernel import kt_kernel_ext
    from kt_kernel.experts_partial import PartialMoEWrapper

    E, topk = shape.experts, shape.topk
    H, I = shape.hidden, shape.inter
    # numa_map이 주어지면 그 노드들만 쓴다 — 실모델의 SGLANG_PRISM_NUMA_MAP과
    # 같은 경로(WorkerPoolConfig)다. 소켓당 스레드 수가 offcore 큐 점유를
    # 결정하고 그것이 프리페처 부호를 정하므로, 이 축을 못 고정하면 측정이
    # 실모델과 다른 동작점에 놓인다.
    if numa_map:
        cfg_pool = kt_kernel_ext.WorkerPoolConfig()
        cfg_pool.subpool_count = len(numa_map)
        cfg_pool.subpool_numa_map = list(numa_map)
        cfg_pool.subpool_thread_count = [
            threads // len(numa_map) + (1 if i < threads % len(numa_map) else 0)
            for i in range(len(numa_map))
        ]
        cpuinfer = kt_kernel_ext.CPUInfer(cfg_pool)
    else:
        cpuinfer = kt_kernel_ext.CPUInfer(threads)
    cfg = kt_kernel_ext.moe.MOEConfig(E, topk, H, I, 0)
    cfg.max_len = 1
    cfg.layer_idx = 0
    cfg.partial.enabled = True
    cfg.partial.n_total = I

    rows = {}
    for proj, ki in (("gate", cfg.partial.gate), ("up", cfg.partial.up),
                     ("down", cfg.partial.down)):
        axis = shape.k_axis(proj)
        kc = k_cold[proj]
        rows[proj] = kc
        if band:
            # 퇴화형: 전 expert가 같은 연속 밴드. kt가 gather를 건너뛴다.
            ki.offset = axis - kc          # 축 끝에 붙인 밴드
            ki.rows = kc
        else:
            ki.row_off = [e * kc for e in range(E + 1)]
            # split_index: up만 다른 인덱스를 준다 → dual_pack()의 `idx ==` 비교가
            # 첫 원소에서 조기 종료한다. 그 대가로 kt가 activation을 두 번
            # pack하므로(K5 dual-pack) 일이 **늘어난다** — 그런데도 총시간이
            # 줄면 비교 자체가 비용이었다는 확증이다.
            sd = seed + (7919 if (split_index and proj == "up") else 0)
            ki.idx = tier_index(axis, kc, skip=axis - kc, seed=sd) \
                .to(torch.int32).repeat(E).tolist()

    align = _N_ALIGN[kernel_key]
    gu_off, gu_rows = node_table(I, numa_split, nodes, align)
    dn_off, dn_rows = node_table(H, numa_split, nodes, align)
    cfg.partial.node_gateup_n_offset = gu_off
    cfg.partial.node_gateup_n_rows = gu_rows
    cfg.partial.node_down_n_offset = dn_off
    cfg.partial.node_down_n_rows = dn_rows

    sp = cfg.partial.sparsity
    sp.pmax, sp.grid, sp.ng, sp.renorm_it = PMAX, GRID, NG, RENORM_IT
    sp.p_gate, sp.lam_gate = SPARSITY_P, SPARSITY_LAM
    sp.p_up, sp.lam_up = SPARSITY_P, SPARSITY_LAM
    sp.p_down, sp.lam_down = SPARSITY_P, SPARSITY_LAM
    cfg.pool = cpuinfer.backend_

    weights, tables, keep = {}, {}, {}
    for proj in ("gate", "up", "down"):
        n = shape.n_cols(proj)
        t = torch.empty(E * n * rows[proj], dtype=torch.bfloat16)
        t.normal_(0, 0.02)
        weights[proj] = t.contiguous()
        a, c, thr, frac = sparse_tables(E, rows[proj], sparsity,
                                        pattern=pattern, seed=seed)
        tables[f"{proj}_wn_sq"] = a
        tables[f"{proj}_pair_dot"] = c
        tables[f"thr_{proj}"] = thr
        keep[proj] = frac

    wrapper = PartialMoEWrapper(cfg, cpuinfer, kernel_key=kernel_key)
    wrapper.load_weights_from_tensors(
        weights["gate"], weights["up"], weights["down"], sparsity_tables=tables)
    return wrapper, cpuinfer, rows, keep


def run(args, E: int) -> dict:
    shape = type(args.shape)(experts=E, topk=args.shape.topk,
                             hidden=args.shape.hidden, inter=args.shape.inter)
    k_cold = {p: split_rows(shape.k_axis(p), args.cold_frac)
              for p in ("gate", "up", "down")}
    for p, kc in k_cold.items():
        if kc % K_STEP:
            raise SystemExit(f"{p}: cold rows {kc} not a K_STEP multiple")

    numa_map = [int(x) for x in args.numa_map.split(",") if x.strip()] or None
    # shard table의 항목 수는 tp_count(=subpool 수)와 같아야 한다. numa_map을
    # 주면 subpool이 그 길이만큼만 생기므로 nodes도 같이 좁혀야 한다 —
    # 안 그러면 kt가 "partial shard table size != tp_count"로 즉사한다.
    nodes = len(numa_map) if numa_map else numa_nodes()
    wrapper, cpuinfer, rows, keep = build_cold(
        shape, k_cold, sparsity=args.sparsity, pattern=args.mask_pattern,
        seed=args.seed, numa_split=args.numa_split, threads=args.threads,
        numa_map=numa_map,
        kernel_key=args.cpu_kernel, band=args.band, nodes=nodes,
        split_index=args.split_index)

    topk = shape.topk
    qlen = torch.ones(1, dtype=torch.int32)
    # 스텝마다 다른 expert 조합 — 실제 라우팅의 성질이고, E 의존을 만드는
    # 조건이다 (같은 expert를 반복하면 캐시에 남아 항이 사라진다).
    g = torch.Generator().manual_seed(1234)
    if args.fixed_ids:
        # 판별자: 풀은 E개인데 매 스텝 **같은** 8개만 쓴다. E 의존이 캐시
        # 잔존율(재사용 거리)이면 여기서 E=8 수준으로 붕괴하고, 구조적이면
        # 그대로 남는다.
        one = torch.randperm(E, generator=g)[:topk].to(torch.int64).contiguous()
        ids = [one] * args.iters
    else:
        ids = [torch.randperm(E, generator=g)[:topk].to(torch.int64).contiguous()
               for _ in range(args.iters)]
    w = torch.full((1, topk), 1.0 / topk, dtype=torch.float32)

    if args.proj == "gateup":
        x = torch.ones(1, shape.hidden, dtype=torch.bfloat16)
        out = torch.zeros(1, topk, 2 * shape.inter, dtype=torch.bfloat16)
        call = wrapper.forward_gateup
    else:
        x = torch.ones(1, topk, shape.inter, dtype=torch.bfloat16)
        out = torch.zeros(1, topk, shape.hidden, dtype=torch.bfloat16)
        call = wrapper.forward_down

    def step(i: int) -> None:
        call(ids[i % args.iters].reshape(1, topk), x, out, w)

    for i in range(min(20, args.iters)):   # warmup
        step(i)
    per = []
    for r in range(args.replays):
        t0 = time.perf_counter()
        for i in range(args.iters):
            step(i)
        per.append((time.perf_counter() - t0) / args.iters * 1e6)
    del wrapper
    return {
        "experts": E, "topk": topk, "k_cold": rows, "keep_frac": keep["gate"],
        "band": args.band, "numa_nodes": nodes,
        "us": round(statistics.median(per), 2),
        "min_us": round(min(per), 2), "max_us": round(max(per), 2),
        "replays": args.replays, "iters": args.iters,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shape_args(p)
    p.add_argument("--cold-frac", type=float, default=0.875)
    p.add_argument("--sparsity", type=float, default=1.0,
                   help="기본 1.0 — W 바이트 0으로 두고 준비 작업만 잰다")
    p.add_argument("--proj", default="gateup", choices=("gateup", "down"))
    p.add_argument("--split-index", action="store_true",
                   help="gate != up 인덱스 (dual_pack의 벡터 비교를 조기 종료시킴)")
    p.add_argument("--fixed-ids", action="store_true",
                   help="매 스텝 같은 expert 집합 (재사용 거리 판별자)")
    p.add_argument("--band", action="store_true",
                   help="밴드 퇴화형 주입 (kt의 gather zero-copy 경로)")
    p.add_argument("--sweep-experts", default=None,
                   help="쉼표로 구분한 E 목록 — A 고정, E만 훑는다")
    p.add_argument("--mask-pattern", default="random", choices=("random", "block"))
    p.add_argument("--cpu-kernel", default="kt_tile_k2_bf16",
                   choices=tuple(_N_ALIGN))
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
    a.shape = shape_from_args(a)
    a.threads = a.threads or int(os.environ.get(
        "SGLANG_PRISM_CPUINFER_THREADS", max(2, (os.cpu_count() or 4) // 2 - 2)))

    es = ([int(x) for x in a.sweep_experts.split(",")] if a.sweep_experts
          else [a.shape.experts])
    results = [run(a, E) for E in es]
    emit({
        "bench": "cold_cpu",
        "kernel": a.cpu_kernel,
        "shape": a.shape.as_dict(),
        "params": {
            "cold_frac": a.cold_frac, "sparsity": a.sparsity, "proj": a.proj,
            "band": a.band, "fixed_ids": a.fixed_ids, "split_index": a.split_index, "mask_pattern": a.mask_pattern,
            "numa_split": a.numa_split, "cpuinfer_threads": a.threads,
            "iters": a.iters, "replays": a.replays, "seed": a.seed,
        },
        "results": results,
    }, a.out)


if __name__ == "__main__":
    main()
