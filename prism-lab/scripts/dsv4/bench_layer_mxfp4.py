#!/usr/bin/env python
"""DSV4-Flash 치수의 mxfp4 층 하나 — 로더 시간/메모리 + decode(M=1)·prefill(M) 층당 시간.

e2e 전에 커널·로더를 실제 치수에서 잰다. 예측(계획 §1.5): warm 87.5% dense decode =
6 expert × 12.75 MiB × 0.875 ≈ 67 MiB/층 → 51 GB/s에서 ~1.4 ms/층.
사용: bench_layer_mxfp4.py <plan.json> [--m 1 2048] [--reps 20]
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, "/home/um3maru/prism-sglang/sglang/test/prism")
from mxfp4_ref import random_expert_ckpt  # noqa: E402

from sglang.srt.layers.moe.prism.executor import PrismExecutor
from sglang.srt.layers.moe.prism.plan import parse_plan, validate_static
from sglang.srt.layers.moe.prism.resources import ExecutionResources, ResourceSpec
from sglang.srt.layers.moe.prism.weights import prepare_layer_weights
from sglang.srt.layers.moe.prism.numa import gpu_numa_node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--m", type=int, nargs="*", default=[1, 2048])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--grouped-min-m", type=int, default=None)
    args = ap.parse_args()
    plan = parse_plan(args.plan)
    validate_static(plan)
    d = plan.dims
    E, H, I, K = d.num_experts, d.hidden_size, d.intermediate_size, d.top_k
    dev = torch.device("cuda")

    g = torch.Generator().manual_seed(0)
    # 랜덤 fp4 체크포인트 층 (int8 nibble + fp32 배율)
    w13 = torch.randint(-128, 128, (E, 2 * I, H // 2), generator=g, dtype=torch.int8)
    w2 = torch.randint(-128, 128, (E, H, I // 2), generator=g, dtype=torch.int8)
    e13 = torch.randint(118, 126, (E, 2 * I, H // 32), generator=g)
    e2 = torch.randint(118, 126, (E, H, I // 32), generator=g)
    s13 = torch.ldexp(torch.ones(e13.shape), e13 - 127)
    s2 = torch.ldexp(torch.ones(e2.shape), e2 - 127)

    t0 = time.perf_counter()
    prepared = prepare_layer_weights(0, w13, w2, plan, device=dev, warm_node=gpu_numa_node(dev),
                                     w13_scale=s13, w2_scale=s2)
    t1 = time.perf_counter()
    hot_b = sum(t.numel() * t.element_size() for sh in (prepared.hot.gate, prepared.hot.up, prepared.hot.down)
                if sh is not None for t in (sh.w_flat, sh.s_flat) if t is not None) if prepared.hot else 0
    warm_b = sum(t.numel() * t.element_size() for sh in (prepared.warm.gate, prepared.warm.up, prepared.warm.down)
                 if sh is not None for t in (sh.w_flat, sh.s_flat) if t is not None)
    print(f"[load] prepare_layer_weights {t1 - t0:.2f} s  hot {hot_b / 2**20:.0f} MiB  warm(pinned) {warm_b / 2**20:.0f} MiB")

    spec = ResourceSpec.from_plan(plan, max_tokens=max(args.m), device=dev)
    ex = PrismExecutor(plan, ExecutionResources(spec), None, grouped_min_m=args.grouped_min_m)
    ex.register_layer(0, prepared)

    for m in args.m:
        x = (torch.randn(m, H, generator=g) / 4).to(torch.bfloat16).cuda()
        ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(m)]).cuda()
        w = torch.rand(m, K, generator=g).cuda()
        for _ in range(3):
            ex.run_layer(0, x, ids, w, swiglu_limit=10.0)
        torch.cuda.synchronize()
        reps = args.reps if m <= 64 else max(3, args.reps // 5)
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(reps):
            ex.run_layer(0, x, ids, w, swiglu_limit=10.0)
        ev1.record(); torch.cuda.synchronize()
        ms = ev0.elapsed_time(ev1) / reps
        # 이 호출이 읽는 바이트 (warm dense 기준: 활성 expert의 warm 행 전부)
        active = len(set(ids.reshape(-1).tolist()))
        per_e = sum((sh.w_flat.numel() + sh.s_flat.numel()) * 1 for sh in
                    (prepared.warm.gate, prepared.warm.up, prepared.warm.down) if sh is not None) / E
        touched = (m * K if m == 1 else active) * per_e
        print(f"[layer] M={m:5d}  {ms:8.3f} ms/층  warm bytes {touched / 2**20:7.1f} MiB → "
              f"{touched / ms / 1e6:6.1f} GB/s (PCIe 실효)   ×43층 = {ms * 43:7.1f} ms")


if __name__ == "__main__":
    main()
