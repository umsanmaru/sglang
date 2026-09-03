#!/usr/bin/env python3
"""dense cold(kt) 퇴화 경로 실측 — 인스턴스 비용과 처리량.

0단계 실측(2026-09-01)에서 `MOEConfig(E=1, k=1, hidden=N_proj, intermediate=K_proj)`의
**down 슬롯만** 쓰면 `forward_down`이 정확히 `x @ W_cold.T`를 낸다는 것이 확인됐다.
C++ `PartialDenseWrapper` 없이 갈 수 있다는 뜻인데, 대가가 둘이다:

  · **빈 슬롯을 못 쓴다** — gate/up에 최소 타일(32행) 더미가 필요하다 (`moe.hpp:513`
    "no weight source"). 슬롯당 K_proj × 32 × 2 B.
  · **슬롯당 인스턴스 하나** — MoE는 layer당 1개(총 64)인데 dense는 슬롯당이라 350+개다.
    각 인스턴스가 버퍼·plan arena를 들고 같은 스레드풀에 붙는다.

이 스크립트가 재는 것: 인스턴스당 메모리·생성 시간, 그리고 M별 forward 시간.
과다구독이 submit/sync 고정비를 폭증시킨다는 것은 MoE 실측이 이미 보였으므로
(16코어 60스레드 → sync 1.85 ms), **인스턴스 수 스케일링**이 핵심 질문이다.

    python scripts/qwen38/bench_cold_degenerate.py --threads 14
"""
import argparse, gc, os, resource, time

import torch
from kt_kernel import kt_kernel_ext as ext
from kt_kernel.experts_partial import PartialMoEWrapper

TILE = 32            # kt_amx_bf16의 cold_pack_tile_rows
DUMMY = TILE         # 빈 슬롯 대체용 최소 타일

# Qwen3.8-27B 한 층의 슬롯 (K, N, 층수) — cold 비율은 인자로
SLOTS = [
    ("mlp.gate_up_proj/gate", 5120, 17408, 64),
    ("mlp.gate_up_proj/up",   5120, 17408, 64),
    ("mlp.down_proj",        17408,  5120, 64),
    ("linear_attn.in_proj_qkvz", 5120, 16384, 48),
    ("linear_attn.out_proj",  6144,  5120, 48),
    ("self_attn.qkv_proj/q",  5120, 12288, 16),
    ("self_attn.qkv_proj/k",  5120,  1024, 16),
    ("self_attn.qkv_proj/v",  5120,  1024, 16),
    ("self_attn.o_proj",      6144,  5120, 16),
]


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def make(cpuinfer, nodes, k_cold, n, kernel="kt_amx_bf16"):
    """한 슬롯의 kt 인스턴스. down 슬롯만 쓰고 gate/up은 더미."""
    cfg = ext.moe.MOEConfig(1, 1, n, k_cold, 0)      # hidden=N(출력), intermediate=K(입력)
    cfg.max_len = 2048
    cfg.layer_idx = 0
    cfg.partial.enabled = True
    for slot, rows in ((cfg.partial.gate, DUMMY), (cfg.partial.up, DUMMY),
                       (cfg.partial.down, k_cold)):
        slot.row_off = [0, rows]
        slot.idx = list(range(rows))
    cfg.partial.n_total = k_cold
    def split(total):
        step = (total // nodes) // TILE * TILE
        off, rows, cur = [], [], 0
        for i in range(nodes):
            end = total if i == nodes - 1 else cur + step
            off.append(cur); rows.append(end - cur); cur = end
        return off, rows
    go, gr = split(k_cold)      # gateup의 N = intermediate = k_cold
    do, dr = split(n)           # down의 N = hidden = n
    cfg.partial.node_gateup_n_offset, cfg.partial.node_gateup_n_rows = go, gr
    cfg.partial.node_down_n_offset,  cfg.partial.node_down_n_rows  = do, dr
    cfg.pool = cpuinfer.backend_
    wr = PartialMoEWrapper(cfg, cpuinfer, kernel_key=kernel)
    dummy = torch.zeros(k_cold * DUMMY, dtype=torch.bfloat16)
    w = torch.randn(n, k_cold, dtype=torch.bfloat16)
    wr.load_weights_from_tensors(dummy, dummy, w.contiguous())
    return wr, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--nodes", type=int, default=2)
    ap.add_argument("--cold", type=float, default=0.75, help="cold 비율")
    ap.add_argument("--layers", type=int, default=2, help="몇 층 분량을 만들지")
    ap.add_argument("--m", type=int, nargs="+", default=[1, 8, 64])
    a = ap.parse_args()

    pc = ext.WorkerPoolConfig()
    pc.subpool_count = a.nodes
    pc.subpool_numa_map = list(range(a.nodes))
    per = a.threads // a.nodes
    pc.subpool_thread_count = [per + (1 if i < a.threads % a.nodes else 0)
                               for i in range(a.nodes)]
    cpuinfer = ext.CPUInfer(pc)
    print(f"CPUInfer: {a.threads} threads / {a.nodes} nodes {pc.subpool_thread_count}")
    print(f"cold {a.cold:.0%} · {a.layers}층 분량\n")

    base = rss_gb()
    made, t0 = [], time.perf_counter()
    for name, K, N, _ in SLOTS:
        k_cold = int(K * a.cold) // TILE * TILE
        for _ in range(a.layers):
            made.append((name, k_cold, N, *make(cpuinfer, a.nodes, k_cold, N)))
    t_make = time.perf_counter() - t0
    mem = rss_gb() - base
    n_inst = len(made)
    w_gb = sum(k * n * 2 for _, k, n, _, _ in made) / 1e9
    print(f"인스턴스 {n_inst}개: 생성 {t_make:6.2f}s ({t_make/n_inst*1000:5.1f} ms/개)")
    print(f"  RSS 증가 {mem:6.2f} GB / weight {w_gb:.2f} GB → 오버헤드 {mem-w_gb:+.2f} GB "
          f"({(mem-w_gb)/n_inst*1000:.1f} MB/개)\n")

    print(f"  {'M':>5s} {'슬롯당':>10s} {'층당(9슬롯)':>12s} {'64층 추정':>12s} {'유효 대역':>12s}")
    for M in a.m:
        tot, gb = 0.0, 0.0
        for name, k_cold, N, wr, w in made:
            x = torch.randn(M, k_cold, dtype=torch.bfloat16)
            out = torch.zeros(M, 1, N, dtype=torch.bfloat16)
            ids = torch.zeros(M, 1, dtype=torch.int64)
            wr.forward_down(ids, x.contiguous(), out)          # 워밍업
            t = time.perf_counter()
            for _ in range(3):
                wr.forward_down(ids, x.contiguous(), out)
            dt = (time.perf_counter() - t) / 3
            tot += dt; gb += k_cold * N * 2 / 1e9
        per_slot = tot / n_inst
        layer = tot / a.layers
        print(f"  {M:5d} {per_slot*1000:9.2f}ms {layer*1000:11.2f}ms "
              f"{layer*64*1000/1000:11.2f}s {gb/tot:11.1f} GB/s")


if __name__ == "__main__":
    main()
