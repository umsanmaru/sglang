#!/usr/bin/env python
"""warm(GPU) + cold(CPU) sparse GEMV의 실행시간 — 하드웨어 프로파일링 ②.

재는 것은 **두 티어가 실제로 부르는 그 커널들**이다:

  warm — `gemv_worklist_indexed_pinned_sparse` (tiers.py `SparsePinnedDirectTier`;
         W가 pinned host, GPU가 UVA로 제자리 읽고 죽은 페어의 로드를 발행하지
         않는다 → 건너뛴 만큼이 PCIe 절약)
  cold — kt `forward_{gateup,down}_partial` (기본 `kt_tile_k2_bf16` = TileK2BF16_MOE;
         s → thr → 점수 → 마스크 → masked GEMV가 전부 C++ 안에서 끝난다)

네 개의 숫자를 낸다 (전부 iteration당 µs):

  warm_only      — 커널 reps개를 CUDA graph에 캡처 → replay/reps
  cold_only      — host 루프 reps회 (submit+sync). CPU 단독 시간
  combined       — graph 하나에 reps × (ids D2H → cold submit → warm launch →
                   cold sync). kt 호출이 host node로 캡처되는 실제 graph decode
                   경로 그대로이고, 이 값이 **겹침 이후의 벽시계**다
  combined_eager — 같은 것을 eager로 (graph host node 경로의 교차검증)

`--with-staging`을 주면 combined에 x/router-가중 D2H와 partial out H2D까지 포함한
변형이 하나 더 붙는다 — cold가 실제로 지불하는 왕복 비용이다.

sparsity는 **정확히 실현된다**: 점수 재료(a, c)와 threshold 곡선을 역산해
심으므로 (bench_common의 docstring) GPU와 CPU가 같은 페어 집합을 살린다.
`--check`는 그 마스크로 계산한 레퍼런스와 두 커널의 출력을 대조한다 —
마스킹이 조용히 사라져도 성능만 달라지므로 이 대조가 유일한 검출기다.

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
import os
from dataclasses import dataclass
from typing import Optional

import torch

from bench_common import (
    GRID,
    K_STEP,
    NG,
    PAIR_GROUP,
    PMAX,
    RENORM_IT,
    SPARSITY_LAM,
    SPARSITY_P,
    Shape,
    add_shape_args,
    emit,
    env_stamp,
    nvtx,
    graph_stats,
    host_stats,
    select_device,
    shape_from_args,
    sparse_tables,
    split_rows,
    tier_index,
)

from sglang.jit_kernel.prism_gemv import (
    gemv_worklist_indexed_pinned,
    gemv_worklist_indexed_pinned_sparse,
    warmup_jit,
)
from sglang.srt.layers.moe.prism.numa import (
    alloc_pinned_on_node,
    gpu_numa_node,
    numa_node_count,
)
from sglang.srt.layers.moe.prism.resources import ExecutionResources, ResourceSpec
from sglang.srt.layers.moe.prism.tiers import SparseSpec

GROUPS = {"gateup": ("gate", "up"), "down": ("down",)}


# ─── 행 분할 ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Split:
    """한 proj의 K축 분할. hot은 이 벤치의 관심사가 아니므로 남은 행은 버린다
    (feature ①이 그 몫을 잰다) — warm_frac + cold_frac < 1이면 그 차이가 hot이다."""

    proj: str
    axis: int
    n_cols: int
    warm_rows: torch.Tensor   # int32 [k_warm] — K축 행 번호
    cold_rows: torch.Tensor   # int32 [k_cold]

    @property
    def k_warm(self) -> int:
        return int(self.warm_rows.numel())

    @property
    def k_cold(self) -> int:
        return int(self.cold_rows.numel())

    def as_dict(self) -> dict:
        return {"proj": self.proj, "k_axis": self.axis, "n_cols": self.n_cols,
                "k_warm": self.k_warm, "k_cold": self.k_cold}


def make_split(shape: Shape, proj: str, warm_frac: float, cold_frac: float,
               *, shuffle: bool, seed: int) -> Split:
    axis = shape.k_axis(proj)
    kw = split_rows(axis, warm_frac)
    kc = split_rows(axis, cold_frac)
    if kw + kc > axis:
        raise SystemExit(
            f"{proj}: warm {kw} + cold {kc} rows exceed axis {axis} "
            f"(K_STEP={K_STEP} 반올림 후)")
    # 같은 시드의 같은 순열에서 앞을 warm, 그 다음을 cold가 갖는다 → 서로소.
    return Split(
        proj=proj, axis=axis, n_cols=shape.n_cols(proj),
        warm_rows=tier_index(axis, kw, shuffle=shuffle, seed=seed),
        cold_rows=tier_index(axis, kc, skip=kw, shuffle=shuffle, seed=seed),
    )


# ─── warm (GPU, pinned W) ─────────────────────────────────────────────────
class WarmTier:
    """한 proj의 warm 스토어 + sparse 인자. weights.py의 warm shard와 같은 형태:
    w_flat [Σₑ k(e), N] pinned, row_off/k_index는 device 상주."""

    def __init__(self, shape: Shape, sp: Split, *, sparsity: float, pattern: str,
                 seed: int, device, node: Optional[int]):
        E = shape.experts
        kw = sp.k_warm
        self.split = sp
        self.w_flat = alloc_pinned_on_node(
            (E * kw, sp.n_cols), torch.bfloat16, node, f"warm {sp.proj} store")
        self.w_flat.normal_(0, 0.02)
        self.row_off = (torch.arange(E + 1, dtype=torch.int32) * kw).to(device)
        self.k_index = sp.warm_rows.to(torch.uint16).repeat(E).contiguous().to(device)
        a, c, thr, self.keep_frac = sparse_tables(
            E, kw, sparsity, pattern=pattern, seed=seed)
        self.a_host = a.reshape(E, kw)
        self.spec = SparseSpec(
            a=a.to(device), c=c.to(device), thr=thr.to(device),
            p=SPARSITY_P, lam=SPARSITY_LAM, pmax=PMAX, grid=GRID,
            ng=NG, renorm_it=RENORM_IT,
        )

    def launch(self, x, ids, topk_w, out, col_off: int, *, masking: bool) -> None:
        stream = torch.cuda.current_stream()
        if masking:
            gemv_worklist_indexed_pinned_sparse(
                x, ids, topk_w, self.w_flat, self.row_off, self.k_index, out,
                self.spec, col_off, self.split.proj == "down", stream)
        else:
            gemv_worklist_indexed_pinned(
                x, ids, self.w_flat, self.row_off, self.k_index, out,
                col_off, self.split.proj == "down", stream)


# ─── cold (CPU, kt) ───────────────────────────────────────────────────────
# 노드 N shard의 정렬 요구 — **커널이 정한다**.
#
# tile_k2의 `gemv_slab`은 타일 컬럼 stride를 `c * TILE_ELEMS`(= N_BLOCK 256 ×
# K_STEP 32)로 계산하므로 마지막 부분 블록을 표현할 수 없고, 그래서 스스로
# `assert(n % N_BLOCK == 0)`을 건다. Release 빌드는 NDEBUG라 그 assert가 없어
# **조용히 남의 열을 읽고 segfault한다** (2026-08-26 실측: I=768을 2노드 384씩
# 나눴을 때). 그래서 이 정렬은 벤치가 입력 단계에서 막는다.
#
# AMX(kt_amx_bf16)의 mat_mul/amx_kernel은 부분 N_BLOCK을 처리하므로 N_STEP(32)만
# 지키면 된다. 노드 하나가 받는 shard는 kt가 rows > 0을 요구한다.
_N_ALIGN = {"kt_tile_k2_bf16": 256, "kt_amx_bf16": 32}


def _node_table(n_total: int, frac: float, nodes: int, align: int) -> tuple:
    """N축을 노드에 나눈다 → (offset 테이블, rows 테이블, 실현된 node 0 비율).

    2노드는 frac을 쓰고 (align 블록 단위로 반올림), 그 이상은 균등이다 — 실
    plan(cold_shards)도 노드당 1 shard다.
    """
    if n_total % align:
        raise SystemExit(
            f"N={n_total}이 커널의 N 정렬 {align}의 배수가 아니다 — "
            f"이 커널로는 이 치수를 잴 수 없다")
    blocks = n_total // align
    if blocks < nodes:
        raise SystemExit(
            f"N={n_total}은 {align}짜리 블록 {blocks}개뿐인데 노드가 {nodes}개다 "
            f"(kt는 노드마다 rows > 0을 요구한다) — 더 큰 --inter/--hidden이나 "
            f"--cpu-kernel kt_amx_bf16")
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
    return off, rows, round(rows[0] / n_total, 4)


class ColdTier:
    """kt partial 인스턴스 하나 (레이어 1개 분량). cold_backend.KtColdBackend가
    Plan에서 굽는 것과 같은 config를 여기서는 CLI에서 굽는다 — 이 벤치는 Plan을
    읽지 않는다 (프로파일 결과가 Plan의 입력이므로 역방향 의존을 만들지 않는다)."""

    def __init__(self, shape: Shape, splits: dict, *, sparsity: float,
                 pattern: str, seed: int, numa_split: float, threads: int,
                 kernel_key: str):
        from kt_kernel import kt_kernel_ext
        from kt_kernel.experts_partial import PartialMoEWrapper

        E, topk = shape.experts, shape.topk
        H, I = shape.hidden, shape.inter
        self.nodes = numa_node_count()
        self.cpuinfer = kt_kernel_ext.CPUInfer(threads)
        cfg = kt_kernel_ext.moe.MOEConfig(E, topk, H, I, 0)
        cfg.max_len = 1              # sparse는 decode 전용 (qlen==1)
        cfg.layer_idx = 0
        cfg.partial.enabled = True
        cfg.partial.n_total = I
        for proj, ki in (("gate", cfg.partial.gate), ("up", cfg.partial.up),
                         ("down", cfg.partial.down)):
            sp = splits[proj]
            ki.row_off = [e * sp.k_cold for e in range(E + 1)]
            ki.idx = sp.cold_rows.to(torch.int32).repeat(E).tolist()
            # real_rows는 비워 둔다 — k_cold가 K_STEP 배수라 패딩이 없다.
        align = _N_ALIGN[kernel_key]
        gu_off, gu_rows, gu_frac = _node_table(I, numa_split, self.nodes, align)
        dn_off, dn_rows, dn_frac = _node_table(H, numa_split, self.nodes, align)
        cfg.partial.node_gateup_n_offset = gu_off
        cfg.partial.node_gateup_n_rows = gu_rows
        cfg.partial.node_down_n_offset = dn_off
        cfg.partial.node_down_n_rows = dn_rows
        self.node_tables = {
            "n_align": align,
            "gateup": {"offset": gu_off, "rows": gu_rows, "node0_frac": gu_frac},
            "down": {"offset": dn_off, "rows": dn_rows, "node0_frac": dn_frac},
        }

        s = cfg.partial.sparsity
        s.pmax, s.grid, s.ng, s.renorm_it = PMAX, GRID, NG, RENORM_IT
        s.p_gate, s.lam_gate = SPARSITY_P, SPARSITY_LAM
        s.p_up, s.lam_up = SPARSITY_P, SPARSITY_LAM
        s.p_down, s.lam_down = SPARSITY_P, SPARSITY_LAM
        cfg.pool = self.cpuinfer.backend_

        # cold 스토어: expert 블록이 ckpt 방향 [N, k_cold]인 1-D flat
        # (weights.py `_cold_flat`과 같은 레이아웃).
        self.w = {}
        for proj in ("gate", "up", "down"):
            sp = splits[proj]
            t = torch.empty(E * sp.n_cols * sp.k_cold, dtype=torch.bfloat16)
            t.normal_(0, 0.02)
            self.w[proj] = t.contiguous()

        tables, self.keep_frac, self.a_host = {}, {}, {}
        for proj in ("gate", "up", "down"):
            sp = splits[proj]
            a, c, thr, frac = sparse_tables(
                E, sp.k_cold, sparsity, pattern=pattern, seed=seed)
            tables[f"{proj}_wn_sq"] = a
            tables[f"{proj}_pair_dot"] = c
            tables[f"thr_{proj}"] = thr
            self.keep_frac[proj] = frac
            self.a_host[proj] = a.reshape(E, sp.k_cold)

        self.wrapper = PartialMoEWrapper(cfg, self.cpuinfer, kernel_key=kernel_key)
        self.wrapper.load_weights_from_tensors(
            self.w["gate"], self.w["up"], self.w["down"], sparsity_tables=tables)


# ─── 벤치 본체 ─────────────────────────────────────────────────────────────
def _ids(shape: Shape, reps: int, device, seed: int) -> tuple:
    """iteration마다 다른 top_k expert 배정 (M=1). GPU 커널과 cold staging이
    같은 int64 버퍼를 공유하므로 dtype은 staging(_expert_ids)에 맞춘다."""
    g = torch.Generator().manual_seed(seed)
    host = [torch.randperm(shape.experts, generator=g)[: shape.topk]
            .to(torch.int64).reshape(1, shape.topk).contiguous()
            for _ in range(reps)]
    return host, [t.to(device) for t in host]


def bench_group(group: str, shape: Shape, warm: dict, cold: ColdTier, res,
                *, reps: int, replays: int, device, with_staging: bool,
                masking: bool, qlen_ptr: int,
                only: Optional[frozenset] = None) -> dict:
    """gateup(gate+up 두 launch, cold 한 번) 또는 down(각각 한 번)의 네 숫자."""
    projs = GROUPS[group]
    topk = shape.topk
    st = res.staging
    ids_host, ids_dev = _ids(shape, reps, device, seed=1234)

    # warm 입력. x는 1.0으로 채운다 — sparsity 합성이 x0=x1=1을 전제한다.
    if group == "gateup":
        x = torch.ones(1, shape.hidden, dtype=torch.bfloat16, device=device)
        out = torch.zeros(1, topk, 2 * shape.inter, dtype=torch.bfloat16,
                          device=device)
        cols = {"gate": 0, "up": shape.inter}
        submit = cold.wrapper.submit_forward_gateup
        cold_in = st.x_ptr()
        cold_out = st.partial_gateup_ptr()
        st.fill_x(torch.ones(1, shape.hidden, dtype=torch.bfloat16))
    else:
        x = torch.ones(topk, shape.inter, dtype=torch.bfloat16, device=device)
        out = torch.zeros(1, topk, shape.hidden, dtype=torch.bfloat16,
                          device=device)
        cols = {"down": 0}
        submit = cold.wrapper.submit_forward_down
        cold_in = st.act_ptr()
        cold_out = st.partial_down_ptr()
        st.fill_act(torch.ones(1, topk, shape.inter, dtype=torch.bfloat16))
    topk_w_dev = torch.full((1, topk), 1.0 / topk, dtype=torch.float32,
                            device=device)
    st.fill_topk_w(topk_w_dev)
    w_ptr = st.topk_w_ptr() if masking else 0

    def warm_launch(i: int) -> None:
        for proj in projs:
            warm[proj].launch(x, ids_dev[i], topk_w_dev, out, cols[proj],
                              masking=masking)

    def cold_step(i: int, stream: Optional[int]) -> None:
        submit(qlen_ptr, topk, st.expert_ids_ptr(), cold_in, cold_out,
               stream, w_ptr)
        cold.wrapper.sync(stream)

    out_stats = {}
    # `only`가 주어지면 그 변형만 돈다 — nsys 트레이스를 한 변형으로 좁히려면
    # 나머지가 타임라인에 없어야 한다 (전부 돌리면 커널이 이름 없이 섞인다).
    def want(name: str) -> bool:
        return not only or name in only

    if want("warm_only"):
        with nvtx(f"{group}/warm_only"):
            out_stats["warm_only"] = graph_stats(warm_launch, reps, replays=replays)

    def cold_only(i: int) -> None:
        st.fill_expert_ids(ids_host[i])
        cold_step(i, None)

    if want("cold_only"):
        with nvtx(f"{group}/cold_only"):
            out_stats["cold_only"] = host_stats(cold_only, reps, replays=replays)

    def combined(i: int) -> None:
        # graph 경로의 expert_ids 조달: device → pinned async D2H (캡처 가능).
        st.fill_expert_ids(ids_dev[i], non_blocking=True)
        stream = torch.cuda.current_stream().cuda_stream
        submit(qlen_ptr, topk, st.expert_ids_ptr(), cold_in, cold_out,
               stream, w_ptr)
        warm_launch(i)
        cold.wrapper.sync(stream)

    if want("combined"):
        with nvtx(f"{group}/combined"):
            out_stats["combined"] = graph_stats(combined, reps, replays=replays)

    def combined_eager(i: int) -> None:
        # eager 경로의 단계별 구간 — 이 변형이 nsys로 들여다보는 대상이므로
        # cold submit / warm launch / cold sync를 각각 표시한다. 세 구간의
        # 폭이 곧 "무엇이 임계경로인가"의 답이다.
        with nvtx("eager/ids"):
            st.fill_expert_ids(ids_host[i])
        with nvtx("eager/cold_submit"):
            submit(qlen_ptr, topk, st.expert_ids_ptr(), cold_in, cold_out, None, w_ptr)
        with nvtx("eager/warm_launch"):
            warm_launch(i)
        with nvtx("eager/cold_sync"):
            cold.wrapper.sync(None)

    if want("combined_eager"):
        with nvtx(f"{group}/combined_eager"):
            out_stats["combined_eager"] = host_stats(
                combined_eager, reps, replays=replays, sync_cuda=True)

    def combined_eager_latency(i: int) -> None:
        """eager지만 iteration마다 GPU까지 기다린다.

        `combined_eager`는 host 루프의 **throughput**이다: cold sync는 CPU만
        기다리고 warm 커널은 큐에 쌓이므로 iteration i의 GPU가 i+1의 CPU와
        겹친다. 실제 executor는 rejoin이 두 결과를 같이 먹으므로 그 겹침이
        불가능하다 — 그 차이가 이 값과 위 값의 간격이고, `combined`(graph)와의
        비교에서 "host node 디스패치 비용"과 "파이프라이닝 상실"을 분리한다.
        """
        combined_eager(i)
        torch.cuda.current_stream().synchronize()

    if want("combined_eager_latency"):
        with nvtx(f"{group}/combined_eager_latency"):
            out_stats["combined_eager_latency"] = host_stats(
                combined_eager_latency, reps, replays=replays)

    if with_staging:
        def combined_staged(i: int) -> None:
            st.fill_expert_ids(ids_dev[i], non_blocking=True)
            if group == "gateup":
                st.fill_x(x, non_blocking=True)
            else:
                st.fill_act(x.reshape(1, topk, shape.inter), non_blocking=True)
            st.fill_topk_w(topk_w_dev, non_blocking=True)
            stream = torch.cuda.current_stream().cuda_stream
            submit(qlen_ptr, topk, st.expert_ids_ptr(), cold_in, cold_out,
                   stream, w_ptr)
            warm_launch(i)
            cold.wrapper.sync(stream)
            src = st.gateup_out(1) if group == "gateup" else st.down_out(1)
            src.to(device, non_blocking=True)

        with nvtx(f"{group}/combined_staged"):
            out_stats["combined_staged"] = graph_stats(
                combined_staged, reps, replays=replays)

    warm_bytes = sum(topk * warm[p].split.k_warm * warm[p].split.n_cols * 2
                     for p in projs)
    cold_bytes = sum(topk * cold_stat_rows(cold, p) * shape.n_cols(p) * 2
                     for p in projs)
    out_stats["shape"] = {
        "group": group,
        "warm": [warm[p].split.as_dict() for p in projs],
        "warm_bytes_dense": warm_bytes,
        "cold_bytes_dense": cold_bytes,
        "warm_keep_frac": round(warm[projs[0]].keep_frac, 4),
        "cold_keep_frac": round(cold.keep_frac[projs[0]], 4),
    }
    return out_stats


def cold_stat_rows(cold: ColdTier, proj: str) -> int:
    return int(cold.a_host[proj].shape[1])


# ─── 검증 ─────────────────────────────────────────────────────────────────
def check_masks(group: str, shape: Shape, warm: dict, cold: ColdTier, res,
                *, device, qlen_ptr: int) -> dict:
    """합성한 마스크로 계산한 레퍼런스와 두 커널의 출력을 대조한다.

    x ≡ 1이므로 레퍼런스는 "살아있는 행의 W 합"이다 — 마스킹이 빠지면(전부
    dense) sparsity만큼 값이 커져 즉시 드러난다.
    """
    projs = GROUPS[group]
    topk = shape.topk
    st = res.staging
    ids = torch.arange(topk, dtype=torch.int64).reshape(1, topk) % shape.experts
    ids_dev = ids.to(device)
    topk_w = torch.full((1, topk), 1.0 / topk, dtype=torch.float32, device=device)
    st.fill_expert_ids(ids)
    st.fill_topk_w(topk_w)
    report = {}

    # warm
    if group == "gateup":
        x = torch.ones(1, shape.hidden, dtype=torch.bfloat16, device=device)
        out = torch.zeros(1, topk, 2 * shape.inter, dtype=torch.bfloat16,
                          device=device)
        cols = {"gate": 0, "up": shape.inter}
        width = shape.inter
    else:
        x = torch.ones(topk, shape.inter, dtype=torch.bfloat16, device=device)
        out = torch.zeros(1, topk, shape.hidden, dtype=torch.bfloat16,
                          device=device)
        cols = {"down": 0}
        width = shape.hidden
    for proj in projs:
        warm[proj].launch(x, ids_dev, topk_w, out, cols[proj], masking=True)
    torch.cuda.synchronize()
    for proj in projs:
        tier = warm[proj]
        kw = tier.split.k_warm
        errs = []
        for j in range(topk):
            e = int(ids[0, j])
            keep = tier.a_host[e] > 0
            block = tier.w_flat[e * kw:(e + 1) * kw].float()
            ref = block[keep].sum(0)
            got = out[0, j, cols[proj]:cols[proj] + width].float().cpu()
            errs.append(float((got - ref).abs().max()
                              / ref.abs().max().clamp_min(1e-6)))
        report[f"warm_{proj}_max_rel_err"] = round(max(errs), 5)

    # cold
    submit = (cold.wrapper.submit_forward_gateup if group == "gateup"
              else cold.wrapper.submit_forward_down)
    if group == "gateup":
        st.fill_x(torch.ones(1, shape.hidden, dtype=torch.bfloat16))
        cold_in, cold_out = st.x_ptr(), st.partial_gateup_ptr()
    else:
        st.fill_act(torch.ones(1, topk, shape.inter, dtype=torch.bfloat16))
        cold_in, cold_out = st.act_ptr(), st.partial_down_ptr()
    submit(qlen_ptr, topk, st.expert_ids_ptr(), cold_in, cold_out, None,
           st.topk_w_ptr())
    cold.wrapper.sync(None)
    got_all = (st.gateup_out(1) if group == "gateup" else st.down_out(1)).float()
    for pi, proj in enumerate(projs):
        kc = cold_stat_rows(cold, proj)
        n = shape.n_cols(proj)
        blocks = cold.w[proj].float().reshape(shape.experts, n, kc)
        errs = []
        for j in range(topk):
            e = int(ids[0, j])
            keep = cold.a_host[proj][e] > 0
            ref = blocks[e][:, keep].sum(1)
            got = got_all[0, j, pi * n:(pi + 1) * n]
            errs.append(float((got - ref).abs().max()
                              / ref.abs().max().clamp_min(1e-6)))
        report[f"cold_{proj}_max_rel_err"] = round(max(errs), 5)
    return report


# ─── main ─────────────────────────────────────────────────────────────────
def footprint(shape: Shape, splits: dict) -> dict:
    E = shape.experts
    warm = sum(E * splits[p].k_warm * splits[p].n_cols * 2
               for p in ("gate", "up", "down"))
    cold = sum(E * splits[p].k_cold * splits[p].n_cols * 2
               for p in ("gate", "up", "down"))
    return {"warm_pinned_mb": round(warm / 1e6, 1),
            "cold_mb": round(cold / 1e6, 1),
            "note": "cold는 주입 텐서와 kt packed 사본이 동시에 사는 순간이 load "
                    "중 한 번 있다 (pack 완료 후 주입본 해제 — 계약 ③)"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shape_args(p)
    p.add_argument("--sparsity", type=float, default=0.9,
                   help="죽이는 페어의 비율 (0=dense, 0.9=90%% 스킵)")
    p.add_argument("--warm-frac", type=float, default=0.125,
                   help="K축에서 warm이 소유하는 비율")
    p.add_argument("--cold-frac", type=float, default=None,
                   help="cold 비율 (기본 1-warm_frac; 합이 1보다 작으면 차이가 hot)")
    p.add_argument("--numa-split", type=float, default=0.5,
                   help="cold N축에서 node 0의 몫")
    p.add_argument("--groups", default="gateup,down")
    p.add_argument("--mask-pattern", default="random", choices=("random", "block"))
    p.add_argument("--cpu-kernel", default="kt_tile_k2_bf16",
                   choices=("kt_tile_k2_bf16", "kt_amx_bf16"))
    p.add_argument("--threads", type=int, default=None,
                   help="CPUInfer 스레드 (기본 cpu_count//2-2, method.py와 같은 관례)")
    p.add_argument("--dense", action="store_true",
                   help="마스킹 없이 (sparse 커널 대신 dense 커널) — 절감 기준선")
    p.add_argument("--with-staging", action="store_true",
                   help="combined에 x/가중 D2H + partial H2D 포함 변형 추가")
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
    p.add_argument("--only", default="",
                   help="쉼표로 구분한 변형만 돈다 "
                        "(warm_only,cold_only,combined,combined_eager,"
                        "combined_eager_latency) — nsys 트레이스를 한 변형으로 "
                        "좁힐 때 쓴다")
    p.add_argument("--out")
    a = p.parse_args()

    shape = shape_from_args(a)
    cold_frac = 1.0 - a.warm_frac if a.cold_frac is None else a.cold_frac
    splits = {
        proj: make_split(shape, proj, a.warm_frac, cold_frac,
                         shuffle=a.shuffle_index, seed=a.seed)
        for proj in ("gate", "up", "down")
    }
    for sp in splits.values():
        if sp.k_cold % K_STEP or sp.k_warm % PAIR_GROUP:
            raise SystemExit(
                f"{sp.proj}: cold rows must be a K_STEP({K_STEP}) multiple and "
                f"warm rows even — got cold={sp.k_cold} warm={sp.k_warm}")
    if a.dry_run:
        emit({"bench": "warm_cold_sparse", "shape": shape.as_dict(),
              "splits": {k: v.as_dict() for k, v in splits.items()},
              "footprint": footprint(shape, splits)}, a.out)
        return

    device = select_device(a.device)
    warmup_jit()
    threads = a.threads or int(os.environ.get(
        "SGLANG_PRISM_CPUINFER_THREADS", max(2, (os.cpu_count() or 4) // 2 - 2)))
    warm_node = a.warm_node if a.warm_node is not None else gpu_numa_node(a.device)

    warm = {
        proj: WarmTier(shape, splits[proj], sparsity=a.sparsity,
                       pattern=a.mask_pattern, seed=a.seed, device=device,
                       node=warm_node)
        for proj in ("gate", "up", "down")
    }
    cold = ColdTier(shape, splits, sparsity=a.sparsity, pattern=a.mask_pattern,
                    seed=a.seed, numa_split=a.numa_split, threads=threads,
                    kernel_key=a.cpu_kernel)
    res = ExecutionResources(ResourceSpec(
        max_tokens=1, top_k=shape.topk, hidden_size=shape.hidden,
        intermediate_size=shape.inter, device=device))
    qlen_pin = torch.ones(1, dtype=torch.int32).pin_memory()

    results = {}
    for group in [g.strip() for g in a.groups.split(",") if g.strip()]:
        if group not in GROUPS:
            raise SystemExit(f"unknown group {group!r} (gateup|down)")
        results[group] = bench_group(
            group, shape, warm, cold, res, reps=a.reps, replays=a.replays,
            device=device, with_staging=a.with_staging, masking=not a.dense,
            qlen_ptr=qlen_pin.data_ptr(),
            only=frozenset(v.strip() for v in a.only.split(",") if v.strip())
            or None)
        if a.check:
            results[group]["check"] = check_masks(
                group, shape, warm, cold, res, device=device,
                qlen_ptr=qlen_pin.data_ptr())

    emit({
        "bench": "warm_cold_sparse",
        "kernels": {
            "warm": ("gemv_worklist_indexed_pinned_sparse" if not a.dense
                     else "gemv_worklist_indexed_pinned"),
            "cold": a.cpu_kernel,
        },
        "shape": shape.as_dict(),
        "params": {
            "sparsity": a.sparsity, "warm_frac": a.warm_frac,
            "cold_frac": cold_frac, "numa_split": a.numa_split,
            "mask_pattern": a.mask_pattern, "masking": not a.dense,
            "m": 1, "reps": a.reps, "replays": a.replays,
            "cpuinfer_threads": threads, "numa_nodes": cold.nodes,
            "warm_node": warm_node, "node_tables": cold.node_tables,
            "shuffle_index": a.shuffle_index, "seed": a.seed,
        },
        "splits": {k: v.as_dict() for k, v in splits.items()},
        "footprint": footprint(shape, splits),
        "results": results,
        "env": env_stamp(device),
    }, a.out)


if __name__ == "__main__":
    main()
