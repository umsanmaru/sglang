"""cold 티어만, GPU 없이 (프로파일 ③).

`warm_cold`의 cold 절반만 떼어낸 것이다. CUDA를 전혀 건드리지 않으므로
(a) GPU가 남에게 점유돼 있어도 돌고, (b) `perf`로 심볼·캐시 이벤트를 바로 볼 수
있다. 재는 것은 kt의 **동기** 진입점 `forward_{gateup,down}_partial` 한 번이고,
그것이 곧 prism의 cold 호출이다 (async 경로의 submit/sync 왕복 ~4 µs가 빠진다).

원인 규명용 개입 플래그 넷 — 2026-08-26에 cold 비용의 77%가
`expert_num × activated_expert`에 비례하는 것을 이걸로 갈랐다:

  sweep(experts=[...])  A 고정, E만 훑는다. 중첩 루프면 E에 **선형**,
                        메모리면 캐시 경계에서 **무릎**.
  band=True             cold 인덱스를 밴드 퇴화형으로 준다 (row_off/idx를 비우고
                        offset/rows만) → kt의 `dense_rows`가 zero-copy로 떨어져
                        gather가 사라진다.
  fixed_ids=True        매 스텝 같은 expert 집합. E 의존이 캐시 잔존율이면
                        여기서 붕괴하고, 구조적이면 그대로 남는다.
  split_index=True      gate ≠ up 인덱스 → `dual_pack()`의 `idx ==` 벡터 비교가
                        첫 원소에서 조기 종료한다. 그 대가로 activation을 두 번
                        pack하므로 **일이 늘어나는데도** 총시간이 줄면 비교
                        자체가 비용이었다는 확증이다.

그 네 개입이 지목한 범인은 `dual_pack()`이었고 (`same_as` → `std::vector` 전체
비교, 활성 expert 루프 안에서 458 KB memcmp), kt `eb780a4`에서 생성 시점에
굳히는 것으로 수정됐다.

**sparsity 기본값이 0.9인 이유**: 1.0은 W 바이트를 0으로 만들어 준비 작업만
남기는 **진단용** 값이다. 모르고 쓰면 cold 비용이 아닌 숫자를 cold 비용으로
읽게 되므로 기본값으로 두지 않는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch

from sglang.srt.layers.moe.prism.profile.common import (
    GRID,
    SparseGemv,
    K_STEP,
    NG,
    PMAX,
    PROJS,
    RENORM_IT,
    SPARSITY_LAM,
    SPARSITY_P,
    Shape,
    Timing,
    default_cpuinfer_threads,
    env_stamp,
    numa_nodes,
    sparse_tables,
    split_rows,
    tier_index,
)
from sglang.srt.layers.moe.prism.profile.common import store_of
from sglang.srt.layers.moe.prism.profile.warm_cold import N_ALIGN, node_table


@dataclass(frozen=True)
class ColdCpuReport:
    experts: int
    topk: int
    k_cold: Mapping[str, int]
    keep_frac: float
    band: bool
    fixed_ids: bool
    split_index: bool
    numa_nodes: int
    timing: Timing
    iters: int

    @property
    def us(self) -> float:
        return self.timing.us

    def as_dict(self) -> dict:
        d = dict(self.timing.as_dict())
        d.update(experts=self.experts, topk=self.topk, k_cold=dict(self.k_cold),
                 keep_frac=self.keep_frac, band=self.band,
                 fixed_ids=self.fixed_ids, split_index=self.split_index,
                 numa_nodes=self.numa_nodes, iters=self.iters)
        return d


class ColdCpuProfiler:
    """kt partial 인스턴스 하나를 CUDA 없이 만들고 동기 호출을 잰다.

    생성이 비싸다 (weight 패킹). `close()` 또는 with 문으로 해제한다.
    """

    def __init__(self, shape: Shape, *, cold_frac: float = 0.875,
                 sparsity: float = 0.9, proj: str = "gateup",
                 band: bool = False, split_index: bool = False,
                 mask_pattern: str = "random", numa_split: float = 0.5,
                 threads: Optional[int] = None,
                 cpu_kernel: Optional[str] = None, dtype: str = "bf16",
                 seed: int = 0, numa_map: Optional[Sequence[int]] = None):
        from kt_kernel import kt_kernel_ext
        from kt_kernel.experts_partial import PartialMoEWrapper

        if proj not in ("gateup", "down"):
            raise ValueError(f"proj must be gateup|down, got {proj!r}")
        # dtype이 백엔드(kt 커널 키)와 K 정렬을 정한다 — cpu_kernel은 그 안에서만 고른다.
        self.store = store_of(dtype)
        cpu_kernel = cpu_kernel or self.store.cpu_kernel
        if cpu_kernel not in N_ALIGN:
            raise ValueError(f"unknown cpu kernel {cpu_kernel!r}")
        if cpu_kernel not in self.store.cpu_kernels:
            raise ValueError(f"cpu kernel {cpu_kernel!r} cannot consume a "
                             f"{self.store.name} cold store "
                             f"(compatible: {list(self.store.cpu_kernels)})")
        self.cpu_kernel = cpu_kernel
        self.shape = shape
        self.proj = proj
        self.band = band
        self.split_index = split_index
        self.sparsity = sparsity
        self.threads = threads or default_cpuinfer_threads()
        # numa_map: 실모델의 SGLANG_PRISM_NUMA_MAP과 같은 경로. shard table 길이가
        # tp_count(=subpool 수)와 같아야 하므로 nodes도 같이 좁힌다.
        self.numa_map = list(numa_map) if numa_map else None
        self.nodes = len(self.numa_map) if self.numa_map else numa_nodes()

        E, topk = shape.experts, shape.topk
        H, I = shape.hidden, shape.inter
        self.k_cold = {}
        step = self.store.rows_step()
        for p in PROJS:
            kc = split_rows(shape.k_axis(p), cold_frac, step=step)
            if kc % step:
                raise ValueError(f"{p}: cold rows {kc} not a multiple of {step} "
                                 f"({self.store.name} 배율 블록)")
            self.k_cold[p] = kc

        if self.numa_map:
            pool_cfg = kt_kernel_ext.WorkerPoolConfig()
            pool_cfg.subpool_count = len(self.numa_map)
            pool_cfg.subpool_numa_map = list(self.numa_map)
            pool_cfg.subpool_thread_count = [
                self.threads // len(self.numa_map)
                + (1 if i < self.threads % len(self.numa_map) else 0)
                for i in range(len(self.numa_map))
            ]
            self.cpuinfer = kt_kernel_ext.CPUInfer(pool_cfg)
        else:
            self.cpuinfer = kt_kernel_ext.CPUInfer(self.threads)
        cfg = kt_kernel_ext.moe.MOEConfig(E, topk, H, I, 0)
        cfg.max_len = 1
        cfg.layer_idx = 0
        cfg.partial.enabled = True
        cfg.partial.n_total = I
        for p, ki in (("gate", cfg.partial.gate), ("up", cfg.partial.up),
                      ("down", cfg.partial.down)):
            axis, kc = shape.k_axis(p), self.k_cold[p]
            if band:
                # 퇴화형: 전 expert가 같은 연속 밴드. kt가 gather를 건너뛴다.
                ki.offset = axis - kc          # 축 끝에 붙인 밴드
                ki.rows = kc
            else:
                ki.row_off = [e * kc for e in range(E + 1)]
                sd = seed + (7919 if (split_index and p == "up") else 0)
                ki.idx = tier_index(axis, kc, skip=axis - kc, seed=sd) \
                    .to(torch.int32).repeat(E).tolist()

        align = N_ALIGN[cpu_kernel]
        gu_off, gu_rows, gu_frac = node_table(I, numa_split, self.nodes, align)
        dn_off, dn_rows, _ = node_table(H, numa_split, self.nodes, align)
        # 요청값과 실현값을 둘 다 들고 있는다 — 정렬 반올림이 이 둘을 벌린다.
        self.numa_split = numa_split
        self.node_gateup_rows = tuple(gu_rows)
        self.node_gateup_frac = gu_frac
        self.node_down_rows = tuple(dn_rows)
        cfg.partial.node_gateup_n_offset = gu_off
        cfg.partial.node_gateup_n_rows = gu_rows
        cfg.partial.node_down_n_offset = dn_off
        cfg.partial.node_down_n_rows = dn_rows

        sp = cfg.partial.sparsity
        sp.pmax, sp.grid, sp.ng, sp.renorm_it = PMAX, GRID, NG, RENORM_IT
        sp.p_gate, sp.lam_gate = SPARSITY_P, SPARSITY_LAM
        sp.p_up, sp.lam_up = SPARSITY_P, SPARSITY_LAM
        sp.p_down, sp.lam_down = SPARSITY_P, SPARSITY_LAM
        cfg.pool = self.cpuinfer.backend_

        weights, scales, tables, keep = {}, {}, {}, {}
        for p in PROJS:
            n = shape.n_cols(p)
            weights[p], scales[p] = self.store.cold_store(
                E, n, self.k_cold[p], seed=seed + hash(p) % 97)
            a, c, thr, frac = sparse_tables(E, self.k_cold[p], sparsity,
                                            pattern=mask_pattern, seed=seed)
            tables[f"{p}_wn_sq"] = a
            tables[f"{p}_pair_dot"] = c
            tables[f"thr_{p}"] = thr
            keep[p] = frac
        self.keep_frac = keep

        self.wrapper = PartialMoEWrapper(cfg, self.cpuinfer, kernel_key=cpu_kernel)
        scale_kw = ({} if scales["gate"] is None else
                    dict(gate_scale=scales["gate"], up_scale=scales["up"],
                         down_scale=scales["down"]))
        self.wrapper.load_weights_from_tensors(
            weights["gate"], weights["up"], weights["down"],
            sparsity_tables=tables, **scale_kw)

    def close(self) -> None:
        self.wrapper = None
        self.cpuinfer = None

    def __enter__(self) -> "ColdCpuProfiler":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def measure(self, *, iters: int = 100, replays: int = 8,
                fixed_ids: bool = False, seed: int = 1234) -> ColdCpuReport:
        shape, topk = self.shape, self.shape.topk
        E = shape.experts
        g = torch.Generator().manual_seed(seed)
        if fixed_ids:
            # 판별자: 풀은 E개인데 매 스텝 **같은** top_k만 쓴다.
            one = torch.randperm(E, generator=g)[:topk].to(torch.int64).contiguous()
            ids = [one] * iters
        else:
            ids = [torch.randperm(E, generator=g)[:topk].to(torch.int64).contiguous()
                   for _ in range(iters)]
        w = torch.full((1, topk), 1.0 / topk, dtype=torch.float32)

        if self.proj == "gateup":
            x = torch.ones(1, shape.hidden, dtype=torch.bfloat16)
            out = torch.zeros(1, topk, 2 * shape.inter, dtype=torch.bfloat16)
            call = self.wrapper.forward_gateup
        else:
            x = torch.ones(1, topk, shape.inter, dtype=torch.bfloat16)
            out = torch.zeros(1, topk, shape.hidden, dtype=torch.bfloat16)
            call = self.wrapper.forward_down

        def step(i: int) -> None:
            call(ids[i].reshape(1, topk), x, out, w)

        for i in range(min(20, iters)):   # warmup
            step(i)
        per = []
        for _ in range(replays):
            t0 = time.perf_counter()
            for i in range(iters):
                step(i)
            per.append((time.perf_counter() - t0) / iters * 1e6)

        return ColdCpuReport(
            experts=E, topk=topk, k_cold=dict(self.k_cold),
            keep_frac=self.keep_frac["gate"], band=self.band,
            fixed_ids=fixed_ids, split_index=self.split_index,
            numa_nodes=self.nodes, timing=Timing.of(per), iters=iters,
        )


def cold_cpu(shape: Shape, *, iters: int = 100, replays: int = 8,
             fixed_ids: bool = False, **kw) -> ColdCpuReport:
    """한 번 쓰고 버리는 편의 함수."""
    with ColdCpuProfiler(shape, **kw) as prof:
        return prof.measure(iters=iters, replays=replays, fixed_ids=fixed_ids)


def cold_cpu_sweep(shape: Shape, experts: Sequence[int], *, iters: int = 100,
                   replays: int = 8, fixed_ids: bool = False, **kw) -> dict:
    """A(top_k)와 치수를 고정하고 expert 풀 크기만 훑는다 — E 의존의 판별자.

    반환값은 CLI가 쓰는 리포트 dict다 (`results`가 E별 항목의 리스트).
    """
    results = []
    for E in experts:
        rep = cold_cpu(shape.replace(experts=E), iters=iters, replays=replays,
                       fixed_ids=fixed_ids, **kw)
        results.append(rep.as_dict())
    return {
        "bench": "cold_cpu",
        "kernel": kw.get("cpu_kernel") or store_of(kw.get("dtype", "bf16")).cpu_kernel,
        "shape": shape.as_dict(),
        "params": {
            "cold_frac": kw.get("cold_frac", 0.875),
            "sparsity": kw.get("sparsity", 0.9),
            "proj": kw.get("proj", "gateup"),
            "band": kw.get("band", False),
            "fixed_ids": fixed_ids,
            "split_index": kw.get("split_index", False),
            "mask_pattern": kw.get("mask_pattern", "random"),
            "numa_split": kw.get("numa_split", 0.5),
            "cpuinfer_threads": kw.get("threads") or default_cpuinfer_threads(),
            "numa_map": list(kw["numa_map"]) if kw.get("numa_map") else None,
            "iters": iters, "replays": replays, "seed": kw.get("seed", 0),
        },
        "results": results,
        "env": env_stamp(None),
    }


def cold_sparse_gemv(k: int, n: int, sparsity: float = 0.9, *, iters: int = 100,
                     replays: int = 8, mask_pattern: str = "random",
                     numa_split: float = 0.5, threads: Optional[int] = None,
                     cpu_kernel: Optional[str] = None, dtype: str = "bf16", seed: int = 0,
                     numa_map: Optional[Sequence[int]] = None) -> SparseGemv:
    """[k, n] weight 하나의 cold sparse GEMV — shape과 sparsity만 받는다. CUDA 불필요.

        cold_sparse_gemv(1792, 768, 0.9).us

    `warm_sparse_gemv`의 CPU 짝이다: expert/top_k를 1로 접고 kt의 동기 진입점
    `forward_gateup_partial`을 부른다 (`cold_backend`가 부르는 그 경로). s → thr →
    점수 → 마스크 → masked GEMV가 전부 C++ 안에서 끝난다.

    **주의 — 이 축약은 warm보다 cold에 훨씬 위험하다.** cold 비용에는 **활성
    expert당 ~3.9 µs** 항이 있어서(버퍼 carve + BufferA pack + pair mask + plan
    인코딩) top_k를 8에서 1로 접으면 그만큼이 사라진다. 실측: 같은 바이트에서
    cold gateup이 260.7 → 36.4 µs였다 (kt eb780a4 이전 측정). 커널 상한을 볼 때만
    쓰고, 티어 비용은 `ColdCpuProfiler`/`WarmColdProfiler`로 잰다.

    `n`은 커널의 N 정렬을 지켜야 한다 — tile_k2는 노드당 256의 배수를 요구하므로
    작은 `n`은 `cpu_kernel="kt_amx_bf16"`(32) 또는 `numa_map=[0]`을 쓴다.
    """
    store = store_of(dtype)
    if k % store.rows_step():
        raise ValueError(f"{store.name}: k must be a multiple of {store.rows_step()}, got {k}")
    shape = Shape(experts=1, topk=1, hidden=k, inter=n)
    with ColdCpuProfiler(shape, cold_frac=1.0, sparsity=sparsity, proj="gateup",
                         mask_pattern=mask_pattern, numa_split=numa_split,
                         threads=threads, cpu_kernel=cpu_kernel, dtype=store,
                         seed=seed, numa_map=numa_map) as prof:
        rep = prof.measure(iters=iters, replays=replays)
        rows = prof.node_gateup_rows   # proj="gateup" 고정
    return SparseGemv(where="cold", k_rows=k, n_cols=n, sparsity=sparsity,
                      keep_frac=rep.keep_frac,
                      dense_bytes=store.store_bytes(1, k, n),
                      timing=rep.timing, numa_split=numa_split,
                      node_rows=rows)
