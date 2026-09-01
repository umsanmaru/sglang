"""full-layer 프로파일 (프로파일 ④) — 세 티어를 **실제 구성으로 동시에** 돌린 한
레이어 decode 시간.

①(hot)·②(warm+cold)는 티어를 따로 재고, planner는 그 값들을 "expert 하나·proj
하나"로 정규화한 뒤 활성 expert 수만큼 곱해 더한다. 이 모듈은 그 근사가 실제와
얼마나 벌어지는지 보려고 **근사하지 않은 값**을 만든다: hot·warm·cold가 한
iteration 안에서 같이 돌고, GPU 두 티어는 한 CUDA graph에 담기고, cold는
executor와 같은 방식으로(submit → GPU 일 → sync) 겹친다.

실제와 맞춘 것 (프로파일 ①②가 균일로 단순화했던 것들):

  · **expert마다 다른 티어 경계** — 중요도 곡선이 expert마다 다르므로 실제 plan의
    hot/warm/cold 행 수는 expert마다 다르다. `hot_spread`/`warm_spread`로 준다
    (행 수는 스토어 정렬의 배수로만 잘린다 — fp8은 128).
  · **expert마다 다른 sparsity** — 라우터 분포가 expert마다 다르다.
    `sparsity_spread`로 준다 (평균은 정확히 `sparsity`).
  · **iteration마다 다른 top_k** — 같은 expert를 반복하면 W가 캐시에 남는다.
  · **hot의 L2 flush** — 측정으로 확인한 것: hot 스토어(E=128, fp8, gateup 151 MB)가
    GPU L2 96 MB를 겨우 넘어서, E=128에서도 시간이 아직 saturate하지 않는다
    (E=128 10.33 µs → E=256 11.41 µs). 실제로는 이 레이어가 수십 레이어 뒤에 다시
    오므로 상주가 0이다. 그래서 iteration마다 L2보다 큰 버퍼를 훑고 **차분**한다.
    warm은 flush가 필요 없다 — host 상주 W 읽기는 GPU L2에 캐시되지 않는다
    (sp=0.0에서 풀을 4 MB→134 MB로 키워도 90.6 µs로 불변). cold는 GPU flush의
    대상이 아니고(CPU 캐시라) 회전에 기댄다 — 회전이 실제로 크게 먹는다는 것은
    실측했다(같은 top_k 반복 52.5 µs vs 회전 86.4 µs, **+64%**). 다만 회전만으로
    L3 상주가 **전부** 사라지는지는 아직 검증하지 않았다 — 그것을 보려면 CPU측
    flush(dense `forward_down`을 타이밍 밖에서 여러 번)가 필요하다.

**한계**: 이 프로파일은 실사용 계획기가 될 수 없다 — 구성 하나마다 스토어를 굽고
kt 인스턴스를 만들어야 하므로 경우의 수를 훑을 수 없다. 근사의 오차를 보는
계측기다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch

from sglang.srt.layers.moe.prism.profile.common import (
    GRID,
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
    graph_timing,
    host_timing,
    numa_nodes,
    nvtx,
    select_device,
    sparse_tables,
    split_rows_varied,
    spread_values,
    store_of,
    tier_indices,
)
from sglang.srt.layers.moe.prism.profile.warm_cold import (
    ColdTier,
    N_ALIGN,
    Split,
)

GROUPS = {"gateup": ("gate", "up"), "down": ("down",)}


# ─── 티어 분할 ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ProjSplit:
    """한 proj의 K축 3티어 분할. 행 수는 expert마다 다르고, 세 티어는 expert별로
    서로소다 (같은 순열의 앞·중간·뒤를 나눠 갖는다)."""

    proj: str
    k_axis: int
    n_cols: int
    hot: tuple              # expert당 행 수
    warm: tuple
    cold: tuple
    hot_idx: torch.Tensor   # int32 [Σₑ hot(e)] — expert 블록 이어붙인 K축 행 번호
    warm_idx: torch.Tensor
    cold_idx: torch.Tensor

    def rows(self, tier: str) -> tuple:
        return {"hot": self.hot, "warm": self.warm, "cold": self.cold}[tier]

    def idx(self, tier: str) -> torch.Tensor:
        return {"hot": self.hot_idx, "warm": self.warm_idx,
                "cold": self.cold_idx}[tier]

    def as_dict(self) -> dict:
        E = len(self.hot)
        d = {"proj": self.proj, "k_axis": self.k_axis, "n_cols": self.n_cols}
        for tier in ("hot", "warm", "cold"):
            r = self.rows(tier)
            d[tier] = {"frac": round(sum(r) / (E * self.k_axis), 4),
                       "min": min(r), "max": max(r),
                       "mean_rows": round(sum(r) / E, 1)}
        return d


def build_splits(shape: Shape, *, hot_frac: float, warm_frac: float,
                 cold_frac: Optional[float] = None, hot_spread: float = 0.0,
                 warm_spread: float = 0.0, cold_spread: float = 0.0,
                 dtype: str = "bf16", shuffle: bool = False,
                 seed: int = 0) -> dict:
    """proj별 `ProjSplit`. `cold_frac=None`이면 hot·warm이 안 가진 행 전부가 cold다.

    세 티어에 **같은 `seed`** 로 `tier_indices`를 부르고 `skip`을 누적해서 준다 —
    expert e의 순열 하나에서 앞을 hot, 그 다음을 warm, 그 다음을 cold가 갖는다.
    `skip`이 expert마다 다른 것이 핵심이다 (앞 티어의 행 수가 expert마다 다르므로).
    """
    store = store_of(dtype)
    step = store.rows_step()
    E = shape.experts
    out = {}
    for proj in PROJS:
        axis = shape.k_axis(proj)
        hot = split_rows_varied(axis, hot_frac, E, spread=hot_spread,
                                step=step, seed=seed + 11)
        warm = split_rows_varied(axis, warm_frac, E, spread=warm_spread,
                                 step=step, seed=seed + 22)
        if cold_frac is None:
            cold = tuple(axis - h - w for h, w in zip(hot, warm))
        else:
            cold = split_rows_varied(axis, cold_frac, E, spread=cold_spread,
                                     step=step, seed=seed + 33)
        for e, (h, w, c) in enumerate(zip(hot, warm, cold)):
            if h + w + c > axis:
                raise ValueError(
                    f"{proj} expert {e}: hot {h} + warm {w} + cold {c} > axis {axis} "
                    f"(step={step} 반올림 후) — frac 합을 줄이거나 spread를 줄여라")
            if c < 0:
                raise ValueError(f"{proj} expert {e}: cold rows {c} < 0")
        skip_w = tuple(hot)
        skip_c = tuple(h + w for h, w in zip(hot, warm))
        hot_idx, _ = tier_indices(axis, hot, E, skip=0, shuffle=shuffle, seed=seed)
        warm_idx, _ = tier_indices(axis, warm, E, skip=skip_w, shuffle=shuffle,
                                   seed=seed)
        cold_idx, _ = tier_indices(axis, cold, E, skip=skip_c, shuffle=shuffle,
                                   seed=seed)
        out[proj] = ProjSplit(proj=proj, k_axis=axis, n_cols=shape.n_cols(proj),
                              hot=hot, warm=warm, cold=cold, hot_idx=hot_idx,
                              warm_idx=warm_idx, cold_idx=cold_idx)
    return out


# ─── GPU 티어 (hot = device dense, warm = pinned sparse) ───────────────────
class GpuTier:
    """한 proj의 GPU 티어. hot과 warm은 거처(`node`)와 masking만 다르고 나머지
    계약이 같다 — `tiers.py`의 두 구현이 그렇게 갈리는 것과 같은 축이다."""

    def __init__(self, shape: Shape, split: ProjSplit, tier: str, *,
                 sparsity=0.0, pattern: str = "random", device,
                 node: Optional[int] = None, dtype: str = "bf16",
                 seed: int = 0):
        from sglang.srt.layers.moe.prism.tiers import SparseSpec

        E = shape.experts
        self.store = store_of(dtype)
        self.tier = tier
        self.proj = split.proj
        self.rows = split.rows(tier)
        self.n_cols = split.n_cols
        self.masking = tier == "warm"
        self.pinned = node is not None
        # 스토어: hot은 device 상주, warm은 pinned + NUMA 바인딩 (계약 ③).
        self.parts = self.store.gpu_store(
            E, self.rows, self.n_cols, device=(None if self.pinned else device),
            node=node, seed=seed)
        off = torch.zeros(E + 1, dtype=torch.int32)
        off[1:] = torch.tensor(self.rows, dtype=torch.int32).cumsum(0)
        self.row_off = off.to(device)
        self.k_index = split.idx(tier).to(torch.uint16).contiguous().to(device)
        self.keep_frac = 1.0
        self.spec = None
        self.a_host = None
        if self.masking:
            a, c, thr, self.keep_frac = sparse_tables(
                E, self.rows, sparsity, pattern=pattern, seed=seed)
            self.a_host = a           # check의 레퍼런스가 마스크로 쓴다
            self.spec = SparseSpec(a=a.to(device), c=c.to(device), thr=thr.to(device),
                                   p=SPARSITY_P, lam=SPARSITY_LAM, pmax=PMAX,
                                   grid=GRID, ng=NG, renorm_it=RENORM_IT)

    @property
    def total_rows(self) -> int:
        return sum(self.rows)

    def bytes_per_launch(self, topk: int) -> int:
        """활성 expert `topk`개가 읽는 바이트 (마스킹 전). 행 수가 expert마다
        다르므로 평균 행 수로 센다 — 어느 expert가 뽑히는지는 iteration마다 다르다."""
        mean = sum(self.rows) / len(self.rows)
        return int(topk * mean * self.n_cols * self.store.elem_bytes)

    def launch(self, x, ids, tw, out, col_off: int, *,
               x_row_is_pair: bool) -> None:
        fn = self.store.fmt.gemv(pinned=self.pinned, sparse=self.masking)
        stream = torch.cuda.current_stream()
        if self.masking:
            self.store.call(fn, (x, ids, tw, *self.parts, self.row_off,
                                 self.k_index, out, self.spec, col_off,
                                 x_row_is_pair, stream))
        else:
            self.store.call(fn, (x, ids, *self.parts, self.row_off, self.k_index,
                                 out, col_off, x_row_is_pair, stream))

    @staticmethod
    def launch_gateup(gate: "GpuTier", up: "GpuTier", x, ids, tw, out,
                      inter: int) -> None:
        """gate와 up을 한 커널로 — executor가 그렇게 발행한다 (`tiers.ResidentGateUp`,
        `tiers.SparsePinnedGateUp`). 두 번 launch하면 시스템보다 비싼 값이 나온다."""
        store = gate.store
        fn = store.fmt.gemv_gateup(pinned=gate.pinned, sparse=gate.masking)
        if fn is None:                      # 그 조합의 융합 진입점이 없으면 2 launch
            gate.launch(x, ids, tw, out, 0, x_row_is_pair=False)
            up.launch(x, ids, tw, out, inter, x_row_is_pair=False)
            return
        stream = torch.cuda.current_stream()
        if gate.masking:
            args = (x, ids, tw, *gate.parts, gate.row_off, gate.k_index,
                    *up.parts, up.row_off, up.k_index, out, gate.spec, up.spec,
                    0, inter, False, stream)
        else:
            args = (x, ids, *gate.parts, gate.row_off, gate.k_index,
                    *up.parts, up.row_off, up.k_index, out, 0, inter, False, stream)
        store.call(fn, args)


# ─── L2 flush ──────────────────────────────────────────────────────────────
class L2Flush:
    """iteration마다 GPU L2를 씻는다 — CUDA graph에 캡처되는 D2D 복사 하나.

    L2보다 큰 버퍼를 통째로 복사하면 읽기·쓰기 양쪽으로 L2를 덮으므로 앞
    iteration의 W 라인이 남지 않는다. flush 자체가 iteration 시간에 섞이니
    `graph(flush+work)`와 `graph(flush)`를 각각 재서 **차분**해야 한다
    (`FullLayerProfiler._timed`가 그렇게 한다).
    """

    def __init__(self, device, mb: Optional[int] = None):
        l2 = torch.cuda.get_device_properties(device).L2_cache_size
        want = int(mb * 2 ** 20) if mb else int(1.25 * l2)
        n = max(1, want // 4)
        # **쓰기 전용**이다: 쓰기도 L2 라인을 할당하므로 L2보다 큰 영역을 0으로
        # 채우기만 해도 앞 iteration의 W가 전부 밀려난다. 복사(읽기+쓰기)는 같은
        # 축출에 2배의 시간을 쓰고, 그 시간이 차분의 잡음이 된다 (실측: 복사
        # 288 MB 197 µs vs 쓰기 120 MB ~60 µs — 재는 값이 한 자리 µs다).
        self.dst = torch.empty(n, dtype=torch.float32, device=device)
        self.mb = round(n * 4 / 2 ** 20, 1)
        self.l2_mb = round(l2 / 2 ** 20, 1)

    def __call__(self) -> None:
        self.dst.zero_()


# ─── 리포트 ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LayerGroupReport:
    group: str
    timings: Mapping[str, Timing]
    info: dict

    def us(self, name: str) -> float:
        return self.timings[name].us

    def as_dict(self) -> dict:
        d = {k: v.as_dict() for k, v in self.timings.items()}
        d["info"] = self.info
        return d


# ─── 프로파일러 ────────────────────────────────────────────────────────────
class FullLayerProfiler:
    """hot + warm + cold를 한 번에 굽고 group(gateup|down)별로 잰다.

    생성이 비싸다 (세 티어의 스토어 + kt weight 패킹). `close()` 또는 with 문.
    """

    def __init__(self, shape: Shape, *, hot_frac: float = 0.375,
                 warm_frac: float = 0.125, cold_frac: Optional[float] = None,
                 sparsity: float = 0.5, sparsity_spread: float = 0.3,
                 hot_spread: float = 0.0, warm_spread: float = 0.0,
                 cold_spread: float = 0.0, dtype: str = "bf16", device=0,
                 numa_split: float = 0.5, mask_pattern: str = "random",
                 cpu_kernel: Optional[str] = None, threads: Optional[int] = None,
                 numa_map: Optional[Sequence[int]] = None,
                 warm_node: Optional[int] = None, shuffle_index: bool = False,
                 flush_mb: Optional[int] = None, seed: int = 0):
        from sglang.srt.layers.moe.prism.numa import gpu_numa_node
        from sglang.srt.layers.moe.prism.resources import (
            ExecutionResources,
            ResourceSpec,
        )

        self.shape = shape
        self.device = select_device(device)
        self.store = store_of(dtype)
        self.store.fmt.warmup()          # JIT 컴파일을 캡처 밖으로
        E = shape.experts
        cpu_kernel = cpu_kernel or self.store.cpu_kernel
        self.threads = threads or default_cpuinfer_threads()
        self.warm_node = (warm_node if warm_node is not None
                          else gpu_numa_node(self.device.index or 0))
        # expert마다 다른 sparsity — 평균은 정확히 `sparsity`다 (대칭 쌍 + 순열).
        self.sparsity = spread_values(sparsity, E, spread=sparsity_spread,
                                      seed=seed + 44)
        self.splits = build_splits(
            shape, hot_frac=hot_frac, warm_frac=warm_frac, cold_frac=cold_frac,
            hot_spread=hot_spread, warm_spread=warm_spread,
            cold_spread=cold_spread, dtype=self.store, shuffle=shuffle_index,
            seed=seed)

        self.hot = {p: GpuTier(shape, self.splits[p], "hot", device=self.device,
                               dtype=self.store, seed=seed + 1) for p in PROJS}
        self.warm = {p: GpuTier(shape, self.splits[p], "warm",
                                sparsity=self.sparsity, pattern=mask_pattern,
                                device=self.device, node=self.warm_node,
                                dtype=self.store, seed=seed + 2) for p in PROJS}
        # cold는 `ColdTier`가 굽는다 (kt config 배관을 한 곳에 둔다). 가변 행 수는
        # rows/index를 따로 넘겨서 준다 — kt는 row_off[-1] 기준으로 검증한다.
        cold_splits = {p: Split(proj=p, axis=self.splits[p].k_axis,
                                n_cols=self.splits[p].n_cols,
                                warm_rows=self.splits[p].warm_idx,
                                cold_rows=self.splits[p].cold_idx)
                       for p in PROJS}
        self.cold = ColdTier(
            shape, cold_splits, sparsity=self.sparsity, pattern=mask_pattern,
            seed=seed, numa_split=numa_split, threads=self.threads,
            kernel_key=cpu_kernel, dtype=self.store, numa_map=numa_map,
            rows_per_expert={p: self.splits[p].cold for p in PROJS},
            index_per_expert={p: self.splits[p].cold_idx for p in PROJS})

        self.res = ExecutionResources(ResourceSpec(
            max_tokens=1, top_k=shape.topk, hidden_size=shape.hidden,
            intermediate_size=shape.inter, device=self.device))
        self._qlen = torch.ones(1, dtype=torch.int32).pin_memory()
        self.flush = L2Flush(self.device, flush_mb)
        self.params = {
            "dtype": self.store.name, "m": 1,
            "hot_frac": hot_frac, "warm_frac": warm_frac, "cold_frac": cold_frac,
            "hot_spread": hot_spread, "warm_spread": warm_spread,
            "sparsity_mean": round(sum(self.sparsity) / E, 6),
            "sparsity_min": round(min(self.sparsity), 4),
            "sparsity_max": round(max(self.sparsity), 4),
            "sparsity_spread": sparsity_spread,
            "mask_pattern": mask_pattern, "cpu_kernel": cpu_kernel,
            "cpuinfer_threads": self.threads, "numa_split": numa_split,
            "numa_nodes": self.cold.nodes, "numa_map": list(numa_map) if numa_map else None,
            "warm_node": self.warm_node, "node_tables": self.cold.node_tables,
            "flush_mb": self.flush.mb, "l2_mb": self.flush.l2_mb,
            "shuffle_index": shuffle_index, "seed": seed,
        }

    # ── 수명 ─────────────────────────────────────────────────────────────
    def close(self) -> None:
        self.hot = self.warm = {}
        self.cold = None
        self.res = None
        self.flush = None
        torch.cuda.empty_cache()

    def __enter__(self) -> "FullLayerProfiler":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def footprint(self) -> dict:
        E = self.shape.experts
        f = {}
        for tier, tiers in (("hot_device", self.hot), ("warm_pinned", self.warm)):
            f[f"{tier}_mb"] = round(sum(
                self.store.store_bytes(E, t.rows, t.n_cols)
                for t in tiers.values()) / 1e6, 1)
        f["cold_mb"] = round(sum(
            self.store.store_bytes(E, self.cold.rows[p], self.splits[p].n_cols)
            for p in PROJS) / 1e6, 1)
        return f

    # ── 측정 ─────────────────────────────────────────────────────────────
    def _ids(self, reps: int, seed: int = 1234) -> tuple:
        """iteration마다 다른 top_k (M=1). staging이 int64를 받으므로 dtype을 맞춘다."""
        g = torch.Generator().manual_seed(seed)
        host = [torch.randperm(self.shape.experts, generator=g)[: self.shape.topk]
                .to(torch.int64).reshape(1, self.shape.topk).contiguous()
                for _ in range(reps)]
        return host, [t.to(self.device) for t in host]

    def _timed(self, fn, reps: int, replays: int, flush: bool,
               base: Optional[Timing] = None) -> Timing:
        """flush를 걸면 `graph(flush+work) − graph(flush)`로 차분한다.

        중앙값끼리 빼는 것이라 분산은 보존되지 않는다 — min/max/p90도 같은 방식으로
        빼서 남기되, 대표값은 어디까지나 두 중앙값의 차다."""
        if not flush:
            return graph_timing(fn, reps, replays=replays)

        def with_flush(i: int) -> None:
            self.flush()
            fn(i)

        work = graph_timing(with_flush, reps, replays=replays)
        b = base if base is not None else self.flush_timing(reps, replays)
        return Timing(us=round(work.us - b.us, 3),
                      min_us=round(work.min_us - b.min_us, 3),
                      max_us=round(work.max_us - b.max_us, 3),
                      p90_us=round(work.p90_us - b.p90_us, 3),
                      replays=work.replays)

    def flush_timing(self, reps: int = 100, replays: int = 20) -> Timing:
        with nvtx("full/flush"):
            return graph_timing(lambda i: self.flush(), reps, replays=replays)

    def measure(self, group: str = "gateup", *, reps: int = 50,
                replays: int = 10, flush: bool = True, rounds: int = 3,
                only: Optional[Sequence[str]] = None) -> LayerGroupReport:
        """group의 변형들을 잰다.

        변형:
          hot_only / warm_only  — GPU 티어 하나씩 (flush 차분)
          gpu_only              — hot + warm (executor가 decode에서 한 스트림에 발행)
          cold_only             — kt 동기 호출만 (host 루프; CUDA graph에 담을 수 없다)
          cold_graph            — graph 안의 cold만 (겹침 판정의 기준선)
          combined              — cold submit → GPU 두 티어 → cold sync (executor의 decode 형태)
          combined_split        — cold를 곁 스트림으로 뺀 것 (가짜 의존 제거)
          layer / layer_split   — 각각 + cold partial H2D + rejoin (phase 전체)

        `rounds`는 **변형 목록 전체를 그만큼 반복**하고 변형별 중앙값을 쓴다. 한
        변형을 연속으로 반복하지 않고 라운드를 바깥에 두는 이유는 드리프트다 —
        공유 머신에서 같은 측정이 191 vs 101 µs로 튄 적이 있고, 그때 변형별로
        몰아 재면 그 드리프트가 변형 간 차이로 둔갑한다. 라운드별 원값은
        `info["rounds"]`에 남는다.
        """
        if group not in GROUPS:
            raise ValueError(f"unknown group {group!r} (gateup|down)")
        from sglang.srt.layers.moe.prism.rejoin import rejoin_down, rejoin_gateup

        shape, topk = self.shape, self.shape.topk
        projs = GROUPS[group]
        st = self.res.staging
        qlen_ptr = self._qlen.data_ptr()
        dev = self.device
        ids_host, ids_dev = self._ids(reps)
        tw = torch.full((1, topk), 1.0 / topk, dtype=torch.float32, device=dev)
        st.fill_topk_w(tw)
        w_ptr = st.topk_w_ptr()

        # x ≡ 1 — sparsity 합성이 x0=x1=1을 전제한다 (common.py의 역산).
        if group == "gateup":
            x = torch.ones(1, shape.hidden, dtype=torch.bfloat16, device=dev)
            width = 2 * shape.inter
            submit = self.cold.wrapper.submit_forward_gateup
            cold_in, cold_out = st.x_ptr(), st.partial_gateup_ptr()
            st.fill_x(torch.ones(1, shape.hidden, dtype=torch.bfloat16))
            pair = False
        else:
            x = torch.ones(topk, shape.inter, dtype=torch.bfloat16, device=dev)
            width = shape.hidden
            submit = self.cold.wrapper.submit_forward_down
            cold_in, cold_out = st.act_ptr(), st.partial_down_ptr()
            st.fill_act(torch.ones(1, topk, shape.inter, dtype=torch.bfloat16))
            pair = True
        out_hot = torch.zeros(1, topk, width, dtype=torch.bfloat16, device=dev)
        out_warm = torch.zeros(1, topk, width, dtype=torch.bfloat16, device=dev)

        def gpu_launch(tiers, out, i: int) -> None:
            if group == "gateup":
                GpuTier.launch_gateup(tiers["gate"], tiers["up"], x, ids_dev[i],
                                      tw, out, shape.inter)
            else:
                tiers["down"].launch(x, ids_dev[i], tw, out, 0, x_row_is_pair=True)

        def hot_only(i: int) -> None:
            gpu_launch(self.hot, out_hot, i)

        def warm_only(i: int) -> None:
            gpu_launch(self.warm, out_warm, i)

        def gpu_only(i: int) -> None:
            hot_only(i)
            warm_only(i)

        def cold_only(i: int) -> None:
            st.fill_expert_ids(ids_host[i])
            submit(qlen_ptr, topk, st.expert_ids_ptr(), cold_in, cold_out, None, w_ptr)
            self.cold.wrapper.sync(None)

        def combined(i: int) -> None:
            # executor와 같은 순서: cold를 먼저 던지고 GPU 일을 한 뒤 거둔다.
            st.fill_expert_ids(ids_dev[i], non_blocking=True)
            stream = torch.cuda.current_stream().cuda_stream
            submit(qlen_ptr, topk, st.expert_ids_ptr(), cold_in, cold_out, stream, w_ptr)
            gpu_only(i)
            self.cold.wrapper.sync(stream)

        def cold_graph(i: int) -> None:
            # 진단용: graph 안의 cold만 (GPU 티어 없음). `combined`와의 차가 곧
            # "GPU 일이 cold와 겹쳤는가"의 답이다 — 겹쳤으면 차가 0에 가깝고,
            # 직렬화면 gpu_only만큼 늘어난다. `cold_only`(eager)와의 차는 host
            # node 디스패치 비용이다.
            st.fill_expert_ids(ids_dev[i], non_blocking=True)
            stream = torch.cuda.current_stream().cuda_stream
            submit(qlen_ptr, topk, st.expert_ids_ptr(), cold_in, cold_out, stream, w_ptr)
            self.cold.wrapper.sync(stream)

        def combined_split(i: int) -> None:
            """cold를 **곁 스트림**으로 뺀 combined — P2(가짜 의존)의 대책.

            `combined`에서는 sync host node가 같은 스트림의 hot·warm 커널 완료까지
            기다린다. sync가 실제로 기다려야 하는 것은 CPU 완료뿐이므로 그건 가짜
            의존이고, cold 사슬(D2H → submit → sync)을 곁 스트림에 두면 GPU 커널과
            병렬로 흐른다. 마지막에 event로 다시 합류한다 (rejoin이 둘 다 먹어야
            하므로 합류 자체는 없앨 수 없다).

            executor는 이 형태를 `res.cold_stream` + `_cold_phase_async`로 갖고
            있지만 **graph 경로에서는 꺼 둔다** (`executor.py`의 `cold_async`에
            `not flow.graph_flow`). 그 결정을 되돌릴 값어치가 있는지가 여기서 나온다.
            """
            main = torch.cuda.current_stream()
            side = self.res.cold_stream
            side.wait_stream(main)                 # x·topk·ids 준비 완료 시점
            with torch.cuda.stream(side):
                st.fill_expert_ids(ids_dev[i], non_blocking=True)
                ss = side.cuda_stream
                submit(qlen_ptr, topk, st.expert_ids_ptr(), cold_in, cold_out, ss, w_ptr)
                self.cold.wrapper.sync(ss)
            gpu_only(i)                            # main 스트림
            main.wait_stream(side)

        def _with_rejoin(inner):
            def run(i: int) -> None:
                inner(i)
                src = st.gateup_out(1) if group == "gateup" else st.down_out(1)
                cold_part = src.to(dev, non_blocking=True)
                if group == "gateup":
                    rejoin_gateup([out_hot, out_warm, cold_part], shape.inter)
                else:
                    rejoin_down([out_hot, out_warm, cold_part], tw)
            return run

        # **flush는 cold가 없는 변형에만 건다.** cold를 포함한 변형에서 flush는
        # 얻는 것보다 잃는 것이 크다: GPU L2를 씻어도 kt 계산은 그대로인데,
        # ~75 µs짜리 flush를 빼는 차분의 잡음(±3 µs)이 그 변형에서 hot의 L2 상주
        # (1–3 µs)보다 크다. 실측에서 그 잡음이 `cold_graph`와 `cold_only`의 실제
        # 6 µs 차를 18 µs로 부풀렸다.
        #
        # 더 중요한 것은 **겹침 판정이 뺄셈**이라는 점이다 — `combined − cold_graph`.
        # 한쪽만 차분하면 잡음이 한쪽에만 실려 그 뺄셈이 무의미해진다. 그래서
        # cold를 포함한 넷은 `flush` 인자와 무관하게 항상 flush 없이 잰다.
        VARIANTS = (
            ("hot_only", hot_only, flush),
            ("warm_only", warm_only, flush),
            ("gpu_only", gpu_only, flush),
            ("cold_graph", cold_graph, False),
            ("combined", combined, False),
            ("combined_split", combined_split, False),
            ("layer", _with_rejoin(combined), False),
            ("layer_split", _with_rejoin(combined_split), False),
        )

        want = (lambda name: not only or name in only)
        if rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}")
        samples: dict = {}

        def keep(name: str, t: Timing) -> None:
            samples.setdefault(name, []).append(t)

        for _ in range(rounds):
            base = None
            if flush and any(use for nm, _, use in VARIANTS if want(nm) and use):
                base = self.flush_timing(reps, replays)
                keep("flush", base)
            for name, fn, use_flush in VARIANTS:
                if not want(name):
                    continue
                with nvtx(f"full/{group}/{name}"):
                    keep(name, self._timed(fn, reps, replays, use_flush, base))
            if want("cold_only"):
                with nvtx(f"full/{group}/cold_only"):
                    keep("cold_only", host_timing(cold_only, reps, replays=replays))

        def median_of(ts: Sequence[Timing]) -> Timing:
            """라운드 간 중앙값. 대표값만 고르지 않고 통계 전체를 그 라운드에서
            가져온다 — 항목별로 다른 라운드를 섞으면 없는 실행을 지어내게 된다."""
            return sorted(ts, key=lambda t: t.us)[len(ts) // 2]

        timings = {k: median_of(v) for k, v in samples.items()}
        info = {
            "group": group,
            "splits": [self.splits[p].as_dict() for p in projs],
            "hot_bytes_dense": sum(self.hot[p].bytes_per_launch(topk) for p in projs),
            "warm_bytes_dense": sum(self.warm[p].bytes_per_launch(topk) for p in projs),
            "cold_bytes_dense": sum(
                int(topk * (sum(self.cold.rows[p]) / shape.experts)
                    * shape.n_cols(p) * self.store.elem_bytes) for p in projs),
            "warm_keep_frac": round(self.warm[projs[0]].keep_frac, 4),
            "cold_keep_frac": round(self.cold.keep_frac[projs[0]], 4),
            "reps": reps, "replays": replays, "flush": flush, "rounds": rounds,
            # 라운드별 원값 — 이 폭이 곧 결론을 얼마나 믿을 수 있는지다.
            "rounds_us": {k: [round(t.us, 3) for t in v] for k, v in samples.items()},
            "exposed_us": None,
        }
        if "combined" in timings and "cold_graph" in timings:
            info["exposed_us"] = {
                "combined": round(timings["combined"].us - timings["cold_graph"].us, 3),
                "combined_split": (
                    round(timings["combined_split"].us - timings["cold_graph"].us, 3)
                    if "combined_split" in timings else None),
            }
        return LayerGroupReport(group=group, timings=timings, info=info)

    # ── 검증 ─────────────────────────────────────────────────────────────
    def check(self, group: str = "gateup", *, seed: int = 7) -> dict:
        """세 티어 partial의 합을 합성 마스크 레퍼런스와 대조한다.

        x ≡ 1이므로 레퍼런스는 "살아있는 행의 W 합"이다 — hot은 전 행(dense),
        warm·cold는 keep 마스크가 산 행만. 마스킹이 조용히 사라지면 성능만
        달라지므로 이 대조가 유일한 검출기다. 티어 경계가 expert마다 다른
        구성에서는 `row_off`가 어긋나도 여기서 드러난다.
        """
        if group not in GROUPS:
            raise ValueError(f"unknown group {group!r}")
        if self.store.name == "mxfp4":
            raise NotImplementedError(
                "mxfp4는 codes 행이 k-페어라 expert별 행 슬라이스가 한 단계 더 "
                "필요하다 — bf16/fp8만 대조한다")
        from sglang.srt.layers.moe.prism.profile.common import _e4m3_table

        shape, topk = self.shape, self.shape.topk
        projs = GROUPS[group]
        E = shape.experts
        st = self.res.staging
        dev = self.device
        g = torch.Generator().manual_seed(seed)
        ids_host = (torch.randperm(E, generator=g)[:topk]
                    .to(torch.int64).reshape(1, topk).contiguous())
        ids_dev = ids_host.to(dev)
        tw = torch.full((1, topk), 1.0 / topk, dtype=torch.float32, device=dev)
        st.fill_topk_w(tw)

        if group == "gateup":
            x = torch.ones(1, shape.hidden, dtype=torch.bfloat16, device=dev)
            width = 2 * shape.inter
            submit = self.cold.wrapper.submit_forward_gateup
            cold_in, cold_out = st.x_ptr(), st.partial_gateup_ptr()
            st.fill_x(torch.ones(1, shape.hidden, dtype=torch.bfloat16))
        else:
            x = torch.ones(topk, shape.inter, dtype=torch.bfloat16, device=dev)
            width = shape.hidden
            submit = self.cold.wrapper.submit_forward_down
            cold_in, cold_out = st.act_ptr(), st.partial_down_ptr()
            st.fill_act(torch.ones(1, topk, shape.inter, dtype=torch.bfloat16))
        out_hot = torch.zeros(1, topk, width, dtype=torch.bfloat16, device=dev)
        out_warm = torch.zeros(1, topk, width, dtype=torch.bfloat16, device=dev)

        # eager 1회 — graph 없이 (레퍼런스 대조는 시간을 재지 않는다)
        if group == "gateup":
            GpuTier.launch_gateup(self.hot["gate"], self.hot["up"], x, ids_dev,
                                  tw, out_hot, shape.inter)
            GpuTier.launch_gateup(self.warm["gate"], self.warm["up"], x, ids_dev,
                                  tw, out_warm, shape.inter)
        else:
            self.hot["down"].launch(x, ids_dev, tw, out_hot, 0, x_row_is_pair=True)
            self.warm["down"].launch(x, ids_dev, tw, out_warm, 0, x_row_is_pair=True)
        st.fill_expert_ids(ids_host)
        submit(self._qlen.data_ptr(), topk, st.expert_ids_ptr(), cold_in,
               cold_out, None, st.topk_w_ptr())
        self.cold.wrapper.sync(None)
        torch.cuda.synchronize()
        got_cold = (st.gateup_out(1) if group == "gateup" else st.down_out(1)).float()
        got = out_hot.float().cpu() + out_warm.float().cpu() + got_cold.cpu()

        def deq(codes: torch.Tensor) -> torch.Tensor:
            """합성 배율은 전부 1.0이라 dequant는 코드 값 그 자체다."""
            if self.store.name == "bf16":
                return codes.float()
            return _e4m3_table()[codes.long()]

        report = {"group": group, "ids": ids_host[0].tolist()}
        for pi, proj in enumerate(projs):
            n = self.splits[proj].n_cols
            cold_blocks = self.cold.dequant_blocks(proj, n)
            errs = []
            for j in range(topk):
                e = int(ids_host[0, j])
                ref = torch.zeros(n, dtype=torch.float32)
                for tier in (self.hot[proj], self.warm[proj]):
                    beg = sum(tier.rows[:e])
                    end = beg + tier.rows[e]
                    block = deq(tier.parts[0][beg:end].cpu())      # [k(e), n]
                    if tier.a_host is None:
                        ref += block.sum(0)
                    else:
                        keep = tier.a_host[beg:end] > 0
                        ref += block[keep].sum(0)
                keep_c = self.cold.a_host[proj][e] > 0
                ref += cold_blocks[e][:, keep_c].sum(1)
                col = got[0, j, pi * n:(pi + 1) * n]
                errs.append(float((col - ref).abs().max()
                                  / ref.abs().max().clamp_min(1e-6)))
            report[f"{proj}_max_rel_err"] = round(max(errs), 5)
        return report

    def report(self, groups: Sequence[str] = ("gateup", "down"), *,
               reps: int = 50, replays: int = 10, flush: bool = True,
               rounds: int = 3, only: Optional[Sequence[str]] = None) -> dict:
        results = {g: self.measure(g, reps=reps, replays=replays, flush=flush,
                                   rounds=rounds, only=only).as_dict() for g in groups}

        def total(name):
            if not all(name in results[g] for g in groups):
                return None
            return round(sum(results[g][name]["us"] for g in groups), 3)

        layer_us, layer_split_us = total("layer"), total("layer_split")
        return {
            "bench": "full_layer",
            "shape": self.shape.as_dict(),
            "params": self.params,
            "footprint": self.footprint,
            "groups": results,
            "layer_us": layer_us,
            # P2(cold를 곁 스트림으로)를 적용했을 때의 같은 값 — production에
            # 넘길 숫자는 이쪽이다.
            "layer_split_us": layer_split_us,
            "layer_saving_us": (None if None in (layer_us, layer_split_us)
                                else round(layer_us - layer_split_us, 3)),
            "env": env_stamp(self.device),
        }


# ─── planner 근사식 재현 ───────────────────────────────────────────────────
def planner_model(shape: Shape, splits: Mapping[str, ProjSplit], group: str, *,
                  sparsity: float, dtype: str = "bf16", device=0,
                  anchor_topk: int = 1, reps: int = 50, replays: int = 10,
                  numa_split: float = 0.5, threads: Optional[int] = None,
                  cpu_kernel: Optional[str] = None,
                  numa_map: Optional[Sequence[int]] = None,
                  seed: int = 0) -> dict:
    """planner가 쓰는 선형 근사를 **그 절차 그대로** 재현한다.

    절차: 티어마다 "weight [k, n] 하나"를 `anchor_topk`개 활성 expert로 재고, 그
    값을 단위 수(`anchor_topk × proj 수`)로 나눠 "expert 하나 · proj 하나"를 만든
    뒤, 실제 활성 expert 수만큼 선형으로 곱한다. 마지막에 GPU 합과 cold를 겹쳐
    `max`를 취한다.

    `anchor_topk=1`이 planner의 현재 형태다 (`cold_sparse_gemv`/`warm_sparse_gemv`의
    접힌 기본값). `anchor_topk=shape.topk`를 주면 같은 절차를 실제 활성 수에서
    앵커해 "선형 확장만 남긴" 값이 나온다 — 두 값의 차가 곧 **앵커 위치가 만드는
    오차**이고, 실측과의 차가 **선형 가정 자체의 오차**다.

    `k`는 이 구성의 expert별 평균 행 수로 준다 (planner도 단일 k를 쓴다).
    """
    from sglang.srt.layers.moe.prism.profile.cold_cpu import cold_sparse_gemv
    from sglang.srt.layers.moe.prism.profile.hot import dense_gemv
    from sglang.srt.layers.moe.prism.profile.warm_cold import warm_sparse_gemv

    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r}")
    projs = GROUPS[group]
    E, topk = shape.experts, shape.topk
    nproj = len(projs)
    units = topk * nproj                     # 이 phase가 실제로 도는 GEMV 수
    anchor_units_gpu = anchor_topk           # GPU 헬퍼는 proj 하나만 돈다
    anchor_units_cold = anchor_topk * 2      # forward_gateup은 gate+up 둘을 돈다
    ref = splits[projs[0]]
    n = ref.n_cols
    pair = group == "down"

    def mean_rows(tier: str) -> int:
        r = splits[projs[0]].rows(tier)
        step = store_of(dtype).rows_step()
        m = int(round(sum(r) / len(r) / step)) * step
        return max(step, m)

    k_hot, k_warm, k_cold = (mean_rows(t) for t in ("hot", "warm", "cold"))
    # cold 헬퍼는 fake shape (hidden=k, inter=n)으로 kt를 굽는다 — 그래서 **k도**
    # 그 커널의 N 정렬을 넘겨야 한다 (tile_k2는 노드당 256의 배수, 노드마다 rows>0).
    # 못 맞추는 k는 측정 가능한 값으로 올려 재고 행 수에 **선형으로** 되돌린다:
    # 선형성은 이 모델 자신의 가정이므로 새 가정을 들이는 것이 아니고, 되돌리기
    # 전 값(`cold_us_raw`)과 쓴 k(`k_cold_used`)를 리포트에 남긴다.
    align = N_ALIGN[cpu_kernel or store_of(dtype).cpu_kernel]
    nodes = len(numa_map) if numa_map else numa_nodes()
    k_used = max(align * nodes, -(-k_cold // align) * align)

    hot = dense_gemv(k_hot, n, device=device, experts=E, topk=anchor_topk,
                     reps=reps, replays=replays, dtype=dtype, seed=seed)
    warm = warm_sparse_gemv(k_warm, n, sparsity, device=device, reps=reps,
                            replays=replays, dtype=dtype, experts=E,
                            topk=anchor_topk, x_row_is_pair=pair, seed=seed)
    cold = cold_sparse_gemv(k_used, n, sparsity, dtype=dtype, experts=E,
                            topk=anchor_topk, iters=max(50, reps),
                            replays=replays, numa_split=numa_split,
                            threads=threads, cpu_kernel=cpu_kernel,
                            numa_map=numa_map, seed=seed)
    cold_us = cold.us * (k_cold / k_used)
    terms = {
        "hot_us": round(hot.us / anchor_units_gpu * units, 3),
        "warm_us": round(warm.us / anchor_units_gpu * units, 3),
        "cold_us": round(cold_us / anchor_units_cold * units, 3),
        "cold_us_raw": round(cold.us / anchor_units_cold * units, 3),
        "k_cold_used": k_used,
    }
    terms["gpu_us"] = round(terms["hot_us"] + terms["warm_us"], 3)
    terms["overlap_us"] = round(max(terms["gpu_us"], terms["cold_us"]), 3)
    terms.update(anchor_topk=anchor_topk, units=units,
                 k_hot=k_hot, k_warm=k_warm, k_cold=k_cold, n_cols=n,
                 anchor_raw={"hot_us": hot.us, "warm_us": warm.us, "cold_us": cold.us})
    return terms


def compare(shape: Shape, *, groups: Sequence[str] = ("gateup", "down"),
            anchors: Sequence[int] = (1,), reps: int = 50, replays: int = 10,
            flush: bool = True, rounds: int = 3, **profiler_kw) -> dict:
    """실측과 planner 근사를 나란히 놓은 리포트.

    **순서가 중요하다**: 모델 항을 먼저 재고 그 다음에 full-layer 인스턴스를
    만든다. 둘이 동시에 살아 있으면 CPUInfer가 두 벌 떠서 (각각 스레드 전부를
    잡으므로) 서로의 cold 시간을 왜곡한다.

    `anchors`에 1과 실제 top_k를 함께 주면 두 오차가 분리된다:
      anchor_topk=1        planner의 현재 형태 — 접힌 값에서 선형 확장
      anchor_topk=shape.topk  실제 활성 수에서 앵커 — 선형 가정만 남는다
    """
    splits = build_splits(
        shape, hot_frac=profiler_kw.get("hot_frac", 0.375),
        warm_frac=profiler_kw.get("warm_frac", 0.125),
        cold_frac=profiler_kw.get("cold_frac"),
        hot_spread=profiler_kw.get("hot_spread", 0.0),
        warm_spread=profiler_kw.get("warm_spread", 0.0),
        cold_spread=profiler_kw.get("cold_spread", 0.0),
        dtype=profiler_kw.get("dtype", "bf16"),
        shuffle=profiler_kw.get("shuffle_index", False),
        seed=profiler_kw.get("seed", 0))
    store = store_of(profiler_kw.get("dtype", "bf16"))
    model = {}
    for group in groups:
        model[group] = {}
        for a in anchors:
            model[group][f"anchor_topk_{a}"] = planner_model(
                shape, splits, group,
                sparsity=profiler_kw.get("sparsity", 0.5),
                dtype=store, device=profiler_kw.get("device", 0),
                anchor_topk=a, reps=reps, replays=replays,
                numa_split=profiler_kw.get("numa_split", 0.5),
                threads=profiler_kw.get("threads"),
                cpu_kernel=profiler_kw.get("cpu_kernel"),
                numa_map=profiler_kw.get("numa_map"),
                seed=profiler_kw.get("seed", 0))

    with FullLayerProfiler(shape, **profiler_kw) as p:
        out = {"bench": "full_layer_vs_model", "shape": shape.as_dict(),
               "params": p.params, "footprint": p.footprint, "groups": {}}
        for group in groups:
            rep = p.measure(group, reps=reps, replays=replays, flush=flush,
                            rounds=rounds)
            meas = rep.timings["combined"].us
            for m in model[group].values():
                m["err_vs_combined"] = round(m["overlap_us"] / meas - 1.0, 4)
                if "layer" in rep.timings:
                    m["err_vs_layer"] = round(
                        m["overlap_us"] / rep.timings["layer"].us - 1.0, 4)
            out["groups"][group] = {
                "measured": {k: v.us for k, v in rep.timings.items()},
                "info": rep.info, "model": model[group]}
        out["env"] = env_stamp(p.device)
    return out
