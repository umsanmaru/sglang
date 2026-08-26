"""warm(GPU) + cold(CPU) sparse GEMV의 실행시간 (프로파일 ②).

재는 것은 **두 티어가 실제로 부르는 그 커널들**이다:

  warm — `gemv_worklist_indexed_pinned_sparse` (tiers.py `SparsePinnedDirectTier`;
         W가 pinned host, GPU가 UVA로 제자리 읽고 죽은 페어의 로드를 발행하지
         않는다 → 건너뛴 만큼이 PCIe 절약)
  cold — kt `forward_{gateup,down}_partial` (기본 `kt_tile_k2_bf16` = TileK2BF16_MOE;
         s → thr → 점수 → 마스크 → masked GEMV가 전부 C++ 안에서 끝난다)

`measure()`가 내는 값들 (전부 iteration당 µs):

  warm_only              커널 reps개를 CUDA graph에 캡처 → replay/reps
  cold_only              host 루프 reps회 (submit+sync). CPU 단독 시간
  combined               graph 하나에 reps × (ids D2H → cold submit → warm launch
                         → cold sync). kt 호출이 host node로 캡처되는 실제 graph
                         decode 경로 그대로이고, 이 값이 겹침 이후의 벽시계다
  combined_eager         같은 것을 eager로 (host 루프 throughput)
  combined_eager_latency eager + iteration마다 GPU까지 대기 (latency)
  combined_staged        with_staging=True일 때, x/가중 D2H와 partial H2D까지 포함

스토어 로딩이 비싸다 (cold는 1 GB 패킹). 그래서 이 클래스는 **한 번 만들어 여러
번 질의**하는 형태다 — sparsity/warm_frac을 바꾸려면 새 인스턴스를 만들어야
한다 (점수 테이블과 weight가 로드 타임에 굳으므로).

sparse는 decode 전용이라 M=1 고정이다 (executor의 masking 조건).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch

from sglang.srt.layers.moe.prism.profile.common import (
    GRID,
    SparseGemv,
    K_STEP,
    NG,
    PAIR_GROUP,
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
    nvtx,
    numa_nodes,
    select_device,
    split_rows,
    tier_index,
)

GROUPS = {"gateup": ("gate", "up"), "down": ("down",)}

VARIANTS = ("warm_only", "cold_only", "combined", "combined_eager",
            "combined_eager_latency", "combined_staged")

# 노드 N shard의 정렬 요구 — **커널이 정한다**.
#
# tile_k2의 `gemv_slab`은 타일 컬럼 stride를 `c * TILE_ELEMS`(= N_BLOCK 256 ×
# K_STEP 32)로 계산하므로 마지막 부분 블록을 표현할 수 없고, 그래서 스스로
# `assert(n % N_BLOCK == 0)`을 건다. Release 빌드는 NDEBUG라 그 assert가 없어
# **조용히 남의 열을 읽고 segfault한다** (2026-08-26 실측: I=768을 2노드 384씩
# 나눴을 때). 그래서 이 정렬은 입력 단계에서 막는다.
#
# AMX(kt_amx_bf16)의 mat_mul/amx_kernel은 부분 N_BLOCK을 처리하므로 N_STEP(32)만
# 지키면 된다. 노드 하나가 받는 shard는 kt가 rows > 0을 요구한다.
N_ALIGN = {"kt_tile_k2_bf16": 256, "kt_amx_bf16": 32}


# ─── 행 분할 ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Split:
    """한 proj의 K축 분할. hot은 이 프로파일의 관심사가 아니므로 남은 행은
    버린다 (①이 그 몫을 잰다) — warm_frac + cold_frac < 1이면 그 차이가 hot이다."""

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
               *, shuffle: bool = False, seed: int = 0) -> Split:
    axis = shape.k_axis(proj)
    kw = split_rows(axis, warm_frac)
    kc = split_rows(axis, cold_frac)
    if kw + kc > axis:
        raise ValueError(
            f"{proj}: warm {kw} + cold {kc} rows exceed axis {axis} "
            f"(K_STEP={K_STEP} 반올림 후)")
    if kc % K_STEP or kw % PAIR_GROUP:
        raise ValueError(
            f"{proj}: cold rows must be a K_STEP({K_STEP}) multiple and warm "
            f"rows even — got cold={kc} warm={kw}")
    # 같은 시드의 같은 순열에서 앞을 warm, 그 다음을 cold가 갖는다 → 서로소.
    return Split(
        proj=proj, axis=axis, n_cols=shape.n_cols(proj),
        warm_rows=tier_index(axis, kw, shuffle=shuffle, seed=seed),
        cold_rows=tier_index(axis, kc, skip=kw, shuffle=shuffle, seed=seed),
    )


def footprint(shape: Shape, splits: Mapping[str, Split]) -> dict:
    E = shape.experts
    warm = sum(E * splits[p].k_warm * splits[p].n_cols * 2 for p in PROJS)
    cold = sum(E * splits[p].k_cold * splits[p].n_cols * 2 for p in PROJS)
    return {"warm_pinned_mb": round(warm / 1e6, 1),
            "cold_mb": round(cold / 1e6, 1),
            "note": "cold는 주입 텐서와 kt packed 사본이 동시에 사는 순간이 load "
                    "중 한 번 있다 (pack 완료 후 주입본 해제 — 계약 ③)"}


# ─── warm (GPU, pinned W) ─────────────────────────────────────────────────
class WarmTier:
    """한 proj의 warm 스토어 + sparse 인자. weights.py의 warm shard와 같은 형태:
    w_flat [Σₑ k(e), N] pinned, row_off/k_index는 device 상주."""

    def __init__(self, shape: Shape, sp: Split, *, sparsity: float, pattern: str,
                 seed: int, device, node: Optional[int]):
        from sglang.srt.layers.moe.prism.numa import alloc_pinned_on_node
        from sglang.srt.layers.moe.prism.profile.common import sparse_tables
        from sglang.srt.layers.moe.prism.tiers import SparseSpec

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
        from sglang.jit_kernel.prism_gemv import (
            gemv_worklist_indexed_pinned,
            gemv_worklist_indexed_pinned_sparse,
        )

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
def node_table(n_total: int, frac: float, nodes: int, align: int) -> tuple:
    """N축을 노드에 나눈다 → (offset 테이블, rows 테이블, 실현된 node 0 비율).

    2노드는 frac을 쓰고 (align 블록 단위로 반올림), 그 이상은 균등이다 — 실
    plan(cold_shards)도 노드당 1 shard다.
    """
    if n_total % align:
        raise ValueError(
            f"N={n_total}이 커널의 N 정렬 {align}의 배수가 아니다 — "
            f"이 커널로는 이 치수를 잴 수 없다")
    blocks = n_total // align
    if blocks < nodes:
        raise ValueError(
            f"N={n_total}은 {align}짜리 블록 {blocks}개뿐인데 노드가 {nodes}개다 "
            f"(kt는 노드마다 rows > 0을 요구한다) — 더 큰 inter/hidden이나 "
            f"cpu_kernel='kt_amx_bf16'")
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
    Plan에서 굽는 것과 같은 config를 여기서는 인자에서 굽는다 — 이 패키지는
    Plan을 읽지 않는다 (프로파일 결과가 Plan의 입력이므로)."""

    def __init__(self, shape: Shape, splits: Mapping[str, Split], *,
                 sparsity: float, pattern: str, seed: int, numa_split: float,
                 threads: int, kernel_key: str,
                 numa_map: Optional[Sequence[int]] = None):
        from kt_kernel import kt_kernel_ext
        from kt_kernel.experts_partial import PartialMoEWrapper

        from sglang.srt.layers.moe.prism.profile.common import sparse_tables

        if kernel_key not in N_ALIGN:
            raise ValueError(f"unknown cpu kernel {kernel_key!r} "
                             f"(known: {sorted(N_ALIGN)})")
        E, topk = shape.experts, shape.topk
        H, I = shape.hidden, shape.inter
        # numa_map이 주어지면 그 노드들만 쓴다 — 실모델의 SGLANG_PRISM_NUMA_MAP과
        # 같은 경로(WorkerPoolConfig)다. shard table의 항목 수는 tp_count(=subpool
        # 수)와 같아야 하므로 nodes도 같이 좁힌다 — 안 그러면 kt가
        # "partial shard table size != tp_count"로 즉사한다.
        if numa_map:
            pool_cfg = kt_kernel_ext.WorkerPoolConfig()
            pool_cfg.subpool_count = len(numa_map)
            pool_cfg.subpool_numa_map = list(numa_map)
            pool_cfg.subpool_thread_count = [
                threads // len(numa_map) + (1 if i < threads % len(numa_map) else 0)
                for i in range(len(numa_map))
            ]
            self.cpuinfer = kt_kernel_ext.CPUInfer(pool_cfg)
            self.nodes = len(numa_map)
        else:
            self.cpuinfer = kt_kernel_ext.CPUInfer(threads)
            self.nodes = numa_nodes()
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
        align = N_ALIGN[kernel_key]
        gu_off, gu_rows, gu_frac = node_table(I, numa_split, self.nodes, align)
        dn_off, dn_rows, dn_frac = node_table(H, numa_split, self.nodes, align)
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
        for proj in PROJS:
            sp = splits[proj]
            t = torch.empty(E * sp.n_cols * sp.k_cold, dtype=torch.bfloat16)
            t.normal_(0, 0.02)
            self.w[proj] = t.contiguous()

        tables, self.keep_frac, self.a_host = {}, {}, {}
        for proj in PROJS:
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

    def rows_of(self, proj: str) -> int:
        return int(self.a_host[proj].shape[1])


# ─── 리포트 ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GroupReport:
    """한 group(gateup | down)의 측정 결과."""

    group: str
    timings: Mapping[str, Timing]
    info: dict
    check: Optional[dict] = None

    def __getitem__(self, name: str) -> Timing:
        return self.timings[name]

    def us(self, name: str) -> float:
        return self.timings[name].us

    def as_dict(self) -> dict:
        d = {k: v.as_dict() for k, v in self.timings.items()}
        d["shape"] = self.info
        if self.check is not None:
            d["check"] = self.check
        return d


class WarmColdProfiler:
    """warm 스토어(pinned) + cold 인스턴스(kt) + staging을 한 번 만들어 두고
    여러 group을 질의한다. `close()` 또는 with 문으로 해제한다."""

    def __init__(self, shape: Shape, *, warm_frac: float,
                 cold_frac: Optional[float] = None, sparsity: float = 0.9,
                 numa_split: float = 0.5, mask_pattern: str = "random",
                 cpu_kernel: str = "kt_tile_k2_bf16",
                 threads: Optional[int] = None, device=0, masking: bool = True,
                 warm_node: Optional[int] = None, shuffle_index: bool = False,
                 seed: int = 0, numa_map: Optional[Sequence[int]] = None):
        from sglang.jit_kernel.prism_gemv import warmup_jit
        from sglang.srt.layers.moe.prism.numa import gpu_numa_node
        from sglang.srt.layers.moe.prism.resources import (
            ExecutionResources,
            ResourceSpec,
        )

        self.shape = shape
        self.device = select_device(device)
        self.masking = masking
        self.sparsity = sparsity
        cold_frac = 1.0 - warm_frac if cold_frac is None else cold_frac
        self.splits = {
            proj: make_split(shape, proj, warm_frac, cold_frac,
                             shuffle=shuffle_index, seed=seed)
            for proj in PROJS
        }
        self.threads = threads or default_cpuinfer_threads()
        self.warm_node = (warm_node if warm_node is not None
                          else gpu_numa_node(self.device.index or 0))
        warmup_jit()   # JIT 컴파일을 캡처 밖으로

        self.warm = {
            proj: WarmTier(shape, self.splits[proj], sparsity=sparsity,
                           pattern=mask_pattern, seed=seed, device=self.device,
                           node=self.warm_node)
            for proj in PROJS
        }
        self.cold = ColdTier(shape, self.splits, sparsity=sparsity,
                             pattern=mask_pattern, seed=seed,
                             numa_split=numa_split, threads=self.threads,
                             kernel_key=cpu_kernel, numa_map=numa_map)
        self.res = ExecutionResources(ResourceSpec(
            max_tokens=1, top_k=shape.topk, hidden_size=shape.hidden,
            intermediate_size=shape.inter, device=self.device))
        # cold task가 역참조하는 qlen — 주소 고정 (계약 ④의 포인터 경유)
        self._qlen = torch.ones(1, dtype=torch.int32).pin_memory()
        self.params = {
            "sparsity": sparsity, "warm_frac": warm_frac,
            "cold_frac": cold_frac, "numa_split": numa_split,
            "mask_pattern": mask_pattern, "masking": masking, "m": 1,
            "cpu_kernel": cpu_kernel, "cpuinfer_threads": self.threads,
            "numa_nodes": self.cold.nodes, "warm_node": self.warm_node,
            "numa_map": list(numa_map) if numa_map else None,
            "node_tables": self.cold.node_tables,
            "shuffle_index": shuffle_index, "seed": seed,
        }

    # ── 수명 ─────────────────────────────────────────────────────────────
    def close(self) -> None:
        self.warm = {}
        self.cold = None
        self.res = None
        torch.cuda.empty_cache()

    def __enter__(self) -> "WarmColdProfiler":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def footprint(self) -> dict:
        return footprint(self.shape, self.splits)

    # ── 측정 ─────────────────────────────────────────────────────────────
    def _ids(self, reps: int, seed: int = 1234) -> tuple:
        """iteration마다 다른 top_k expert 배정 (M=1). GPU 커널과 cold staging이
        같은 int64 버퍼를 공유하므로 dtype은 staging(_expert_ids)에 맞춘다."""
        g = torch.Generator().manual_seed(seed)
        host = [torch.randperm(self.shape.experts, generator=g)[: self.shape.topk]
                .to(torch.int64).reshape(1, self.shape.topk).contiguous()
                for _ in range(reps)]
        return host, [t.to(self.device) for t in host]

    def measure(self, group: str = "gateup", *, reps: int = 100,
                replays: int = 20, with_staging: bool = False,
                only: Optional[Sequence[str]] = None,
                fused: bool = True) -> GroupReport:
        """group의 변형들을 잰다. `only`를 주면 그 변형만 돈다 — nsys 트레이스를
        한 변형으로 좁히려면 나머지가 타임라인에 없어야 한다 (전부 돌리면 커널이
        이름 없이 섞인다)."""
        if group not in GROUPS:
            raise ValueError(f"unknown group {group!r} (gateup|down)")
        if only is not None:
            unknown = set(only) - set(VARIANTS)
            if unknown:
                raise ValueError(f"unknown variants {sorted(unknown)}")
            only = frozenset(only)

        shape, warm, cold = self.shape, self.warm, self.cold
        projs = GROUPS[group]
        topk = shape.topk
        st = self.res.staging
        qlen_ptr = self._qlen.data_ptr()
        masking = self.masking
        device = self.device
        ids_host, ids_dev = self._ids(reps)

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

        # gateup을 **시스템과 같은 형태로** 발행한다: executor는 gate와 up을 한
        # 커널로 보낸다 (`tiers.SparsePinnedGateUp`). 두 번 launch해서 재면
        # 시스템보다 비싼 값이 나온다 — 실측 31.7 vs 22.8 µs. `fused=False`로
        # 옛 경로를 재서 그 차이를 볼 수 있다. 융합 진입점이 (pinned, sparse)
        # 조합에만 있으므로 dense(prefill) 경로는 자동으로 2 launch다.
        fuse_now = fused and group == "gateup" and masking
        fused_fn = None
        if fuse_now:
            from sglang.jit_kernel.prism_gemv import (
                gemv_worklist_indexed_pinned_sparse_gateup,
            )
            fused_fn = gemv_worklist_indexed_pinned_sparse_gateup

        def warm_launch(i: int) -> None:
            if fused_fn is not None:
                wg, wu = warm["gate"], warm["up"]
                fused_fn(x, ids_dev[i], topk_w_dev,
                         wg.w_flat, wg.row_off, wg.k_index,
                         wu.w_flat, wu.row_off, wu.k_index,
                         out, wg.spec, wu.spec, 0, shape.inter, False,
                         torch.cuda.current_stream())
                return
            for proj in projs:
                warm[proj].launch(x, ids_dev[i], topk_w_dev, out, cols[proj],
                                  masking=masking)

        def want(name: str) -> bool:
            return not only or name in only

        timings: dict = {}

        if want("warm_only"):
            with nvtx(f"{group}/warm_only"):
                timings["warm_only"] = graph_timing(warm_launch, reps,
                                                    replays=replays)

        def cold_only(i: int) -> None:
            st.fill_expert_ids(ids_host[i])
            submit(qlen_ptr, topk, st.expert_ids_ptr(), cold_in, cold_out,
                   None, w_ptr)
            cold.wrapper.sync(None)

        if want("cold_only"):
            with nvtx(f"{group}/cold_only"):
                timings["cold_only"] = host_timing(cold_only, reps,
                                                   replays=replays)

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
                timings["combined"] = graph_timing(combined, reps, replays=replays)

        def combined_eager(i: int) -> None:
            # eager 경로의 단계별 구간 — 이 변형이 nsys로 들여다보는 대상이므로
            # cold submit / warm launch / cold sync를 각각 표시한다. 세 구간의
            # 폭이 곧 "무엇이 임계경로인가"의 답이다.
            with nvtx("eager/ids"):
                st.fill_expert_ids(ids_host[i])
            with nvtx("eager/cold_submit"):
                submit(qlen_ptr, topk, st.expert_ids_ptr(), cold_in, cold_out,
                       None, w_ptr)
            with nvtx("eager/warm_launch"):
                warm_launch(i)
            with nvtx("eager/cold_sync"):
                cold.wrapper.sync(None)

        if want("combined_eager"):
            with nvtx(f"{group}/combined_eager"):
                timings["combined_eager"] = host_timing(
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
                timings["combined_eager_latency"] = host_timing(
                    combined_eager_latency, reps, replays=replays)

        if with_staging and want("combined_staged"):
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
                timings["combined_staged"] = graph_timing(
                    combined_staged, reps, replays=replays)

        info = {
            "group": group,
            "warm": [self.splits[p].as_dict() for p in projs],
            "warm_bytes_dense": sum(
                topk * self.splits[p].k_warm * self.splits[p].n_cols * 2
                for p in projs),
            "cold_bytes_dense": sum(
                topk * cold.rows_of(p) * shape.n_cols(p) * 2 for p in projs),
            "warm_keep_frac": round(warm[projs[0]].keep_frac, 4),
            "cold_keep_frac": round(cold.keep_frac[projs[0]], 4),
            "reps": reps, "replays": replays, "warm_fused": fuse_now,
        }
        return GroupReport(group=group, timings=timings, info=info)

    # ── 검증 ─────────────────────────────────────────────────────────────
    def check(self, group: str = "gateup") -> dict:
        """합성한 마스크로 계산한 레퍼런스와 두 커널의 출력을 대조한다.

        x ≡ 1이므로 레퍼런스는 "살아있는 행의 W 합"이다 — 마스킹이 빠지면(전부
        dense) sparsity만큼 값이 커져 즉시 드러난다. 마스킹이 조용히 사라져도
        성능만 달라지므로 이 대조가 유일한 검출기다.
        """
        if group not in GROUPS:
            raise ValueError(f"unknown group {group!r}")
        shape, warm, cold = self.shape, self.warm, self.cold
        projs = GROUPS[group]
        topk = shape.topk
        st = self.res.staging
        device = self.device
        ids = torch.arange(topk, dtype=torch.int64).reshape(1, topk) % shape.experts
        ids_dev = ids.to(device)
        topk_w = torch.full((1, topk), 1.0 / topk, dtype=torch.float32,
                            device=device)
        st.fill_expert_ids(ids)
        st.fill_topk_w(topk_w)
        report = {}

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

        submit = (cold.wrapper.submit_forward_gateup if group == "gateup"
                  else cold.wrapper.submit_forward_down)
        if group == "gateup":
            st.fill_x(torch.ones(1, shape.hidden, dtype=torch.bfloat16))
            cold_in, cold_out = st.x_ptr(), st.partial_gateup_ptr()
        else:
            st.fill_act(torch.ones(1, topk, shape.inter, dtype=torch.bfloat16))
            cold_in, cold_out = st.act_ptr(), st.partial_down_ptr()
        submit(self._qlen.data_ptr(), topk, st.expert_ids_ptr(), cold_in,
               cold_out, None, st.topk_w_ptr())
        cold.wrapper.sync(None)
        got_all = (st.gateup_out(1) if group == "gateup" else st.down_out(1)).float()
        for pi, proj in enumerate(projs):
            kc = cold.rows_of(proj)
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

    # ── 전체 리포트 (CLI가 쓰는 형태) ────────────────────────────────────
    def report(self, groups: Sequence[str] = ("gateup", "down"), *,
               reps: int = 100, replays: int = 20, with_staging: bool = False,
               only: Optional[Sequence[str]] = None,
               do_check: bool = False, fused: bool = True) -> dict:
        results = {}
        for group in groups:
            rep = self.measure(group, reps=reps, replays=replays,
                               with_staging=with_staging, only=only, fused=fused)
            d = rep.as_dict()
            if do_check:
                d["check"] = self.check(group)
            results[group] = d
        warm_kernel = ("gemv_worklist_indexed_pinned_sparse" if self.masking
                       else "gemv_worklist_indexed_pinned")
        return {
            "bench": "warm_cold_sparse",
            "kernels": {"warm": warm_kernel,
                        "cold": self.params["cpu_kernel"]},
            "shape": self.shape.as_dict(),
            "params": {**self.params, "reps": reps, "replays": replays},
            "splits": {k: v.as_dict() for k, v in self.splits.items()},
            "footprint": self.footprint,
            "results": results,
            "env": env_stamp(self.device),
        }


def warm_cold_sparse(shape: Shape, *, groups: Sequence[str] = ("gateup", "down"),
                     reps: int = 100, replays: int = 20,
                     with_staging: bool = False, do_check: bool = False,
                     only: Optional[Sequence[str]] = None, fused: bool = True,
                     **kw) -> dict:
    """한 번 쓰고 버리는 편의 함수 — 스토어를 만들고 재고 해제한다.

    같은 스토어로 여러 번 질의하려면 `WarmColdProfiler`를 직접 쓴다 (cold 패킹이
    1 GB급이라 생성이 비싸다)."""
    with WarmColdProfiler(shape, **kw) as prof:
        return prof.report(groups, reps=reps, replays=replays,
                           with_staging=with_staging, only=only,
                           do_check=do_check, fused=fused)


def single_expert_warm_cold(hidden: int, inter: int, *, warm_frac: float = 0.125,
                            sparsity: float = 0.9, **kw) -> WarmColdProfiler:
    """expert 1개 · top_k 1로 접은 `WarmColdProfiler` — 치수만 주면 된다.

        with single_expert_warm_cold(2048, 768, sparsity=0.9, device=1) as p:
            p.measure("gateup").us("cold_only")
            p.report()          # CLI가 찍는 것과 같은 dict

    라우팅을 지우고 **같은 바이트를 한 expert의 GEMV로** 재는 형태다 (E=8, k=8을
    "8배 넓은 expert 1개"로 접는 것과 같은 축약: warm 바이트는
    `top_k × k_warm × N`이므로 top_k를 1로 줄인 만큼 N을 키우면 보존된다).

    **어디까지 유효한가** (2026-08-26 실측):
      warm dense   — 정확하다. 131.5 vs 131.6 µs. pinned UVA 읽기는 GPU L2
                     재사용이 없어 워킹셋이 줄어도 PCIe를 온전히 낸다.
      warm sparse  — ~17% 낙관. pair 블록마다 thr을 top_k 루프(+renorm)로 다시
                     구하므로 그 몫이 top_k에 비례해 사라진다 (30.5 → 25.4 µs).
      cold         — 낙관적이다. cold 비용에는 **활성 expert당 ~3.9 µs** 항이
                     있어서 (버퍼 carve + BufferA pack + pair mask + plan 인코딩)
                     A를 8에서 1로 줄이면 그만큼 빠진다. 커널 상한을 볼 때는
                     쓸 수 있지만, 티어 비율을 정하는 cost model에는 실제
                     `experts`/`topk`를 줘야 한다.
    """
    return WarmColdProfiler(
        Shape(experts=1, topk=1, hidden=hidden, inter=inter),
        warm_frac=warm_frac, sparsity=sparsity, **kw)


def warm_sparse_gemv(k: int, n: int, sparsity: float = 0.9, *, m: int = 1,
                     reps: int = 100, replays: int = 20, device=0,
                     mask_pattern: str = "random", seed: int = 0,
                     warm_node: Optional[int] = None,
                     x_row_is_pair: bool = False) -> SparseGemv:
    """[k, n] weight 하나의 warm sparse GEMV — shape과 sparsity만 받는다.

        warm_sparse_gemv(1792, 768, 0.9, device=1).us

    `dense_gemv`의 sparse 짝이다: expert/top_k를 1로 접고, W는 pinned host에 두고
    GPU가 UVA로 제자리 읽는다 (`gemv_worklist_indexed_pinned_sparse` —
    `tiers.SparsePinnedGateUp`이 부르는 그 커널). 죽은 페어의 로드를 발행하지
    않으므로 건너뛴 만큼이 그대로 PCIe 절약이다.

    **주의 — 커널 상한이지 warm 티어 비용이 아니다.** top_k를 1로 접으면
    (a) 블록이 `ceil(n/64)`뿐이라 SM이 덜 붙고 — outstanding 요청 한계가 SM당이라
    실측에서 12블록은 PCIe 피크의 9%밖에 못 뽑았다 — (b) pair 블록마다 하던 thr
    계산이 1회로 줄어 ~17% 낙관적이다. 티어 비용은 `WarmColdProfiler`로 잰다.
    바이트를 보존해 비교하려면 top_k를 줄인 만큼 `n`을 키우면 된다.
    """
    from sglang.jit_kernel.prism_gemv import (
        gemv_worklist_indexed_pinned_sparse,
        warmup_jit,
    )

    from sglang.srt.layers.moe.prism.numa import alloc_pinned_on_node, gpu_numa_node
    from sglang.srt.layers.moe.prism.profile.common import sparse_tables
    from sglang.srt.layers.moe.prism.tiers import SparseSpec

    dev = select_device(device)
    warmup_jit()
    if k % PAIR_GROUP:
        raise ValueError(f"k must be even (pair group), got {k}")
    node = warm_node if warm_node is not None else gpu_numa_node(dev.index or 0)

    w = alloc_pinned_on_node((k, n), torch.bfloat16, node, "warm_sparse_gemv store")
    w.normal_(0, 0.02)
    row_off = torch.tensor([0, k], dtype=torch.int32, device=dev)
    kidx = torch.arange(k, dtype=torch.int32).to(torch.uint16).to(dev)
    a, c, thr, keep = sparse_tables(1, k, sparsity, pattern=mask_pattern, seed=seed)
    spec = SparseSpec(a=a.to(dev), c=c.to(dev), thr=thr.to(dev),
                      p=SPARSITY_P, lam=SPARSITY_LAM, pmax=PMAX, grid=GRID,
                      ng=NG, renorm_it=RENORM_IT)
    # x ≡ 1 — sparsity 합성이 x0=x1=1을 전제한다 (common.py의 역산).
    x = torch.ones(m, k, dtype=torch.bfloat16, device=dev)
    ids = torch.zeros(m, 1, dtype=torch.int32, device=dev)
    tw = torch.ones(m, 1, dtype=torch.float32, device=dev)
    out = torch.zeros(m, 1, n, dtype=torch.bfloat16, device=dev)

    def launch(i: int) -> None:
        gemv_worklist_indexed_pinned_sparse(
            x, ids, tw, w, row_off, kidx, out, spec, 0, x_row_is_pair,
            torch.cuda.current_stream())

    try:
        with nvtx("warm/sparse_gemv"):
            timing = graph_timing(launch, reps, replays=replays)
    finally:
        del w, x, out
        torch.cuda.empty_cache()

    return SparseGemv(where="warm", k_rows=k, n_cols=n, sparsity=sparsity,
                      keep_frac=keep, dense_bytes=m * k * n * 2, timing=timing)
