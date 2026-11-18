"""구성 요소 프로파일 — MoE decode 레이어 한 phase를 **조각마다 따로** 잰다.

closure.py의 세 티어 호출(hot / warm / cold host 루프)만으로는 한 레이어가 재현되지
않는다 (fp8 down: 티어 셋의 합성 46 µs vs 실측 layer 104 µs). 티어 사이의 조각 —
입력 D2H, cold host node의 submit/sync 왕복, partial H2D, rejoin — 이 크기 때문이다.
이 모듈은 그 조각을 하나씩 잰다. 합치는 규칙은 `closure.compose_exec`(순수 함수)다.

## 같은 스토어, 같은 회전

모든 조각은 `FullLayerProfiler` 인스턴스 **하나**의 스토어·kt 인스턴스·staging 위에서
잰다 (`self.fl`). 비교 대상(full-layer `combined`/`layer`)도 같은 인스턴스가 재므로,
조각의 합과 기준의 차이에 "스토어가 달라서"나 "CPU 풀이 두 벌이라서"가 섞이지 않는다.
expert 회전은 full-layer와 같은 `_ids(reps)`(seed 1234)다.

## 조각 (group = gateup | down)

  GPU 커널 (graph, reps개를 한 graph에)
    hot          hot 티어 커널 (gateup은 gate+up 융합 한 launch)
    warm         warm 티어 커널 — 런타임이 고르는 경로 (`warm_path`, 아래)
    gpu_seq      hot → warm 연속 (더해지는지 확인용)
    rejoin       rejoin_gateup (act 포함) / rejoin_down (router 가중합 포함)
  전송 (graph)
    d2h_ids      expert_ids device → pinned (64 B)
    d2h_x        gateup 입력 x (bf16 [1, H])           — 런타임만, full-layer는 graph 밖
    d2h_topk_w   router 가중 fp32 [1, k]               — 런타임만 (gateup, masking)
    d2h_act      down 입력 act (bf16 [1, k, I])         — 런타임만
    h2d_partial  cold partial pinned → device
  cold (CPU kt)
    cold_host    host 루프 동기 forward (= ColdCpuProfiler가 보고하는 형태)
    cold_graph   graph 안 [d2h_ids → submit host node → sync host node], 사이 GPU 일 없음
    cold_graph_tail  cold_graph 뒤에 partial H2D + rejoin을 붙인 것 (`_h2d`, `_rejoin`은 하나씩)
                 — host node 완료 뒤 첫 GPU 노드가 내는 전이 비용까지 담은 합류 꼬리
    cold_sleep   같은 것 + submit과 sync 사이에 길이를 아는 GPU 커널(`torch.cuda._sleep`)
                 — 길이 L을 훑어 "GPU가 길 때 남는 고정비"(handshake_gpu)를 읽는다
    cold_host_pcie  host 루프 cold를 **warm 커널이 PCIe로 pinned를 읽는 동안** 잰 것
                 (곁 스트림에서 warm graph를 돌려 둔다) — CPU DRAM 대역폭 경쟁의 크기
  기준선
    empty        1원소 zero_ 커널 하나 (graph 노드 하나의 고정비)
    flush        L2Flush 단독 (차분의 잡음 = 두 번 잰 값의 차)

## 조건 (요인으로 다룬다, 판단으로 고르지 않는다)

  rotate   expert 회전 on/off (off = 모든 iteration이 같은 top_k)
  flush    GPU 조각 앞 L2 flush (차분) on/off. cold를 포함한 조각에는 걸지 않는다
           (full-layer와 같은 규약 — `FullLayerProfiler.measure`의 VARIANTS 주석).
  cpu_flush  cold host 루프 사이에 L3보다 큰 버퍼를 훑는가 (CPU 캐시 대응물)
그리고 kt 인스턴스 쪽 조건(cold를 다시 굽는다): `max_len`, gate/up 인덱스 독립 여부.

## warm 경로

bf16은 런타임 기본이 packed warm-kt slab(`tiers.PackedWarmGateUp`)이다
(`runtime/config.py warm_kt_enabled`). 이 모듈은 런타임 설정에 묻고(`closure.runtime_path`)
그 결과를 리포트에 적는다. bf16이면 `build_packed_warm()`이 런타임과 같은 packed slab을
따로 굽고 `warm_packed`(+ `gpu_seq_packed`)를 함께 잰다. full-layer 기준은 row-major warm을
쓰므로 그 기준과의 합성에는 `warm`을, 런타임 기준에는 `warm_packed`를 쓴다
(`closure.exec_variants`의 `runtime+packed`).
"""

from __future__ import annotations

import gc
import os
import time
from typing import Callable, Mapping, Optional, Sequence

import torch

from sglang.srt.layers.moe.prism.profile.common import (
    Shape,
    Timing,
    default_cpuinfer_threads,
    env_stamp,
    graph_timing,
    host_timing,
    nvtx,
    store_of,
)
from sglang.srt.layers.moe.prism.profile.full_layer import GROUPS

# cold_sleep가 훑는 L — cold_graph(C)의 배수. 1.0 위쪽 점들이 handshake_gpu를 준다.
SLEEP_FRACS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
FACTORS = ("rotate", "flush")


def _l3_bytes() -> int:
    try:
        raw = open("/sys/devices/system/cpu/cpu0/cache/index3/size").read().strip()
    except OSError:
        return 0
    mult = {"K": 1024, "M": 1024 ** 2}.get(raw[-1], 1)
    return int(raw[:-1] if raw[-1] in "KM" else raw) * mult


def _diff(work: Timing, base: Timing) -> Timing:
    return Timing(us=round(work.us - base.us, 3), min_us=round(work.min_us - base.min_us, 3),
                  max_us=round(work.max_us - base.max_us, 3),
                  p90_us=round(work.p90_us - base.p90_us, 3), replays=work.replays)


def _median_of(ts: Sequence[Timing]) -> Timing:
    """라운드 간 중앙값 — full_layer와 같이 한 라운드의 통계를 통째로 가져온다."""
    return sorted(ts, key=lambda t: t.us)[len(ts) // 2]


class ComponentProfiler:
    """한 점(shape, 티어 비율, sparsity, dtype)의 구성 요소 프로파일러.

    생성이 비싸다 (FullLayerProfiler = 세 티어 스토어 + kt 패킹). with 문으로 쓴다.
    `max_len`/`independent_gateup`은 kt cold 인스턴스를 그 조건으로 다시 굽는다 —
    full-layer 기준도 같은 인스턴스를 쓰므로 양쪽에 함께 걸린다.
    """

    def __init__(self, shape: Shape, *, hot_frac: float, warm_frac: float,
                 sparsity: float, dtype: str = "bf16", cpu_kernel: Optional[str] = None,
                 threads: Optional[int] = None, device=0, numa_split: float = 0.5,
                 sparsity_spread: float = 0.0, seed: int = 0,
                 max_len: int = 1, independent_gateup: bool = False, n_layers: int = 1):
        from sglang.srt.layers.moe.prism.profile.full_layer import FullLayerProfiler

        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.shape = shape
        self.store = store_of(dtype)
        self.cpu_kernel = cpu_kernel or self.store.cpu_kernel
        self.threads = threads or default_cpuinfer_threads()
        self.fl = FullLayerProfiler(
            shape, hot_frac=hot_frac, warm_frac=warm_frac, cold_frac=None,
            sparsity=sparsity, sparsity_spread=sparsity_spread, dtype=dtype,
            device=device, numa_split=numa_split, cpu_kernel=self.cpu_kernel,
            threads=self.threads, seed=seed)
        self.device = self.fl.device
        self.conditions = {"max_len": int(max_len), "independent_gateup": bool(independent_gateup),
                           "numa_split": numa_split, "threads": self.threads,
                           "sparsity_spread": sparsity_spread, "n_layers": int(n_layers)}
        if max_len != 1 or independent_gateup:
            self._rebuild_cold(max_len=max_len, independent_gateup=independent_gateup,
                               numa_split=numa_split, seed=seed)
        if max_len != 1:
            # 런타임은 max_tokens 하나를 kt max_len과 staging 용량에 함께 쓴다
            # (runtime/context.py → KtColdBackend(max_tokens) · ResourceSpec(max_tokens)).
            from sglang.srt.layers.moe.prism.resources import ExecutionResources, ResourceSpec

            self.fl.res = ExecutionResources(ResourceSpec(
                max_tokens=int(max_len), top_k=shape.topk, hidden_size=shape.hidden,
                intermediate_size=shape.inter, device=self.device))
        # 층 0 = full-layer 인스턴스의 티어. 나머지 층은 같은 분할·다른 seed로 따로 굽고
        # kt 풀은 공유한다 (런타임: 층마다 인스턴스, CPUInfer 하나).
        self.layers = [{"hot": self.fl.hot, "warm": self.fl.warm, "cold": self.fl.cold}]
        for l in range(1, n_layers):
            self.layers.append(self._build_layer(l, max_len=max_len,
                                                 independent_gateup=independent_gateup,
                                                 numa_split=numa_split, seed=seed))
        self.point = {"hot_frac": hot_frac, "warm_frac": warm_frac, "sparsity": sparsity,
                      "dtype": self.store.name, "cpu_kernel": self.cpu_kernel}
        self._sleep_cycles_per_us: Optional[float] = None

    # ── 수명 ─────────────────────────────────────────────────────────────
    def close(self) -> None:
        self._packed = None
        self._packed_wrappers = None
        self.layers = None
        if self.fl is not None:
            self.fl.close()
        self.fl = None
        gc.collect()

    def __enter__(self) -> "ComponentProfiler":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── 층 여러 개 (캐시 조건) ────────────────────────────────────────────
    def _cold_index(self, independent_gateup: bool, seed: int) -> dict:
        from sglang.srt.layers.moe.prism.profile.common import tier_indices

        fl = self.fl
        idx = {p: fl.splits[p].cold_idx for p in fl.splits}
        if independent_gateup:
            sp = fl.splits["up"]
            skip = tuple(h + w for h, w in zip(sp.hot, sp.warm))
            # cold 밴드(skip 뒤 행들) 안에서만 다른 순서로 고른다 — 행 **집합**의 크기는 같고
            # 원소가 gate와 달라진다 (shuffle=True, 다른 seed).
            idx["up"], _ = tier_indices(sp.k_axis, sp.cold, self.shape.experts, skip=skip,
                                        shuffle=True, seed=seed + 7919)
        return idx

    def _make_cold(self, idx: dict, *, max_len: int, numa_split: float, cpuinfer=None):
        from sglang.srt.layers.moe.prism.profile.warm_cold import ColdTier, Split

        fl = self.fl
        cold_splits = {p: Split(proj=p, axis=fl.splits[p].k_axis, n_cols=fl.splits[p].n_cols,
                                warm_rows=fl.splits[p].warm_idx, cold_rows=idx[p])
                       for p in fl.splits}
        return ColdTier(
            self.shape, cold_splits, sparsity=fl.sparsity, pattern="random", seed=0,
            numa_split=numa_split, threads=self.threads, kernel_key=self.cpu_kernel,
            dtype=self.store, rows_per_expert={p: fl.splits[p].cold for p in fl.splits},
            index_per_expert=idx, x_levels=fl.x_levels, max_len=max_len, cpuinfer=cpuinfer)

    def _build_layer(self, l: int, *, max_len: int, independent_gateup: bool,
                     numa_split: float, seed: int) -> dict:
        """층 l의 hot/warm/cold — 분할(행 집합)은 층 0과 같고 weight 값(seed)만 다르다.
        cold는 층 0의 CPUInfer 풀을 공유한다 (풀 두 벌은 서로의 cold를 왜곡한다)."""
        from sglang.srt.layers.moe.prism.profile.full_layer import GpuTier

        fl = self.fl
        hot = {p: GpuTier(self.shape, fl.splits[p], "hot", device=self.device,
                          dtype=self.store, seed=seed + 1 + 100 * l) for p in fl.splits}
        warm = {p: GpuTier(self.shape, fl.splits[p], "warm", sparsity=fl.sparsity,
                           pattern="random", device=self.device, node=fl.warm_node,
                           dtype=self.store, seed=seed + 2 + 100 * l, per_k=fl.per_k,
                           x_levels=None if fl.x_levels is None else fl.x_levels[p])
                for p in fl.splits}
        cold = self._make_cold(self._cold_index(independent_gateup, seed), max_len=max_len,
                               numa_split=numa_split, cpuinfer=fl.cold.cpuinfer)
        return {"hot": hot, "warm": warm, "cold": cold}

    # ── kt 조건 ──────────────────────────────────────────────────────────
    def _rebuild_cold(self, *, max_len: int, independent_gateup: bool,
                      numa_split: float, seed: int) -> None:
        """cold kt 인스턴스를 다른 조건으로 다시 굽는다 (fl.cold를 교체 — full-layer 기준도
        이 인스턴스를 쓴다).

        independent_gateup: up의 cold 인덱스를 gate와 다른 순열로 (CONTRACTS — 런타임
        인덱스는 proj마다 독립이다; build_splits는 같은 seed로 gate/up에 같은 인덱스를 준다).
        max_len: kt MOEConfig.max_len (런타임은 max_tokens) — ColdTier의 인자로 넘긴다.
        """
        fl = self.fl
        idx = self._cold_index(independent_gateup, seed)
        old = fl.cold
        fl.cold = None
        del old
        gc.collect()
        fl.cold = self._make_cold(idx, max_len=max_len, numa_split=numa_split)

    # ── warm-kt (packed slab) ────────────────────────────────────────────
    def build_packed_warm(self) -> dict:
        """bf16 warm을 런타임 warm-kt와 같은 형태로 굽는다 → {"gateup": PackedWarmGateUp,
        "down": PackedWarmTier}.

        `KtColdBackend.load_warm_layer`(cold_backend.py:285-322)를 Plan 없이 옮긴 것이다:
        warm 행으로 kt partial 인스턴스를 하나 더 만들되 N shard를 GPU-local 노드에 전량
        두고(다른 노드 0행), kt가 pack한 slab(`cold_slab_views`)을 host-register해서 GPU가
        packed 레이아웃으로 읽는다 (`cold_gpu.build_cold_gpu_layer`의 ColdSlab 구성과 같음).
        warm 행 수는 `fl.splits`의 warm 그대로이고 sparse 재료(a/c/thr)는 row-major warm
        티어와 **같은 SparseSpec**을 쓴다 — 두 경로가 같은 페어 집합을 본다.
        행 수가 kt 타일(32)의 배수가 아니면 패딩이 필요해 ValueError다.
        """
        if self.store.name != "bf16":
            raise ValueError(f"warm-kt packed path exists for bf16 only, not {self.store.name}")
        if getattr(self, "_packed", None) is not None:
            return self._packed
        from kt_kernel import kt_kernel_ext
        from kt_kernel.experts_partial import PartialMoEWrapper

        from sglang.srt.layers.moe.prism.cold_gpu import ColdSlab, _register_host
        from sglang.srt.layers.moe.prism.formats import FORMATS
        from sglang.srt.layers.moe.prism.plan import Proj
        from sglang.srt.layers.moe.prism.profile.common import numa_nodes
        from sglang.srt.layers.moe.prism.tiers import PackedWarmGateUp, PackedWarmTier

        fl, shape = self.fl, self.shape
        E, H, I = shape.experts, shape.hidden, shape.inter
        local = int(fl.warm_node)
        nodes = numa_nodes()
        tile = 32
        for p in ("gate", "up", "down"):
            if any(r % tile for r in fl.splits[p].warm):
                raise ValueError(f"{p}: warm rows {sorted(set(fl.splits[p].warm))} not a multiple "
                                 f"of kt tile {tile} — packed warm needs padding (not modeled)")

        def table(full):
            return ([full if i != local else 0 for i in range(nodes)],
                    [0 if i != local else full for i in range(nodes)])

        self._packed_wrappers = []           # slab 메모리의 소유자 — 살려 둔다
        self._packed = [self._packed_layer(L["warm"], l, local, nodes, table)
                        for l, L in enumerate(self.layers)]
        return self._packed

    def _packed_layer(self, warm_tiers, l: int, local: int, nodes: int, table) -> dict:
        from kt_kernel import kt_kernel_ext
        from kt_kernel.experts_partial import PartialMoEWrapper

        from sglang.srt.layers.moe.prism.cold_gpu import ColdSlab, _register_host
        from sglang.srt.layers.moe.prism.formats import FORMATS
        from sglang.srt.layers.moe.prism.plan import Proj
        from sglang.srt.layers.moe.prism.tiers import PackedWarmGateUp, PackedWarmTier

        fl, shape = self.fl, self.shape
        E, H, I = shape.experts, shape.hidden, shape.inter
        cfg = kt_kernel_ext.moe.MOEConfig(E, shape.topk, H, I, 0)
        cfg.max_len = self.conditions["max_len"]
        cfg.layer_idx = 0
        cfg.partial.enabled = True
        cfg.partial.n_total = I
        for p, ki in (("gate", cfg.partial.gate), ("up", cfg.partial.up),
                      ("down", cfg.partial.down)):
            rows = fl.splits[p].warm
            off = [0]
            for r in rows:
                off.append(off[-1] + r)
            ki.row_off = off
            ki.idx = fl.splits[p].warm_idx.to(torch.int32).tolist()
        (gu_off, gu_rows), (dn_off, dn_rows) = table(I), table(H)
        cfg.partial.node_gateup_n_offset = gu_off
        cfg.partial.node_gateup_n_rows = gu_rows
        cfg.partial.node_down_n_offset = dn_off
        cfg.partial.node_down_n_rows = dn_rows
        cfg.pool = fl.cold.cpuinfer.backend_
        wrapper = PartialMoEWrapper(cfg, fl.cold.cpuinfer, kernel_key=self.cpu_kernel)
        w = {p: self.store.cold_store(E, shape.n_cols(p), fl.splits[p].warm,
                                      seed=5 + i + 100 * l)[0]
             for i, p in enumerate(("gate", "up", "down"))}
        wrapper.load_weights_from_tensors(w["gate"], w["up"], w["down"])
        fmt = FORMATS["bf16"]
        pmap = {0: Proj.GATE, 1: Proj.UP, 2: Proj.DOWN}
        names = {Proj.GATE: "gate", Proj.UP: "up", Proj.DOWN: "down"}
        slabs = {}
        for v in wrapper.cold_slab_views():
            if int(v["node"]) != local:
                continue
            proj = pmap[int(v["proj"])]
            sp = fl.splits[names[proj]]
            k_rows = list(sp.warm)
            if list(v["k"]) != k_rows:
                raise ValueError(f"{proj.value}: kt packed k {list(v['k'])[:4]} != warm rows "
                                 f"{k_rows[:4]}")
            expert_off = [int(o) for o in v["expert_off"]]
            ptr, nbytes = int(v["ptr"]), int(v["bytes"])
            _register_host(ptr, nbytes)
            slab, blk_off = fmt.cold_slab(ptr, nbytes, expert_off, self.device)
            ro = torch.tensor([0] + list(torch.tensor(k_rows).cumsum(0).tolist()),
                              dtype=torch.int32)
            slabs[proj] = ColdSlab(
                slab=slab, blk_off=blk_off,
                expert_offsets=tuple(o // slab.element_size() for o in expert_off),
                row_off=ro.to(self.device), k_index=sp.warm_idx.to(torch.uint16).to(self.device),
                n=int(v["n"]), n_start=0, n_block=int(v["n_block"]), k_block=int(v["k_block"]),
                node=local, proj=proj, k_max=max(k_rows), fmt=fmt, layout="kt_bf16_nc",
                rows_host=tuple(ro.tolist()))
        self._packed_wrappers.append(wrapper)
        return {
            "gateup": PackedWarmGateUp(gate=slabs[Proj.GATE], up=slabs[Proj.UP],
                                       gate_spec=warm_tiers["gate"].spec,
                                       up_spec=warm_tiers["up"].spec),
            "down": PackedWarmTier(slab=slabs[Proj.DOWN], spec=warm_tiers["down"].spec),
        }

    def _layer_of(self, i: int) -> int:
        """iteration i가 쓰는 층 — 런타임 graph처럼 층을 돌아가며 (expert 회전과 함께)."""
        return i % len(self.layers)

    def _packed_launch(self, group: str, io, i: int, out) -> None:
        pk = self._packed[self._layer_of(i)][group]
        if group == "gateup":
            pk.run(io["x"], io["ids_dev"][i], io["tw"], out, self.shape.inter, masking=True)
        else:
            pk.run(io["x"], io["ids_dev"][i], io["tw"], out, 0, x_row_is_pair=True,
                   masking=True)

    # ── 입력 ─────────────────────────────────────────────────────────────
    def _io(self, group: str, reps: int, rotate: bool) -> dict:
        fl, shape = self.fl, self.shape
        topk = shape.topk
        st = fl.res.staging
        dev = self.device
        ids_host, ids_dev = fl._ids(reps)
        if not rotate:
            ids_host = [ids_host[0]] * reps
            ids_dev = [ids_dev[0]] * reps
        tw = torch.full((1, topk), 1.0 / topk, dtype=torch.float32, device=dev)
        st.fill_topk_w(tw)
        xh = fl._x_host(group)
        if group == "gateup":
            x = xh.to(dev)
            width = 2 * shape.inter
            cold_in, cold_out = st.x_ptr(), st.partial_gateup_ptr()
            st.fill_x(xh)
            src = st.gateup_out(1)
        else:
            x = xh.reshape(topk, shape.inter).to(dev)
            width = shape.hidden
            cold_in, cold_out = st.act_ptr(), st.partial_down_ptr()
            st.fill_act(xh)
            src = st.down_out(1)
        return {
            "st": st, "ids_host": ids_host, "ids_dev": ids_dev, "tw": tw, "x": x,
            "x_act3": x.reshape(1, topk, shape.inter) if group == "down" else None,
            "cold_in": cold_in, "cold_out": cold_out, "src": src,
            "out_hot": torch.zeros(1, topk, width, dtype=torch.bfloat16, device=dev),
            "out_warm": torch.zeros(1, topk, width, dtype=torch.bfloat16, device=dev),
            "cold_dev": torch.zeros(1, topk, width, dtype=torch.bfloat16, device=dev),
            "qlen_ptr": fl._qlen.data_ptr(), "w_ptr": st.topk_w_ptr(),
        }

    def _gpu(self, group: str, tier: str, io, i: int, out) -> None:
        from sglang.srt.layers.moe.prism.profile.full_layer import GpuTier

        tiers = self.layers[self._layer_of(i)][tier]

        if group == "gateup":
            GpuTier.launch_gateup(tiers["gate"], tiers["up"], io["x"], io["ids_dev"][i],
                                  io["tw"], out, self.shape.inter)
        else:
            tiers["down"].launch(io["x"], io["ids_dev"][i], io["tw"], out, 0,
                                 x_row_is_pair=True)

    # ── 측정 원시 ────────────────────────────────────────────────────────
    def sleep_cycles_per_us(self) -> float:
        """`torch.cuda._sleep`의 cycle→µs 환산 (graph 안에서 실측)."""
        if self._sleep_cycles_per_us is None:
            cyc = 200_000
            t = graph_timing(lambda i: torch.cuda._sleep(cyc), 20, replays=10)
            self._sleep_cycles_per_us = cyc / max(t.us, 1e-3)
        return self._sleep_cycles_per_us

    def _cold_steps(self, io, group: str) -> tuple:
        st = io["st"]
        name = "submit_forward_gateup" if group == "gateup" else "submit_forward_down"
        wrappers = [L["cold"].wrapper for L in self.layers]

        def submit(i: int) -> None:
            st.fill_expert_ids(io["ids_dev"][i], non_blocking=True)
            s = torch.cuda.current_stream().cuda_stream
            getattr(wrappers[self._layer_of(i)], name)(
                io["qlen_ptr"], self.shape.topk, st.expert_ids_ptr(), io["cold_in"],
                io["cold_out"], s, io["w_ptr"])

        def sync(i: int) -> None:
            wrappers[self._layer_of(i)].sync(torch.cuda.current_stream().cuda_stream)

        def host(i: int) -> None:
            st.fill_expert_ids(io["ids_host"][i])
            w = wrappers[self._layer_of(i)]
            getattr(w, name)(io["qlen_ptr"], self.shape.topk, st.expert_ids_ptr(),
                             io["cold_in"], io["cold_out"], None, io["w_ptr"])
            w.sync(None)

        return submit, sync, host

    def _cpu_flush_buf(self) -> torch.Tensor:
        l3 = _l3_bytes() or 32 * 2 ** 20
        return torch.empty(2 * l3 // 4, dtype=torch.float32)

    def measure_group(self, group: str, *, reps: int = 50, replays: int = 10,
                      rotate: bool = True, flush: bool = False,
                      cpu_flush: bool = False, sleep_fracs: Sequence[float] = SLEEP_FRACS,
                      with_pcie: bool = True, with_seq: bool = True) -> dict:
        """한 group의 조각 전부를 한 번 (한 라운드) 잰다. 반환 {name: Timing}, 보조 정보."""
        if group not in GROUPS:
            raise ValueError(f"unknown group {group!r} (gateup|down)")
        from sglang.srt.layers.moe.prism.rejoin import rejoin_down, rejoin_gateup

        fl, shape = self.fl, self.shape
        io = self._io(group, reps, rotate)
        st = io["st"]
        submit, sync, host = self._cold_steps(io, group)
        tim: dict = {}
        base = fl.flush_timing(reps, replays) if flush else None

        def gtime(name: str, fn: Callable[[int], None], use_flush: bool = False) -> None:
            with nvtx(f"comp/{group}/{name}"):
                if use_flush and base is not None:
                    def wf(i: int) -> None:
                        fl.flush()
                        fn(i)
                    tim[name] = _diff(graph_timing(wf, reps, replays=replays), base)
                else:
                    tim[name] = graph_timing(fn, reps, replays=replays)

        tiny = torch.zeros(1, device=self.device)
        gtime("empty", lambda i: tiny.zero_())
        gtime("hot", lambda i: self._gpu(group, "hot", io, i, io["out_hot"]), flush)
        gtime("warm", lambda i: self._gpu(group, "warm", io, i, io["out_warm"]), flush)
        if getattr(self, "_packed", None) is not None:
            gtime("warm_packed", lambda i: self._packed_launch(group, io, i, io["out_warm"]),
                  flush)

            def gpu_seq_packed(i: int) -> None:
                self._gpu(group, "hot", io, i, io["out_hot"])
                self._packed_launch(group, io, i, io["out_warm"])
            gtime("gpu_seq_packed", gpu_seq_packed, flush)

        def gpu_seq(i: int) -> None:
            self._gpu(group, "hot", io, i, io["out_hot"])
            self._gpu(group, "warm", io, i, io["out_warm"])
        gtime("gpu_seq", gpu_seq, flush)

        parts = [io["out_hot"], io["out_warm"], io["cold_dev"]]
        if group == "gateup":
            gtime("rejoin", lambda i: rejoin_gateup(parts, shape.inter), flush)
        else:
            gtime("rejoin", lambda i: rejoin_down(parts, io["tw"]), flush)

        gtime("d2h_ids", lambda i: st.fill_expert_ids(io["ids_dev"][i], non_blocking=True))
        if group == "gateup":
            gtime("d2h_x", lambda i: st.fill_x(io["x"], non_blocking=True))
            gtime("d2h_topk_w", lambda i: st.fill_topk_w(io["tw"], non_blocking=True))
        else:
            gtime("d2h_act", lambda i: st.fill_act(io["x_act3"], non_blocking=True))
        gtime("h2d_partial", lambda i: io["src"].to(self.device, non_blocking=True))

        # cold — 세 형태
        buf = self._cpu_flush_buf() if cpu_flush else None

        def host_step(i: int) -> None:
            if buf is not None:
                buf.add_(1.0)             # L3보다 큰 버퍼를 읽고 쓴다 (타이밍 안 — 아래서 뺀다)
            host(i)

        tim["cold_host"] = host_timing(host_step, reps, replays=replays)
        if buf is not None:
            tim["cpu_flush"] = host_timing(lambda i: buf.add_(1.0), reps, replays=replays)
            tim["cold_host"] = _diff(tim["cold_host"], tim["cpu_flush"])

        def cold_graph(i: int) -> None:
            submit(i)
            sync(i)
        gtime("cold_graph", cold_graph)

        # 합류 꼬리: sync host node **뒤에** partial H2D·rejoin이 붙을 때의 실효 비용.
        # 단독 h2d_partial/rejoin은 GPU 노드끼리 이어진 값이라 host node 완료 → 다음 노드
        # 전이 비용을 담지 못한다. 그래서 cold_graph에 꼬리를 붙여 차분한다 (GPU 티어 없음).
        def tail_h2d(i: int) -> None:
            cold_graph(i)
            io["src"].to(self.device, non_blocking=True)

        def tail_full(i: int) -> None:
            cold_graph(i)
            cp = io["src"].to(self.device, non_blocking=True)
            ps = [io["out_hot"], io["out_warm"], cp]
            if group == "gateup":
                rejoin_gateup(ps, shape.inter)
            else:
                rejoin_down(ps, io["tw"])

        def tail_rejoin(i: int) -> None:
            cold_graph(i)
            if group == "gateup":
                rejoin_gateup(parts, shape.inter)
            else:
                rejoin_down(parts, io["tw"])
        gtime("cold_graph_h2d", tail_h2d)
        gtime("cold_graph_rejoin", tail_rejoin)
        gtime("cold_graph_tail", tail_full)

        cpu = self.sleep_cycles_per_us()
        C = tim["cold_graph"].us
        sleeps = []
        for f in sleep_fracs:
            if f <= 0:
                continue
            cyc = int(f * C * cpu)

            def sl(i: int, cyc=cyc) -> None:
                torch.cuda._sleep(cyc)

            def cs(i: int, cyc=cyc) -> None:
                submit(i)
                torch.cuda._sleep(cyc)
                sync(i)
            t_sl = graph_timing(sl, reps, replays=replays)
            t_cs = graph_timing(cs, reps, replays=replays)
            sleeps.append({"frac": f, "sleep_us": t_sl.us, "cold_sleep_us": t_cs.us,
                           "cold_sleep": t_cs.as_dict()})
        if sleeps and max(sleep_fracs) >= 2.0:
            # GPU가 임계일 때의 합류 꼬리: 가장 긴 sleep 점에 H2D + rejoin을 붙여 차분한다.
            # (CPU가 임계일 때의 꼬리 = cold_graph_tail − cold_graph와 다를 수 있다 — sync
            # host node가 GPU 완료 직후에 도는 경우와 CPU 완료를 기다리는 경우.)
            cyc = int(max(sleep_fracs) * C * cpu)

            def cs_tail(i: int, cyc=cyc) -> None:
                submit(i)
                torch.cuda._sleep(cyc)
                sync(i)
                cp = io["src"].to(self.device, non_blocking=True)
                ps = [io["out_hot"], io["out_warm"], cp]
                if group == "gateup":
                    rejoin_gateup(ps, shape.inter)
                else:
                    rejoin_down(ps, io["tw"])
            tim["cold_sleep_tail"] = graph_timing(cs_tail, reps, replays=replays)
            tim["cold_sleep_long"] = Timing(**{**sleeps[-1]["cold_sleep"]})

        if with_pcie:
            tim["cold_host_pcie"] = self._cold_under_warm(group, io, host, reps, replays)
            # graph 안의 간섭: submit과 sync 사이에 **진짜 warm 커널**을 둔 것과 같은 길이의
            # sleep을 둔 것의 차. GPU 갈래 구조(H·의존)는 같고 다른 것은 warm의 pinned
            # 읽기 하나뿐이라, 차가 곧 cold가 warm 트래픽 때문에 늦어진 몫이다 (CPU 임계일 때).
            wcyc = int(tim["warm"].us * cpu)

            def c_warm(i: int) -> None:
                submit(i)
                self._gpu(group, "warm", io, i, io["out_warm"])
                sync(i)

            def c_wsleep(i: int, cyc=wcyc) -> None:
                submit(i)
                torch.cuda._sleep(cyc)
                sync(i)
            gtime("cold_warm_graph", c_warm)
            gtime("cold_wsleep_graph", c_wsleep)
        if with_seq:
            # 상호 간섭(진단 v3): executor 순서 그대로의 phase 사슬에서 GPU 갈래를 **진짜 hot·warm**
            # 으로 둔 것과 **같은 길이 sleep**으로 둔 것의 차. 두 그래프는 입력 D2H·submit·sync·
            # H2D·rejoin이 같고 다른 것은 GPU 커널의 메모리/PCIe 트래픽 하나뿐이다 — CPU 임계면
            # kt가 그 트래픽에 늦어진 몫, GPU 임계면 warm이 kt 트래픽에 늦어진 몫이 나온다.
            gcyc = int((tim["hot"].us + tim["warm"].us) * cpu)

            def seq_inputs(i: int) -> None:
                if group == "gateup":
                    st.fill_x(io["x"], non_blocking=True)
                    st.fill_topk_w(io["tw"], non_blocking=True)
                else:
                    st.fill_act(io["x_act3"], non_blocking=True)

            def seq_tail() -> None:
                ps = [io["out_hot"], io["out_warm"], io["src"].to(self.device, non_blocking=True)]
                if group == "gateup":
                    rejoin_gateup(ps, shape.inter)
                else:
                    rejoin_down(ps, io["tw"])

            def seq_full(i: int) -> None:
                seq_inputs(i)
                submit(i)
                self._gpu(group, "hot", io, i, io["out_hot"])
                self._gpu(group, "warm", io, i, io["out_warm"])
                sync(i)
                seq_tail()

            def seq_sleep(i: int, cyc=gcyc) -> None:
                seq_inputs(i)
                submit(i)
                torch.cuda._sleep(cyc)
                sync(i)
                seq_tail()
            gtime("seq_full", seq_full)
            gtime("seq_sleep", seq_sleep)
        return {"timings": tim, "sleep": sleeps,
                "flush_base": base.as_dict() if base is not None else None}

    def _cold_under_warm(self, group: str, io, host, reps: int, replays: int) -> Timing:
        """cold host 루프를 warm 커널이 곁 스트림에서 pinned를 읽는 동안 잰다.

        warm graph 하나(reps launch)를 루프 길이보다 길게 반복 replay해 둔다 — 루프가
        끝날 때까지 PCIe 읽기가 계속되게. 루프 시간만 잰다 (GPU 대기는 밖에서).
        """
        from sglang.srt.layers.moe.prism.profile.common import capture

        fl = self.fl
        side = torch.cuda.Stream()
        g = capture(lambda i: self._gpu(group, "warm", io, i, io["out_warm"]), reps)
        # warm 한 graph 길이 추정 → 루프(reps × cold)를 덮을 replay 수
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        g_us = (time.perf_counter() - t0) * 1e6
        for i in range(reps):
            host(i)
        t0 = time.perf_counter()
        for i in range(reps):
            host(i)
        loop_us = (time.perf_counter() - t0) * 1e6
        n_rep = max(2, int(2 * loop_us / max(g_us, 1.0)) + 1)
        per = []
        for _ in range(replays):
            torch.cuda.synchronize()
            with torch.cuda.stream(side):
                for _ in range(n_rep):
                    g.replay()
            time.sleep(20e-6)
            t0 = time.perf_counter()
            for i in range(reps):
                host(i)
            per.append((time.perf_counter() - t0) / reps * 1e6)
            torch.cuda.synchronize()
        del g
        return Timing.of(per)

    # ── 보고 ─────────────────────────────────────────────────────────────
    def bytes_moved(self, group: str) -> dict:
        fl, shape = self.fl, self.shape
        topk, E = shape.topk, shape.experts
        projs = GROUPS[group]
        width = 2 * shape.inter if group == "gateup" else shape.hidden
        warm_keep = fl.warm[projs[0]].keep_frac
        out = {
            "hot": sum(fl.hot[p].bytes_per_launch(topk) for p in projs),
            "warm_dense": sum(fl.warm[p].bytes_per_launch(topk) for p in projs),
            "warm_kept": int(sum(fl.warm[p].bytes_per_launch(topk) for p in projs) * warm_keep),
            "cold_dense": sum(int(topk * (sum(fl.cold.rows[p]) / E) * shape.n_cols(p)
                                  * self.store.elem_bytes) for p in projs),
            "d2h_ids": topk * 8,
            "h2d_partial": topk * width * 2,
            "rejoin": 4 * topk * width * 2,      # 파트 3 읽기 + 출력 (상한)
        }
        if group == "gateup":
            out.update(d2h_x=shape.hidden * 2, d2h_topk_w=topk * 4)
        else:
            out.update(d2h_act=topk * shape.inter * 2)
        return out

    def store_info(self) -> dict:
        fp = self.fl.footprint
        return {**fp, "l2_mb": self.fl.flush.l2_mb, "flush_mb": self.fl.flush.mb,
                "l3_mb_per_socket": round(_l3_bytes() / 2 ** 20, 1)}


# ─── 묶음 측정 ─────────────────────────────────────────────────────────────
def handshake_from_sleep(sleeps: Sequence[Mapping], d2h_ids_us: float,
                         cold_graph_us: float) -> dict:
    """cold_sleep 곡선에서 handshake_gpu를 읽는다 (순수 함수).

    모형: f(L) = d2h_ids + max(H + L, C), C = cold_graph − d2h_ids (GPU 일 없을 때 submit→sync).
    L ≥ C인 점들(GPU가 임계)에서 H = f(L) − L − d2h_ids의 중앙값을 쓴다. 그런 점이 없으면
    가장 긴 L 점 하나로 둔다 (`source`). 각 점에 대해 모형 잔차를 남긴다.
    """
    C = cold_graph_us - d2h_ids_us
    pts = [s for s in sleeps if s["sleep_us"] > 0]
    if not pts:
        raise ValueError("no sleep points")
    gpu_bound = [s for s in pts if s["sleep_us"] >= C * 1.2]
    use = gpu_bound or [max(pts, key=lambda s: s["sleep_us"])]
    hs = sorted(s["cold_sleep_us"] - s["sleep_us"] - d2h_ids_us for s in use)
    H = hs[len(hs) // 2]
    fit = [{"sleep_us": s["sleep_us"], "measured": s["cold_sleep_us"],
            "model": round(d2h_ids_us + max(H + s["sleep_us"], C), 3),
            "resid": round(s["cold_sleep_us"] - (d2h_ids_us + max(H + s["sleep_us"], C)), 3)}
           for s in pts]
    return {"handshake_gpu_us": round(H, 3), "cold_eff_us": round(C, 3),
            "source": "gpu_bound" if gpu_bound else "longest_only",
            "n_points": len(use), "fit": fit}


def measure_components(prof: ComponentProfiler, *, groups: Sequence[str] = tuple(GROUPS),
                       reps: int = 50, replays: int = 10, rounds: int = 3,
                       factors: Optional[Sequence[str]] = None,
                       reference: Optional[Callable[[ComponentProfiler, str], dict]] = None
                       ) -> dict:
    """조각 전부를 `rounds` 라운드 재고 라운드 간 중앙값을 낸다.

    라운드가 바깥이다 (full_layer.measure와 같은 이유 — 드리프트가 조각 간 차이로 둔갑하지
    않게). `reference(prof, group) -> {"us": {name: µs}}`를 주면 **같은 라운드 안에서**
    조각 바로 뒤에 기준을 재고, 기준도 라운드 간 중앙값을 낸다 — 공유 머신에서 cold가
    프로세스 안에서도 5% 넘게 흐르므로(실측 118 vs 125 µs), 조각과 기준을 떨어뜨려
    재면 그 흐름이 합성 오차로 보인다.

    factors: 기준 조건(rotate=True, flush=False) 밖에 추가로 잴 조건들 —
    "no_rotate"(회전 끔), "flush"(GPU 조각 L2 flush 차분), "cpu_flush"(cold host 루프 사이
    L3 훑기). 요인 측정은 1라운드만 한다 (민감도 확인용; 리포트에 rounds가 남는다).
    """
    factors = tuple(factors or ())
    unknown = set(factors) - {"no_rotate", "flush", "cpu_flush"}
    if unknown:
        raise ValueError(f"unknown factors {sorted(unknown)}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    samples: dict = {g: {} for g in groups}
    sleeps: dict = {g: [] for g in groups}
    ref_samples: dict = {g: {} for g in groups}
    for _ in range(rounds):
        for g in groups:
            r = prof.measure_group(g, reps=reps, replays=replays)
            for k, t in r["timings"].items():
                samples[g].setdefault(k, []).append(t)
            sleeps[g].append(r["sleep"])
            if reference is not None:
                for k, v in reference(prof, g)["us"].items():
                    ref_samples[g].setdefault(k, []).append(float(v))
    out = {}
    for g in groups:
        tim = {k: _median_of(v) for k, v in samples[g].items()}
        # sleep 곡선은 점마다 라운드 중앙값
        merged = []
        for j, s in enumerate(sleeps[g][0]):
            vals = sorted(rr[j]["cold_sleep_us"] for rr in sleeps[g])
            sls = sorted(rr[j]["sleep_us"] for rr in sleeps[g])
            merged.append({"frac": s["frac"], "sleep_us": sls[len(sls) // 2],
                           "cold_sleep_us": vals[len(vals) // 2]})
        hs = handshake_from_sleep(merged, tim["d2h_ids"].us, tim["cold_graph"].us)
        # 합류 꼬리는 **라운드 안에서** 짝지은 차의 중앙값 (cold가 라운드마다 ±10% 흐르므로
        # 중앙값끼리의 차는 그 흐름을 그대로 싣는다).
        tails = {}
        for name, (a, b) in {"join_tail": ("cold_graph_tail", "cold_graph"),
                             "join_tail_gpu": ("cold_sleep_tail", "cold_sleep_long"),
                             "warm_interference": ("cold_warm_graph", "cold_wsleep_graph"),
                             "gpu_interference": ("seq_full", "seq_sleep")}.items():
            if a in samples[g] and b in samples[g]:
                d = sorted(x.us - y.us for x, y in zip(samples[g][a], samples[g][b]))
                tails[name] = {"us": round(d[len(d) // 2], 3), "rounds_us": [round(x, 3) for x in d]}
        fac = {}
        if "no_rotate" in factors:
            fac["no_rotate"] = {k: t.us for k, t in prof.measure_group(
                g, reps=reps, replays=replays, rotate=False, with_pcie=False,
                sleep_fracs=())["timings"].items()}
        if "flush" in factors:
            r1 = prof.measure_group(g, reps=reps, replays=replays, flush=True,
                                    with_pcie=False, sleep_fracs=())
            r2_base = prof.fl.flush_timing(reps, replays)
            fac["flush"] = {k: t.us for k, t in r1["timings"].items()}
            fac["flush_noise_us"] = round(abs(r1["flush_base"]["us"] - r2_base.us), 3)
            fac["flush_base_us"] = r1["flush_base"]["us"]
        if "cpu_flush" in factors:
            fac["cpu_flush"] = {k: t.us for k, t in prof.measure_group(
                g, reps=reps, replays=replays, cpu_flush=True, with_pcie=False,
                sleep_fracs=())["timings"].items() if k in ("cold_host", "cpu_flush")}
        out[g] = {
            "components": {k: t.as_dict() for k, t in tim.items()},
            "rounds_us": {k: [t.us for t in v] for k, v in samples[g].items()},
            "sleep_curve": merged, "handshake": hs, "tails": tails,
            "bytes": prof.bytes_moved(g),
            "factors": fac,
            "condition": {"rotate": True, "gpu_flush": False, "cpu_flush": False,
                          "reps": reps, "replays": replays, "rounds": rounds},
        }
        if reference is not None:
            rs = ref_samples[g]
            out[g]["reference"] = {
                "us": {k: sorted(v)[len(v) // 2] for k, v in rs.items()},
                "rounds_us": rs, "interleaved": True}
    return out


def measure_chain(prof: ComponentProfiler, *, reps: int = 48, replays: int = 10,
                  rounds: int = 3) -> dict:
    """phase 경계 항 — gateup 꼬리 뒤에 down이 **곧바로** 이어질 때 생기는 추가 비용.

    run_layer(executor.py:570-599)는 rejoin#1 직후 같은 스트림에서 down의 fill_act D2H →
    submit host node를 낸다. phase를 따로 잰 그래프에는 이 경계(직전 sync host node·rejoin
    뒤의 첫 host node, kt 풀이 방금 끝난 gateup 직후 down을 받는 상태)가 없다. GPU 티어 없이
    cold 사슬만으로 세 그래프를 같은 라운드에 잰다:

      A = gateup [d2h_x·d2h_ids·d2h_topk_w → submit → sync → H2D → rejoin_gateup]
      B = down   [d2h_act → submit → sync → H2D → rejoin_down]
      AB = A 다음 B (층 하나)
      chain = AB − A − B   (라운드 짝 중앙값)
    """
    from sglang.srt.layers.moe.prism.rejoin import rejoin_down, rejoin_gateup

    io = {g: prof._io(g, reps, True) for g in GROUPS}
    steps = {g: prof._cold_steps(io[g], g) for g in GROUPS}
    st = io["gateup"]["st"]
    shape = prof.shape

    def A(i: int) -> None:
        st.fill_x(io["gateup"]["x"], non_blocking=True)
        st.fill_topk_w(io["gateup"]["tw"], non_blocking=True)
        steps["gateup"][0](i)            # ids D2H + submit
        steps["gateup"][1](i)
        cp = io["gateup"]["src"].to(prof.device, non_blocking=True)
        rejoin_gateup([io["gateup"]["out_hot"], io["gateup"]["out_warm"], cp], shape.inter)

    def B(i: int) -> None:
        d = io["down"]
        st.fill_act(d["x_act3"], non_blocking=True)
        s = torch.cuda.current_stream().cuda_stream
        name = "submit_forward_down"
        w = prof.layers[prof._layer_of(i)]["cold"].wrapper
        getattr(w, name)(d["qlen_ptr"], shape.topk, st.expert_ids_ptr(), d["cold_in"],
                         d["cold_out"], s, d["w_ptr"])
        w.sync(s)
        cp = d["src"].to(prof.device, non_blocking=True)
        rejoin_down([d["out_hot"], d["out_warm"], cp], d["tw"])

    def AB(i: int) -> None:
        A(i)
        B(i)

    res = {"A": [], "B": [], "AB": []}
    for _ in range(rounds):
        for k, f in (("A", A), ("B", B), ("AB", AB)):
            res[k].append(graph_timing(f, reps, replays=replays).us)
    ch = sorted(ab - a - b for a, b, ab in zip(res["A"], res["B"], res["AB"]))
    return {"chain_us": round(ch[len(ch) // 2], 3), "rounds_us": res,
            "chain_rounds_us": [round(x, 3) for x in ch]}


def full_layer_reference(prof: ComponentProfiler, group: str, *, reps: int = 50,
                         replays: int = 10, rounds: int = 1) -> dict:
    """기본 비교 대상 — 같은 인스턴스의 FullLayerProfiler `combined`/`layer`."""
    rep = prof.fl.measure(group, reps=reps, replays=replays, flush=False, rounds=rounds,
                          only=("cold_graph", "combined", "layer", "gpu_only",
                                "hot_only", "warm_only"))
    return {"source": "full_layer", "us": {k: t.us for k, t in rep.timings.items()},
            "rounds_us": rep.info.get("rounds_us")}


def env_info(prof: ComponentProfiler) -> dict:
    return {"env": env_stamp(prof.device), "store": prof.store_info(),
            "pid": os.getpid()}
