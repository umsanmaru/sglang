#!/usr/bin/env python
"""N층 × [gateup → down]을 **한 CUDA graph**에 담아 cold 배관을 nsys로 본다.

`nsys_cold_graph.py`는 phase 하나를 격리해 launch마다 replay한다. production의
CudaGraphRunner는 forward 전체를 한 graph에 담으므로 한 launch 안에 cold submit/sync
쌍이 층 수 × 2개 들어가고, 그 사이에 rejoin·다음 층이 끼어 있다. 격리 하니스에서는
원리적으로 안 보이는 것 셋을 여기서 본다:

  1. 콜백 스레드 포화 — host node는 드라이버 콜백 스레드 **하나**에서 직렬로 실행되고
     kt의 sync_는 그 안에서 블로킹한다. 노드 2개가 아니라 4N개일 때 kt.submit /
     kt.sync(wait)가 줄지어 밀리는가.
  2. 층 간 겹침 — L층의 cold가 도는 동안 GPU가 L의 rejoin·L+1의 커널을 진행하는가.
  3. launch 상각 — launch 1회가 phase 하나가 아니라 2N phase에 걸쳐 나뉜다.

층마다 executor와 같은 순서를 밟는다 (executor.py Phase 1/2, `_cold_split_*`):

  gateup: fill_x(D2H) · fill_expert_ids(D2H) · fill_topk_w(D2H) → submit host node
          → hot+warm gateup 커널 → sync host node → partial HtoD → rejoin_gateup
  down:   fill_act(D2H) → submit → hot+warm down 커널 → sync → partial HtoD → rejoin_down

입력은 층마다 x ≡ 1, act ≡ 1을 **새로 먹인다** (격리 하니스와 같은 전제). rejoin 출력을
다음 층에 체인하면 합성 가중치로 2층 만에 값이 0/NaN으로 무너지고, warm·cold의
K-sparse GEMV가 0인 입력 열을 건너뛰어 warm 커널이 150 → 8 µs로 사라진다 (2026-09-05
첫 트레이스에서 실측; dense인 hot은 그대로였다). 층 사이의 실행 의존은 한 스트림에
캡처되므로 데이터를 체인하지 않아도 graph 안에서 직렬 사슬로 남는다. cold 인스턴스
(weight)는 N층이 같은 것을 재사용하므로 CPU 시간은 캐시 덕에 실제보다 짧다 — 배관
구조를 보는 목적에는 CPU가 짧을수록 배관 비용이 도드라지니 오히려 낫다.

변형 (한 run에 전부, 각각 graph 하나):
  gpu_only   N × [hot+warm gateup, rejoin, hot+warm down, rejoin]  — cold 없음 (기준선)
  combined   cold 사슬을 main 스트림에 (production graph 경로, cold_split 꺼진 상태)
  split      cold 사슬을 곁 스트림에 (P2 = executor의 `_cold_side_stream_submit/_join`)

graph 안의 NVTX는 캡처 시점에만 찍히므로 층 경계는 커널 순서로 읽는다: 층 하나 =
[gateup 커널(hot, warm) · _rejoin_gateup · down 커널(hot, warm) · _rejoin_down].
콜백 스레드 행에서는 kt.submit / kt.sync(wait)가 replay당 2N개씩 나온다.

사용:
  nsys profile -t cuda,nvtx --cuda-graph-trace=node \
    --sample=process-tree --cpuctxsw=process-tree \
    -c cudaProfilerApi --capture-range-end=repeat --force-overwrite=true \
    -o <out> python test/prism/nsys_cold_layers.py [n_layers] [warm_fracs] [replays] [variants] \
                                                  [dtype] [hot_frac] [E,topk,hidden,inter] [shuffle]
  예: ... nsys_cold_layers.py 8 0.5 6
      ... nsys_cold_layers.py 8 0.125,0.25,0.375,0.5 6 gpu_only,split   (warm 비율 스윕, 구성마다 파일 1개)
      ... nsys_cold_layers.py 8 0.1,0.2,0.3,0.4,0.5 6 gpu_only,split bf16 0.2 512,10,2560,640  (Qwen3.8-Flash-Next)
      ... nsys_cold_layers.py 8 0.1875 6 gpu_only,combined,split bf16 0.06 512,10,2560,640 1
          (plans/qwen38fn/qwen38_next_exp14gb_block2_sp50.json layer 24의 티어 비율, block2 근사)
"""
from __future__ import annotations

import sys

import torch

from sglang.srt.layers.moe.prism.profile import Shape, FullLayerProfiler
from sglang.srt.layers.moe.prism.profile.common import nvtx

N_LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
WARM_FRACS = ([float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2
              else [0.5])
REPLAYS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
VARIANTS = (sys.argv[4].split(",") if len(sys.argv) > 4
            else ["gpu_only", "combined", "split"])
DTYPE = sys.argv[5] if len(sys.argv) > 5 else "fp8"
HOT_FRAC = float(sys.argv[6]) if len(sys.argv) > 6 else 0.375
# shape "E,topk,hidden,inter" — 기본은 합성 128-expert. Qwen3.8-Flash-Next(text_config):
# experts 512, top-k 10, hidden 2560, moe_intermediate 640 → "512,10,2560,640".
SHAPE = ([int(v) for v in sys.argv[7].split(",")] if len(sys.argv) > 7
         else [128, 8, 2048, 768])
# shuffle=1: 티어 행을 K축에서 무작위(k_align 단위)로 뽑는다 — 컴파일 plan의 block2
# interleave 밴드(예: plans/qwen38fn/*_block2_sp50.json)에 가깝다. 0이면 연속 구간.
SHUFFLE = bool(int(sys.argv[8])) if len(sys.argv) > 8 else False


def run_one(shape: Shape, warm_frac: float) -> None:
    from sglang.srt.layers.moe.prism.profile.full_layer import GpuTier
    from sglang.srt.layers.moe.prism.rejoin import rejoin_down, rejoin_gateup

    with FullLayerProfiler(shape, hot_frac=HOT_FRAC, warm_frac=warm_frac, sparsity=0.5,
                           sparsity_spread=0.3, dtype=DTYPE, device=0, seed=0,
                           shuffle_index=SHUFFLE) as p:
        st, dev, topk, inter, hidden = (p.res.staging, p.device, shape.topk,
                                        shape.inter, shape.hidden)
        qlen = p._qlen.data_ptr()
        cold = p.cold.wrapper
        side = p.res.cold_stream
        assert side is not None, "res.cold_stream이 없다 — split 변형을 담을 수 없다"

        tw = torch.full((1, topk), 1.0 / topk, dtype=torch.float32, device=dev)
        st.fill_topk_w(tw)
        w_ptr = st.topk_w_ptr()
        # 층마다 **서로 겹치지 않는** expert 묶음. 독립 randperm으로 뽑으면 층 사이에
        # expert가 겹쳐 한 replay의 warm footprint가 GPU L2(5090: 96 MB)에 들어가고,
        # 층 3 이후 warm 커널이 PCIe(60~160 µs)가 아니라 L2(8 µs)로 돈다 — 2026-09-05
        # 첫 트레이스에서 실측. production은 층마다 가중치가 다르니 그 상주가 없다.
        # N·topk ≤ E이면 disjoint로 자르고, 넘으면 randperm을 이어 붙인다.
        E = shape.experts
        g = torch.Generator().manual_seed(1234)
        pool = torch.cat([torch.randperm(E, generator=g)
                          for _ in range(-(-N_LAYERS * topk // E))])
        ids_d = [pool[L * topk:(L + 1) * topk].to(torch.int64).reshape(1, topk)
                 .contiguous().to(dev) for L in range(N_LAYERS)]
        x0 = torch.ones(1, hidden, dtype=torch.bfloat16, device=dev)
        act0 = torch.ones(1, topk, inter, dtype=torch.bfloat16, device=dev)

        # 출력 버퍼는 층마다 따로 — 같은 버퍼를 재사용하면 graph가 없는 의존(WAR)을
        # 만들지는 않지만(스트림 순서라 어차피 직렬) 읽기 편하게 분리해 둔다.
        out_gu = [(torch.zeros(1, topk, 2 * inter, dtype=torch.bfloat16, device=dev),
                   torch.zeros(1, topk, 2 * inter, dtype=torch.bfloat16, device=dev))
                  for _ in range(N_LAYERS)]
        out_dn = [(torch.zeros(1, topk, hidden, dtype=torch.bfloat16, device=dev),
                   torch.zeros(1, topk, hidden, dtype=torch.bfloat16, device=dev))
                  for _ in range(N_LAYERS)]

        def gpu_gateup(tiers, x, ids, out):
            GpuTier.launch_gateup(tiers["gate"], tiers["up"], x, ids, tw, out, inter)

        def gpu_down(tiers, act, ids, out):
            tiers["down"].launch(act.view(topk, inter), ids, tw, out, 0,
                                 x_row_is_pair=True)

        # ── cold 한 phase의 두 반쪽: [입력 D2H → submit]과 [sync → partial HtoD] ──
        def cold_submit(phase: str, x_or_act, ids, stream):
            s = stream.cuda_stream
            if phase == "gateup":
                st.fill_x(x_or_act, non_blocking=True)
                st.fill_expert_ids(ids, non_blocking=True)
                st.fill_topk_w(tw, non_blocking=True)
                cold.submit_forward_gateup(qlen, topk, st.expert_ids_ptr(), st.x_ptr(),
                                           st.partial_gateup_ptr(), s, w_ptr)
            else:
                st.fill_act(x_or_act, non_blocking=True)
                cold.submit_forward_down(qlen, topk, st.expert_ids_ptr(), st.act_ptr(),
                                         st.partial_down_ptr(), s, w_ptr)

        def cold_join(phase: str, stream):
            cold.sync(stream.cuda_stream)
            src = st.gateup_out(1) if phase == "gateup" else st.down_out(1)
            return src.to(dev, non_blocking=True)                # HtoD

        # ── 층 하나: mode ∈ {gpu_only, combined, split} ──────────────────
        def layer(L: int, x, mode: str):
            ids = ids_d[L]
            x = x0                          # 체인하지 않는다 — 위 docstring
            main = torch.cuda.current_stream()
            hot_gu, warm_gu = out_gu[L]
            hot_dn, warm_dn = out_dn[L]

            # gateup
            cold_gu = None
            if mode == "combined":
                cold_submit("gateup", x, ids, main)
            elif mode == "split":
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    cold_submit("gateup", x, ids, side)
            gpu_gateup(p.hot, x, ids, hot_gu)
            gpu_gateup(p.warm, x, ids, warm_gu)
            if mode == "combined":
                cold_gu = cold_join("gateup", main)
            elif mode == "split":
                with torch.cuda.stream(side):
                    cold_gu = cold_join("gateup", side)
                main.wait_stream(side)
            rejoin_gateup([hot_gu, warm_gu, cold_gu], inter)   # 출력은 안 쓴다 (값 퇴화)
            act = act0

            # down
            cold_dn = None
            if mode == "combined":
                cold_submit("down", act, ids, main)
            elif mode == "split":
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    cold_submit("down", act, ids, side)
            gpu_down(p.hot, act, ids, hot_dn)
            gpu_down(p.warm, act, ids, warm_dn)
            if mode == "combined":
                cold_dn = cold_join("down", main)
            elif mode == "split":
                with torch.cuda.stream(side):
                    cold_dn = cold_join("down", side)
                main.wait_stream(side)
            return rejoin_down([hot_dn, warm_dn, cold_dn], tw)

        def forward(mode: str):
            x = x0
            for L in range(N_LAYERS):
                x = layer(L, x, mode)
            return x

        variants = list(VARIANTS)
        graphs = []
        for mode in variants:
            fn = lambda mode=mode: forward(mode)
            s0 = torch.cuda.Stream(); s0.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s0):
                for _ in range(2):
                    fn()
            torch.cuda.current_stream().wait_stream(s0); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            for _ in range(2):
                g.replay()
            torch.cuda.synchronize()
            graphs.append((mode, g))

        sp = p.splits["down"]
        rows = lambda t: round(sum(sp.rows(t)) / E)
        tag = (f"{DTYPE}_E{shape.experts}k{shape.topk}h{shape.hidden}i{shape.inter}_"
               f"w{warm_frac:g}_hot{rows('hot')}_warm{rows('warm')}_cold{rows('cold')}"
               + ("_shuf" if SHUFFLE else ""))

        # 벽시계도 같이 남긴다 — 트레이스 없이도 N-스케일 비교가 되게.
        # launch 비용도 잰다 — host node가 든 graph는 cudaGraphLaunch가 **앞선 launch가
        # 끝날 때까지 블로킹**한다 (2026-09-05 nsys: launch 길이 ≈ replay 길이). 그래서
        # 첫 launch(GPU 유휴 상태)만 순수 CPU 발행 비용이고 나머지는 대기가 섞인다.
        import time
        wall, launch_us = {}, {}
        for mode, g in graphs:
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); ts = []
            for _ in range(REPLAYS):
                c0 = time.perf_counter(); g.replay(); ts.append((time.perf_counter() - c0) * 1e6)
            e1.record(); torch.cuda.synchronize()
            wall[mode] = e0.elapsed_time(e1) * 1000 / REPLAYS
            launch_us[mode] = [round(t) for t in ts]

        torch.cuda.profiler.start()
        for mode, g in graphs:
            with nvtx(f"NSYS/layers{N_LAYERS}/{tag}/{mode}"):
                for _ in range(REPLAYS):
                    g.replay()
                torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        per = {m: round(v / N_LAYERS, 1) for m, v in wall.items()}
        print(f"[nsys] footprint {p.footprint}", flush=True)
        print(f"[nsys] cudaGraphLaunch CPU µs per call: {launch_us}", flush=True)
        print(f"[nsys] layers={N_LAYERS} {tag}: {variants} × {REPLAYS} replays; "
              f"µs/launch {dict((m, round(v, 1)) for m, v in wall.items())}; "
              f"µs/layer {per}", flush=True)


def main() -> None:
    shape = Shape(experts=SHAPE[0], topk=SHAPE[1], hidden=SHAPE[2], inter=SHAPE[3])
    for w in WARM_FRACS:
        run_one(shape, w)
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
