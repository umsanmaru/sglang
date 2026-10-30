#!/usr/bin/env python
"""CUDA graph 안의 cold 배관을 nsys로 들여다보는 하니스.

`--cuda-graph-trace=node`로 떠야 graph가 통짜 막대가 아니라 **노드별**로 펼쳐진다
(nsys 기본은 `graph`라 안이 안 보인다).

한 변형당 graph에 **iteration 1개**만 담고 그것을 여러 번 replay한다 — reps를 여러
개 담으면 같은 노드가 반복돼 한 phase의 경계를 읽기 어렵다.

보고 싶은 것:
  cold.submit   host node가 앞선 D2H 완료를 관측하느라 기다리는 구간
  kt.submit     kt 콜백 **안**에서 찍히는 마커 = CPU 작업이 실제로 시작된 시점
  kt.sync(wait) 그 range의 폭 = 노출된 CPU 대기
  hot/warm 커널 위 두 마커와 겹치는가 (겹치면 숨은 것)

warm 비율을 여러 개 주면 **한 run에** 전부 담는다. 구성마다 스토어를 다시 구워야
하므로(warm 행 수가 스토어에 굳는다) 한 번에 하나만 살려 두고, 빌드 구간은
capture range 밖으로 뺀다 (`--capture-range-end=repeat`이 있어야 여러 구간이 뜬다).

사용:
  nsys profile -t cuda,nvtx --cuda-graph-trace=node \
    -c cudaProfilerApi --capture-range-end=repeat --force-overwrite=true \
    -o <out> python test/prism/nsys_cold_graph.py [group] [warm_fracs]
  예: ... nsys_cold_graph.py down 0.167,0.333,0.5
"""
from __future__ import annotations

import sys

import torch

from sglang.srt.layers.moe.prism.profile import Shape, FullLayerProfiler
from sglang.srt.layers.moe.prism.profile.common import nvtx

GROUP = sys.argv[1] if len(sys.argv) > 1 else "down"
WARM_FRACS = ([float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2
              else [0.167, 0.333, 0.5])
REPLAYS = 12


def run_one(shape: Shape, warm_frac: float) -> None:
    with FullLayerProfiler(shape, hot_frac=0.375, warm_frac=warm_frac, sparsity=0.5,
                           sparsity_spread=0.3, dtype="fp8", device=0, seed=0) as p:
        st, dev, topk = p.res.staging, p.device, shape.topk
        qlen = p._qlen.data_ptr()
        tw = torch.full((1, topk), 1.0 / topk, dtype=torch.float32, device=dev)
        st.fill_topk_w(tw)
        w_ptr = st.topk_w_ptr()
        ids_h, ids_d = p._ids(1)
        st.fill_expert_ids(ids_h[0])

        if GROUP == "gateup":
            x = torch.ones(1, shape.hidden, dtype=torch.bfloat16, device=dev)
            width, pair = 2 * shape.inter, False
            submit = p.cold.wrapper.submit_forward_gateup
            cin, cout = st.x_ptr(), st.partial_gateup_ptr()
            st.fill_x(torch.ones(1, shape.hidden, dtype=torch.bfloat16))
        else:
            x = torch.ones(topk, shape.inter, dtype=torch.bfloat16, device=dev)
            width, pair = shape.hidden, True
            submit = p.cold.wrapper.submit_forward_down
            cin, cout = st.act_ptr(), st.partial_down_ptr()
            st.fill_act(torch.ones(1, topk, shape.inter, dtype=torch.bfloat16))
        out_hot = torch.zeros(1, topk, width, dtype=torch.bfloat16, device=dev)
        out_warm = torch.zeros(1, topk, width, dtype=torch.bfloat16, device=dev)

        def gpu(tiers, out):
            from sglang.srt.layers.moe.prism.profile.full_layer import GpuTier
            if GROUP == "gateup":
                GpuTier.launch_gateup(tiers["gate"], tiers["up"], x, ids_d[0], tw,
                                      out, shape.inter)
            else:
                tiers["down"].launch(x, ids_d[0], tw, out, 0, x_row_is_pair=pair)

        def combined():
            st.fill_expert_ids(ids_d[0], non_blocking=True)
            s = torch.cuda.current_stream().cuda_stream
            submit(qlen, topk, st.expert_ids_ptr(), cin, cout, s, w_ptr)
            gpu(p.hot, out_hot); gpu(p.warm, out_warm)
            p.cold.wrapper.sync(s)

        def combined_split():
            main = torch.cuda.current_stream()
            side = p.res.cold_stream
            side.wait_stream(main)
            with torch.cuda.stream(side):
                st.fill_expert_ids(ids_d[0], non_blocking=True)
                ss = side.cuda_stream
                submit(qlen, topk, st.expert_ids_ptr(), cin, cout, ss, w_ptr)
                p.cold.wrapper.sync(ss)
            gpu(p.hot, out_hot); gpu(p.warm, out_warm)
            main.wait_stream(side)

        def cold_only():
            st.fill_expert_ids(ids_d[0], non_blocking=True)
            s = torch.cuda.current_stream().cuda_stream
            submit(qlen, topk, st.expert_ids_ptr(), cin, cout, s, w_ptr)
            p.cold.wrapper.sync(s)

        def gpu_only():
            gpu(p.hot, out_hot); gpu(p.warm, out_warm)

        def _rejoin():
            """cold partial H2D + rejoin — executor가 phase 끝에서 실제로 하는 것.
            여기까지 담아야 HtoD가 임계경로에 어떻게 붙는지 보인다."""
            from sglang.srt.layers.moe.prism.rejoin import rejoin_down, rejoin_gateup
            src = st.gateup_out(1) if GROUP == "gateup" else st.down_out(1)
            cold_part = src.to(dev, non_blocking=True)          # ← HtoD
            if GROUP == "gateup":
                rejoin_gateup([out_hot, out_warm, cold_part], shape.inter)
            else:
                rejoin_down([out_hot, out_warm, cold_part], tw)

        def layer():
            combined(); _rejoin()

        def layer_split():
            combined_split(); _rejoin()

        variants = [("cold_only", cold_only), ("gpu_only", gpu_only),
                    ("combined", combined), ("combined_split", combined_split),
                    ("layer", layer), ("layer_split", layer_split)]
        graphs = []
        for name, fn in variants:
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
            graphs.append((name, g))

        # 실현된 행 수를 이름에 넣는다 — 비율은 스토어 정렬로 반올림되므로
        # 트레이스가 스스로 무엇을 재는지 말해야 한다.
        proj = "down" if GROUP == "down" else "gate"
        sp = p.splits[proj]
        E = shape.experts
        rows = lambda t: round(sum(sp.rows(t)) / E)
        tag = (f"w{warm_frac:g}_hot{rows('hot')}_warm{rows('warm')}_cold{rows('cold')}")

        torch.cuda.profiler.start()          # 여기서부터 nsys가 뜬다 (스토어 로딩 제외)
        for name, g in graphs:
            with nvtx(f"NSYS/{GROUP}/{tag}/{name}"):
                for _ in range(REPLAYS):
                    g.replay()
                torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        print(f"[nsys] {GROUP} {tag}: {[n for n, _ in graphs]} × {REPLAYS} replays", flush=True)


def main() -> None:
    shape = Shape(experts=128, topk=8, hidden=2048, inter=768)
    for w in WARM_FRACS:
        run_one(shape, w)
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
