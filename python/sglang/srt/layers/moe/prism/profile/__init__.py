"""Prism 하드웨어 프로파일 API — "이 치수에서 이 연산이 몇 µs냐"에 답한다.

세 가지를 잰다. 전부 **prism이 실제로 부르는 그 커널**이고, 대체 구현이 아니다:

  hot_dense_gemv       `gemv_worklist_indexed` (device 상주 W, tiers.ResidentTier)
  WarmColdProfiler     warm = `gemv_worklist_indexed_pinned_sparse` (pinned W, UVA)
                       cold = kt `forward_{gateup,down}_partial` (tile_k2 / AMX)
  ColdCpuProfiler      cold만, CUDA 미사용 (GPU가 점유돼 있어도 돌고 perf 가능)

## 쓰는 법

    from sglang.srt.layers.moe.prism.profile import (
        Shape, hot_dense_gemv, WarmColdProfiler, cold_cpu,
    )

    shape = Shape(experts=128, topk=8, hidden=2048, inter=768)

    # ① hot dense GEMV
    r = hot_dense_gemv(shape, hot_frac=0.375, device=1)
    r.us("gate")            # 17.78
    r.layer_gemv_us         # 47.1  (gate*2 + down)
    r.as_dict()             # JSON 그대로

    # ② warm + cold — 스토어 로딩이 비싸니 한 번 만들어 여러 번 질의한다
    with WarmColdProfiler(shape, warm_frac=0.125, sparsity=0.9, device=1) as p:
        g = p.measure("gateup")
        g.us("cold_only")   # 107.2
        g.us("combined")    # 133.0
        p.check("gateup")   # 마스크 레퍼런스 대조 (rel err)
        p.footprint         # {'warm_pinned_mb': 151.0, 'cold_mb': 1057.0, ...}

    # ③ cold만 (GPU 불필요)
    cold_cpu(shape, sparsity=0.9).us      # 103.5

## 규약

- 이 패키지는 **Plan을 읽지 않는다.** 프로파일 결과가 Plan을 만드는 입력이므로
  역방향 의존을 만들지 않는다. 티어 비율은 `hot_frac`/`warm_frac` 같은 인자로
  받고, 스토어 형태만 `weights.py`와 같게 흉내낸다.
- 잘못된 입력은 `ValueError`다 (`SystemExit`이 아니다) — 남의 프로그램에 심었을
  때 프로세스를 죽이지 않기 위해서다.
- sparsity는 점수 재료(a = wn², c = pair_dot)와 threshold 곡선을 역산해 심으므로
  **요청값이 정확히 실현**되고 GPU와 CPU가 같은 페어 집합을 본다. 자세한 역산은
  `common.py`의 docstring.
- sparse는 decode 전용(M=1)이다 (executor의 masking 조건).
- 측정은 커널 `reps`개를 CUDA graph 하나에 담아 replay/reps로 낸다. iteration마다
  다른 expert를 태우는 것이 필수다 — 같은 launch를 반복하면 W가 L2에 남아 실제
  decode보다 낙관적인 값이 나온다 (실측 차이 30%).

## 알려진 제약

- `kt_tile_k2_bf16`은 노드별 N shard가 **256(N_BLOCK)의 배수**여야 한다.
  `gemv_slab`이 그 stride를 전제하고 Release 빌드는 assert가 없어, 어기면 조용히
  남의 메모리를 읽는다. `numa_split`은 그 블록 단위로 반올림되고 실현값이
  리포트의 `node_tables`에 찍힌다.
- 공유 머신에서는 절대값이 load에 흔들린다. 비교는 같은 세션 안에서만 유효하다.

CLI 껍데기는 `test/prism/bench_*.py`다 — 여기 API를 argparse로 감싼 것뿐이다.
"""

from sglang.srt.layers.moe.prism.profile.cold_cpu import (
    ColdCpuProfiler,
    ColdCpuReport,
    cold_cpu,
    cold_cpu_sweep,
    cold_sparse_gemv,
)
from sglang.srt.layers.moe.prism.profile.common import (
    GRID,
    K_STEP,
    NG,
    PMAX,
    PROJS,
    RENORM_IT,
    SPARSITY_LAM,
    SPARSITY_P,
    THR_CONST,
    Shape,
    SparseGemv,
    Timing,
    default_cpuinfer_threads,
    emit,
    env_stamp,
    gbps,
    graph_timing,
    host_timing,
    numa_nodes,
    nvtx,
    select_device,
    sparse_tables,
    split_rows,
    tier_index,
)
from sglang.srt.layers.moe.prism.profile.hot import (
    HotGemvReport,
    ProjGemv,
    dense_gemv,
    hot_dense_gemv,
    measure_gateup,
    measure_proj,
)
from sglang.srt.layers.moe.prism.profile.warm_cold import (
    GROUPS,
    N_ALIGN,
    VARIANTS,
    ColdTier,
    GroupReport,
    Split,
    WarmColdProfiler,
    WarmTier,
    footprint,
    make_split,
    node_table,
    single_expert_warm_cold,
    warm_sparse_gemv,
    warm_cold_sparse,
)

__all__ = [
    # shape / 결과 타입
    "Shape", "Timing", "ProjGemv", "HotGemvReport", "GroupReport", "Split",
    "ColdCpuReport", "SparseGemv",
    # 측정 진입점
    "hot_dense_gemv", "measure_proj", "measure_gateup", "dense_gemv",
    "WarmColdProfiler", "warm_cold_sparse", "single_expert_warm_cold",
    "ColdCpuProfiler", "cold_cpu", "cold_cpu_sweep",
    "warm_sparse_gemv", "cold_sparse_gemv",
    # 구성 요소 (직접 조립할 때)
    "WarmTier", "ColdTier", "make_split", "footprint", "node_table",
    "sparse_tables", "tier_index", "split_rows",
    "graph_timing", "host_timing", "nvtx",
    # 헬퍼 / 상수
    "emit", "env_stamp", "gbps", "select_device", "numa_nodes",
    "default_cpuinfer_threads",
    "GROUPS", "VARIANTS", "N_ALIGN", "PROJS", "K_STEP",
    "NG", "GRID", "PMAX", "RENORM_IT", "THR_CONST",
    "SPARSITY_P", "SPARSITY_LAM",
]
