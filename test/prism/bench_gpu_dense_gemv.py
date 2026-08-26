#!/usr/bin/env python
"""GPU dense GEMV(hot 티어)의 실행시간 — 하드웨어 프로파일링 ①.

재는 것은 **hot 티어가 실제로 부르는 그 커널**이다: `gemv_worklist_indexed`
(tiers.py `ResidentTier` → device 상주 W, 인덱스 경로). 밴드/bmm 변형은 폐기
됐으므로 (kernels.py의 registry 주석) 재현할 대상이 아니다.

방법: 커널 `--reps`(기본 100)개를 CUDA graph 하나에 캡처하고 replay 시간을
reps로 나눈다. launch 고정비가 지워지고 커널 시간만 남는다. **iteration마다
다른 topk_ids 버퍼**를 태우는 것이 핵심이다 — 같은 expert를 100번 반복하면
W가 L2에 남아 실제 decode보다 낙관적인 시간이 나온다.

    # 35B(H=2048, I=768, E=128, k=8) 전체 K가 hot일 때
    python test/prism/bench_gpu_dense_gemv.py --device 1

    # h375 (K의 37.5%만 hot), bs=1
    python test/prism/bench_gpu_dense_gemv.py --device 1 --hot-frac 0.375

    # 치수만 직접 주는 경우 (weight shape = [k, n])
    python test/prism/bench_gpu_dense_gemv.py --device 1 --k 768 --n 512
"""

from __future__ import annotations

import argparse

import torch

from bench_common import (
    Shape,
    add_shape_args,
    emit,
    env_stamp,
    gbps,
    graph_stats,
    select_device,
    shape_from_args,
    split_rows,
    tier_index,
)

from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed, warmup_jit


def _ids(shape: Shape, m: int, reps: int, device, seed: int) -> list:
    """iteration마다 다른 (token, expert) 배정. 토큰 안에서는 중복 없는 top_k
    (라우터의 성질) — reps×m×top_k가 E를 넘으면 자연히 전 expert를 훑는다."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(reps):
        rows = [torch.randperm(shape.experts, generator=g)[: shape.topk]
                for _ in range(m)]
        out.append(torch.stack(rows).to(torch.int32).to(device))
    return out


def bench_proj(shape: Shape, proj: str, k_rows: int, *, m: int, vec: int,
               reps: int, replays: int, device, shuffle: bool, seed: int,
               label: str | None = None) -> dict:
    """한 (proj, k_rows) 조합의 launch당 µs.

    스토어·인덱스·출력 레이아웃은 weights.py의 hot shard와 같다:
    w_flat [Σₑ k(e), N] device, row_off [E+1] int32 device,
    k_index uint16 device, out3d [M, top_k, N].
    """
    E, topk = shape.experts, shape.topk
    n = shape.n_cols(proj)
    axis = shape.k_axis(proj)
    x_row_is_pair = shape.x_row_is_pair(proj)

    w_flat = torch.empty(E * k_rows, n, dtype=torch.bfloat16, device=device)
    w_flat.normal_(0, 0.02)
    row_off = (torch.arange(E + 1, dtype=torch.int32) * k_rows).to(device)
    rows = tier_index(axis, k_rows, shuffle=shuffle, seed=seed)
    kidx = rows.to(torch.uint16).repeat(E).contiguous().to(device)
    x_rows = m * topk if x_row_is_pair else m
    x = torch.empty(x_rows, axis, dtype=torch.bfloat16, device=device)
    x.normal_(0, 1.0)
    out = torch.zeros(m, topk, n, dtype=torch.bfloat16, device=device)
    ids = _ids(shape, m, reps, device, seed + 1)

    def launch(i: int) -> None:
        gemv_worklist_indexed(x, ids[i], w_flat, row_off, kidx, out, 0,
                              x_row_is_pair, torch.cuda.current_stream(), vec)

    stats = graph_stats(launch, reps, replays=replays)
    w_bytes = m * topk * k_rows * n * 2
    stats.update(
        proj=label or proj, k_rows=k_rows, n_cols=n, k_axis=axis, m=m,
        w_bytes_per_launch=w_bytes, w_store_mb=round(E * k_rows * n * 2 / 1e6, 1),
        gbps=gbps(w_bytes, stats["us"]),
    )
    del w_flat, x, out, ids
    torch.cuda.empty_cache()
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shape_args(p)
    p.add_argument("--hot-frac", type=float, default=1.0,
                   help="K축에서 hot이 소유하는 비율 (기본 1.0 = 전체)")
    p.add_argument("--projs", default="gate,down",
                   help="측정할 proj (기본 gate,down — up은 gate와 같은 치수)")
    p.add_argument("--k", type=int, help="raw 모드: weight 행 수 (proj 치수 대신)")
    p.add_argument("--n", type=int, help="raw 모드: weight 열 수")
    p.add_argument("--m", type=int, default=1, help="토큰 수 (decode=1)")
    p.add_argument("--vec", type=int, default=0, choices=(0, 1, 4, 8),
                   help="W 로드 폭 (0=자동)")
    p.add_argument("--reps", type=int, default=100, help="그래프 하나에 담을 launch 수")
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--shuffle-index", action="store_true",
                   help="티어 인덱스를 정렬하지 않는다 (gather 최악 경우)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out", help="JSON 리포트 경로")
    a = p.parse_args()

    device = select_device(a.device)
    shape = shape_from_args(a)
    warmup_jit()  # JIT 컴파일을 캡처 밖으로

    results = []
    if a.k or a.n:
        if not (a.k and a.n):
            raise SystemExit("raw 모드는 --k와 --n을 함께 준다")
        raw = Shape(experts=shape.experts, topk=shape.topk, hidden=a.k, inter=a.n)
        results.append(bench_proj(
            raw, "gate", a.k, m=a.m, vec=a.vec, reps=a.reps, replays=a.replays,
            device=device, shuffle=a.shuffle_index, seed=a.seed, label="raw"))
    else:
        for proj in [s.strip() for s in a.projs.split(",") if s.strip()]:
            if proj not in ("gate", "up", "down"):
                raise SystemExit(f"unknown proj {proj!r}")
            k_rows = split_rows(shape.k_axis(proj), a.hot_frac)
            if k_rows == 0:
                raise SystemExit("--hot-frac 0은 잴 것이 없다")
            results.append(bench_proj(
                shape, proj, k_rows, m=a.m, vec=a.vec, reps=a.reps,
                replays=a.replays, device=device, shuffle=a.shuffle_index,
                seed=a.seed))

    payload = {
        "bench": "gpu_dense_gemv",
        "kernel": "gemv_worklist_indexed (hot / device-resident W)",
        "shape": shape.as_dict(),
        "params": {
            "hot_frac": a.hot_frac, "m": a.m, "vec": a.vec, "reps": a.reps,
            "replays": a.replays, "shuffle_index": a.shuffle_index,
            "seed": a.seed,
        },
        "results": results,
        "env": env_stamp(device),
    }
    by = {r["proj"]: r["us"] for r in results}
    if "gate" in by and "down" in by:
        # 한 레이어의 GPU dense GEMV 총합 (gate + up + down, up == gate 치수).
        payload["layer_gemv_us"] = round(2 * by["gate"] + by["down"], 3)
    emit(payload, a.out)


if __name__ == "__main__":
    main()
