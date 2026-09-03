#!/usr/bin/env python
"""GPU busy 시간 = 커널 구간의 **합집합** / 벽시계 시간.

SM utilization(얼마나 넓게 쓰는가)이 아니라 "GPU가 쉬지 않은 시간"이다. 스트림이
여럿이면 구간이 겹치므로 총합이 아니라 합집합을 써야 한다 — 겹친 시간을 두 번
세면 100%를 넘는다.

입력은 둘 중 하나:
  - nsys sqlite  (`nsys export --type sqlite`) — CUPTI_ACTIVITY_KIND_KERNEL 등
  - torch profiler chrome trace (.json / .json.gz) — cat이 kernel/gpu_mem* 인 X 이벤트

CUDA graph 주의: nsys는 `--cuda-graph-trace=node`로 떠야 그래프 **안**의 커널이
개별 구간으로 잡힌다. 기본(graph 단위)이면 host node 대기까지 한 구간에 삼켜서
busy가 과대평가된다 (prism decode 그래프에는 host node가 들어 있다).

사용:
  gpu_busy.py <trace>            [--no-copy] [--from T --to T] [--gaps N] [--window auto|span]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sqlite3
import sys

KERNEL_TABLES = ["CUPTI_ACTIVITY_KIND_KERNEL"]
COPY_TABLES = ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]
CHROME_KERNEL = {"kernel", "Kernel"}
CHROME_COPY = {"gpu_memcpy", "gpu_memset", "Memcpy", "Memset"}


def from_sqlite(path, with_copy):
    db = sqlite3.connect(path)
    tables = KERNEL_TABLES + (COPY_TABLES if with_copy else [])
    out = []
    for t in tables:
        try:
            out += db.execute(f"SELECT start, end FROM {t}").fetchall()  # ns
        except sqlite3.OperationalError:
            pass  # 그 종류의 활동이 트레이스에 없다
    if not out:
        raise SystemExit(f"{path}: CUPTI 커널 테이블이 비었다 — -t cuda로 떴는지 확인")
    return [(s / 1e3, e / 1e3) for s, e in out]  # µs


def from_chrome(path, with_copy):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        doc = json.load(f)
    cats = CHROME_KERNEL | (CHROME_COPY if with_copy else set())
    out = []
    for ev in doc.get("traceEvents", doc if isinstance(doc, list) else []):
        if ev.get("ph") != "X" or ev.get("cat") not in cats:
            continue
        ts, dur = ev.get("ts"), ev.get("dur")
        if ts is None or not dur:
            continue
        out.append((float(ts), float(ts) + float(dur)))  # µs
    if not out:
        raise SystemExit(f"{path}: GPU 커널 이벤트가 없다 — activities에 GPU를 넣었는지 확인")
    return out


def union(spans):
    """겹치는 구간을 합쳐 (busy, merged) 반환. 입력은 (start, end) µs."""
    spans = sorted(spans)
    merged = []
    cs, ce = spans[0]
    for s, e in spans[1:]:
        if s > ce:
            merged.append((cs, ce))
            cs, ce = s, e
        else:
            ce = max(ce, e)
    merged.append((cs, ce))
    return sum(e - s for s, e in merged), merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--no-copy", action="store_true",
                    help="memcpy/memset을 busy에서 뺀다 (커널만; copy engine은 SM과 별개다)")
    ap.add_argument("--from", dest="t0", type=float, default=None, help="µs, 이 시각 이후만")
    ap.add_argument("--to", dest="t1", type=float, default=None, help="µs, 이 시각 이전만")
    ap.add_argument("--gaps", type=int, default=10, help="가장 긴 idle 구간 N개를 보여준다")
    a = ap.parse_args()

    loader = from_sqlite if a.trace.endswith((".sqlite", ".db", ".sqlite3")) else from_chrome
    spans = loader(a.trace, not a.no_copy)
    if a.t0 is not None:
        spans = [(max(s, a.t0), e) for s, e in spans if e > a.t0]
    if a.t1 is not None:
        spans = [(s, min(e, a.t1)) for s, e in spans if s < a.t1]
    if not spans:
        raise SystemExit("구간이 비었다 — --from/--to 범위를 확인")

    busy, merged = union(spans)
    # 벽시계는 "첫 커널 시작 ~ 마지막 커널 끝"이다. 측정 창을 명시하려면 --from/--to.
    t0, t1 = merged[0][0], merged[-1][1]
    wall = t1 - t0
    gaps = [(merged[i + 1][0] - merged[i][1], merged[i][1]) for i in range(len(merged) - 1)]
    gaps.sort(reverse=True)

    print(f"trace      : {a.trace}")
    print(f"구간 수     : {len(spans)} (합친 뒤 {len(merged)})")
    print(f"창         : {wall / 1e3:.3f} ms  [{t0:.1f}, {t1:.1f}] µs")
    print(f"busy       : {busy / 1e3:.3f} ms")
    print(f"idle       : {(wall - busy) / 1e3:.3f} ms  (gap {len(gaps)}개)")
    print(f"**GPU busy : {100 * busy / wall:.2f} %**"
          f"{'  (커널만)' if a.no_copy else '  (커널+copy)'}")
    if gaps and a.gaps:
        print(f"\n가장 긴 idle 구간 {min(a.gaps, len(gaps))}개 (µs, 시작시각):")
        for g, at in gaps[: a.gaps]:
            print(f"  {g:10.1f}  @ {at:.1f}")
        tot = sum(g for g, _ in gaps)
        print(f"  상위 {min(a.gaps, len(gaps))}개가 전체 idle의 "
              f"{100 * sum(g for g, _ in gaps[:a.gaps]) / tot:.1f}%")


if __name__ == "__main__":
    main()
