#!/usr/bin/env python
"""DSV4 서버 벤치: (1) greedy 정성 확인, (2) prefill TTFT (프롬프트 길이 스윕), (3) decode ms/tok.

스트리밍 /generate로 첫 토큰 시각(TTFT)과 토큰 간 간격(decode)을 잰다.
사용: bench_dsv4.py --port 30111 [--prefill 672 2048] [--decode-tokens 128]
"""
import argparse
import json
import statistics
import time

import requests


def stream_generate(port, text=None, input_ids=None, max_new_tokens=1):
    payload = {"sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens,
                                   "ignore_eos": True}, "stream": True}
    if input_ids is not None:
        payload["input_ids"] = input_ids
    else:
        payload["text"] = text
    t0 = time.perf_counter()
    r = requests.post(f"http://127.0.0.1:{port}/generate", json=payload, stream=True, timeout=3600)
    stamps, last = [], None
    for line in r.iter_lines():
        if not line or not line.startswith(b"data:"):
            continue
        body = line[5:].strip()
        if body == b"[DONE]":
            break
        last = json.loads(body)
        stamps.append(time.perf_counter() - t0)
    return stamps, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30111)
    ap.add_argument("--prefill", type=int, nargs="*", default=[672, 2048])
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    # 1) 정성 확인
    stamps, last = stream_generate(args.port, text="Explain quantum computing in two sentences:",
                                   max_new_tokens=48)
    print("[text]", repr(last["text"][:400]))
    print("[meta]", {k: last["meta_info"].get(k) for k in ("prompt_tokens", "completion_tokens")})

    # 2) decode: 짧은 프롬프트 + N 토큰. 첫 토큰 이후 간격의 중앙값 = decode ms/tok.
    for _ in range(args.repeat):
        stamps, last = stream_generate(args.port, text="The history of Rome begins",
                                       max_new_tokens=args.decode_tokens)
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        if gaps:
            print(f"[decode] n={len(stamps)} ttft={stamps[0]*1e3:.0f} ms  "
                  f"median {statistics.median(gaps)*1e3:.2f} ms/tok  "
                  f"mean {statistics.mean(gaps)*1e3:.2f}  → {1/statistics.median(gaps):.1f} tok/s")

    # 3) prefill TTFT: 정확한 길이의 input_ids (토큰 id 1000..)로 max_new_tokens=1
    for n in args.prefill:
        ids = [1000 + (i * 7919) % 50000 for i in range(n)]
        best = None
        for _ in range(args.repeat):
            stamps, last = stream_generate(args.port, input_ids=ids, max_new_tokens=1)
            t = stamps[0] if stamps else float("nan")
            best = t if best is None else min(best, t)
        print(f"[prefill] {n} tok: TTFT {best*1e3:.0f} ms  ({best*1e3/n:.2f} ms/tok)")


if __name__ == "__main__":
    main()
