#!/usr/bin/env python
"""P0 uniform plan 생성기 (개발 도구 — 실제 생성기는 코드베이스 밖).

모델 config.json에서 dims를 읽어 hot=∅ / warm=선두 fraction(64-정렬) /
cold=나머지의 uniform plan을 만든다. NUMA shard는 32-정렬 반반.

사용: python gen_uniform_plan.py <model_dir> <out.json> [--warm-frac 0.1]
"""

import argparse
import json
from pathlib import Path

ROW_GROUP = 64
COL_GROUP = 32


def bands(K: int, warm_frac: float):
    warm = int(K * warm_frac) // ROW_GROUP * ROW_GROUP
    out = []
    if warm > 0:
        out.append([0, warm, "warm"])
    if warm < K:
        out.append([warm, K, "cold"])
    return out, warm < K


def shards(N: int, num_nodes: int = 2):
    half = (N // num_nodes) // COL_GROUP * COL_GROUP
    cuts = [0] + [half * (i + 1) for i in range(num_nodes - 1)] + [N]
    return [[i, cuts[i], cuts[i + 1]] for i in range(num_nodes)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--warm-frac", type=float, default=0.1)
    ap.add_argument("--numa-nodes", type=int, default=2)
    args = ap.parse_args()

    cfg = json.loads((Path(args.model_dir) / "config.json").read_text())
    hidden = cfg["hidden_size"]
    inter = cfg["moe_intermediate_size"]
    experts = (cfg.get("num_experts") or cfg.get("n_routed_experts")
               or cfg.get("num_local_experts"))
    top_k = cfg.get("num_experts_per_tok") or cfg.get("top_k")
    layers = cfg["num_hidden_layers"]

    gu_bands, gu_cold = bands(hidden, args.warm_frac)
    dn_bands, dn_cold = bands(inter, args.warm_frac)
    gate_up = {"bands": gu_bands, "cold_shards": shards(inter, args.numa_nodes) if gu_cold else []}
    down = {"bands": dn_bands, "cold_shards": shards(hidden, args.numa_nodes) if dn_cold else []}

    plan = {
        "schema_version": 1,
        "model_id": cfg.get("_name_or_path") or Path(args.model_dir).name,
        "dims": {
            "hidden_size": hidden,
            "intermediate_size": inter,
            "num_layers": layers,
            "num_experts": experts,
            "top_k": top_k,
            "dtype": "bfloat16",
        },
        "kernels": {"gpu_warm": "torch_bmm", "cpu_cold": "kt_amx_bf16"},
        "default": {"gate": gate_up, "up": dict(gate_up), "down": down},
    }
    Path(args.out).write_text(json.dumps(plan, indent=1))
    print(f"plan written: {args.out}  (gateup {gu_bands}, down {dn_bands})")


if __name__ == "__main__":
    main()
