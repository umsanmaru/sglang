#!/usr/bin/env python
"""P0 uniform plan 생성기 (개발 도구 — 실제 생성기는 코드베이스 밖).

모델 config.json에서 dims를 읽어 hot=∅ / warm=선두 fraction(64-정렬) /
cold=나머지의 uniform plan을 만든다. NUMA shard는 32-정렬 반반.

--calib을 주면 schema_version 2 (sparsity) plan을 만든다: pmax/grid/ng/
renorm_it과 λ0는 **자산에서 읽어** 넣는다 (하드코딩하면 Plan과 자산이
조용히 어긋난다). torch import는 이때만 발생한다.

사용:
  python gen_uniform_plan.py <model_dir> <out.json> [--warm-frac 0.1]
  python gen_uniform_plan.py <model_dir> <out.json> \
      --calib assets/qwen35/gatedyn_calib.pt [--target-p 0.5] [--lam 8.0]
"""

import argparse
import hashlib
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


def read_calib(path: str, target_p: float, lam):
    """자산에서 sparsity 블록 필드를 읽는다. 지연 import (torch는 여기서만).

    pmax/grid/ng/renorm_it을 자산에서 가져오는 이유: Plan에 하드코딩하면
    자산을 재생성했을 때 둘이 조용히 어긋난다. λ는 --lam이 우선이고
    없으면 자산의 lam0 (result.md 5.2: λ0가 P=0.4에서 최적이 아니었다).
    """
    import torch  # noqa: PLC0415 — 개발 도구의 optional 경로

    blob = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if lam is None:
        lam = float(blob["lam0"])
    spec = {
        "score": "k2wl2",
        "calib": {"path": str(path), "sha256": digest},
        "pmax": float(blob["PMAX"]),
        "grid": float(blob["GRID"]),
        "ng": int(blob["NG"]),
        "renorm_it": int(blob["RENORM_IT"]),
    }
    print(f"[calib] {path}  pmax={spec['pmax']} grid={spec['grid']} "
          f"ng={spec['ng']} renorm_it={spec['renorm_it']} lam0={float(blob['lam0']):.4f}")
    return spec, float(target_p), float(lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--warm-frac", type=float, default=0.1)
    ap.add_argument("--numa-nodes", type=int, default=2)
    ap.add_argument("--calib", help="gatedyn calib .pt (주면 schema_version 2)")
    ap.add_argument("--target-p", type=float, default=0.5, help="목표 sparsity")
    ap.add_argument("--lam", type=float, default=None,
                    help="gate-dynamic λ (기본: 자산의 lam0)")
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

    sparsity = None
    if args.calib:
        sparsity, p, lam = read_calib(args.calib, args.target_p, args.lam)
        for entry in (gate_up, down):
            entry["p"] = p
            entry["lambda"] = lam

    plan = {
        "schema_version": 2 if sparsity else 1,
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
    if sparsity:
        plan["sparsity"] = sparsity
    Path(args.out).write_text(json.dumps(plan, indent=1))
    tag = f"  sparsity={sparsity['score']} p={gate_up['p']} λ={gate_up['lambda']}" if sparsity else ""
    print(f"plan written: {args.out}  (gateup {gu_bands}, down {dn_bands}){tag}")


if __name__ == "__main__":
    main()
