#!/usr/bin/env python
"""P0 uniform plan 생성기 (개발 도구 — 실제 생성기는 코드베이스 밖).

모델 config.json에서 dims를 읽어 K축 선두부터
hot(--hot-frac) / warm(--warm-frac) / cold(나머지)로 자른 uniform plan을
만든다. 경계는 전부 64-정렬, NUMA shard는 32-정렬 반반.

hot 비율은 VRAM 예산이 정한다: expert weight 총량 = NL·NE·(2·I·H + H·I)·2B
이고 hot이 그 fraction만큼 상주한다 (Qwen3.6-35B-A3B는 60 GiB → f=0.5면
30 GiB). --hot-frac 0이면 기존 2-tier plan과 완전히 동일한 출력이다.

--calib을 주면 schema_version 2 (sparsity) plan을 만든다: pmax/grid/ng/
renorm_it과 λ0는 **자산에서 읽어** 넣는다 (하드코딩하면 Plan과 자산이
조용히 어긋난다). torch import는 이때만 발생한다.

사용:
  python gen_uniform_plan.py <model_dir> <out.json> [--hot-frac 0] [--warm-frac 0.1]
  python gen_uniform_plan.py <model_dir> <out.json> \
      --calib assets/qwen35/gatedyn_calib.pt [--target-p 0.5] [--lam 8.0]
"""

import argparse
import hashlib
import json
from pathlib import Path

ROW_GROUP = 64
COL_GROUP = 32


def bands(K: int, hot_frac: float, warm_frac: float):
    """K축을 [hot | warm | cold] 순으로 자른다. 빈 티어는 밴드를 안 만든다.

    잘림(floor)은 항상 cold로 흘러간다 — hot/warm이 계획보다 커지는 쪽으로
    반올림하면 VRAM/pinned 예산을 조용히 초과한다.
    """
    hot = int(K * hot_frac) // ROW_GROUP * ROW_GROUP
    warm = int(K * warm_frac) // ROW_GROUP * ROW_GROUP
    if hot + warm > K:
        raise SystemExit(f"hot+warm={hot + warm} exceeds K={K}")
    out = []
    if hot > 0:
        out.append([0, hot, "hot"])
    if warm > 0:
        out.append([hot, hot + warm, "warm"])
    if hot + warm < K:
        out.append([hot + warm, K, "cold"])
    return out, hot + warm < K


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
    global ROW_GROUP
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--hot-frac", type=float, default=0.0,
                    help="VRAM 상주 비율 (0이면 기존 2-tier plan과 동일)")
    ap.add_argument("--warm-frac", type=float, default=0.1)
    ap.add_argument("--numa-nodes", type=int, default=2)
    ap.add_argument("--calib", help="gatedyn calib .pt (주면 schema_version 2)")
    ap.add_argument("--target-p", type=float, default=0.5, help="목표 sparsity")
    ap.add_argument("--lam", type=float, default=None,
                    help="gate-dynamic λ (기본: 자산의 lam0)")
    ap.add_argument("--gpu-kernel", default="torch_bmm",
                    choices=["torch_bmm", "gemv_worklist", "gemv_worklist_mxfp4"],
                    help="plan kernels.gpu_warm (worklist는 bs>1 graph decode용; "
                         "gemv_worklist_mxfp4 = MXFP4 pair-row 스토어, K 정렬 32, cold 불가)")
    ap.add_argument("--cpu-kernel", default="kt_amx_bf16",
                    choices=["kt_amx_bf16", "kt_tile_k2_bf16", "kt_amx_fp4"])
    ap.add_argument("--k-align", type=int, default=ROW_GROUP,
                    help="밴드 경계 정렬 (기본 64; mxfp4는 32 배수여야 한다 — 64는 만족)")
    args = ap.parse_args()
    if args.gpu_kernel == "gemv_worklist_mxfp4" and args.k_align % 32:
        raise SystemExit("mxfp4 needs --k-align multiple of 32")
    ROW_GROUP = args.k_align

    raw = json.loads((Path(args.model_dir) / "config.json").read_text())
    # VLM config(Qwen3.5/3.6 계열)는 언어모델 치수를 text_config 아래에 둔다.
    cfg = raw.get("text_config", raw)
    hidden = cfg["hidden_size"]
    inter = cfg["moe_intermediate_size"]
    experts = (cfg.get("num_experts") or cfg.get("n_routed_experts")
               or cfg.get("num_local_experts"))
    top_k = cfg.get("num_experts_per_tok") or cfg.get("top_k")
    layers = cfg["num_hidden_layers"]

    gu_bands, gu_cold = bands(hidden, args.hot_frac, args.warm_frac)
    dn_bands, dn_cold = bands(inter, args.hot_frac, args.warm_frac)
    if args.gpu_kernel == "gemv_worklist_mxfp4" and (gu_cold or dn_cold) and args.cpu_kernel != "kt_amx_fp4":
        raise SystemExit("mxfp4 store needs --cpu-kernel kt_amx_fp4 for its cold tier "
                         f"(gateup {gu_bands}, down {dn_bands})")
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
        "model_id": raw.get("_name_or_path") or cfg.get("_name_or_path") or Path(args.model_dir).name,
        "dims": {
            "hidden_size": hidden,
            "intermediate_size": inter,
            "num_layers": layers,
            "num_experts": experts,
            "top_k": top_k,
            "dtype": "bfloat16",
        },
        "kernels": {"gpu_warm": args.gpu_kernel, "cpu_cold": args.cpu_kernel},
        "default": {"gate": gate_up, "up": dict(gate_up), "down": down},
    }
    if sparsity:
        plan["sparsity"] = sparsity
    Path(args.out).write_text(json.dumps(plan, indent=1))
    tag = f"  sparsity={sparsity['score']} p={gate_up['p']} λ={gate_up['lambda']}" if sparsity else ""
    # VRAM/pinned 예산을 눈으로 확인할 수 있게 실제 밴드에서 역산해 찍는다
    # (요청한 fraction이 아니라 64-정렬 후의 값이라야 예산과 맞는다).
    def _bytes(bs, tier):
        tot = 0
        for (st, en, t) in bs:
            if t == tier:
                tot += (en - st)
        return tot
    # row 1개의 바이트: bf16 2 B/원소, mxfp4 0.5 B 코드 + 1/32 B 배율 = 0.53125 B/원소
    bpe = (0.5 + 1.0 / 32) if args.gpu_kernel == "gemv_worklist_mxfp4" else 2.0
    per_row = experts * inter * bpe                    # gate/up: row 1개 = [E, I]
    dn_row = experts * hidden * bpe                    # down:  row 1개 = [E, H]
    for tier in ("hot", "warm"):
        gib = layers * (2 * _bytes(gu_bands, tier) * per_row
                        + _bytes(dn_bands, tier) * dn_row) / 2**30
        if gib:
            print(f"[budget] {tier:4s} = {gib:6.2f} GiB")
    print(f"plan written: {args.out}  (gateup {gu_bands}, down {dn_bands}){tag}")


if __name__ == "__main__":
    main()
