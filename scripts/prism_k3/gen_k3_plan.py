#!/usr/bin/env python
"""Kimi K3용 uniform prism plan 생성기.

test/prism/gen_uniform_plan.py는 config의 hidden_size를 expert K로 쓰는데, K3의 routed expert는
**latent 공간**(routed_expert_hidden_size=3584)에서 돈다 — FusedMoE hidden_size가 3584이고 prism도
그 치수를 본다. 그래서 dims를 직접 채운다. 나머지(밴드 자르기, NUMA shard)는 원 도구를 재사용.

사용: gen_k3_plan.py <model_dir> <out.json> [--hot-frac 0.03] [--warm-frac 0.05] [--numa-nodes 2]
"""
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 이 스크립트는 <repo>/scripts/prism_k3/ 에 있다 → TREE(=sglang 체크아웃) 는 그 두 단계 위.
TREE = os.environ.get("TREE", str(HERE.parent.parent))
sys.path.insert(0, f"{TREE}/test/prism")
sys.path.insert(0, f"{TREE}/python")
import gen_uniform_plan as g  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("out")
ap.add_argument("--hot-frac", type=float, default=0.03)
ap.add_argument("--warm-frac", type=float, default=0.05)
ap.add_argument("--numa-nodes", type=int, default=2)
ap.add_argument("--gpu-kernel", default="gemv_worklist_mxfp4")
ap.add_argument("--cpu-kernel", default="kt_tile_k2_mxfp4")
ap.add_argument("--calib", default=None, help="calib .pt (gen_mock_calib.py 산출물 포함) → schema_version 2 sparsity plan")
ap.add_argument("--p", type=float, default=0.5, help="sparsity 예산 p (mock calib에서는 keep에 영향 없음)")
ap.add_argument("--lam", type=float, default=None, help="λ (기본: 자산의 lam0)")
a = ap.parse_args()

raw = json.loads((Path(a.model_dir) / "config.json").read_text())
cfg = raw.get("text_config", raw)
assert cfg.get("hidden_act") == "situ", "K3 plan generator expects a Kimi K3 config"
hidden = cfg["routed_expert_hidden_size"] or cfg["hidden_size"]   # latent MoE 치수
inter = cfg["moe_intermediate_size"]
experts = cfg.get("num_experts") or cfg.get("n_routed_experts")
top_k = cfg["num_experts_per_token"]
layers = cfg["num_hidden_layers"]

g.ROW_GROUP = 32  # mxfp4 k_align
gu_bands, gu_cold = g.bands(hidden, a.hot_frac, a.warm_frac)
dn_bands, dn_cold = g.bands(inter, a.hot_frac, a.warm_frac)
gate_up = {"bands": gu_bands, "cold_shards": g.shards(inter, a.numa_nodes) if gu_cold else []}
down = {"bands": dn_bands, "cold_shards": g.shards(hidden, a.numa_nodes) if dn_cold else []}
sparsity = None
if a.calib:
    sparsity, p, lam = g.read_calib(a.calib, a.p, a.lam)
    for entry in (gate_up, down):
        entry["p"] = p
        entry["lambda"] = lam
plan = {
    "schema_version": 2 if sparsity else 1,
    "model_id": Path(a.model_dir).name,
    "dims": {"hidden_size": hidden, "intermediate_size": inter, "num_layers": layers,
             "num_experts": experts, "top_k": top_k, "dtype": "bfloat16"},
    "kernels": {"gpu_warm": a.gpu_kernel, "cpu_cold": a.cpu_kernel},
    "default": {"gate": gate_up, "up": dict(gate_up), "down": down},
    "provenance": {"generator": os.path.basename(__file__), "hot_frac": a.hot_frac, "warm_frac": a.warm_frac,
                   "note": "K3 latent MoE: hidden = routed_expert_hidden_size; layer 0 is dense (plan entry unused)"},
}
if sparsity:
    plan["sparsity"] = sparsity
from sglang.srt.layers.moe.prism.plan import parse_plan, validate_static  # noqa: E402
_plan = parse_plan(plan)
if sparsity:
    from sglang.srt.layers.moe.prism.calib import CalibTables  # noqa: E402
    _calib = CalibTables.load(_plan.sparsity)
    _calib.check_dims(_plan.dims, _plan.sparsity)
    validate_static(_plan, calib_probe=_calib.probe())
else:
    validate_static(_plan)
Path(a.out).write_text(json.dumps(plan, indent=1))
bpe = 0.5 + 1 / 32
moe_layers = layers - int(cfg.get("first_k_dense_replace", 0))
def _b(bs, t): return sum(en - st for st, en, tt in bs if tt == t)
for tier in ("hot", "warm", "cold"):
    gib = moe_layers * (2 * _b(gu_bands, tier) * experts * inter + _b(dn_bands, tier) * experts * hidden) * bpe / 2**30
    print(f"[budget] {tier:4s} = {gib:7.2f} GiB  ({moe_layers} MoE layers)")
print(f"plan written: {a.out}  gateup {gu_bands}  down {dn_bands}")
