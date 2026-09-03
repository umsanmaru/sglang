#!/usr/bin/env python
"""fp8 cold sparse 경로 단독 검증 — 기존 테스트가 덮지 않는 조합.

배경 (2026-08-31): GLM-5.3-Flash를 prism fp8 3-tier로 서빙하면 sparsity p=0.5에서
출력이 반복 루프로 무너지고, **동일 티어 dense**에서는 정상이다. 그런데
`test_sparsity.py`는 `kt_amx_bf16`(bf16 store)만 돌고 `test_fp8_executor.py`/
`test_fp8_prefill.py`에는 sparsity가 한 번도 나오지 않는다 — 즉 fp8 cold의
마스킹 경로는 어떤 테스트도 통과한 적이 없다.

여기서 보는 것은 계약의 두 극단이다 (threshold 곡선을 상수로 채워 s→격자 조회를 우회):
  thr = 0      → 전량 keep → **dense와 같아야 한다** (bf16판 test_zero_threshold_matches_dense_path)
  thr = 매우 큼 → 전량 drop → cold 기여가 0이어야 한다
bf16을 같은 형상으로 함께 돌려 대조군으로 쓴다.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

REPO = Path("/home/um3maru/prism-sglang/sglang")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "test" / "prism"))

from sglang.srt.layers.moe.prism.calib import CalibTables  # noqa: E402
from sglang.srt.layers.moe.prism.executor import PrismExecutor  # noqa: E402
from sglang.srt.layers.moe.prism.plan import (  # noqa: E402
    CalibRef, Proj, SparsitySpec, Tier, parse_plan, validate_static,
)
from sglang.srt.layers.moe.prism.resources import ExecutionResources, ResourceSpec  # noqa: E402
from sglang.srt.layers.moe.prism.weights import prepare_layer_weights  # noqa: E402

import fp8_ref  # noqa: E402  (test/prism)

# fp8 제약을 만족하는 최소 형상: K 정렬 128, cold 노드 shard 정렬 256(2노드 → N/2 = 256)
E, H, I, TOPK = 8, 512, 512, 2
WARM = 128                      # warm 밴드 (128의 배수여야 한다)
NG, GRID, PMAX, RENORM = 201, 0.005, 0.9, 3
MAX_TOKENS = 8


def make_plan(store: str, *, sparse: bool):
    gpu, cpu = (("gemv_worklist_fp8", "kt_tile_k2_fp8b128") if store == "fp8"
                else ("gemv_worklist", "kt_amx_bf16"))
    def entry(bands, N):
        e = {"bands": bands, "cold_shards": [[0, 0, N // 2], [1, N // 2, N]]}
        if sparse:
            e["p"], e["lambda"] = 0.5, 1.0
        return e
    raw = {
        "schema_version": 2 if sparse else 1,
        "model_id": "probe",
        "dims": {"hidden_size": H, "intermediate_size": I, "num_layers": 1,
                 "num_experts": E, "top_k": TOPK, "dtype": "bfloat16"},
        "kernels": {"gpu_warm": gpu, "cpu_cold": cpu},
        "default": {
            "gate": entry([[0, WARM, "warm"], [WARM, H, "cold"]], I),
            "up":   entry([[0, WARM, "warm"], [WARM, H, "cold"]], I),
            "down": entry([[0, WARM, "warm"], [WARM, I, "cold"]], H),
        },
    }
    if sparse:
        raw["sparsity"] = {"score": "k2wl2",
                           "calib": {"path": "unused", "sha256": "0" * 64},
                           "pmax": PMAX, "grid": GRID, "ng": NG, "renorm_it": RENORM}
    plan = parse_plan(raw)
    validate_static(plan)
    return plan


def make_calib(tmp: Path, thr_fill: float, seed=5):
    g = torch.Generator().manual_seed(seed)
    blob = {
        "wn_g": torch.rand(1, E, H, generator=g) + 0.5,
        "wn_u": torch.rand(1, E, H, generator=g) + 0.5,
        "wn_d": torch.rand(1, E, I, generator=g) + 0.5,
        "cg": (torch.rand(1, E, H // 2, generator=g) - 0.5) * 0.1,
        "cu": (torch.rand(1, E, H // 2, generator=g) - 0.5) * 0.1,
        "cd": (torch.rand(1, E, I // 2, generator=g) - 0.5) * 0.1,
    }
    for key in ("tg2l", "tu2l", "td2l"):
        blob[key] = torch.full((1, E, NG), float(thr_fill))
    path = tmp / f"calib_{thr_fill}.pt"
    torch.save(blob, path)
    spec = SparsitySpec(score="k2wl2", calib=CalibRef(path=str(path), sha256="0" * 64),
                        pmax=PMAX, grid=GRID, ng=NG, renorm_it=RENORM)
    return CalibTables.load(spec, verify_digest=False)


def weights(store: str, seed=0):
    if store == "fp8":
        g = torch.Generator().manual_seed(seed)
        w13, s13 = fp8_ref.random_expert_ckpt(2 * I, H, g)
        w2, s2 = fp8_ref.random_expert_ckpt(H, I, g)
        w13 = w13.unsqueeze(0).repeat(E, 1, 1).contiguous()
        s13 = s13.unsqueeze(0).repeat(E, 1, 1).contiguous()
        w2 = w2.unsqueeze(0).repeat(E, 1, 1).contiguous()
        s2 = s2.unsqueeze(0).repeat(E, 1, 1).contiguous()
        return dict(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2)
    torch.manual_seed(seed)
    return dict(w13=(torch.randn(E, 2 * I, H) / 10).to(torch.bfloat16),
                w2=(torch.randn(E, H, I) / 10).to(torch.bfloat16))


def build(plan, wkw, calib=None):
    from sglang.srt.layers.moe.prism.cold_backend import KtColdBackend
    from sglang.srt.layers.moe.prism.numa import numa_node_count
    from sglang.srt.layers.moe.prism.kernels import gpu_store_format, cold_pack_tile_rows

    fmt = gpu_store_format(plan.kernels.gpu_warm)
    prepared = prepare_layer_weights(
        0, plan=plan, device=torch.device("cuda"), calib=calib, fmt=fmt,
        cold_tile_rows=cold_pack_tile_rows(plan.kernels.cpu_cold), **wkw)
    cold = KtColdBackend(plan, max_tokens=MAX_TOKENS, num_numa_nodes=numa_node_count())
    cold.load_layer(0, prepared.cold, prepared.thr)
    ex = PrismExecutor(plan, ExecutionResources(
        ResourceSpec.from_plan(plan, max_tokens=MAX_TOKENS, device=torch.device("cuda"))), cold)
    ex.register_layer(0, prepared)
    return ex


def rel(a, b):
    return (torch.mean(torch.abs(a.float() - b.float()))
            / (torch.mean(torch.abs(b.float())) + 1e-8)).item()


def main():
    # 마스킹은 executor.py:303의 `masking = self._sparse and m == 1` 때문에
    # **m==1에서만** 켜진다 (디코드 전용). m>1로 재면 dense와 비트 동일해 무의미하다.
    torch.manual_seed(31)
    M = 1
    x = (torch.randn(M, H) / 10).to(torch.bfloat16).cuda()
    ids = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M)]).cuda()
    w = torch.rand(M, TOPK, dtype=torch.float32).cuda()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for store in ("bf16", "fp8"):
            wkw = weights(store)
            dense = build(make_plan(store, sparse=False), wkw).run_layer(0, x, ids, w).cpu()
            zero = build(make_plan(store, sparse=True), wkw, make_calib(tmp, 0.0)).run_layer(0, x, ids, w).cpu()
            big = build(make_plan(store, sparse=True), wkw, make_calib(tmp, 1e6)).run_layer(0, x, ids, w).cpu()
            d0, d1 = rel(zero, dense), rel(big, dense)
            print(f"[{store:4s}] thr=0  vs dense : rel={d0:.3e}   {'OK (전량 keep)' if d0 < 1e-5 else '*** 불일치 ***'}")
            print(f"[{store:4s}] thr=1e6 vs dense: rel={d1:.3e}   {'OK (cold 제거됨)' if d1 > 1e-3 else '*** 마스킹 무효 ***'}")


if __name__ == "__main__":
    main()
