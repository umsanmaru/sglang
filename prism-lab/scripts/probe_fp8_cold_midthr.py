#!/usr/bin/env python
"""fp8 cold(kt_tile_k2_fp8b128) 중간 threshold 정합성 — 마지막 미검증 구간.

이미 확인된 것:
  * fp8 **GPU** sparse GEMV: test_fp8_kernels.py가 thr=0 비트일치 + thr=0.5 레퍼런스 대조까지 한다.
  * fp8 **cold** sparse: 양극단(thr=0 → dense 비트일치, thr=1e6 → cold 전량 제거)만 확인했다
    (scripts/probe_fp8_cold_sparsity.py). 양극단은 **행 순서가 틀려도 통과한다**.

여기서 보는 것: 중간 threshold(≈50% keep)에서 kt cold의 출력이 파이썬 레퍼런스와 맞는가.
fp8 cold pack은 타일 레이아웃(올림 128)이라 bf16(32)과 행 배열이 다를 수 있고, 점수 배열
(wn²/pair_dot)이 같은 순서로 gather되지 않으면 **엉뚱한 행에 마스크가 걸린다** — 출력은
그럴듯하지만 조용히 틀린다. 마스킹 티어를 cold **하나로** 두어 GPU 경로를 배제한다.

bf16을 같은 형상·같은 마스크로 함께 돌려 대조군으로 쓴다: 두 store의 레퍼런스 대비 오차가
같은 급이면 fp8 cold는 무죄, fp8만 크게 벌어지면 그 경로가 범인이다.
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
    CalibRef, Proj, SparsitySpec, parse_plan, validate_static,
)
from sglang.srt.layers.moe.prism.resources import ExecutionResources, ResourceSpec  # noqa: E402
from sglang.srt.layers.moe.prism.weights import prepare_layer_weights  # noqa: E402

import fp8_ref  # noqa: E402

E, H, I, TOPK, M = 8, 512, 512, 2, 1
NG, GRID, PMAX, RENORM = 201, 0.005, 0.9, 3
MAX_TOKENS = 8
PAIR = 2


def make_plan(store: str, *, sparse: bool):
    gpu, cpu = (("gemv_worklist_fp8", "kt_tile_k2_fp8b128") if store == "fp8"
                else ("gemv_worklist", "kt_amx_bf16"))
    def entry(K, N):                      # 전 구간 cold — 마스킹 주체가 kt 하나다
        e = {"bands": [[0, K, "cold"]], "cold_shards": [[0, 0, N // 2], [1, N // 2, N]]}
        if sparse:
            e["p"], e["lambda"] = 0.5, 0.0        # λ=0 → s=p 고정, 격자 조회를 단순화
        return e
    raw = {"schema_version": 2 if sparse else 1, "model_id": "probe",
           "dims": {"hidden_size": H, "intermediate_size": I, "num_layers": 1,
                    "num_experts": E, "top_k": TOPK, "dtype": "bfloat16"},
           "kernels": {"gpu_warm": gpu, "cpu_cold": cpu},
           "default": {"gate": entry(H, I), "up": entry(H, I), "down": entry(I, H)}}
    if sparse:
        raw["sparsity"] = {"score": "k2wl2", "calib": {"path": "u", "sha256": "0" * 64},
                           "pmax": PMAX, "grid": GRID, "ng": NG, "renorm_it": RENORM}
    plan = parse_plan(raw); validate_static(plan)
    return plan


def score_tables(seed=5):
    g = torch.Generator().manual_seed(seed)
    return {
        "wn_g": torch.rand(1, E, H, generator=g) + 0.5,
        "wn_u": torch.rand(1, E, H, generator=g) + 0.5,
        "wn_d": torch.rand(1, E, I, generator=g) + 0.5,
        "cg": (torch.rand(1, E, H // 2, generator=g) - 0.5) * 0.1,
        "cu": (torch.rand(1, E, H // 2, generator=g) - 0.5) * 0.1,
        "cd": (torch.rand(1, E, I // 2, generator=g) - 0.5) * 0.1,
    }


def importance(vec, wn_row, dot_row):
    """imp_j = sqrt(max(a0x0² + a1x1² + 2c x0x1, 0)),  a = wn²  (계약 ①)"""
    a = (wn_row * wn_row).float()
    x0, x1 = vec[0::2].float(), vec[1::2].float()
    return torch.sqrt(torch.clamp(a[0::2] * x0 * x0 + a[1::2] * x1 * x1
                                  + 2.0 * dot_row.float() * x0 * x1, min=0.0))


def build_calib(tmp: Path, blob, thr_per_proj):
    b = {k: v.clone() for k, v in blob.items()}
    for key, proj in (("tg2l", Proj.GATE), ("tu2l", Proj.UP), ("td2l", Proj.DOWN)):
        b[key] = torch.full((1, E, NG), float(thr_per_proj[proj]))
    path = tmp / "calib.pt"
    torch.save(b, path)
    spec = SparsitySpec(score="k2wl2", calib=CalibRef(path=str(path), sha256="0" * 64),
                        pmax=PMAX, grid=GRID, ng=NG, renorm_it=RENORM)
    return CalibTables.load(spec, verify_digest=False)


def weights(store, seed=0):
    if store == "fp8":
        g = torch.Generator().manual_seed(seed)
        w13, s13 = fp8_ref.random_expert_ckpt(2 * I, H, g)
        w2, s2 = fp8_ref.random_expert_ckpt(H, I, g)
        rep = lambda t: t.unsqueeze(0).repeat(E, *([1] * t.dim())).contiguous()
        return dict(w13=rep(w13), w2=rep(w2), w13_scale=rep(s13), w2_scale=rep(s2))
    torch.manual_seed(seed)
    return dict(w13=(torch.randn(E, 2 * I, H) / 10).to(torch.bfloat16),
                w2=(torch.randn(E, H, I) / 10).to(torch.bfloat16))


def dequant(store, wkw):
    """fp32 [E, 2I, H], [E, H, I] 로 펼친다 (레퍼런스용)."""
    if store == "fp8":
        return (torch.stack([fp8_ref.dequant_ckpt(wkw["w13"][e], wkw["w13_scale"][e]) for e in range(E)]).float(),
                torch.stack([fp8_ref.dequant_ckpt(wkw["w2"][e], wkw["w2_scale"][e]) for e in range(E)]).float())
    return wkw["w13"].float(), wkw["w2"].float()


def build(plan, wkw, calib=None):
    from sglang.srt.layers.moe.prism.cold_backend import KtColdBackend
    from sglang.srt.layers.moe.prism.numa import numa_node_count
    from sglang.srt.layers.moe.prism.kernels import gpu_store_format, cold_pack_tile_rows
    fmt = gpu_store_format(plan.kernels.gpu_warm)
    prep = prepare_layer_weights(0, plan=plan, device=torch.device("cuda"), calib=calib, fmt=fmt,
                                 cold_tile_rows=cold_pack_tile_rows(plan.kernels.cpu_cold), **wkw)
    cold = KtColdBackend(plan, max_tokens=MAX_TOKENS, num_numa_nodes=numa_node_count())
    cold.load_layer(0, prep.cold, prep.thr)
    ex = PrismExecutor(plan, ExecutionResources(
        ResourceSpec.from_plan(plan, max_tokens=MAX_TOKENS, device=torch.device("cuda"))), cold)
    ex.register_layer(0, prep)
    return ex


def reference(x, ids, w, w13f, w2f, blob, thr):
    """마스크를 정준 행순서로 적용한 fp32 레퍼런스 + 실현 keep 비율."""
    gate_w, up_w = w13f[:, :I, :], w13f[:, I:, :]
    out = torch.zeros(M, H)
    kept = {p: [] for p in Proj}
    for m in range(M):
        for j in range(ids.shape[1]):
            e = int(ids[m, j])
            def mask(vec, wn, dot, proj):
                imp = importance(vec, wn[0, e], dot[0, e])
                keep = (imp >= thr[proj]).repeat_interleave(PAIR)
                kept[proj].append(float(keep.float().mean()))
                return vec.float() * keep.float()
            xg = mask(x[m], blob["wn_g"], blob["cg"], Proj.GATE)
            xu = mask(x[m], blob["wn_u"], blob["cu"], Proj.UP)
            act = torch.nn.functional.silu(xg @ gate_w[e].t()) * (xu @ up_w[e].t())
            ad = mask(act, blob["wn_d"], blob["cd"], Proj.DOWN)
            out[m] += float(w[m, j]) * (ad @ w2f[e].t())
    return out, {p: sum(v) / len(v) for p, v in kept.items()}


def rel(a, b):
    return (torch.mean(torch.abs(a.float() - b.float()))
            / (torch.mean(torch.abs(b.float())) + 1e-8)).item()


def main():
    torch.manual_seed(31)
    x = (torch.randn(M, H) / 10).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M)])
    w = torch.rand(M, TOPK, dtype=torch.float32)
    xc, idc, wc = x.cuda(), ids.cuda(), w.cuda()
    blob = score_tables()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for store in ("bf16", "fp8"):
            wkw = weights(store)
            w13f, w2f = dequant(store, wkw)
            # threshold = 실제 imp 분포의 중앙값 → 약 50% keep (proj별)
            gate_w, up_w = w13f[:, :I, :], w13f[:, I:, :]
            thr = {}
            for proj, wn, dot in ((Proj.GATE, "wn_g", "cg"), (Proj.UP, "wn_u", "cu")):
                v = torch.cat([importance(x[0], blob[wn][0, int(ids[0, j])], blob[dot][0, int(ids[0, j])])
                               for j in range(TOPK)])
                thr[proj] = float(v.median())
            acts = []
            for j in range(TOPK):
                e = int(ids[0, j])
                acts.append(importance(torch.nn.functional.silu(x[0].float() @ gate_w[e].t())
                                       * (x[0].float() @ up_w[e].t()),
                                       blob["wn_d"][0, e], blob["cd"][0, e]))
            thr[Proj.DOWN] = float(torch.cat(acts).median())

            calib = build_calib(tmp, blob, thr)
            got = build(make_plan(store, sparse=True), wkw, calib).run_layer(0, xc, idc, wc).cpu()
            ref, keep = reference(x, ids, w, w13f, w2f, blob, thr)
            dense = build(make_plan(store, sparse=False), wkw).run_layer(0, xc, idc, wc).cpu()
            print(f"[{store:4s}] keep: gate {keep[Proj.GATE]:.2f} up {keep[Proj.UP]:.2f} down {keep[Proj.DOWN]:.2f}"
                  f" | prism vs 레퍼런스 rel={rel(got, ref):.3e}"
                  f" | dense vs 레퍼런스 rel={rel(dense, ref):.3e}")


if __name__ == "__main__":
    main()
