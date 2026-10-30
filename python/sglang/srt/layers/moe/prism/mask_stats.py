"""effective sparsity 실측 — decode 마스크(k2wl2)가 실제로 살리는 행의 비율.

계약 ①은 "cold 밴드의 nnz는 실측 대상"이라고만 말하고 재는 도구가 없었다. 이 모듈은
executor의 decode 경로(`masking = sparse and m == 1`)에서 커널이 보는 **같은 입력**
(hidden / act, topk_ids, topk_weights)으로 커널과 **같은 식**을 파이썬에서 다시 계산해
(layer, proj, tier)별 keep 행 수를 누적한다. 커널의 마스크를 읽어오는 것이 아니라
재계산이므로 계약 ①의 식이 곧 이 파일의 식이다:

    s      = clip(p − λ(g_e − ḡ), 0, pmax), renorm_it회 재정규        (moe_base.hpp slot_sparsity)
    thr    = table[e, lrint(s / grid)]                                 (moe_base.hpp thr_of)
    e_j    = a[2j]x0² + a[2j+1]x1² + 2c[j]x0x1,   a = wn²              (pair_mask.hpp)
    keep_j = e_j >= thr²                                               (sqrt 회피형, kt와 동일)

마스킹되는 티어는 cold(kt가 항상) + `tiers.SPARSE_TIERS`(warm)이고 hot은 dense다. 여기서는
세 티어 모두의 keep을 세되(hot도 "마스크가 걸렸다면" 얼마였을지), 보고서의 effective
sparsity는 마스킹 티어(warm+cold) 기준으로 낸다.

비용: 층당 3 proj × [k, K] CPU 연산 + D2H 동기 한 번. 측정 모드 전용이며 graph 경로에서는
파이썬이 replay되지 않으므로 **eager(`--disable-cuda-graph`)로 띄워야 값이 쌓인다**.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import time
from typing import Optional

import torch

from sglang.srt.layers.moe.prism.plan import PAIR_GROUP, Plan, Proj, Tier

logger = logging.getLogger(__name__)

_TIER_ID = {Tier.HOT: 0, Tier.WARM: 1, Tier.COLD: 2}
_TIER_NAME = ("hot", "warm", "cold")
_PROJ_ID = {Proj.GATE: 0, Proj.UP: 1, Proj.DOWN: 2}


def slot_sparsity(w: torch.Tensor, p: float, lam: float, pmax: float,
                  renorm_it: int) -> torch.Tensor:
    """`moe_base.hpp slot_sparsity`의 torch 이식. w: fp32 [k] 라우터 가중(비정규화 가능)."""
    w = w.to(torch.float32)
    k = w.numel()
    total = float(w.sum())
    inv = 1.0 / (total if total > 1e-9 else 1e-9)
    g = w * inv
    gbar = float(g.sum()) / k
    s = (p - lam * (g - gbar)).clamp_(0.0, pmax)
    for _ in range(renorm_it):
        mean = max(float(s.sum()) / k, 1e-6)
        s = (s * (p / mean)).clamp_(0.0, pmax)
    return s


class _ProjTables:
    """한 (layer, proj)의 점수 재료 — 전부 CPU fp32, full-K."""

    def __init__(self, plan: Plan, calib, layer: int, proj: Proj):
        dims = plan.dims
        K = dims.k_of(proj)
        E = dims.num_experts
        band = calib.slice_band(layer, proj, 0, K, where=f"mask_stats L{layer} {proj.value}")
        self.wn_sq = band.wn_sq                   # [E, K]
        self.pair_dot = band.pair_dot             # [E, K/2]
        self.thr = calib.thr(layer, proj)         # [E, ng]
        ep0 = plan.expert(layer, 0).proj(proj)
        self.p = float(ep0.sparsity_p)
        self.lam = float(ep0.sparsity_lambda)
        # 페어 단위 티어 라벨 (밴드 경계는 페어 정렬이 보증된다 — validate_static).
        self.label = torch.full((E, K // PAIR_GROUP), -1, dtype=torch.int8)
        for e in range(E):
            for b in plan.expert(layer, e).proj(proj).bands:
                self.label[e, b.start // PAIR_GROUP : b.end // PAIR_GROUP] = _TIER_ID[b.tier]
        if bool((self.label < 0).any()):
            raise AssertionError(f"mask_stats L{layer} {proj.value}: bands do not cover K")


class MaskStats:
    """decode 마스크의 실현 keep 누적기. executor가 층 등록·관측을 호출한다."""

    def __init__(self, plan: Plan, calib, out_path: str, *, every: int = 16,
                 nan_probe: bool = False):
        if plan.sparsity is None or calib is None:
            raise ValueError("mask_stats requires a sparse plan with calib tables")
        self.plan, self.calib = plan, calib
        self.spec = plan.sparsity
        self.out_path = out_path
        self.every = max(1, int(every))
        L = plan.dims.num_layers
        self._tables: dict[tuple[int, Proj], _ProjTables] = {}
        # [L, proj, tier] 페어 수 (행 수는 ×PAIR_GROUP)
        self.keep = torch.zeros(L, 3, 3, dtype=torch.int64)
        self.total = torch.zeros(L, 3, 3, dtype=torch.int64)
        self.s_sum = torch.zeros(L, 3, dtype=torch.float64)
        self.slots = torch.zeros(L, dtype=torch.int64)
        self.tokens = 0
        self.nan_slots = 0      # s가 non-finite였던 (slot, proj) 수 — thr[0] 사용으로 처리
        # NaN 진단(옵션): prefill 포함 모든 호출에서 층 입력/act의 유한성을 본다.
        # 층마다 GPU 리덕션 + 동기라 측정용 런에서만 켠다.
        self.nan_probe = bool(nan_probe)
        self.first_nan = None   # {"layer", "phase", "m", "call"} — 최초 1회만
        self.probe_calls = 0
        self.disabled = False   # 내부 예외 1회 → 관측 중단 (서버는 계속)
        self._first_layer: Optional[int] = None
        self._t0 = time.time()
        self._dumped_tokens = -1
        atexit.register(self._atexit)
        logger.info("[prism] mask_stats on → %s (dump every %d tokens)", out_path, self.every)

    # ── 등록 ────────────────────────────────────────────────────────────
    def register_layer(self, layer: int) -> None:
        for proj in Proj:
            self._tables[(layer, proj)] = _ProjTables(self.plan, self.calib, layer, proj)
        if self._first_layer is None or layer < self._first_layer:
            self._first_layer = layer

    # ── 관측 ────────────────────────────────────────────────────────────
    def observe_gateup(self, layer, hidden, topk_ids, topk_weights) -> None:
        """decode 1토큰: hidden [1, H], topk_ids [1, k], topk_weights [1, k]."""
        self._guard(self._observe_gateup, layer, hidden, topk_ids, topk_weights)

    def observe_down(self, layer, act, topk_ids, topk_weights) -> None:
        """decode 1토큰: act [1, k, I] (rejoin#1 출력 = kt fill_act가 받는 텐서)."""
        self._guard(self._observe_down, layer, act, topk_ids, topk_weights)

    @torch.no_grad()
    def probe(self, layer: int, phase: str, t: torch.Tensor, m: int) -> None:
        """층 입력(gateup) / rejoin 출력(down)의 유한성 — 최초 위반 지점만 기록.

        prism이 NaN을 어디서 처음 만드는지는 마스크 통계로는 알 수 없다(관측이
        decode 전용이라 prefill을 못 본다). 이 프로브는 그 구멍을 메운다.
        """
        if not self.nan_probe or self.first_nan is not None:
            return
        try:
            if bool(torch.isfinite(t).all()):
                return
        except Exception:
            return
        self.first_nan = {"layer": int(layer), "phase": phase, "m": int(m),
                          "call": int(self.probe_calls)}
        logger.error("[prism] mask_stats nan_probe: first non-finite at layer %d phase %s (m=%d)",
                     layer, phase, m)

    def _guard(self, fn, *args) -> None:
        # 측정기의 결함이 서빙을 죽이면 안 된다 — 1회 로그 후 관측을 끈다.
        if self.disabled:
            return
        try:
            fn(*args)
        except Exception:
            self.disabled = True
            logger.exception("[prism] mask_stats disabled after internal error (tokens=%d)", self.tokens)

    @torch.no_grad()
    def _observe_gateup(self, layer: int, hidden: torch.Tensor, topk_ids: torch.Tensor,
                        topk_weights: torch.Tensor) -> None:
        if layer == self._first_layer:
            self.tokens += 1
        x = hidden[0].to("cpu", torch.float32)
        ids = topk_ids[0].to("cpu", torch.int64)
        w = topk_weights[0].to("cpu", torch.float32)
        xk = x.unsqueeze(0).expand(ids.numel(), -1)
        for proj in (Proj.GATE, Proj.UP):
            self._observe(layer, proj, xk, ids, w)
        self.slots[layer] += ids.numel()

    @torch.no_grad()
    def _observe_down(self, layer: int, act: torch.Tensor, topk_ids: torch.Tensor,
                      topk_weights: torch.Tensor) -> None:
        a = act[0].to("cpu", torch.float32)
        ids = topk_ids[0].to("cpu", torch.int64)
        w = topk_weights[0].to("cpu", torch.float32)
        self._observe(layer, Proj.DOWN, a, ids, w)
        if self.tokens % self.every == 0 and self.tokens != self._dumped_tokens \
                and layer == max(l for (l, _) in self._tables):
            self.dump()

    def _observe(self, layer: int, proj: Proj, x: torch.Tensor, ids: torch.Tensor,
                 w: torch.Tensor) -> None:
        t = self._tables[(layer, proj)]
        sp = self.spec
        s = slot_sparsity(w, t.p, t.lam, sp.pmax, sp.renorm_it)              # [k]
        # kt thr_of: lrint(NaN)은 음수(INT_MIN)로 떨어져 idx 0 → thr[...,0](=0, 전량 keep)을 쓴다.
        # 라우터 가중에 NaN/inf가 오면 여기서도 같은 결정을 하고 횟수만 기록한다.
        bad = ~torch.isfinite(s)
        if bool(bad.any()):
            self.nan_slots += int(bad.sum())
            if self.nan_slots <= 8:
                logger.warning("[prism] mask_stats L%d %s: non-finite s from topk_weights=%s",
                               layer, proj.value, w.tolist())
            s = torch.where(bad, torch.zeros_like(s), s)
        gi = torch.round(s / sp.grid).clamp_(0, sp.ng - 1).to(torch.int64)   # lrint ≡ round-half-even
        thr = t.thr[ids, gi]                                                 # [k]
        a = t.wn_sq[ids]                                                     # [k, K]
        c = t.pair_dot[ids]                                                  # [k, K/2]
        x0, x1 = x[:, 0::2], x[:, 1::2]
        e2 = a[:, 0::2] * x0 * x0 + a[:, 1::2] * x1 * x1 + 2.0 * c * x0 * x1
        keep = e2 >= (thr * thr).unsqueeze(1)                                # [k, K/2]
        lab = t.label[ids].to(torch.int64).reshape(-1)
        pi = _PROJ_ID[proj]
        self.keep[layer, pi].index_add_(0, lab, keep.reshape(-1).to(torch.int64))
        self.total[layer, pi].index_add_(0, lab, torch.ones_like(lab))
        self.s_sum[layer, pi] += float(s.sum())

    # ── 보고 ────────────────────────────────────────────────────────────
    def report(self) -> dict:
        masked = (1, 2)   # warm, cold — hot은 dense
        layers = {}
        for l in sorted({l for (l, _) in self._tables}):
            if int(self.slots[l]) == 0:
                continue
            d = {"slots": int(self.slots[l])}
            for proj in Proj:
                pi = _PROJ_ID[proj]
                pd = {"s_mean": self.s_sum[l, pi].item() / max(int(self.slots[l]), 1)}
                for ti, name in enumerate(_TIER_NAME):
                    tot = int(self.total[l, pi, ti])
                    if tot == 0:
                        continue
                    kp = int(self.keep[l, pi, ti])
                    pd[name] = {"rows": tot * PAIR_GROUP, "keep_rows": kp * PAIR_GROUP,
                                "keep_frac": kp / tot}
                mt = sum(int(self.total[l, pi, ti]) for ti in masked)
                mk = sum(int(self.keep[l, pi, ti]) for ti in masked)
                pd["masked_keep_frac"] = (mk / mt) if mt else None
                d[proj.value] = pd
            layers[str(l)] = d

        def agg(pis, tis):
            tot = int(self.total[:, pis][:, :, tis].sum())
            kp = int(self.keep[:, pis][:, :, tis].sum())
            return {"rows": tot * PAIR_GROUP, "keep_frac": (kp / tot) if tot else None,
                    "effective_sparsity": (1 - kp / tot) if tot else None}

        total = {}
        for proj in Proj:
            pi = [_PROJ_ID[proj]]
            total[proj.value] = {"masked": agg(pi, list(masked)), "warm": agg(pi, [1]),
                                 "cold": agg(pi, [2])}
        total["all"] = {"masked": agg([0, 1, 2], list(masked)), "warm": agg([0, 1, 2], [1]),
                        "cold": agg([0, 1, 2], [2]), "hot_if_masked": agg([0, 1, 2], [0])}
        return {
            "score": self.spec.score, "pmax": self.spec.pmax, "grid": self.spec.grid,
            "plan": os.environ.get("SGLANG_PRISM_PLAN"), "calib": self.spec.calib.path,
            "decode_tokens": self.tokens, "elapsed_s": round(time.time() - self._t0, 1),
            "nan_slots": self.nan_slots, "disabled": self.disabled,
            "nan_probe": self.nan_probe, "first_nan": self.first_nan,
            "total": total, "layers": layers,
        }

    def dump(self) -> None:
        rep = self.report()
        tmp = self.out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rep, f, indent=1)
        os.replace(tmp, self.out_path)
        self._dumped_tokens = self.tokens
        m = rep["total"]["all"]
        logger.info("[prism] mask_stats dump: tokens=%d masked keep=%.4f (warm %.4f / cold %.4f)",
                    self.tokens, m["masked"]["keep_frac"] or float("nan"),
                    m["warm"]["keep_frac"] or float("nan"), m["cold"]["keep_frac"] or float("nan"))

    def _atexit(self) -> None:
        try:
            if self.tokens != self._dumped_tokens:
                self.dump()
            if self.tokens == 0:
                logger.warning("[prism] mask_stats: no decode tokens observed — "
                               "graph 경로에서는 파이썬이 돌지 않는다 (--disable-cuda-graph 필요)")
        except Exception as err:  # atexit에서 예외를 올리지 않는다
            logger.warning("[prism] mask_stats final dump failed: %s", err)
