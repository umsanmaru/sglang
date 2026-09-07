#!/usr/bin/env python
"""속도 측정용 **가짜 calib 자산** — 입력과 무관하게 정확히 keep 비율만 실현한다.

실 calib(k2wl2)은 실모델 활성화로 threshold 곡선을 캘리브해야 만들 수 있다. 더미 가중치로는 불가능하다.
대신 profile/common.py `sparse_tables`의 트릭을 자산 형식으로 옮긴다:
  wn(열 노름) ∈ {1, 0}  — 페어 단위로 시드 고정 랜덤 (1 = 살림, 0 = 죽임)
  pair_dot = 0
  thr 곡선 = 상수 t (아주 작은 양수)
kt 마스크는 e = a[2j]·x0² + a[2j+1]·x1² + 2c·x0·x1  >=  t²  (pair_mask.hpp, a = wn²) 이므로
죽인 페어는 e = 0 < t² 로 항상 빠지고, 살린 페어는 |x| >= t 면 항상 남는다. t = 1e-15 (t² = 1e-30, fp32 정규수)
→ 라우터 가중·활성화 분포·p/λ 예산과 무관하게 expert별 sparsity가 그대로 실현된다. 정확도는 물론 무의미하다.

expert별 분포: --sparsity s 를 기준으로 (layer, expert)마다 U[lo·s, hi·s] (기본 --spread 0.4 1.6) 에서 뽑아
[0, 1]로 자른다. gate/up/down 세 proj는 같은 expert면 같은 sparsity(실 plan도 예산은 expert 단위). 기본 spread는
평균이 1.0·s 라 **실현 평균 = s** 이고 expert 간 4배 산포를 준다. 단 hi·s > 1 이면(s > 0.625) 클램프가 걸려
평균이 s보다 낮아진다 — 실현 평균을 항상 출력하니 그 값을 쓸 것.

사용: gen_mock_calib.py <model_dir> <out.pt> [--sparsity 0.5] [--spread 0.4 1.6] [--seed 0] [--layers N]
"""
import argparse, json
from pathlib import Path

import torch

NG, GRID, PMAX, RENORM_IT = 201, 0.005, 0.9, 3   # profile/common.py와 같은 값
THR = 1e-15
PAIR = 2

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("out")
ap.add_argument("--sparsity", type=float, default=0.5, help="기준 sparsity s (죽일 페어 비율)")
ap.add_argument("--spread", type=float, nargs=2, default=(0.4, 1.6), metavar=("LO", "HI"),
                help="expert별 sparsity ∈ U[LO·s, HI·s] (기본 0.4 1.6 → 평균 s). 균일하게 s로 두려면 1 1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--layers", type=int, default=None, help="num_hidden_layers 덮어쓰기 (plan dims.num_layers와 같아야)")
ap.add_argument("--thr", type=float, default=THR)
a = ap.parse_args()

raw = json.loads((Path(a.model_dir) / "config.json").read_text())
cfg = raw.get("text_config", raw)
H = cfg.get("routed_expert_hidden_size") or cfg["hidden_size"]      # K3 latent MoE
I = cfg["moe_intermediate_size"]
E = cfg.get("num_experts") or cfg.get("n_routed_experts")
L = a.layers or cfg["num_hidden_layers"]
assert H % PAIR == 0 and I % PAIR == 0

g = torch.Generator().manual_seed(a.seed)
lo, hi = a.spread
if not (0 <= lo <= hi):
    raise SystemExit(f"--spread needs 0 <= LO <= HI, got {lo} {hi}")
# (layer, expert)별 sparsity — 세 proj 공통
_hi = hi * a.sparsity
if _hi > 1.0:
    print(f"  [warn] HI·s = {_hi:.3f} > 1 — 클램프로 실현 평균이 s({a.sparsity})보다 낮아진다")
sp = torch.empty(L, E).uniform_(lo * a.sparsity, _hi, generator=g).clamp_(0.0, 1.0)
def wn(K):
    keep = (torch.rand(L, E, K // PAIR, generator=g) < (1.0 - sp)[..., None]).to(torch.float32)
    return keep.repeat_interleave(PAIR, dim=-1).contiguous(), 1.0 - keep.mean().item()

wn_g, sg = wn(H); wn_u, su = wn(H); wn_d, sd = wn(I)
thr = torch.full((L, E, NG), a.thr, dtype=torch.float32)
blob = {
    "tg2l": thr, "tu2l": thr.clone(), "td2l": thr.clone(),
    "wn_g": wn_g, "wn_u": wn_u, "wn_d": wn_d,
    "cg": torch.zeros(L, E, H // PAIR), "cu": torch.zeros(L, E, H // PAIR), "cd": torch.zeros(L, E, I // PAIR),
    "PMAX": PMAX, "GRID": GRID, "NG": NG, "RENORM_IT": RENORM_IT, "lam0": 0.0,
    "MOCK": (f"input-independent mask, sparsity={a.sparsity} spread={lo},{hi} seed={a.seed} thr={a.thr}; "
             "NOT a calibration"),
    "mock_sparsity_per_expert": sp,   # [L, E] 진단용 (로더는 안 읽는다)
}
torch.save(blob, a.out)
sz = Path(a.out).stat().st_size / 2**20
print(f"mock calib written: {a.out} ({sz:.0f} MiB)  dims L={L} E={E} H={H} I={I}")
print(f"  expert sparsity ~ U[{lo * a.sparsity:.3f}, {hi * a.sparsity:.3f}]  min/mean/max = "
      f"{sp.min():.3f}/{sp.mean():.3f}/{sp.max():.3f}")
print(f"  realized sparsity gate/up/down = {sg:.4f}/{su:.4f}/{sd:.4f}  (pair-weighted; bytes read ∝ 1 − this)")
