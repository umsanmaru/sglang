#!/usr/bin/env python
"""Kimi K3 config를 N층으로 잘라 별도 디렉터리에 쓴다 — 가중치 없는 속도/경로 하니스용.

`--load-format dummy` 와 짝이다. 체크포인트(1.4 TB) 없이 K3 모델 코드(KDA/MLA/latent MoE/situ)와
prism 경로를 e2e로 돌리려면 config + tokenizer만 있으면 된다:

    hf download moonshotai/Kimi-K3 --local-dir /tmp/k3cfg \
        --include "config.json" "*.py" "tiktoken.model" "tokenizer_config.json" "generation_config.json" \
                  "preprocessor_config.json"
    make_12l_config.py /tmp/k3cfg ./modelcfg/Kimi-K3-12L --layers 12

**함정(실측 2026-09-06)**: `linear_attn_config`의 `kda_layers` / `full_attn_layers`는 **1-indexed**다
(`KimiLinearConfig.is_kda_layer`가 `(layer_idx + 1) in kda_layers`로 본다). 그래서 L층으로 자를 때
남기는 조건은 `i <= L`이지 `i < L`이 아니다. 잘못 자르면 층 종류(KDA/MLA)가 밀려 다른 모델이 된다.
L=12 → kda [1,2,3,5,6,7,9,10,11], full [4,8,12] = 0-indexed MLA 층 3/7/11.
"""
import argparse
import json
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("src", help="전체 K3 config 디렉터리 (가중치 불필요)")
ap.add_argument("dst", help="출력 디렉터리")
ap.add_argument("--layers", type=int, default=12)
a = ap.parse_args()

src, dst = Path(a.src), Path(a.dst)
dst.mkdir(parents=True, exist_ok=True)

# config.json 외의 부수 파일(tokenizer, remote code, processor)은 그대로 복사한다.
# safetensors/index는 dummy 로더가 안 보므로 가져오지 않는다.
for f in sorted(src.iterdir()):
    if not f.is_file() or f.name.startswith(".") or f.name == "config.json":
        continue
    if "safetensors" in f.name:
        continue
    shutil.copy(f, dst / f.name)

cfg = json.loads((src / "config.json").read_text())
text = cfg.get("text_config", cfg)
L = a.layers
full_L = text["num_hidden_layers"]
if not 1 <= L <= full_L:
    raise SystemExit(f"--layers must be in 1..{full_L}, got {L}")
text["num_hidden_layers"] = L

la = text.get("linear_attn_config")
if la:
    # 1-indexed (is_kda_layer: layer_idx + 1 in kda_layers) → i <= L 을 남긴다.
    for key in ("kda_layers", "full_attn_layers"):
        if la.get(key) is not None:
            la[key] = [i for i in la[key] if i <= L]

cfg["_prism_note"] = (
    f"dummy-weight speed/path harness: text_config.num_hidden_layers {full_L}->{L}; "
    "linear_attn_config kda/full_attn lists (1-indexed) filtered to <= L"
)
(dst / "config.json").write_text(json.dumps(cfg, indent=1))

print(f"wrote {dst}/config.json  ({full_L} -> {L} layers)")
if la:
    kda = la.get("kda_layers") or []
    fa = la.get("full_attn_layers") or []
    print(f"  kda(1-idx)={kda}")
    print(f"  full_attn(1-idx)={fa}  -> MLA layers (0-idx) {[i - 1 for i in fa]}")
print(f"  copied {len(list(dst.iterdir())) - 1} auxiliary files (tokenizer / remote code)")
