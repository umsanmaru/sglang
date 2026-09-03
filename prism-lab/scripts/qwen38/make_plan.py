#!/usr/bin/env python3
"""Qwen3.8-27B dense prism plan 생성기.

체크포인트에서 확인한 치수(2026-09-01, safetensors 헤더 직독):

    64층 = linear_attention 48 + full_attention 16 (층 3, 7, 11, …, 63)
    hidden 5120 · intermediate 17408

    layer 종류        projection                sglang 모듈        K      N       calib
    ───────────────────────────────────────────────────────────────────────────────────
    공통 (64층)       gate_proj + up_proj    →  mlp.gate_up_proj  5120  34816   g / u
                      down_proj                 mlp.down_proj    17408   5120   d
    full_attn (16층)  q+k+v_proj             →  self_attn.qkv_proj 5120  14336   q/k/v
                      o_proj                    self_attn.o_proj   6144   5120   o
    linear_attn(48층) in_proj_qkvz              linear_attn.…      5120  16384   — (없음)
                      out_proj                                     6144   5120   — (없음)

`in_proj_qkvz`는 `[key, key, value, value]` = `[2048, 2048, 6144, 6144]`(q,k,v,z) 융합이다.
**이름은 실행 트리(upstream main) 기준**이다 — 포크는 같은 것을 넷으로 쪼개 놨다.

`q_proj`가 `[12288, 5120]`인 것은 `attn_output_gate: true`라 q가 게이트와 함께
`24 heads × 2 × 256`으로 실리기 때문이다. sglang의 `QKVParallelLinear`가 q/k/v를
`[12288, 1024, 1024]`로 묶으므로 plan의 분할도 그 경계를 따른다.

**linear_attn 3종에는 calib이 없다.** 자산의 `q`/`k`/`v`/`o`는 `attn_layers`(16개
full_attention 층)에서만 채워져 있고 나머지 48층은 전부 0이다 — 그대로 마스킹하면
`imp = 0 >= thr = 0`이라 전부 살아남아 sparsity가 조용히 0이 된다. 그래서 그 셋은
`"sparse": false`로 명시한다 (빠뜨림과 구분하기 위해 명시가 필수다).

`in_proj_ba`(N=96)와 `conv1d`(K=4)는 합쳐서 0.02 G라 오프로드 대상이 아니다.

    python scripts/qwen38/make_plan.py --hot 0.125 --warm 0.125 -o plans/qwen38/x.json
"""
import argparse, hashlib, json, sys
from pathlib import Path

H, I, NL = 5120, 17408, 64
HEAD_DIM, N_HEADS, N_KV, GATE = 256, 24, 4, 1
KEY_DIM, VALUE_DIM = 16 * 128, 48 * 128
ATTN_LAYERS = list(range(3, NL, 4))          # 자산의 attn_layers와 같아야 한다
ROW_GROUP, COL_GROUP = 2, 32


def bands(k, hot, warm):
    """[0, k)를 hot/warm/cold로. 경계는 ROW_GROUP 배수.

    **마지막 티어가 나머지를 흡수한다.** 각 티어를 독립적으로 내림 정렬하면 반올림
    나머지가 남아 의도치 않은 꼬리 밴드가 생긴다 — `--hot 0.1 --warm 0.9`가 K=6144에서
    `[6142, 6144)` 짜리 **2행 cold 밴드**를 만들었고, cold가 미배선이라 서버가 즉사했다
    (2026-09-01). 비율의 합이 1이면 cold가 없어야 하는데 부동소수점 내림이 그걸 깬다.
    """
    cold = max(0.0, 1.0 - hot - warm)
    live = [(t, f) for t, f in (("hot", hot), ("warm", warm), ("cold", cold)) if f > 0]
    if not live: raise SystemExit(f"k={k}: 비율이 전부 0이다")
    out, cur = [], 0
    for i, (tier, f) in enumerate(live):
        last = i == len(live) - 1
        end = k if last else min(k, cur + int(k * f) // ROW_GROUP * ROW_GROUP)
        if end > cur:
            out.append([cur, end, tier]); cur = end
    if cur != k: raise SystemExit(f"k={k}: 밴드가 [0,{cur})만 덮는다")
    return out


def shards(n, nodes):
    """N축을 노드로 등분. 경계는 COL_GROUP 배수, 조각 로컬 좌표."""
    step = (n // nodes) // COL_GROUP * COL_GROUP
    out, cur = [], 0
    for i in range(nodes):
        end = n if i == nodes - 1 else cur + step
        out.append([i, cur, end]); cur = end
    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot", type=float, default=0.125)
    ap.add_argument("--warm", type=float, default=0.125)
    ap.add_argument("--nodes", type=int, default=2)
    ap.add_argument("--p", type=float, default=None, help="sparsity 예산 (미지정 = dense)")
    ap.add_argument("--asset", default="assets/qwen38_27b.pt")
    ap.add_argument("--gpu-warm", default="gemv_worklist")
    ap.add_argument("--cpu-cold", default="kt_tile_k2_bf16")
    ap.add_argument("--attn", action="store_true", help="self_attn qkvo도 오프로드")
    ap.add_argument("--linear-attn", action="store_true", help="linear_attn 3종도 오프로드")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    sparse = a.p is not None
    root = Path(__file__).resolve().parents[2]

    def mk(k, n, calib=None):
        d = {"bands": bands(k, a.hot, a.warm)}
        if any(b[2] == "cold" for b in d["bands"]):
            d["cold_shards"] = shards(n, a.nodes)
        if sparse:
            if calib is None:
                d["sparse"] = False
            else:
                d.update({"calib": calib, "p": a.p, "lambda": 0.0})
        return d

    projs = {
        "mlp.gate_up_proj": {"k": H, "n": 2 * I, "parts": [
            {"name": "gate", "n": I, **mk(H, I, "g")},
            {"name": "up", "n": I, **mk(H, I, "u")}]},
        "mlp.down_proj": {"k": I, "n": H, **mk(I, H, "d")},
    }
    if a.attn:
        # QKVParallelLinear이 [12288, 1024, 1024]로 묶는다 (q에 게이트 포함).
        q, kv = N_HEADS * (1 + GATE) * HEAD_DIM, N_KV * HEAD_DIM
        projs["self_attn.qkv_proj"] = {"k": H, "n": q + 2 * kv, "parts": [
            {"name": "q", "n": q, **mk(H, q, "q")},
            {"name": "k", "n": kv, **mk(H, kv, "k")},
            {"name": "v", "n": kv, **mk(H, kv, "v")}]}
        projs["self_attn.o_proj"] = {"k": N_HEADS * HEAD_DIM, "n": H,
                                     **mk(N_HEADS * HEAD_DIM, H, "o")}
    if a.linear_attn:
        # ⚠ 모듈 이름은 **트리마다 다르다.** upstream main(`sglang-dense`)은 GatedDeltaNet의
        # 입력을 `in_proj_qkvz`(q,k,v,z 융합) + `in_proj_ba` 둘로 두는데, 포크(`sglang`)는
        # 그걸 `in_proj_qkv`/`in_proj_z`/`in_proj_b`/`in_proj_a` 넷으로 쪼갰다
        # ("Split projection layers (following vLLM's implementation)"). 여기 이름은
        # **실행 트리(upstream main)** 기준이다 — 포크 소스를 보고 쓰면 아무것도 매치되지
        # 않고, 그건 method.py의 coverage 게이트가 잡는다 (2026-09-01 실제로 잡혔다).
        #
        # `in_proj_ba`(N=96)와 `conv1d`(K=4)는 오프로드 대상이 아니다.
        # calib 없음 → sparse: false 명시 (0 테이블로 마스킹하면 조용히 sparsity 0)
        for name, k, n in (("linear_attn.in_proj_qkvz", H, 2 * KEY_DIM + 2 * VALUE_DIM),
                           ("linear_attn.out_proj", VALUE_DIM, H)):
            projs[name] = {"k": k, "n": n, **mk(k, n, None)}

    plan = {"schema_version": 1, "model_id": "Qwen/Qwen3.8-27B",
            "dims": {"num_layers": NL, "dtype": "bfloat16"},
            "kernels": {"gpu_warm": a.gpu_warm, "cpu_cold": a.cpu_cold},
            "projs": projs}
    if sparse:
        p = root / a.asset
        plan["sparsity"] = {"score": "k2wl2",
                            "calib": {"path": str(p.resolve()),
                                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()},
                            "pmax": 0.9, "grid": 0.005, "ng": 201, "renorm_it": 3}

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=1))

    sys.path.insert(0, str(root / "sglang/python"))
    from sglang.srt.layers.prism.linear.plan import parse_plan, validate_static
    pl = parse_plan(out); validate_static(pl)
    slots = sum(len(pl.proj(0, n).parts) for n in pl.names())
    print(f"{out}  ({out.stat().st_size/1024:.1f} KB)")
    print(f"  ✅ validate_static — proj {sorted(pl.names())}")
    print(f"  좌표 {len(pl.projs)}개 · 층당 슬롯 {slots}개 · sparsity={'k2wl2 p=%.2f' % a.p if sparse else 'none'}")
    print(f"  ⚠ full_attention 층은 {ATTN_LAYERS[:4]}… 16개뿐 — 나머지 48층에서 self_attn.* 는 매치되지 않는다")


if __name__ == "__main__":
    main()
