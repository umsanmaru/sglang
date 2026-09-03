# Qwen3.8-27B × Prism dense 런북 — nutella3 / RTX 5090 (sm_120)

dense 레인의 첫 실모델. **목적은 성능이 아니라 배선 검증**이다 — GPU 티어만으로는
warm(PCIe) 오프로드뿐이라 이득이 없고, "prism이 실모델에서 서는가"가 이 단계의 전부다.

## 트리·환경

| | |
|---|---|
| 트리 | `sglang-dense` (upstream/main `9e9d26a4a`) |
| env | `prism-dense` — `bash scripts/build_prism_dense_env.sh` |
| 모델 | `~/models/Qwen3.8-27B` (55.6 GB, 샤드 18개) |
| 자산 | `assets/qwen38_27b.pt` (19.6 MB) |

포크(`sglang`)는 개발 트리이고, prism 소스를 여기로 미는 것은
`DST=…/sglang-dense bash scripts/port_prism_to_upstream.sh` 한 명령이다.
**포크에서 dense 코드를 고칠 때마다 다시 밀어야 한다.**

## 이 모델의 구조 (체크포인트 직독으로 확정)

```
64층 = linear_attention 48 + full_attention 16 (층 3, 7, 11, …, 63)
hidden 5120 · intermediate 17408 · prefix `model.language_model.layers.N.`

공통 (64층)        mlp.gate_up_proj   K=5120  N=34816   parts: gate/up   calib g/u
                   mlp.down_proj      K=17408 N= 5120                    calib d
full_attn (16층)   self_attn.qkv_proj K=5120  N=14336   parts: q/k/v     calib q/k/v
                   self_attn.o_proj   K=6144  N= 5120                    calib o
linear_attn (48층) linear_attn.in_proj_qkvz K=5120 N=16384               calib 없음
                   linear_attn.out_proj     K=6144 N= 5120               calib 없음
```

⚠ **모듈 이름은 트리마다 다르다.** upstream main(`sglang-dense`)은 GatedDeltaNet 입력을
`in_proj_qkvz`(q,k,v,z 융합 `[2048,2048,6144,6144]`) + `in_proj_ba` 둘로 두는데, 포크
(`sglang`)는 그걸 `in_proj_qkv`/`in_proj_z`/`in_proj_b`/`in_proj_a` 넷으로 쪼갰다.
plan의 이름은 **실행 트리 기준**이어야 한다 — 포크 소스를 보고 쓰면 아무것도 매치되지
않는다. 2026-09-01에 실제로 그렇게 틀렸고 `check_coverage`가 잡았다:

```
PlanError: prism dense: these planned projections never matched any linear layer:
           ['linear_attn.in_proj_qkv', 'linear_attn.in_proj_z']
```

게이트가 없었으면 그 2.5 G 파라미터가 조용히 오프로드에서 빠진 채 서버가 정상 기동하고
정상 응답했을 것이다 — VRAM만 더 먹고 벤치 결론만 틀린다.

`q_proj`가 `[12288, 5120]`인 것은 `attn_output_gate: true`라 q가 게이트와 함께
`24 heads × 2 × 256`으로 실리기 때문이다.

⚠ **`linear_attn` 3종은 마스킹할 수 없다.** 자산의 `q`/`k`/`v`/`o`가 `attn_layers`
16개 층에서만 채워져 있고 나머지 48층은 전부 0이다. 0 테이블로 마스킹하면
`imp = 0 >= thr = 0`이라 **전부 살아남아** sparsity가 조용히 0이 된다 — 출력은
정확하고 벤치 결론만 틀린다. plan 생성기가 `"sparse": false`로 명시하고,
calib 어댑터가 그래도 새어 들어오면 즉사시킨다.

sparse 커버리지: 층당 9슬롯 중 마스킹 가능은 full_attn 층 7 / linear_attn 층 3 →
**384 중 256 (67%)**. 100%가 필요하면 Muse-Glimmer(52층 전부 표준 attention)를 쓴다.

## Phase 1 — stock 기동 (기준선)

```bash
scripts/qwen38/run_qwen38.sh                 # 터미널 A
scripts/qwen38/smoke.sh                      # 터미널 B
```

게이트:
- 토큰이 나온다
- 로그에서 `use_attn_output_gate` / projection 이름이 위 표와 맞는지
- **이 출력을 적어 둔다** — Phase 2의 대조 기준이다 (greedy라 결정적)

## Phase 2 — dense prism, GPU 티어만

```bash
python scripts/qwen38/make_plan.py --hot 0.125 --warm 0.125 -o plans/qwen38/mlp_h125_w125.json
scripts/qwen38/run_qwen38.sh plans/qwen38/mlp_h125_w125.json
scripts/qwen38/smoke.sh
```

게이트:
- 기동 로그에 `[prism-linear] plan loaded: …` 와
  `[prism-linear] coverage: mlp.down_proj×64, mlp.gate_up_proj×64 (총 128개 등록)`
- **stock과 같은 토큰**이 나온다
- `nvidia-smi`로 VRAM이 stock보다 낮다 (weight 일부가 pinned host로 갔으므로)

그다음 범위를 넓힌다 (`--attn`, `--linear-attn`). `--attn`을 넣으면 coverage가
`self_attn.qkv_proj×16` 처럼 **16**으로 찍혀야 한다 — 64가 나오면 층 판정이 틀린 것이고,
0이면 이름이 틀려 `check_coverage`가 즉사시킨다.

⚠ **cold 밴드가 있는 plan은 아직 못 돈다.** `executor.register()`가
`NotImplementedError: COLD rows are not wired yet`로 즉사한다 — 조용히 그 행을 빼면
값이 틀리므로 일부러 막아 뒀다. `--hot`/`--warm` 합이 1.0이 되게 주면 cold가 안 생긴다:

```bash
python scripts/qwen38/make_plan.py --hot 0.5 --warm 0.5 -o plans/qwen38/nocold.json
```

## Phase 3 이후 (cold / sparse)

`CPUINFER=14`가 붙는 자리는 cold backend가 배선된 뒤다. 지금 넘겨도 쓰이지 않는다.
sparse는 `--p 0.5`로 plan을 뽑되, **cold가 선 뒤에** 켠다 (WARM+COLD가 sparse 티어다).

## 알려진 함정

- **`sglang-dense`에서 돌려야 한다.** `prism-e2e`(포크 env)로 돌리면 파이썬이 포크를
  보고, 거기엔 `muse_glimmer.py`도 최신 upstream도 없다.
- **plan의 경로는 절대경로로 들어간다** (`readlink -f`). calib 자산 경로도 plan 안에
  절대경로 + sha256으로 박히므로, 자산을 재생성하면 plan을 다시 뽑아야 한다.
- **`--mem-fraction-static`을 낮춰야 한다** (스크립트 기본 0.70). hot 밴드가 VRAM을
  먹는데 sglang은 그걸 모른 채 KV 캐시를 잡는다.
