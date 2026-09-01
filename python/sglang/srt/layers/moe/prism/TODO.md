# Prism TODO — 의도적으로 미룬 작업 대장

P0에서 결정만 해두고 구현을 미룬 항목들. 각 항목은 "왜 미뤘는지"와
"구현 시 건드릴 곳"을 함께 기록한다. (P0 범위 자체는 CONTRACTS.md와
커밋 계획 참조)

## dense 확장 — 진행 중 (2026-08-31)

MoE 오프로드와 같은 K-split을 qkvo(`wq_a`/`wq_b`/`wo_a`/`wo_b`)와 dense MLP
(`gate_up_proj`/`down_proj`)에도 적용한다.

### 확정된 설계 (2026-08-31 사용자 결정)

- **plan은 MoE와 분리.** 별도 파일 + 별도 env(`SGLANG_PRISM_LINEAR_PLAN`).
  근거: MoE만 / dense만 / 둘 다를 env 조합으로 스윕할 수 있어야 벤치가 성립한다.
  대가는 VRAM/PCIe/CPU 예산을 두 plan이 따로 정해 합이 하드웨어를 넘을 수 있다는
  것 — planner의 책임이고 런타임은 검사하지 않는다.
- **executor도 별도.** dense는 pair 축이 없어 `[M,k,N]` → `[M,N]`이고
  grouping·worklist·라우터 가중합이 통째로 불필요하다.
- **위치: 공유 코어만 승격.** `layers/prism/`에 축 무관한 것만 올리고 dense를 그
  밑에. MoE 5,000줄은 제자리에서 re-export한다 (기존 42파일 무수정).

### 착지한 것

**① linear 래퍼 훅** — `FusedMoE.__init__`에는 `maybe_wrap_moe_quant_method`
슬롯이 있는데 `LinearBase.__init__`은 `quant_method`를 고르고 그대로 끝났다.
그 사이가 비면 래퍼가 **파라미터를 어디에 할당할지** 정할 수 없다 — 계약 ③
(full 텐서를 host에 받아 K-슬라이스 후 원본 소멸)이 성립하지 않는다.

- `layers/linear_method_registry.py` 신설, `LinearBase.__init__` 마지막 줄에 훅
- MoE registry와 다른 점 셋 (전부 `LinearBase`의 성질에서 강제됨):
  `prefix`가 predicate 인자다(`layer_id`가 없다) · 모든 linear가 지나가므로
  predicate는 모르는 prefix에 반드시 None을 내야 한다 · **`tp_size`는 predicate
  시점에 아직 없다**(서브클래스가 `super().__init__()` 반환 후 대입).
  `server_args`도 None일 수 있다 — linear는 서버 없이도 routinely 생성된다
  (`get_global_server_args()`가 `ValueError`를 던진다).
- `test/prism/test_linear_registry.py` 13종

**② 공유 코어 승격** — `layers/prism/` (505줄). 의존은 아래로만 흐른다:
`moe/prism/` → `prism/`, `prism/linear/` → `prism/`. 역방향 0 (검증됨).

- `numa.py` (266줄, 변경 0) · `kernels.py` (130줄) · `geometry.py` (78줄:
  `Tier`/`BandSpec`/`NumaShard`/`KernelSpec`/`PlanError`/`ROW,COL,PAIR_GROUP`)
- `kernels.gpu_store_format`만 갈렸다: 태그(`"bf16"`/`"mxfp4"`/`"fp8"`)까지는
  공유고, 태그 → `StoreFormat` 객체 해석은 각 오프로드가 한다. 포맷 객체가
  파라미터 이름과 full 텐서 인출을 들고 있고 그것이 MoE(w13/w2)와
  dense(weight)에서 갈리기 때문이다.
- `moe/prism/{numa,kernels}.py`는 shim, `plan.py`는 기하를 re-export →
  `moe.prism.plan.Tier` 같은 기존 경로가 전부 산다. `test_numa.py`만 새 경로로
  옮겼다 (private `_libnuma`를 쓰는데 `import *`가 안 넘긴다).

**③ dense plan** — `layers/prism/linear/plan.py` + `test_linear_plan.py` 36종.

- 좌표가 `(layer, proj)`이고 proj는 **열린 문자열**이다. MoE는 gate/up/down
  enum이면 됐다 — 세 개뿐이고 K가 `hidden`/`intermediate`에서 파생된다. dense는
  모델마다 projection 집합이 다르고 K에 파생 공식이 없어서 **k/n을 plan이 직접
  적고** 로드 타임에 실제 layer와 대조한다(`check_dims`). 안 대조하면 엉뚱한 행을
  슬라이스해 **결과가 조용히 틀린다**.
- `split_prefix("model.layers.7.self_attn.wq_b") → (7, "self_attn.wq_b")`.
  이 규약을 plan에 둔 이유: "무엇이 proj인가"는 plan의 어휘다. 훅은 이걸 부를 뿐
  이름 규칙을 자기가 알지 않는다.
- `projs`에 없는 projection은 안 건드린다. "전부 오프로드" 축약형은 두지 않았다 —
  무엇을 오프로드하는지가 곧 실험 변수라 명시가 기본이다.
- **sparsity는 v1 범위 밖** (의도적): MoE calib 자산이 `[L,E,K]` 축이라 dense의
  `[L,K]`와 포맷이 다르고 자산 생성이 선행되어야 한다.

**④ 인덱스 어휘 승격** — `layers/prism/store.py` (`IDX_DTYPE`/`OFF_DTYPE`/`MAX_K`/
`is_row_run`). dense가 같은 K-인덱스 표현을 쓰므로 두 곳에 두면 드리프트가 곧
**무증상 오답**이다 — 한쪽만 int32로 올라가면 wrap된 인덱스가 전혀 다른 행을 읽는데
예외가 안 난다. `moe/prism/index.py`는 re-export.

**⑤ dense weights** — `layers/prism/linear/weights.py` + `test_linear_weights.py` 27종.

- `PreparedLinear{hot, warm, cold}` — proj 3개 × 티어 3개 = 9 shard였던 것이 티어 3개로.
  `row_off[E+1]`이 **사라졌다**: 그 테이블은 "expert마다 k가 다르다"를 표현하려고
  있었고 dense에는 expert가 없다. `LinearTierShard`는 `w_flat [k_tier, N]` +
  `k_index [k_tier]` + `contiguous`/`k_start`뿐이다.
- 방향 규약은 MoE 그대로: hot/warm은 `[k_tier, N]` **K-major**(전치 없는 정준 방향,
  GPU 커널 공유), cold는 `[N, k_pad]` **ckpt 방향 유지**(kt `from_mat`이 읽는 방향) +
  커널 타일 배수까지 0 패딩. `real_rows`는 `[E]` 텐서가 아니라 **스칼라**다 —
  "E=1 퇴화 MOEConfig"로의 번역은 cold backend 어댑터 한 곳에서 하고 이 구조체는
  dense의 사실만 말한다.
- 테스트의 축은 **계약 ⑤**다: 세 티어 부분합의 fp32 합 == 원래 행렬곱, 그리고 hot/warm/
  cold 경계를 옮긴 두 plan의 재구성 weight가 비트일치(배치 불변성). 실제 배치
  (hot→VRAM, warm→GPU-local NUMA pinned, cold→평범한 host)도 CUDA 테스트로 잡는다 —
  warm이 원격 소켓에 앉는 것은 결과가 정확하고 느리기만 해서 다른 어떤 테스트도 안 잡는다.
**⑥ dense 포맷 (bf16 + blockwise fp8)** — `layers/prism/linear/formats.py`.

- MoE `StoreFormat`을 그대로 못 가져온다: `create_params`/`take_full`이 `w13_weight`/
  `w2_weight`라는 MoE 어휘에 묶여 있다. dense에서 그 역할은 `method.py`가
  `LinearBase`의 `weight`/`weight_scale_inv`를 직접 읽어 넘기는 것이므로, 이 포맷
  객체는 **텐서만 보고 layer를 모른다**.
- 포맷이 정하는 것은 결국 하나다: **K를 어디서 자를 수 있는가.** bf16은 페어(2),
  fp8은 **128** — 배율 하나가 128k×128n 블록을 덮으므로 티어 경계가 블록을 쪼개면
  두 티어가 같은 배율을 나눠 갖는다. `check_rows`가 로드마다 본다 (plan 밴드 검증은
  `ROW_GROUP=2`까지만 보므로 여기가 유일한 게이트다).
- fp8 스토어: GPU 티어 `[k_tier, N]` u8 코드 + `[k_tier/128, N/128]` fp32 배율,
  cold `[N, k_pad]` + `[N/128, k_pad/128]`. **fp8 cold에는 패딩이 없다** — 타일(128)과
  K 정렬(128)이 같기 때문. `float8_e4m3fn`은 index_select가 안 되므로 u8 뷰로 다룬다.
- 테스트는 **비트일치 재구성**(`test_fp8_reconstruction_is_bit_exact`)이 절단의
  정확성을 증명하고, 행렬곱 비교는 재결합 오차 때문에 명시 tolerance를 쓴다 —
  둘을 섞으면 tolerance가 진짜 버그를 숨긴다.
- **mxfp4는 두지 않았다** (dense 대상에 없다는 2026-08-31 사용자 확인). 필요해지면
  `k_align=32` 구현을 더하면 되고 배관은 이미 배율 있는 포맷을 지원한다.

**⑦ proj별 커널 (혼합 포맷)** — plan 스키마 변경.

`projs[name].kernels`가 top-level `kernels`를 덮는다. MoE는 model-global 하나로
족했지만 dense는 **한 모델 안에서 형식이 갈린다**: DSV4의 `wo_a`는
`SGLANG_OPT_FP8_WO_A_GEMM`이 꺼져 있으면 `quant_config=None` +
`params_dtype=bfloat16`으로 만들어져, 같은 모델의 `wq_b`/`wo_b`가 fp8인데 혼자
bf16이다 (`models/deepseek_v4.py:641`). model-global 커널 쌍으로는 그 모델을 아예
표현할 수 없다 — 나중에 발견하면 schema_version bump가 되므로 지금 넣었다.

⚠ **cold backend에 파급된다**: proj마다 kt 커널이 다르면 kt 인스턴스도 proj마다
따로 만들어야 한다. MoE는 한 layer의 gate/up/down이 한 `PartialMoEWrapper`를 공유하는데,
dense는 애초에 proj별 인스턴스이므로 구조적으로는 맞다 — 다만 스레드풀 하나에 인스턴스
수가 늘어나는 비용은 실측해야 한다.

**⑧ dense 레인 분리 — `sglang-dense` 워크트리** (2026-09-01)

타깃이 Muse-Glimmer-30B로 정해지면서 트리를 나눴다. 그 모델 파일이 **upstream main에만**
있고 포크는 7,496커밋 뒤처져 있어서다 (포크 173 모델 파일 / main 221). GLM 레인은
pr-36507 기준인데 거기엔 GLM 전용 tilelang 패치가 섞여 있어 dense에는 불필요하다.

    sglang        f116b6dd9 [prism-orchestration]   포크 = 개발 트리(source of truth)
    sglang-dense  9e9d26a4a [prism-dense]           upstream/main
    sglang-glm    3992c4ce2 [pr-36507]

`scripts/port_prism_to_upstream.sh`를 dense용으로 확장했다:

- `copy srt/layers/linear_method_registry.py`
- **`== 2c) LinearBase 훅`** — 앵커 삽입. 후보 2개 fallback을 둔 이유는 base마다
  `LinearBase.__init__`의 끝 모양이 다르기 때문이다: 최신 upstream은 quant_method 선택
  뒤에 `wrap_method_with_debug_kernel_once` 블록이 있고 구버전 포크에는 없다. main에서는
  첫 후보가 걸렸고 그 **앞**에 붙는다 — debug 래퍼가 prism의 `apply`를 감싸야 계측 대상이
  prism이 된다.
- 검증에 dense 항목 (훅·registry·method 존재, 문법, `keeps_params_on_host` 선언)

**`linear.py`의 훅을 지역 import로 바꿨다.** module-level import면 registry 파일이 없는
트리에서 `linear.py` 자체가 import되지 않아 이식이 성립하지 않는다. 오버헤드는
**0.84 µs/linear**(2000개에 1.7 ms) — 같은 프로세스에서 훅을 no-op으로 치환해 비교한 값이다.

⚠ **`keeps_params_on_host`를 빠뜨렸다가 이식 스크립트를 읽고 발견했다.** MoE는
`method.py:293`에 선언하고 `loader.py:142`가 소비하는데 dense method엔 없었다. 없으면
`device_loading_context`가 layer마다 full weight를 GPU로 올렸다 내린 뒤 버린다 —
**에러 없이 느려지기만** 한다(DSV4 실측 43층 186초). 스크립트의 "선언+소비 양쪽을 센다"
검증이 정확히 이 부류를 잡는 장치다.

**남은 Phase 0**: `scripts/build_prism_dense_env.sh` 실행 (Qwen3.8 다운로드가 대역폭·디스크를
쓰고 있어 그 뒤로 미뤘다). 게이트는 `import muse_glimmer` + `pytest test/prism` 전량.

**⑨ dense cold — C++ `PartialDenseWrapper`로 간다** (2026-09-01 결정)

퇴화 `MOEConfig`(E=1, down 슬롯만) 실측은 **정확히 동작한다** — `forward_down`이
`x @ W_cold.T`를 내고 decode 0.39 s/tok(28스레드, 119 GB/s)로 warm-only의 2.7배다.
인스턴스를 9→72개로 늘려도 슬롯당 시간이 평평하다.

그런데 **빈 슬롯을 못 쓴다** (`moe.hpp:513` "no weight source") — gate/up에 32행 더미가
필수고, 그건 증상이지 원인이 아니다. 원인은 슬롯이 gate/up/down 3개로 고정이라는
구조이고, dense는 층당 5~8슬롯이라 거기 맞출 수가 없다. 대가가 슬롯당 인스턴스
(Qwen3.8 352개, 오버헤드 +25 GB, 로딩 78초)로 나온다.

**계획·발견·게이트 전부 `layers/prism/linear/TODO.md`에 있다.** 그쪽이 dense 인계
문서다. 여기서는 MoE와 공유되는 사실만 적는다:

- kt 출력 dtype은 **bf16**이다 (`resources.py:75`가 이미 적어놨는데 못 보고 fp8로 잡아
  `got[j] ≈ ref[2j+1]` + 50% 0을 한참 디버깅했다)
- shard 테이블은 **NUMA 노드 수만큼** 필요하다 (`partial shard table size != tp_count`)
- 스레드 최적점은 이 박스에서 **28** (물리 16코어). 14→28 +7%, **56은 9배 악화** —
  MoE 메모의 과다구독 현상과 같다
- kt 빌드에 **`libxml2-devel`**이 필요하다 (conda `hwloc.pc`의 `Requires.private`가
  시스템 `.pc`로 폴백해 없는 경로를 만든다)

### 남은 것

- **`executor.py`** — 2-phase가 아니라 **1-phase**다 (gate_up/down이 별개 layer라
  prism이 둘을 한 호출에서 볼 일이 없다). cold submit ∥ hot/warm GEMV → sync →
  fp32 Σ → bf16. rejoin이 라우터 가중합 없이 단순 합이라 MoE `rejoin.py`를 못 쓴다.
- **`method.py`** — `LinearMethodBase` 형태: `create_weights(layer,
  input_size_per_partition, output_partition_sizes, input_size, output_size,
  params_dtype, **attrs)` / `apply(layer, x, bias) -> Tensor`. MoE의
  `dispatch_output`/`CombineInput` 래핑이 없다. predicate에서 `tp_size`를 못 보므로
  TP=1 방어는 `create_weights`에서 (`input_size_per_partition != input_size`면 즉사).
- **② kt cold의 dense 입구 (미해소).** cold는 kt `MOEConfig` + `expert_ids`를
  전제한다. `operators/llamafile/linear.h`의 `Linear`는 (a) python 바인딩이
  `ext_bindings.cpp:678`에서 주석 처리돼 있고 (b) llamafile 경로라 AMX가 아니다.
  **권고: E=1 퇴화 `MOEConfig`로 우회** — 기존 배관(`PartialMoEWrapper`, partial
  K-index, NUMA N-shard)을 그대로 쓰고 `expert_ids`를 전부 0으로 준다. AMX GEMM
  바인딩 신설은 그 다음.
- **최종형으로의 이동.** MoE도 `layers/prism/moe/`로 내리면 이름이 완전히 정직해지고
  upstream rebase 충돌 표면도 준다(prism은 fork 전용 코드). 5,432줄 이동 + import
  재작성 42파일이라 별도 커밋으로 미뤘다.

### TP를 열 때 (지금은 TP=1 전용이라 무관)

축 관계는 다음과 같다 — kt NUMA shard는 **N축**(`plan.NumaShard`)이라
prism band(K축)와 원래 직교다. 부딪히는 것은 TP다.

| | 쪼개는 축 |
|---|---|
| prism band (hot/warm/cold) | K |
| kt NUMA shard | N |
| `ColumnParallelLinear` TP | N |
| `RowParallelLinear` TP | K |

- `RowParallelLinear`(down_proj, wo_b): TP(K) 안에 prism band(K) 중첩.
  정확성은 무해하다 — all_reduce가 `apply` 뒤에 온다. 숙제는 (a) plan의 K
  오프셋이 full-K 절대좌표라 rank별 re-base 필요, (b) 밴드 경계 페어(%2) 정렬이
  K/T 분할 후에도 유지, (c) sparsity calib(`wn`, `pair_dot`)도 K축이라 동반 슬라이스.
- `ColumnParallelLinear`(gate_up, wq_b, wo_a): TP(N) 안에 kt NUMA(N) 중첩.
  `N/TP/nodes`가 `COL_GROUP=32` 배수여야 하며 아니면 `cold_backend._build_config`의
  정렬 검사가 즉사시킨다.
- **진짜 병목은 축이 아니라 프로세스 자원이다.** TP>1은 rank마다 별도 프로세스인데
  `_PrismRuntime`(CPUInfer 스레드풀)은 프로세스 전역 1벌이라 T배 과다구독된다 —
  이 파일 위쪽 "테스트 위생" 절과 `method.py`의 실측(16코어 60스레드 → sync
  1.85ms) 그대로다. `SGLANG_PRISM_CPUINFER_THREADS`를 코어/TP로 낮추고 rank별
  코어 pin이 전제. warm pinned store도 GPU-local 노드에 놓이므로 2소켓 4GPU에서는
  rank 둘이 같은 노드의 host 대역폭을 나눠 쓴다.

## cold 커널 교체 — 두 포팅이 같은 레이아웃을 다른 방법으로 만든다 (2026-08-25 밤)

두 커널이 도착했고 **둘 다 n-contiguous B 레이아웃을 전제**하는데, 그 레이아웃을
만드는 방법이 서로 배타적이다. 이건 어느 한쪽을 조용히 고르면 안 되는 갈래다.

| | AMX (`rmnc-port`) | AVX (`cpu-mm-platform`) |
|---|---|---|
| 방법 | **새 버퍼 타입** `BufferBBF16NContigImpl`. kt의 `BufferBBF16Impl`은 무변경 | kt의 `BufferBBF16Impl`을 **제자리 변경** (`pack_block`/`get_submat`/`to_mat`/`from_bb_transposed` 4곳 + writeback 1곳) |
| 커널 struct | standalone (`GemmKernel224BF16RowMajorNC`), 자기 BufferA/B/C | **상속** (`GemmKernelTileK2BF16 : GemmKernel224BF16`) → kt의 BufferB를 그대로 씀 |
| writeback | 레이아웃 가드로 **거부** | `get_submat` 호출로 치환해 **해결** |

AVX 쪽이 상속으로 버퍼를 공유하는데, 그 버퍼가 제자리 변경되지 않으면 AVX 커널은
구 레이아웃을 읽는다. 반대로 제자리 변경하면 AMX 쪽의 별도 버퍼 타입이 잉여가 된다.

**권고: AVX 안(제자리 변경)**. 근거 셋 —
1. 목표가 "weight 한 벌"인데 버퍼 타입이 둘이면 그 목표가 타입 수준에서 깨진다.
2. AMX 커널의 `BufferB` typedef를 kt 것으로 돌리면 되고, 두 레이아웃의 **2 KB 단위
   내부가 이미 동일**하므로(AMX B-타일 그 자체) 커널 본문은 안 바뀐다.
3. `write_weights_to_buffer`가 가드가 아니라 `get_submat` 치환으로 닫히고, 같은
   함수의 full-K 결함(§10.7)도 같이 닫힌다.

**착수 전 확인 하나**: 제자리 변경은 kt **원본 커널**의 packed 순서도 바꾼다.
§8.2는 AMX 타일 내부가 불변이라 `load_b`/`amx_kernel`이 영향받지 않는다고 하지만,
그건 게이트로 확인할 것 — `test/prism` 39종 + per_commit 306종이 전부 그 판정이다.

### 오늘 밤 붙인 것 / 안 붙인 것

- **AMX NC 커널**: 파일 3개 배치 + 배선 완료. **비트일치 게이트 6/6 통과**
  (`bf16_rowmajor-test`). 단 `WithPartial=false`로 바인딩했다 — partial의 sparse
  분기가 `amx::vec_mul_sparse`를 부르는데 그 오버로드가 `GemmKernel224BF16` 버퍼에
  하드코딩돼 있어 NC 타입에서 인스턴스화가 안 된다. 그래서 **prism은 아직 이 커널을
  쓰지 않는다** (`cpu_cold: kt_amx_bf16` 그대로).
- `do_*_gemm`의 sparse 분기에 `kHasSparseGemv` 컴파일 가드를 넣었다 — sparse 짝이
  없는 커널에서 조용히 dense로 떨어지는 대신 즉사한다 (마스킹이 조용히 사라지면
  성능만 달라져 어떤 테스트도 안 잡는다).
- **AVX tile_k2**: 미착수. 위 갈래가 정해져야 시작할 수 있다.

---

## ~~grouped GEMM prefill~~ — ✅ 완료 (2026-08-26/27, 3단계)

**진단** (35B dims h12.5/w12.5/cold75, H100, 레이어 1개, M=2048): 층 264 ms 중
warm 250 ms — worklist가 pair마다 W를 다시 읽어(중복도 M·k/E = 64배) 층당
12.9 GB를 PCIe로 읽었다 (스토어는 192 MiB). 엔진 prefill 5.1 ms/tok, 2688토큰
TTFT 13.8 s.

**1단계 — grouped GEMM** (`prism_grouped.cuh`, `grouping.py`, tiers/executor
grouped 경로, `GROUPED_MIN_M=16`): pair를 expert로 묶어 W를 expert당 한 번 읽는
wmma bf16 커널. hot/warm 같은 커널(포인터 종류만 다름), gate+up 융합 launch.
M=2048: warm gateup 167 → 2.64 ms(63배, PCIe 51 GB/s = 스토어 1회 읽기 이론치),
hot gateup 3.47 → 0.65 ms. 층 264 → 33 ms. **엔진 TTFT 13.8 → 1.90 s (7.3배),
0.71 ms/tok.** 임계 M: M=8은 worklist 우세, M≥16 grouped 우세 (warm PCIe가 결정).

**2단계 — hot∥warm 스트림 분리** (`resources.warm_stream`, executor
`split_streams`, `SGLANG_PRISM_SPLIT_STREAMS`, warm launch grid 상한
`WARM_MAX_TILES`): 엔진 TTFT 1.90 → 1.74 s (−8%). 단 마이크로벤치(cold 없는
h50/w50)에서는 layer 21.3 → 20.6 ms로 이득이 작다 — warm 그리드 상한을 132/32/16
블록으로 낮춰도 hot이 거의 겹치지 않는다 (아래 미룸 항목 "UVA 커널과 compute
커널의 동시 실행" 참조). warm 커널은 그리드 32 타일(512블록)로도 PCIe를 포화한다.

**3단계 — cold GPU 읽기** (`cold_gpu.py`, kt `alloc_cold_slabs` +
`cold_slab_views()`, 커널 COLD 레이아웃 로더, executor `cold_gpu_min_m`,
`SGLANG_PRISM_COLD_GPU_MIN_M`): kt packed AMX 레이아웃을 cudaHostRegister로
매핑해 GPU가 **재배치 없이 제자리 읽기** — weight 한 벌. cold 75% = 1.15 GiB/층
→ 23.6 ms/층 @ 51 GB/s, M과 무관. CPU cold(14스레드)는 M=512 12 ms / 2048
28 ms / 4096 49 ms → **교차점 M≈1500–2000** (스레드 수·PCIe 세대에 따라 이동).
정합: CPU cold 대비 rel 8e-5 (bf16 누산 순서 차이). **엔진 TTFT** (cold GPU 강제
vs CPU cold, 둘 다 grouped+split): 672tok 806 vs 715 ms, 1344tok 1049 vs 1044,
2688tok **1485 vs 1744 (−15%)**, 4032tok 1925 ms (0.48 ms/tok). 기본 임계
`COLD_GPU_MIN_M = 1536`. 켜면 cold 전량(35B 45 GiB)이 host-register되어 pinned가
된다 — `SGLANG_PRISM_COLD_GPU_MIN_M=0`이 끄는 스위치.

**누적**: prefill 2688tok TTFT 13.8 s → 1.49 s (**9.3배**), 5.14 → 0.55 ms/tok.
측정은 전부 GPU1(H100)이 조용할 때; 08:10 이후 타 사용자 vLLM이 GPU1을 점유해
`WARM_MAX_BLOCKS=128` 기본값의 executor 수준 확인(nocold plan)은 미완 — 커널 단독
실험 근거로 기본값을 뒀다.

**kt 기본 커널 vs prism, 단일 소켓(node0) 6스레드, GPU0 Blackwell, TTFT min/3
(2026-08-27)** — 1-node plan `q36_h125_w0125_1node_wl`:

| tok | kt 기본(cold-only CPU) | prism CPU cold | prism cold GPU(≥1536) |
|---|---|---|---|
| 168 | 453 | 804 | 771 |
| 672 | 720 | 1200 | 1343 |
| 1344 | 1080 | 1675 | 1927 |
| 2688 | 1632 | 2571 | **1602** |
| 4032 | 2236 | 3470 | **2190** |
| 5376 (4096+1280 청크) | 3168 | 5000 | 3796 |
| 8064 (2청크) | 4468 | 6850 | **4277** |

읽는 법: (1) **CPU cold 경로는 6스레드에서 kt 기본에 진다** — 일이 75%인데 층당
cold 69 ms(M=2688) vs kt 기본 전체 41 ms. 원인은 partial 계약의 구조 비용: cold가
(token, slot)별 partial을 bf16 [M,k,2I]/[M,k,H]로 내보내고(kt 기본은 라우터 가중
합 [M,H]만 냄 — 출력 8배), act [M,k,I] D2H, phase마다 sync·GPU rejoin 동안 CPU
유휴. 스레드가 적을수록 CPU 계산이 짧아져 이 고정비가 지배한다. (2) **cold GPU면
kt 기본과 동급~4% 우세** (PCIe 바운드: 청크 4096당 warm+cold 52 GiB ≈ 1.1 s
이론, 실측은 토큰 타일 2개 재읽기로 +30%). (3) 4096 청크 다음의 짧은 꼬리
청크(1280)는 CPU cold로 떨어져 5376이 4032보다 1.6 s 느리다 — 청크 단위가 아니라
**요청 단위**로 cold 경로를 정하거나 hybrid로 가야 한다. (4) 168~1344에서 cold GPU
켠 런이 CPU cold 런보다 10% 느린 것은 마이크로벤치로 재현되지 않았다(slab
등록은 CPU cold 속도에 영향 없음) — 엔진 런 간 노이즈로 본다.

**nsys 리포트 3종** (`profiles/{kt_q36_1s6t_prefill2688, prism_q36_1s6t_prefill2688_cpucold,
prism_q36_1s6t_prefill2688_coldgpu}.nsys-rep`, 2688tok, node0 6thr, GPU0; 드라이버
scratchpad `engine_nsys_prefill.py`, 요약 `summarize_nsys.py`):

| | TTFT | GPU 커널 합 | 주역 |
|---|---|---|---|
| kt 기본 | 1634 ms | 64 ms (4%) | 전부 CPU AMX — 층당 ~39 ms에 expert 100% |
| prism CPU cold | 2712 | 482 (18%) | `cold.*.window` 2075 ms(77%) = CPU 75% 행에 층당 ~50 ms; hot/warm GPU 324 ms는 그 안에 숨음; H2D 97 + D2H 27 |
| prism cold GPU | 1617 | 1547 (96%) | cold grouped GEMM 1113 ms(72%, 층당 28 ms ≈ 43 GB/s), hot/warm 273, rejoin ~100, memcpy 0.5 |

행당 CPU 효율: kt 네이티브 39 ms/100% vs partial 경로 50 ms/75% → **1.7배 열세**
(출력 8배·gather/pack·2-phase). cold GPU 경로의 rejoin/elementwise 100 ms(6%)는
fp32 누산·silu·캐스팅 분리 launch — 융합하면 회수 가능. `s1.topk_d2h`(eager 경로의
`topk_ids.to("cpu")`)는 cold GPU에서 불필요한 sync 포인트다.

**partial export 병렬화 (kt `export_rows_parallel`, 2026-08-27)**: `KT_PARTIAL_TIMING=1`
단계 분해(M=2688, 6thr)에서 export가 **단일 스레드 memcpy**로 gateup 6.4 + down 14.5 =
층당 21 ms(CPU cold 층 60 ms의 1/3)였다 — down이 gateup만큼 걸리던 이유의 절반
(나머지 절반은 K=384 짧은 down GEMM의 효율 2.5 vs 3.5 TFLOPS). subpool 16토큰 블록
work-stealing으로 → export 1.6 + 3.2 ms, 층당 44 ms. 엔진(CPU cold 6thr): 2688tok
2571 → **2099 ms**, 8064tok 6850 → **5332**. kt 기본(1632/4468) 대비 남은 격차 ~1.25배
= 출력 8배 D2H/H2D + down GEMM 효율 + 2-phase sync. kt 기본 경로는 이 함수를 쓰지
않아 무영향. 남은 CPU 단계: gateup pack(x gather) 4.8 ms/층.

**apple-to-apple CPU 커널 (kt 네이티브 `forward_prefill` vs partial 경로, 같은 층·M=2688·
node0 6thr·균등 라우팅, `KT_PARTIAL_TIMING=1`; scratchpad `bench_kt_native_vs_partial.py`,
2026-08-27)**:

| 단계 (ms/층) | kt 네이티브 100% | partial 100% | partial 75% |
|---|---|---|---|
| gate/up GEMM (+to_mat) | 24.3 | **24.3** | 19.1 |
| down GEMM (+to_mat) | 16.0 | **15.6** | 13.3 |
| 나머지 | cpy_input 2.2 + packA 2.9 + act 1.4 + packA_down 0.6 + weight_sum 1.6 = 8.7 | pack 5.1+1.7 + export 1.6+3.2 = 11.6 | 3.9+1.4 + 1.6+3.2 = 10.1 |
| 합 | 49.1 | 51.8 | 43.0 |

**GEMM은 동일하다** — 같은 `do_*_gemm`/`to_mat`/work-stealing 코드라 시간이 소수점까지
같다. CPU 커널이 느린 것이 아니고, partial의 CPU측 초과분은 층당 +2.9 ms(pack이
native의 cpy_input+packA보다 크고, export가 weight_sum보다 큼)뿐이다. 남은 엔진 격차
(2099 vs 1632 = 층당 52 vs 41 ms)는 CPU 밖에 있다: partial 경로는 층마다 **CPU 구간 ↔
GPU 구간(rejoin#1 silu·캐스팅 ~2.5 ms, D2H/H2D 3 ms, attention/dense 1.6 ms)이 직렬**로
번갈아 돌아 CPU가 층당 ~7 ms 유휴다 (40층 × 7 ≈ 280 ms). kt 네이티브는 GPU 구간이
attention 1.6 ms뿐이다. 해법은 CPU/GPU 파이프라이닝 — cold-down deferral(다음 층 attention
아래로) 또는 M을 반으로 쪼개 반쪽 rejoin 동안 다른 반쪽 cold 계산.

주의: 마이크로벤치(균등 라우팅) 네이티브 49 ms vs 엔진 kt 기본 ~41 ms/층 — 엔진 프롬프트
("alpha bravo …" 16단어 반복)는 라우팅이 소수 expert에 몰려 expert당 M이 커 AMX 효율이
높다(M_STEP 패딩↓, B 재사용↑). 두 경로가 같은 프롬프트를 쓰므로 비교는 공정하지만,
절대값은 프롬프트 의존이다 — 실제 텍스트로 재측정할 것.

**"graph를 씌우면?"의 상한 실측** — `SGLANG_PRISM_COLD_STREAM=1`(cold submit/sync를
stream host node로, host 블로킹 제거 = graph가 없앨 수 있는 성분만 제거), CPU cold 6thr:
2688tok 2099 → 1964 ms(−6%), 8064tok 5332 → 5450(±0). 즉 host dispatch 갭은 층당 ≤3 ms고
나머지 격차는 CPU↔GPU 의존 사슬의 직렬화다 — graph로는 안 없어지고 파이프라이닝(deferral
/ M 분할)이 필요하다.

**rejoin 융합 (`rejoin.py`, Triton 2커널, 2026-08-27)**: torch 사슬(캐스팅·add·split·
silu·mul·wsum, 층당 ~30 launch·2.5 ms GPU) → partial을 한 번만 읽는 커널 2개. 207
테스트 통과(정확표현 비트일치 포함). 마이크로벤치 비-CPU 잔여 6.6 → 4.9 ms/층.
엔진(6thr): CPU cold 2688tok 2099 → 2073, 8064 5332 → 5213; cold GPU 2688 1602 →
1509, **8064 4277 → 4016 (kt 기본 4468 대비 −10%)**. 다음은 M 분할 파이프라인
(cold 반쪽씩 → GPU 구간을 다른 반쪽 CPU 시간 아래로).

**host launch 오버헤드 실측 (nsys cuda_api_sum, 2688tok 6thr)**: CPU cold 경로 launch
API 63 ms/4826회(**13.1 µs/회 — 정상 5–6의 2배**, host 스레드가 node0의 kt 워커 6개와
코어 경합 추정) + Python dispatch(추정 ~20 µs/회 ≈ 100 ms) ≈ **160 ms ≈ 4 ms/층 ≈ 8%**.
cold GPU 경로는 5.7 µs/회·28 ms(호스트가 놀아서 정상). 이 성분이 graph/host 선행
발행으로 지울 수 있는 상한이고, `COLD_STREAM=1`의 −6%가 그 대부분이다. 나머지
미설명 ~4 ms/층은 kt 풀의 phase 진입/sync 지연으로 남는다. 저렴한 시도: scheduler
스레드를 kt 워커 코어 밖에 pin(13 → 6 µs면 ~35 ms).

**prefill MoE 층 CUDA graph 실측** (`bench_layer_tiers.py --graph 1`: `force_graph_path`
executor로 캡처, cold는 stream host node, 6thr node0): eager → replay ms/층 — CPU cold
M=672 25.19 → 24.56, M=2688 48.84 → 48.43; cold GPU M=2688 28.76 → 27.91. 출력 동일.
**graph 이득 ≤ 1 ms/층(≤3%)** — prefill은 launch 바운드가 아니다. API 트레이스의 63 ms
launch 시간은 host가 앞서 발행하며 CPU 계산과 겹치는 부분이 대부분이었다. 미설명
잔여는 kt 풀 phase 진입/sync 지연 + sglang 측 층 밖 오버헤드로 남는다.

**cold hybrid (2026-08-27)** — cold GPU 조건에서 expert를 GPU/CPU로 나눠 **동시** 계산.
kt skip 마스크(`gpu_experts_mask`, pinned u8, 호출 직전 host 쓰기; skip 행은 kt가 0-fill)
+ 제한 그룹핑(`build_grouping(expert_mask=)`) + 4-part rejoin. env
`SGLANG_PRISM_COLD_HYBRID_FRAC=gu,dn` (phase별 비율 — down은 CPU export 고정비가 커
GPU 몫이 더 커야 균형). eager·비-stream 호출에서만 켠다. 정합: CPU cold 대비 rel 4e-5.

| M=2688, ms/층 | CPU cold | all-GPU cold | hybrid 최적 |
|---|---|---|---|
| node0 6thr | 47.8 | 28.4 | **26.8** (0.65, 0.85) → −6% |
| 2노드 14thr | 31.3 | 28.6 | **25.0** (0.55, 0.80) → −13% |

분해(6thr, 0.65/0.85): CPU gu 12.8 / dn 6.3 vs GPU(warm+cold) 12.9 / 8.1 — phase 균형은
맞고, 이득이 작은 이유는 (a) 6스레드 CPU가 cold의 1/3만 가져가 GPU측이 27.6 → 21로만
줄고, (b) CPU partial의 H2D(3 ms)·4-part rejoin·zero-fill export(3.1 ms)가 그 절반을
되먹는다. 이득 ∝ CPU 몫. 개선 여지: rejoin이 (m,j) 소유를 마스크로 골라 읽으면
zero-fill·H2D 절반 제거(~2 ms/층).

**dual socket 14thr 엔진 비교 (2026-08-27, 2-node plan `q36_h125_w0125_dense`)**:

| tok | kt 기본 2s14t | prism CPU cold | prism all-GPU cold | prism hybrid(expert, 비용모델) |
|---|---|---|---|---|
| 2688 | **1370** | 1439 (+5%) | 1721 | 1492 |
| 8064 | **3312** | 3764 (+14%) | 5124 | 4074 |

- all-GPU cold가 1노드 plan(4016)보다 훨씬 느린 이유: node1 slab을 GPU0(node0)이
  **원격 소켓 UVA**로 읽는다 — warm의 GPU-local NUMA 규칙이 cold GPU에도 그대로 적용된다.
- hybrid 분할은 **비용 모델**로 바꿨다(`_balance_hybrid`): GPU 비용 ∝ expert 수(W 바이트),
  CPU 비용 ∝ pair 수 → 꼬리 expert를 GPU, 몰린 expert를 CPU. 첫 판(몰린 expert를 GPU)은
  6thr 8064tok에서 7.3 s로 역효과였다. 강한 skew에선 모델이 all-GPU로 수렴한다(active
  expert가 줄어 GPU가 전부 싸짐).
- **다음 = 계층 분할**: 소켓 경계는 N-shard(원격 shard → 그 소켓 CPU 전부, GPU-local
  shard → GPU + local CPU expert 분할), shard 비율은 plan `cold_shards`로 재조정(node0↑).
  필요한 kt 변경: 노드별 `gpu_experts_mask` 포인터(지금은 TP 래퍼 공용). 어림 층당
  ~17 ms @2688 (kt dual ≈ 31·(2688/4096) 환산과 비교 시 ~1.8배).

**W-resident grouped GEMM (2026-08-27, `prism_grouped_gemm_wres`, PCIe 경로 기본 ON
`WRES_PCIE`)**: 블록 = (expert, 열 타일)이 W 슬라이스를 smem에 1회 올리고 토큰 타일을
순회 → pair 수와 무관하게 W 1회 읽기. smem이 부족하면(GPU0 sm_120 99 KB) K를 조각내
fp32 scratch로 누산을 이어간다(BN=64 유지 — BN=16 폴백은 A gather 16배로 회귀했다:
cold gateup 15.8 → 18.7). 비트일치(row-major·cold 레이아웃, skew·pair>128 포함). 속도
(GPU0, uniq 바이트 기준): warm gateup M=4096 skew 8.6 → 2.7 ms(16 → 49 GB/s), cold
gateup skew 12.9 → 8.2, 균등 M=2688 회귀 없음(15.9), **M=4096 층 46.6 → 30.8 ms**.

**packed-layout GEMV (decode, `prism_gemv_packed.cuh`)**: kt 32×32 VNNI 타일을 그대로
읽는 worklist GEMV — dense·gate+up 융합·**k2wl2 sparse**(페어 마스크 = 타일의 16-dword
행 하나) 전부 row-major 커널과 비트일치. M=1 pinned 90 µs (row-major pinned 107).
→ warm을 **kt 포맷 slab 한 벌(pinned)**로 두는 모드 `SGLANG_PRISM_WARM_KT=1`
(loader `warm_kt`, backend `load_warm_layer`(GPU-local 노드 전량·타 노드 0행 — kt 검증
완화), tiers `PackedWarmTier/GateUp`). 다음: prefill에서 warm-kt 인스턴스를 CPU가
계산하는 경로(hot=0 → kt 네이티브와 같은 FLOPs) + warm/cold 공통 hybrid.

**W-resident + hybrid 이중 합산 버그 (2026-08-27 발견·수정)**: `prism_grouped_gemm_wres`가
expert 루프에서 `tile_off`(hybrid의 expert 마스크)를 보지 않아 GPU가 cold 100%를 계산
→ CPU 몫 행이 rejoin에서 두 번 더해졌다. hybrid nsys에서 cold 커널 863 ms(all-GPU
~600)로 드러남. `tile_off[e+1]==tile_off[e]`면 skip으로 수정, 정합 복구(rel 5e-5),
회귀 테스트 `test_cold_hybrid_matches_cpu_cold`. 그 사이의 hybrid+wres 수치(6thr
2688 1411/1510 ms)는 무효.

**warm-kt 판정**: h125 warm-kt(GPU packed/grouped) 8064tok 2602 ms = row-major 2592와
동일; hot=0 전부 GPU 2812; hot=0 전부 CPU(warm-kt+cold) 7785(kt 4468) — CPU 계산은
이 하드웨어에서 GPU 읽기에 못 미친다. **기본 경로는 row-major warm + W-resident +
cold GPU**, warm-kt는 `SGLANG_PRISM_WARM_KT=1` 옵션으로만 유지.

**hybrid nsys (수정 후, 2688tok 6thr, `profiles/prism_q36_1s6t_prefill2688_hybrid`)**:
TTFT 1335 ms(비프로파일) vs all-GPU cold 1061 — **hybrid가 더 느리다**. 분해: GPU cold
wres 703 ms(8.8 ms/launch), CPU `cold.gu.window` 3.9 ms/층, **`cold.dn.h2d_out` 3.5 ms/층**
(88 MB H2D가 이론 1.7의 2배 — GPU의 UVA W 읽기와 같은 H2D 방향 PCIe를 나눠 쓴다),
H2D 합 133 ms. 즉 hybrid의 구조 비용 = CPU partial H2D가 GPU W 읽기와 **링크 경합** +
zero-fill export + 4-part rejoin. 6스레드에서는 CPU 몫(~1/3)의 절감보다 이 비용이 커서
순손실. hybrid는 CPU가 충분히 강하거나(14thr 마이크로벤치 −13%) PCIe가 널널할 때만
의미가 있다 — 기본은 **all-GPU cold(≥임계) / CPU cold(그 아래)**, hybrid는 옵션.
비용 모델 ratio도 보정점 상수(`HYBRID_CALIB_PAIRS`)로 고정했다 (이전엔 현재 분포로
환산해 small-M에서 전부 GPU로 몰았다).

**small-M prefill (100–300 tok × 10, hot=0/warm12.5, 2노드 14thr, TTFT ms min/3, 2026-08-27)**:

| tok | kt dual 14thr | prism CPU cold | **CPU cold + COLD_STREAM=1** | cold GPU(강제) | hybrid(비용모델) |
|---|---|---|---|---|---|
| 102 | 348 | 452 | **353** | 526 | 436 |
| 138 | 275 | 393 | **288** | 604 | 380 |
| 178 | 316 | 407 | **304** | 626 | 406 |
| 217 | 382 | 454 | **322** | 642 | 426 |
| 265 | 421 | 450 | **335** | 661 | 455 |
| 295 | 367 | 468 | **360** | 697 | 481 |

- 이 구간은 층당 CPU 계산 1–3 ms라 **host 고정비가 지배**: CPU cold는 kt의 1.07–1.43배,
  `COLD_STREAM=1`(host 선행 발행)로 **0.80–1.05배** — kt와 동급~우위. 소수 프롬프트에서
  노이즈 큼(kt 138tok < 102tok).
- cold GPU 강제는 1.3–2.2× — PCIe 고정비. 기본 임계(`COLD_GPU_MIN_M=1536`)가 맞다.
- hybrid는 ratio 수정 후 CPU와 동급(GPU pair 몫 1–2%로 스스로 수렴) — 이전 판(현재
  분포로 c/g 환산)은 전부 GPU로 몰아 1.5–1.9배였다.
- 함의: 작은 M의 기본은 CPU cold + COLD_STREAM=1. 단 COLD_STREAM은 host 즉시쓰기
  (qlen pin, hybrid 마스크)가 stream 순서 밖이라 M이 층마다 바뀌는 상황·hybrid와 동시
  사용 전에 그 값들을 stream-ordered로 옮겨야 한다 (아래 남은 것).

**cold_async — 미완, 격리됨 (2026-08-27)**: cold를 전용 stream + 완료 플래그(kt
`signal_task` → pinned int32 → `prism_wait_flag` 커널)로 받아 블로킹 sync 콜백을 없애
hot/warm과의 콜백 스레드 직렬화를 풀려는 시도. 배관(kt binding, resources.cold_stream/
cold_flag, backend.submit_signal, executor._cold_phase_async, env SGLANG_PRISM_COLD_ASYNC)은
들어갔으나 **두 버그로 비활성**: (1) 첫 호출 결과 오답 — partial H2D가 flag wait보다 앞서
실행됨(두 번째 호출은 blocking과 비트일치 → 순수 순서 문제), (2) `prism_wait_flag` 스핀
커널이 wrapper의 stream 컨텍스트를 안 타고 default stream에 실려 GPU hang/segfault. 기본
꺼짐 + 테스트 skip + 생성자 경고. **재구현 방향**: 스핀 커널을 드라이버 네이티브
`cuStreamWaitValue32`로 교체(hang 위험 제거), H2D를 wait 커널 뒤 같은 stream에 명시적으로
순서 넣기. 동기는 여전히 유효 — `COLD_STREAM=1` nsys에서 warm과 cold 콜백이 직렬화됐다.

### 남은 것 (이 작업에서 파생)

- **UVA 커널과 compute 커널의 동시 실행**: 스트림을 나누고 warm의 블록 수를
  줄여도 hot이 그 아래로 숨지 않는다. 가설은 UVA(sysmem) 로드가 SM의 메모리
  파이프라인(MSHR/LSU 큐)을 점유해 같은 SM의 다른 블록도 멈춘다는 것 — 확인은
  nsys로 두 커널의 시간 구간이 실제로 겹치는지, SM당 점유가 어떤지 보는 것.
  겹침이 원리적으로 불가하면 대안은 SM 분할(green context) 또는 hot을 cold GPU
  읽기와 같은 스트림에 두고 "겹침"을 포기하는 것.
- **토큰 타일 재읽기**: kBM=128이라 expert당 pair가 128을 넘으면(M·k/E > 128,
  35B에서 M > 4096) W를 타일 수만큼 다시 읽는다 (M=4096: warm 2.64 → 3.9 ms,
  cold 23.6 → 35 ms). BM=256 변형 또는 K-분할 없는 큰 타일이 해법.
- **hot grouped 커널 효율**: M=2048 h50 gateup 34 GFLOP에 2.6 ms = 13 TFLOPS
  (H100 bf16 dense의 2%). A gather + BK=32 + 파이프라인 없음. cold/warm이 PCIe
  바운드인 지금은 임계경로가 아니지만 hot 비율이 큰 plan(h875)에서는 보인다 —
  cp.async 파이프라인, BK=64, A gather 벡터화.
- **cold hybrid**: 교차점 근처에서는 cold 행을 CPU와 GPU가 **나눠** 읽는 편이
  둘 중 하나보다 빠르다 (M=2048: CPU 28 ∥ GPU 23.6 → 절반씩이면 ~13 ms).
  N shard(노드)를 단위로 한 노드는 CPU, 다른 노드는 GPU로 보내는 것이 가장 싼
  형태 — 노드별 partial이 이미 서로소 열이다.
- `cold_gpu_min_m` 기본값은 이 머신 실측(≈1500)이고 plan/하드웨어 의존이다 —
  planner가 정해 Plan에 적는 것이 맞다 (지금은 env).

## (기록) grouped GEMM prefill — 착수 전 메모

- **현상**: prefill이 decode와 같은 pair-native worklist GEMV를 탄다 (2026-08-25,
  S6). 실측으로 bmm 경로 대비 **1.6~1.9배**다 (35B gate 치수, M=1024~4096:
  worklist 0.98/3.87 ms vs bmm 0.52/2.28 ms). worklist는 라우팅된 pair만
  계산하는데(bmm은 그룹의 전 (토큰, expert) 쌍) 처리량이 1/4이라 나온 값이고,
  원인은 GEMV라 tensor core를 못 타는 것이다.
- **왜 감수했나**: bmm은 **가변 per-expert K를 원리적으로 표현할 수 없다**
  (연속 배치 축 = 배치당 K 하나). 폴백으로 남겨도 이 스키마의 plan을 실행하지
  못하는 경로일 뿐이라, 경로를 하나로 합치고 prefill이 인덱스를 지원하게 하는
  쪽을 택했다.
- **구현**: 토큰을 expert로 묶어(topk argsort) `x[tokens_e][:, idx_e]`를 gather한
  뒤 expert별 GEMM. kt의 prefill 구조(`m_local_pos_` 장부)와 같은 모양이고, GPU
  쪽에서는 grouped/segmented GEMM 커널이 제 형태다. expert별 `torch.mm` 루프는
  정확하지만 launch가 E×proj×layer로 늘어 별도 실측이 필요하다.
- **선행 판단**: prefill의 GPU 티어 시간이 prefill 전체(cold 지배)에서 차지하는
  비율을 먼저 재야 한다. 1.7배가 전체의 몇 %인지 모르는 채로 커널을 쓰는 것은
  이르다.
- **건드릴 곳**: `tiers.py`(세 번째 구현 — Protocol은 그대로), `executor.py`의
  `_run_gateup`/`_run_down` 디스패치. 커널 신규.

## 인덱스 전환 — 남은 작업 (2026-08-25)

sglang 쪽은 S6까지 끝났다 (인덱스 표현 → 커널 → 로더 → executor). 남은 것:

1. **kt 쪽 K3~K5** — `KIndex` + per-expert 접근자, gather-to-dense, dual-pack.
   목록과 line ref는 kt `doc/prism-partial-entrypoints.md` §10.
2. **S7 결합** — `cold_backend`가 인덱스를 kt에 주입. 현재 cold는 밴드 기하로
   남아 있고(`ColdBand.index`가 동행만 한다), `weights.py`가 비밴드 cold plan을
   `NotImplementedError`로 거부한다. K3~K5가 끝나야 풀린다.
3. **calib gather** — `wn`/`pair_dot`을 인덱스로 gather (지금은 밴드 절단).
   소비자가 cold뿐이라 2번과 같이 간다.
4. **S9 셔플 자산** — 자산 생성기 + 로더(`index.py`의 `from_rows`가 입구) +
   **셔플 인덱스에서의 정수 비트일치**. 인덱스 시대의 exact 검출기이고, 지금까지의
   모든 검증은 "연속 인덱스 = 밴드"라는 퇴화형 위에서만 돌았다.
5. **타일 클러스터링** — cold 인덱스를 "같이 죽는 행끼리" 정렬해
   `avx_kernel_4_sparse`의 타일 통째 skip(`mask == 0`)을 발화시키는 순열 최적화.
   keep 0.47에서 랜덤 배치의 발화 확률은 4×10⁻⁵라 지금은 죽은 코드다. 정확성에
   영향이 없고(순열일 뿐) GPU 쪽 대가도 없다 (셔플이 공짜임은 실측).

## 테스트 위생 — CPUInfer 스레드 누적

`test_executor.build_executor`가 호출마다 `KtColdBackend`를 새로 만들고, 기본
`cpuinfer_threads=60`이다. 테스트가 늘수록 스레드풀이 쌓여 스위트가 느려진다
(2026-08-25에 신규 테스트 하나가 스위트를 타임아웃 근처로 밀었다). 픽스처
스코프를 올려 CPUInfer를 공유하면 된다.

## ~~warm 전송 ping-pong~~ — **폐기** (2026-08-25)

warm은 전송되지 않는다 (제자리 UVA 읽기). arena도 그룹 루프도 없어졌으므로
ping/pong 할 대상이 존재하지 않는다. 아래는 폐기 시점의 기록.

### (기록) 그룹 루프 전송 노출 해소

- **현상**: distinct expert > n_slots이면 executor가 n_slots 단위 그룹
  직렬 루프를 돈다. 직렬이라 그룹 g+1의 stage 전송이 그룹 g의 GEMM 뒤에
  시작되어 전송이 완전히 노출된다 (P0는 의도적으로 감수 — 최단순).
- **구현**: arena를 2벌(ping/pong)로 두고, 그룹 g의 run_warm이 도는 동안
  그룹 g+1의 stage를 warm_stream에 선발행. WAR은 "pong 전송 시작 전에
  ping GEMM 완료 event 대기"로 보장.
- **건드릴 곳**: `resources.py` DeviceArena(2벌 할당 + 버퍼 선택),
  `executor.py` 그룹 루프(이벤트 체인). stage/run_warm primitive 시그니처는
  불변 — arena 인자만 ping/pong 중 하나를 받으면 됨.
- **선행 실측**: prefill T별 distinct warm expert 분포 카운터 (P0 실측
  항목) — distinct가 E에 근접하면 ping-pong보다 전량 전송 휴리스틱이
  맞을 수 있으므로, 데이터 보고 우선순위 결정.

## K2 재설계 — down도 N(hidden)축 NUMA 분할 (2026-08-20 사용자 결정)

설계: **전 proj가 각자의 N축으로 NUMA 분할** (gate/up=inter, down=hidden).
kt의 down K(inter)-분할은 "노드가 자기 act 열만 소비"하는 locality 결합의
산물인데, 우리는 gateup partial이 GPU rejoin#1을 거쳐 act가 full-width로
되돌아오므로("중간에 한 번 합치고 분배" = GPU 왕복) 그 결합이 없다.
결과: down도 서로소 열 direct write → **노드-합 소멸**, 전 proj 단일 패턴.

- 스키마: 현행 per-proj `cold_shards`가 이미 이 전제 ✓ (한때 TODO에 있던
  "inter 단일 분할로 개정" 제안은 잘못된 방향이라 철회됨, 2026-08-20).
  **gate/up은 N(shard)·K(밴드) 모두 공유** (2026-08-20 사용자 확정 —
  자유도가 균형 목적상 중복이라 풀 계획 없음). down의 N만 독립.
  P0 validate: gate == up (shards·bands) 요구, 위반 시 로드 즉사.
- 실행(K2): config에 down N-shard 기하 + do_down_gemm N 접근자화 +
  TP ctor 이중 독립 분할(inter/hidden) + down 스테이징 방향 변경(자기
  hidden 행 × cold inter 전체) + down_ba는 노드마다 full cold-inter pack
  (중복 소량) + export는 gateup과 동형. 노트 §2-4의 "노드-합 필요"는
  이 결정으로 폐기.
- 비율 제어(사용자 요구: warm 소켓의 cold 몫 축소): per-proj shards가
  그 노브. 실행 형태 확정(2026-08-20): **tp_configs의 치수를 내부 산식이
  아니라 Plan에서 추출해 주입** — GeneralMOEConfig에 per-node 기하 테이블
  4종 (`node_gateup_n_offset/rows`[inter 축, gate/up 공유],
  `node_down_n_offset/rows`[hidden 축]) 추가. TP ctor는 테이블 있으면
  그대로/없으면 기존 균등(kt 보존), prism cold.py가 cold_shards→테이블
  번역, K1 export의 균등 전제 공식(tp_part_idx×I)도 테이블 읽기로 교체.
  버퍼 할당은 무변경 자동 반영(치수 출처만 바뀜; 스크래치는 노드별 독립
  + max-grow가 레이어 간 차이 흡수, sglang arena도 이미 레이어 최대치 산정).

## ~~partial prefill (qlen>1)~~ — ✅ 완료 (2026-08-20)

gateup/down 양쪽에 prefill(qlen>1) 경로 구현 — kt forward_prefill 동형의
토큰 그룹핑(m_local_pos_) + gather + (m, slot) scatter export. decode와
stage/export 헬퍼 공유 (decode = pos 0 퇴화형). **strided from_mat 변형은
불필요했음** — prefill의 gather memcpy가 밴드 슬라이싱을 흡수. batch>1
eager decode도 함께 해제 (qlen ≤ max_len 범위 guard로 대체). 테스트:
decode/prefill 파라미터화 (중복 expert 필연 구성 + 정수 비트일치가 좌표
뒤섞임 검출) + 불균등 shard × prefill 조합.

## ~~stage 일괄화 (H2D 26→3/층)~~ — 완료 후 **폐기** (2026-08-25: stage 자체가 사라짐)

NVTX 실측(2026-08-20)에서 slot당 `copy_` 산란이 층당 H2D 26조각 + dispatch를
만들어 ~1.4ms/층의 오버헤드였다. `BatchedCopyStager`(host `index_select`로
결집 → H2D 1회/proj, 스크래치는 더블버퍼 + guard event로 WAR 방어)로 치환해
`select_stager`의 eager 기본값이 됐다. 등가성은 `PerSlotCopyStager` 대비
bitwise(`torch.equal`)로 검증 (test/prism/test_stagers.py).

## ~~graph bs=1 (GatherKernelStager)~~ — 완료 후 **폐기** (2026-08-25: bs 제약도 stager도 사라짐)

`GatherKernelStager.stage_from_device`(sel을 device 상주 topk_ids 슬라이스에서
직접 취함 — host copy/H2D/guard 전부 없음)와 cold submit/sync의 stream 통합
(kt `submit_with_cuda_stream`/`sync_with_cuda_stream` = `cudaLaunchHostFunc`
host node)을 조합해 M==1 decode를 캡처 가능하게 만들었다. `SlotOrderGrouping`이
그룹 조성에 host 결정이 없다는 전제(Task 2)와 맞물려 S1(`ids_cpu` D2H)이
decode 경로에서 완전히 소멸한다.

**30B 단일소켓 실측** (아래 표 참조): eager 106.24ms/tok → graph **56.10ms/tok**
(17.8 tok/s) — cold-only graph(55.91ms/tok)와 동급. 즉 warm 10% 오프로드가
주는 CPU 절감(~4.8ms)이 graph 경로의 host-node/오버헤드로 상쇄된다: 단일소켓
8스레드에서는 cold 대역(이론 바닥 ~43ms/tok) 자체가 지금의 성능 상한이고,
이 상한을 낮추지 않는 한 warm 비율을 더 키워도 graph에서는 잘 보이지 않는다.
bs>1 decode는 여전히 eager 폴백(persistent GEMV 전까지 — 아래 순위 참조).

## 다음 후보 순위 (2026-08-21, graph bs=1 완료 후 재평가)

graph bs=1로 "산란 dispatch"와 "S1 host 블록"이라는 두 개의 오버헤드 항목이
사라지면서, 남은 바닥은 거의 전부 cold 자체의 실행 시간이다. 그 관찰을 반영해
순위를 다음으로 재정리한다 (이전 순위는 이 관찰이 없던 상태에서 작성됐음):

1. **cold-down deferral** (아래 항목 상세) — 가장 큰 항목이며 여전히 착수
   전. down partial을 1-layer 지연시켜 다음 층 GEMM 아래로 숨기면, 노출된
   cold 대역 자체가 줄어든다 — graph가 손대지 못하는 유일한 층.
2. **warm 비율 확대 / 스레드(소켓) 확장** — 단일소켓 8스레드에서 cold
   43ms/tok가 이미 이론 바닥이라, graph를 얹어도 56ms/tok에서 멈춘다.
   다음 이득은 cold 대역 자체를 줄이는 데서만 나온다: warm이 흡수하는
   expert 비율을 늘리거나(밴드 재산정), 스레드/소켓을 확장해 cold GEMM의
   실측 시간을 낮춘다. 둘 다 planner/calibration 쪽 작업이라 이 저장소
   범위 밖일 수 있음 — 착수 전 실측으로 어느 쪽이 더 싼지 확인 필요.
3. **persistent GEMV** (warm GEMM 커널 교체, 아래 항목 상세) — bs>1 graph의
   선결 조건이자 torch_bmm placeholder의 낭비(라우팅 안 된 (토큰,expert)
   쌍까지 계산)를 없앤다. bs=1 decode 성능에는 당장 이득이 작음(그룹이
   이미 1개) — bs>1 스코프를 열 때 다시 최우선이 된다.
4. **prefill distinct 카운터** (아래 "warm 전송 ping-pong"의 선행 실측) —
   ping-pong 착수 여부를 결정할 데이터이지만, prefill 경로 자체가 아직
   decode보다 우선순위가 낮아 순위 최하단.

## 기타 미룸 항목 (합의된 것만, 요약)

- **CUDA graph 경로, bs=1** — ✅ 완료 (2026-08-21, 위 "graph bs=1
  (GatherKernelStager)" 절 참조). M=1이면 distinct ≤ top_k라 그룹이 항상
  1개 → 최악치 고정 그룹/패딩 낭비 문제가 공짜로 소멸하고 run_warm/rejoin이
  자명하게 shape-static이라는 2026-08-20 예측이 그대로 구현·실측됨.
  **bs>1 decode는 여전히 eager 폴백** (`--cuda-graph-bs 1`) — 다음 단계는
  persistent GEMV(worklist 네이티브)로 그룹 문제 자체를 제거하는 planir
  방식 (위 "다음 후보 순위" 3번).
- **cold-down deferral**: down partial의 1-layer 지연 합류 (kt deferral
  기계 재사용). Phase 2 노출 해소책.
- **C++ dual-pack (gate ≠ up)** — 2026-08-20에 폐기했다가 **2026-08-25에 부활**
  (gate/up 인덱스 독립 결정). kt A 풀이 2배가 된다 — kt doc §10.3. 스키마의 독립 표현력과 로더의 처리 능력은 공짜
  일반성이라 남겨둠 (계약 아님).
- **티어당 다중 밴드 / hot-warm-cold interleave**: 스키마·검증은 이미 허용
  (validate_static은 disjoint+커버만 요구, 티어 순서·횟수 무제한).
  실행 계층 3곳이 NotImplementedError로 명시 거부 중 — ① loader
  `_single_band`(store를 밴드별 세그먼트 목록으로), ② warm GEMM 계약의
  `k_offset: int`(→ 밴드 목록; 밴드별 bmm 후 fp32 합산 또는 x gather),
  ③ C++ partial 인스턴스(계약 ②-3 덕에 호출 시그니처는 불변 — pack이
  다중 구간을 알고 full-width x에서 구간별로 읽으면 됨). calibration이
  실제로 interleave를 뱉는 시점에 착수.
  **planir 실증 참조**: planir는 티어=임의 index 배열 + gather-to-dense로
  이미 해결 (k_split.cc:130 GPU gather / :114 PackLhsCold / :347 identity
  skip, warm_tier.cc:264 warm gather). 비용 = activation gather O(m×k_tier)
  뿐(weight는 로드 타임 pre-slice), decode m≤8에선 무시 가능. 우리 slice는
  연속-인덱스 퇴화형이므로 "연속이면 slice, 아니면 gather" 디스패치로 확장.
- ~~**warm GEMM 커널 교체: torch_bmm → persistent GEMV**~~ — ✅ 완료
  (worklist GEMV + 인덱스 변형; bmm은 2026-08-25에 삭제): torch_bmm은
  정확성 기준선용 placeholder다. 알려진 placeholder 한계 — ① x를 전
  expert에 broadcast하므로 라우팅 안 된 (토큰, expert) 쌍까지 계산
  (prefill에서 낭비 큼), ② 전역 matmul 플래그 저장/복원이 커널 안에
  있음(진짜 커널은 자체 fp32 누산이라 불필요), ③ 그룹 루프가 launch
  단위라 bs>1 graph에서 최악치 고정 필요. persistent GEMV(planir
  kernels.cu 연장)는 device worklist를 네이티브로 소화해 ①③을 없애고
  가변 k_warm도 공짜 — 교체는 registry 한 줄 (`WarmGemmFn` 계약 유지,
  단 다중 밴드 시 k_offset 인자 진화는 위 interleave 항목 참조).
- ~~**가변 k_warm[e]**~~ — ✅ 완료 (2026-08-25, flat + offset). 원문: loader의 균일성 요구 제거
  (offset 테이블 store) + warm GEMM을 persistent GEMV로 교체 (위 항목).
- ~~**pinned store NUMA 바인딩**~~ — ✅ 완료 (2026-08-25, libnuma 직접 + 배치 검증).
- ~~**hot tier**~~ — **완료** (2026-08-24). HotBand/HotStore + loader VRAM
  배치(`prepare_layer_weights(device=)`) + executor hot 경로(stager·arena
  없이 `index_select` → 같은 warm GEMM). 검증: 3-tier 재조립 bitexact,
  plan 불변성에 `all_hot`/`three_tier` 추가(all_hot ≡ all_warm 수치 일치),
  CUDA graph 캡처·replay bitwise 일치. plan 생성기 `--hot-frac`.
  남은 것: hot 비율 스윕 실측, `index_select` 사본 비용(GEMM 읽기량의 2배
  추가 HBM 트래픽) 측정 후 필요시 gather-free 커널.
- **N-shard rank 분산 (TP>1)**: rank별 자기 inter 샤드만 store/DMA.

## 실측표 (인덱스 worklist GEMV, RTX PRO 6000 Blackwell, 2026-08-25)

밴드 커널 대비 인덱스 커널의 순비용. 치수는 35B-A3B (E=256, top_k=8, Kx=2048/512).
`vN` = W 로드 폭(열/스레드), `shuf` = 무작위 순열 인덱스.

| 구성 | band | idx-v1 | idx-v4 | **idx-v8** | shuf-v8 |
|---|---|---|---|---|---|
| hot gate/up (k=768, N=512) | 10.8 µs | 12.3 (1.14x) | 10.8 (1.00x) | **10.7 (0.99x)** | 11.0 (1.02x) |
| hot down (k=192, N=2048) | 9.6 µs | 9.9 (1.03x) | 9.9 (1.03x) | **10.0 (1.04x)** | 9.9 (1.03x) |
| warm gate/up (pinned/UVA, k=256) | 47.7 µs | 47.9 (1.00x) | 44.0 (0.92x) | **44.6 (0.93x)** | 44.7 (0.94x) |

**층당 순비용 ≈ 0.** 인덱스 지원이 공짜다 (bs=1). warm은 오히려 7% 빨라졌다.

### 여기서 확정된 것

**1. x 산란은 공짜다.** `shuf`가 연속 인덱스와 차이 없다(±3%). x는 한 행이 4 KB라
L1에 상주하므로 gather가 순차 읽기와 같은 값이다. **티어 배치가 원본 순서를 지킬
이유가 없다** — planner가 자유롭게 섞어도 되고, cold 타일 클러스터링 같은 순열
최적화도 GPU 쪽에 대가가 없다.

**2. 인덱스의 첫 비용은 지연이었다 (1.83x).** 내부 루프의 `idx[r] → x[idx[r]]`
의존 로드 사슬. gate/up은 grid가 (8, 8) = 64블록뿐이라 188 SM에서 지연 바운드이고
그 사슬이 1:1로 벽시계에 드러났다. x 행을 블록당 1회 smem에 모아(KTILE=2048)
**1.83 → 1.14배**. 스테이징은 인덱스 경로에만 건다 — 밴드 경로는 x를 순차로 읽어
끊을 사슬이 없고 syncthreads만 늘어 순손실이다 (10.5 → 12.3 µs).

**3. 나머지는 W 로드 폭이었다 (1.14 → 0.99x).** 스레드가 열 1개(bf16 2 B) 대신
V개를 맡아 uint2/uint4로 읽는다. 블록의 열 타일 64는 고정하고 blockDim을
(64, 4) → (64/V, 4V)로 재배치해 **블록 수와 타일 기하를 불변**으로 뒀다 (단순히
V를 곱하면 gate에서 grid.x가 1로 붕괴한다).
- 기전은 대역폭 낭비 회수가 **아니다**. NVIDIA 메모리 요청은 32 B 섹터라 warp의
  연속 64 B는 2섹터를 꽉 채운다 — 버리는 바이트가 없었다. 버는 것은 스레드당
  in-flight 바이트(MLP)와 LSU 명령 수다.
- 정렬은 계약 ①의 `COL_GROUP = 32`가 보증한다 (n_cols % 8 == 0). `row_off[e]`는
  임의여도 된다 — 행 시작이 n_cols 원소의 배수이므로 열 정렬이 곧 행 정렬이다.
- 자동 선택이 정렬이 허용하는 최대(보통 8)를 고른다. **bs ≥ 8에서는 v4가 나을
  때가 있다** (bs=16: v4 20.3 vs v8 22.0) — decode(bs=1) 기준으로 8을 기본값으로
  뒀고, bs>1 스코프를 열 때 재측정할 것.

**4. 가변 k[e]의 부하 불균형이 벡터화로 대부분 사라졌다.** 총량 고정, 분포만 바꾼
실험 (E=256, 평균 768):

| 분포 | max k[e] | v1 | **v8** | v8의 uniform 대비 |
|---|---|---|---|---|
| uniform | 768 | 12.32 µs | 9.7–10.6 | 1.00 |
| bimodal | 1152 (1.5x) | 17.75 µs | 9.8–10.1 | **≈1.00** |
| heavy | 2048 (2.67x) | 29.97 µs | 12.63 | **1.19–1.30** |

v1에서는 벽시계가 `max_e k[e]`에 그대로 비례했다(2.42x). v8은 blockDim.y가 32라
긴 expert의 블록도 내부 병렬이 32-way이고, 그래서 wave 꼬리가 짧아진다. 남은
페널티는 max가 평균의 2.67배인 극단에서 **1.2~1.3배**뿐이다.
- **planner 함의**: per-expert 예산 skew가 사실상 자유롭다. 극단(2.5배 이상)만
  피하면 된다.
- 그래서 **K축 grid 분할은 보류한다.** 그 작업의 명분이던 부하 균형이 이미
  회수됐고, 대가(블록 간 리덕션 → atomicAdd의 run-to-run 비결정 또는 2-pass
  launch)가 남은 1.2배보다 크다. 블록 수가 병목이 아니라는 증거도 있다: gate
  (64블록)와 down (256블록)이 **W 바이트가 정확히 같은데**(6.29 MB) 시간이
  10.8 vs 9.6 µs로 6%밖에 차이 나지 않았다.

---

## 실측표 (Qwen3.6-35B-A3B, RTX PRO 6000 Blackwell 96GB, 2 NUMA, bs=1 greedy, 256tok×5 median, 2026-08-24/25)

치수 NL=40 NE=256 I=512 H=2048 → expert weight 총량 60 GiB. warm-frac 0.125 고정,
`--attention-backend triton --cuda-graph-bs 1 --max-running-requests 1 --context-length 4096`.

**(a) hot 스윕 @ mem-fraction 0.85** — ms/tok

| f_hot | hot VRAM | cold 행(gate/up) | sparse | dense | sparsity 효과 |
|---|---|---|---|---|---|
| 0 | — | 1792 (87.5%) | 24.51 | 27.75 | **−11.7%** |
| 0.0625 | 2.5 GiB | 1664 | 24.63 | — | — |
| 0.125 | 7.5 GiB | 1536 (75%) | 24.08 | 27.65 | **−12.9%** |
| 0.25 | 15 GiB | 1280 (62.5%) | 23.52 | 28.16* | — |
| 0.5 | 30 GiB | 768 (37.5%) | 23.61 | 23.14 | **≈0** |
| 0.75 | 45 GiB | 256 (12.5%) | 24.18 | 23.90 | ≈0 |
| 0.875 | 52.5 GiB | **0** | — | **18.26** | (cold 없음) |
| prism 미사용 (순수 GPU) | — | — | — | **6.58** | — |

<sub>*3회 측정(나머지는 5회)</sub>

VRAM은 `--hot-frac` 예산 예측과 정확히 일치 (h250: `mem usage=20.51 GB` = 5.51 + 15.00).

**(b) 32 GiB 상한 @ mem-fraction 0.335** — 두 구성 모두 실측 33.0 GiB 사용

| f_hot | f_warm | KV 토큰 | sparse | dense |
|---|---|---|---|---|
| 0 | 0.125 | 719k | 24.40 | — |
| 0.375 | 0.125 | 99k | 22.68 | 23.71 |
| 0.375 | **0.25** | 99k | **28.57** | — |
| **0.375** | **0** | 99k | **18.84** | — |
| 0.0625 | 0 | — | 22.86 | — |

토큰당 VRAM 38.1 KiB (KV 자체 20.0 + hybrid GDN state/index pool 18.1) — 2점 fit.

### 여기서 확정된 것

**1. cold sparsity의 실제 효과는 −11.7%다.** 그동안 −42%는 cold 레이어만 떼어낸
합성 벤치였다. 실제 keep ratio도 처음 측정됐다: gate/up 0.4697, down 0.4680
(accuracy-eval `result.md` 8.9절, `calib/qwen36/neuron_freq.py`). 판정식은 kt
런타임과 비트 단위로 맞춰 잰 값이다.

**2. cold 비용은 행 수가 아니라 층당 고정비(~140 µs)에 지배된다.** cold 행을
768 → 256으로 67% 줄여도 이득이 0인데(23.14 → 23.90) 마지막 256행을 없애면
5.6 ms/tok이 빠진다. 소재는 executor의 cold 경로가 current stream에 직렬로
올리는 `fill_x`(D2H) → `submit`(host node) → `sync`(host node) → `h2d_out`(H2D)
로 보이며, phase 2회 × 4개 = 층당 8회 stream 왕복이 cold 행 수와 무관하게 붙는다.
**아직 NVTX로 분해되지 않았다 — 차분에서 얻은 추정이다.**

**3. hot이 커질수록 sparsity 이득이 사라진다.** f_hot 0 → 0.125 → 0.375 → 0.5에서
sparsity 효과가 −11.7% → −12.9% → −4.3% → ≈0. 둘 다 cold 가변 성분만 건드리므로
경쟁 관계다. 같은 이유로 **frequency 기반 행 배치의 기대 이득도 f=0.375에서
0.05 ms/tok(0.2%)** 로 노이즈(산포 1.6 ms) 아래다 — 게다가 gate/up은 K축이 expert
간 공유되는 hidden 채널이라 per-expert permutation이 원리적으로 불가능해 down만
회수 가능하다.

**4. warm 티어는 이 하드웨어에서 순손실이다.** warm H2D는 cold 뒤에 숨지 않고
PCIe 대역폭 그대로 임계경로에 앉는다 — warm 0.125 → 0.25에서 실측 증분 147 µs/층,
이론 PCIe(44.8 GiB/s) 증분 131 µs/층으로 거의 일치. warm 12.5%를 cold로 넘기면
22.68 → 18.84 (**−17%**). warm의 PCIe 총량 5.24 ms vs cold가 같은 행을 처리하는
비용 1.40 ms.
- **arena 캐시("직전에 올렸으면 재전송 생략")로도 못 살린다.** decode step 간
  expert 재사용률 실측 35.2%(W=1, 160 MiB) ~ 66.4%(W=8, 1.28 GiB)인데 — 우연
  수준(3.1%)의 11배로 상관은 확실히 있다 — warm이 cold를 이기려면 적중률
  **73.3%** 가 필요하다. W=8도 부족하고 그 1.28 GiB는 hot에 주는 편이 낫다.
- 단, **cold가 매우 클 때(f_hot=0)는 미검증**이다. cold 창이 크면 warm 전송이
  그 뒤에 숨을 여지가 있고, h375에서 노출된 것은 hot이 cold 창을 줄였기 때문이다.
- 이 결론은 **PCIe 44.8 GiB/s : 이 CPU의 AMX** 라는 비율에서 나온 것이다. 비율이
  다른 조합에서는 뒤집힐 수 있다.

**5. 32 GiB 최선 = hot 37.5% + warm 0 + cold 62.5% sparse → 18.84 ms/tok (53.1 tok/s).**
코드 변경 없이 plan만으로 도달한다.

### 이 측정에서 틀렸던 가설 (반복 방지)

둘 다 "차분으로 얻은 수치를 다른 동작점에 그대로 옮긴" 오류였다:
- "cold는 이미 GPU 뒤에 숨었다" → h875(cold 완전 제거)가 h500보다 21% 빨라 기각.
- "GPU 경로가 병목" → 292 µs/층은 **행 100%가 GPU를 지나는 h875**에서 잰 값이라
  cold 지배 구성에 옮길 수 없었다. cold 지배 영역의 병목은 cold다 (sparsity가
  −11.7%를 내는 것이 그 증거 — GPU 병목이면 cold를 깎아도 벽시계가 안 변한다).
- "warm 전송은 cold 뒤에 숨어 있다" → warm 0.125↔0.25 대조로 기각(위 4항).


## 실측표 (Qwen3.6-35B-A3B, bs 스윕, worklist graph vs eager, RTX PRO 6000 96GB, 2026-08-25)

치수는 위 표와 동일 (NL=40 NE=256 I=512 H=2048). `--attention-backend triton
--context-length 4096 --max-running-requests {bs}`, ntok=64 차분법(warmup 8tok
제외). h375=warm-frac 0(cold-only sparse), h125=warm-frac 0.125. `_wl` = worklist
GEMV 커널, 접미사 없음(torch_bmm) = 현행 placeholder. graph일 때 `--cuda-graph-bs
1 2 4 8`. 측정 중 GPU0에 타 사용자 프로세스(`sglang::scheduler`, 12~29GiB
변동)가 run01~05 구간에 걸쳐 있었다 — bs=1/2/4/8 graph 4건과 wl-eager
bs=2 1건(→ 2.14× 배율의 분자 측정)은 GPU0 공유 상태, 나머지 eager 5건과
h125 교차검증 1건은 GPU0 단독 점유(quiet) 상태에서 측정.

| plan | bs | graph | ms/step | tok/s aggregate |
|---|---|---|---|---|
| h375_w0000_wl | 1 | O | 20.32 | 49.2 |
| h375_w0000_wl | 2 | O | 31.47 | 63.5 |
| h375_w0000_wl | 4 | O | 49.42 | 80.9 |
| h375_w0000_wl | 8 | O | 57.08 | 140.2 |
| h375_w0000_wl | 2 | eager | 67.26 | 29.7 |
| h375_w0000_wl | 4 | eager | 80.37 | 49.8 |
| h375_w0000_wl | 8 | eager | 95.20 | 84.0 |
| h375_w0000 (torch_bmm, 현행 기준선) | 2 | eager | 119.48 | 16.7 |
| h375_w0000 (torch_bmm, 현행 기준선) | 4 | eager | 178.77 | 22.4 |
| h375_w0000 (torch_bmm, 현행 기준선) | 8 | eager | 145.45 | 55.0 |
| h125_w0125_wl (warm/UVA 교차검증) | 4 | O | 67.78 | 59.0 |

**bs=1 회귀 없음.** 20.32ms/step은 기존 실측 대역(17.7–20.9ms, GPU 조용할 때)
안에 있다 — 단, 이 값 자체는 GPU0 공유 상태에서 잰 것이라 대역 상단 쪽으로
치우쳤을 수 있다.

**worklist graph vs 같은 plan eager**: bs=2에서 2.14배(분자 eager
67.26ms·분모 graph 31.47ms 모두 GPU0 공유 상태 측정), bs=4에서 1.63배, bs=8에서 1.67배 —
graph가 여전히 크게 이긴다(GatherKernelStager가 bs>1도 캡처 가능해졌으므로
당연하지만, S1 host 블록 소멸의 효과가 bs가 커져도 유지됨을 확인).

**worklist graph vs torch_bmm eager(현행 기준선) 배율**: bs=2 3.80배, bs=4
3.62배, bs=8 2.55배. bs=8에서 배율이 줄어드는 것은 분자(graph)가 커져서가
아니라 분모인 torch_bmm eager bs=8(145.45ms)이 bs=4(178.77ms)보다 오히려
빠른 비정상 패턴 때문 — placeholder가 전 expert를 broadcast 계산하는 고정비
성격(①번 한계, 위 "warm GEMM 커널 교체" 항목)과 GPU0 비공유 상태였다는 점을
감안해도 재현성 미검증. bs=8 torch_bmm 수치는 그대로 기록하되 액면가로
신뢰하지 말 것 — 재측정 필요.

**bs>1에서 cold는 dense로 돈다.** `executor.py`의 `masking = self._sparse and
m == 1`이 sparsity 마스킹을 M==1(decode 단일 토큰)로 게이트한다 — 즉 이 표의
bs≥2 행은 모두 cold가 sparse가 아니라 dense 경로다. bs 스윕에서 보이는 이득은
전부 "worklist GEMV vs torch_bmm 낭비 계산" 및 "graph vs eager"에서만 오고,
sparsity 자체의 bs>1 효과는 아직 미측정(별도 스코프).

**h125_w0125_wl bs=4 graph(67.78ms)가 h375_w0000_wl bs=4 graph(49.42ms)보다
느리다** — warm 12.5% 추가가 이 조건에서도 순손실이라는 위 §4 결론과 방향이
같다(warm PCIe가 cold보다 비싸다는 관찰이 bs>1·graph 경로에도 이어짐, 크기
비교는 참고용 — 서로 다른 mem-frac/실행 시각).

## 실측표 (Qwen3-30B-A3B, H100, node1 8스레드 단일소켓, batch1 greedy, uniform10-1node plan, 2026-08-20/21)

| 구성 | decode ms/tok | tok/s |
|---|---|---|
| gpu-only eager | 29.95 | 33.4 |
| gpu-only graph | 6.58 | 152 |
| kt cold-only eager (8thr) | 61.99 | 16.1 |
| kt cold-only graph (8thr) | 55.91 | 17.9 |
| prism eager (개선 전, 산란 dispatch 병목) | 106.24 | 9.4 |
| prism graph bs=1 (이번 작업 후) | **56.10** | **17.8** |

prism graph가 cold-only graph와 동급이라는 것은 warm 10% 오프로드가 주는
CPU 절감(~4.8ms)이 graph 경로의 host-node/오버헤드로 상쇄됨을 뜻한다. 단일
소켓 8스레드에서는 cold 대역(이론 바닥 ~43ms/tok)이 이미 성능 상한이므로,
다음 이득은 이 표에 없는 두 방향에서만 나온다: cold-down deferral(cold를
다음 층 아래로 숨겨 노출 대역을 줄임)과 스레드/소켓 확장(cold 자체의 실행
시간을 낮춤). 위 "다음 후보 순위" 절 참조.
