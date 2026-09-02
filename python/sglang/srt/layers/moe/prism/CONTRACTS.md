# Prism 경계 계약 (P0 착수 전 고정본)

Prism = K-split hot/warm/cold 티어링 MoE 오프로드. 패키지명이자 프로젝트명.

구현 전에 문장/API 수준으로 고정한 5개 boundary. 여기 적힌 것을 바꾸는 변경은
"리팩터링"이 아니라 "계약 변경"이며, 의존하는 모든 층을 함께 검토해야 한다.

- 대상 스택: sglang(kvcache-ai fork, 핀 `0f36b26`) + kt-kernel
- 용어: K = 각 projection의 contraction 축, N = 출력 축. M = 토큰 수.
- 용어 (2026-08-25 신설): **select** = expert 축 선택(어느 expert의 슬랩인가,
  런타임) / **gather** = K축 선택(어느 행인가, 인덱스 기반). 두 축이 같은
  단어를 쓰면 이 문서가 읽히지 않는다. 그리고 **k_index**(정적 티어 멤버십)와
  **pair_mask**(동적 skip)는 둘 다 K축 행 선택이지만 성격이 반대다 — 전자는
  로드 타임 고정, 후자는 매 decode 스텝 토큰마다.

**2026-08-25 개정 요약** (밴드 → 인덱스). 바뀐 것만:

| 절 | 변경 |
|---|---|
| ① | 티어를 **거처**로 재정의(HOT/WARM 계산 계약 동일) · 밴드 → **가변 per-expert K 인덱스** · gate/up 독립 · `ROW_GROUP` 폐기, 페어(%2)만 · **Plan=정책 / 자산=기하** 분리 |
| ② | `KRange` → `KIndex` · **시그니처 불변** (sparsity 배치는 2026-08-24 그대로 — 경계를 건너는 것은 라우터 가중 하나) |
| ③ | shard 3종 통일형 · 인덱스/오프셋 소유권 · **warm NUMA 배치 규칙 + startup 검증** |
| ④ | **`stage` primitive 소멸**(arena·stager·grouping 폐기) · `GpuTier` Protocol |
| ⑤ | exact 검출기를 **셔플 인덱스**로 확장 |

---

## ① Plan 최소 계약

**티어 의미 — 거처, 그 하나뿐 (2026-08-25 개정):**

- `HOT` — weight가 **VRAM**에 상주. GPU가 device 포인터로 읽는다.
- `WARM` — weight가 **pinned host**에 상주. GPU가 **UVA로 제자리 읽는다**
  (PCIe read). 스테이징 복사가 아니다 — device 버퍼를 경유하는 구현은
  선택지이지 계약이 아니다.
- `COLD` — weight가 **pageable host(NUMA-local)** 에 상주. CPU가 읽고 계산한다.

HOT과 WARM의 **계산 계약은 완전히 동일하다**: 같은 dense 스토어 방향, 같은
커널, 같은 출력 레이아웃.

> **2026-08-27 prefill 개정.** GPU 티어의 커널은 M에 따라 둘이다 — decode/소배치
> (M < `GROUPED_MIN_M`)는 pair-native worklist GEMV, 그 이상은 **expert-grouped
> GEMM**(`prism_grouped.cuh`; pair를 expert로 정렬해 W를 expert당 한 번 읽는다).
> 둘은 같은 스토어·같은 출력 레이아웃을 읽고 쓰며 정확표현 입력에서 비트일치한다 —
> 커널 선택은 호출 형태의 결정이고 계약이 아니다. 그리고 **COLD도 큰 M에서는 GPU가
> 읽을 수 있다**: kt가 pack한 AMX 레이아웃 slab을 `cudaHostRegister`로 매핑하고
> grouped GEMM의 COLD 레이아웃 로더가 재배치 없이 해석한다 (`cold_gpu.py`). 이때
> cold의 거처(pageable host, C++ 소유)는 그대로이고 **읽는 주체만** CPU에서 GPU로
> 바뀐다 — weight는 여전히 한 벌이다(계약 ③). 선택은 executor의 `cold_gpu_min_m`
> 하나이며, 한 호출에서 cold partial은 CPU 또는 GPU 정확히 한쪽만 낸다. 다른 것은 포인터가 가리키는 메모리의 종류 하나뿐이고,
그것이 `gemv_worklist` ↔ `gemv_worklist_pinned`(`w_on_device` 플래그) 차이의
전부다. 초판의 "step마다 선택된 expert의 밴드만 GPU로 전송되어"는 폐기한다 —
전송하는 경로(arena staging)는 bmm이 연속 배치 축을 요구해서 존재했고, 가변
K가 bmm 자체를 불가능하게 만들면서 함께 사라졌다 (④).

> **2026-08-25 실측 주의 (WARM):** 읽기 방식이 복사에서 제자리 로드로 바뀌어도
> PCIe를 지나는 바이트 수는 같으므로 이 결론은 그대로다. warm 트래픽은 cold 뒤에
> 숨지 않고 PCIe 대역폭 그대로 임계경로에 앉는다(실측 147 µs/층 vs 이론
> 131 µs/층). 현 하드웨어(PCIe 44.8 GiB/s + 2 NUMA AMX)에서는 같은 행을 cold에
> 두는 편이 빨라 `warm-frac 0`이 최선이었다(22.68 → 18.84 ms/tok). WARM은
> **계약상 유효한 티어로 유지**하되(하드웨어 비율이 다르면 뒤집힌다) 이 조합에서
> 기본값으로 쓰지 말 것. 근거와 반증 실험은 TODO.md 실측표 4항.

**좌표계 (2026-08-25 개정 — 밴드 → 인덱스):**

- 티어 멤버십은 **K축 인덱스 집합**이다. 반개구간 밴드 `[start, end)`는 폐기하고,
  그 자리에 "이 티어가 소유하는 K행 번호의 목록"이 들어간다. 밴드는 인덱스가
  연속인 퇴화형이며, 그 등가성이 전환기의 검증 기준이다.
- 인덱스는 **per-(layer, expert, proj)로 독립**이고 **길이도 expert마다 다를 수
  있다**. gate와 up도 독립이다 (초판의 "gate와 up은 밴드·shard를 공유한다"는
  2026-08-25 폐기 — cold 쪽 dual-pack 비용을 지불하고 자유도를 산다).
- 좌표는 여전히 각 proj의 contraction 축 행 번호다:
  `GATE`, `UP`의 K = hidden_size / `DOWN`의 K = intermediate_size.
- N shard(cold의 출력축 NUMA 분할)는 **밴드 그대로**다 — 반개구간, per-proj 독립.
  N은 티어로 쪼개지지 않는다 (한 행을 티어에 올리면 그 행의 전체 N이 따라온다).
- 인덱스 3티어의 합집합은 `[0, K)`의 **순열**이어야 한다. disjoint + 완전 커버가
  순열 하나로 표현되는 것이고, 위반은 조용한 이중계산/누락이므로 최우선 검증이다.

**정렬 규칙 (2026-08-25 개정):**

- `PAIR_GROUP = 2`. **모든 티어의 인덱스는 페어 단위로 움직인다** — 원본에서
  인접한 `(2j, 2j+1)`은 같은 티어에 있어야 하고 gather 후에도 인접해야 한다.
  근거는 k2wl2 점수가 페어의 실제 에너지(교차항 포함)라서 반쪽만 가진 티어는
  점수를 재구성할 수 없기 때문이고, 동시에 cold 커널의 skip 단위가 VNNI 페어
  (인접 k 2행)이기 때문이다. 페어 안에서의 순서는 자유다 (`wn`이 같은 순서로
  동행하면 점수식이 대칭).
- `COL_GROUP = 32`. N shard 경계 (AMX N-타일). N 자체가 32로 나누어떨어져야 한다.
- **`ROW_GROUP`은 폐기한다.** 초판의 64는 요구가 아니라 "K_STEP의 배수라 안전"
  이라는 보수적 선택이었고, 실제 요구는 cold 커널의 packed 저장이 자기 타일
  크기(현 AMX 구현 32행)로 올림된다는 것뿐이다. **그 올림은 인스턴스 내부
  사정이며 plan에 보이지 않는다** (②-3의 "기하는 로드 시점에 구워진다"의 연장).
  커널은 (a) 올림한 패딩 행을 dense 경로에서 0으로 보고, (b) 마스크의 tail
  비트를 `k_real/2` 이상에서 0으로 유지할 책임을 진다.
  이 결정의 값어치는 planner 해상도다: down은 K=512라 %32면 per-expert 크기
  선택지가 16개뿐인데, 가변 per-expert 예산 배분이 이 스키마의 존재 이유다.

**Plan과 자산의 책임 분리 (2026-08-25 신설):**

가변 per-expert 인덱스를 JSON에 적는 것은 불가능하고(40층×256expert×3proj면
수천만 정수) 적을 이유도 없다. 경계를 이렇게 긋는다:

- **Plan = 정책** — 커널 선택, cold N shard, sparsity 예산 `(p, λ)`와
  `SparsitySpec`, 그리고 인덱스 자산의 `path + sha256`.
- **자산 = 기하** — 어느 expert의 어느 K행이 어느 티어인가, 그리고 그 오프셋
  테이블. 자산 생성기는 이 코드베이스 밖이다.

`plan.py`는 순수 stdlib을 유지하며 자산을 열지 않는다. 커버리지·정렬·순열 검증은
자산을 여는 쪽(`index.py`)의 몫이고, `validate_static`은 `index_probe` 주입 시
치수 대조만 한다 (calib_probe와 같은 형태).

**입력기반 sparsity (schema_version 2, 2026-08-24 추가):**

- 마스크는 **각 proj의 K(contraction) 축**에 걸린다. 점수는 그 proj의
  **입력**에서만 나오므로(gate/up은 layer 입력 x, down은 rejoin#1의 act)
  각 티어가 자기 인덱스 안에서 로컬하게 적용할 수 있다 — 티어 간 rejoin을
  기다릴 필요가 없다. (출력 축 마스킹이 아니다 — N shard와 무관.)
- score 변종은 `k2wl2` 하나로 고정 (2026-08-24 사용자 결정). 인접 입력채널
  **페어**의 실제 에너지 `sqrt(a0·x0² + a1·x1² + 2c·x0·x1)`이므로 열 노름
  `wn`과 인접열 내적 `pair_dot` 둘 다를 요구하고, 마스크는 `k/2` 비트다.
  인덱스가 페어 단위라는 규칙(위 정렬 규칙)이 티어 경계가 페어를 쪼개는 것을
  막는다 (쪼개지면 어느 티어도 점수를 재구성할 수 없다).
- **마스킹은 cold에만 적용된다** (warm/hot은 dense — warm GEMM이 latency
  바운드라 마스킹이 순손실이었다).
- **sparsity 전체가 cold 커널의 일이다** (2026-08-24 배치 유지, 2026-08-25
  재확인). 예산·격자·threshold 곡선이 kt config에 구워지고, step마다 경계를
  건너는 것은 **라우터 가중 하나**뿐이다. 커널이 `s → thr → 점수 → 마스크 →
  masked GEMV`를 전부 수행한다:
  `s = clip(p − λ(g_e − ḡ), 0, pmax)` → `thr = table[layer, expert, round(s/grid)]`.
  Plan이 갖는 것은 예산 `(sparsity_p, sparsity_lambda)`(per-(layer, expert, proj)
  스칼라)와 model-global `SparsitySpec`(score, calib 참조, pmax/grid/ng/renorm_it)
  뿐이며 **threshold 값은 Plan에 없다.**
- threshold/노름 테이블은 **full-K 분포로 캘리브된 것을 그대로 쓴다**
  (2026-08-24 사용자 결정 — 티어별 재캘리브 안 함). 마스크는 full-K 기준과
  동일하지만 티어별 nnz 비율은 균일하지 않으므로, **cold 인덱스의 nnz는 실측
  대상**이다 (부하 예측 불가가 이 결정의 대가).
- 테이블 자산은 Plan 밖의 파일이고 Plan은 경로 + sha256만 갖는다(≈130MB
  텐서를 JSON에 넣을 수 없다). `wn`/`pair_dot`은 K축이므로 **weight와 같은
  인덱스로 gather되어** cold 커널에 동행한다. 자산의 shape 검증은
  `validate_static`에 `calib_probe`를 주입해 수행한다 — plan.py는 순수 stdlib을
  유지하며 자산을 열지 않고, "논리 테이블명 → shape" 대조만 소유한다.

**커널 선택 위치 (co-variance 선언):**

- `gpu_warm`, `cpu_cold` 모두 **model-global** (`Plan.kernels`).
  - `gpu_warm`: 이름 하나가 hot/warm 두 티어의 구현을 함께 고른다 (계산 계약이
    같으므로 — 위 티어 의미 참조).
  - `cpu_cold`: CPU 커널의 실체는 CRTP 클래스(gate/up/down GEMM 한 세트)이고,
    레이어당 C++ 인스턴스 1개를 유지하기 위해 전역으로 결정.
- 커널 이름은 startup에 구현체로 resolve되며, 이후 런타임에 문자열/enum
  분기는 존재하지 않는다. **cold의 저장 형식(pack)은 커널 키가 함의한다**.

**검증 불변식 (전부 로드 시 hard error):**

`validate_static`(Plan만 보고 판단 가능한 것):

1. 커널 이름이 registry에 존재 (registry 주입 시).
2. dims가 실제 모델 config와 일치 — 다른 모델/ckpt에 Plan을 적용하는 것이
   이 시스템 최대의 silent failure이므로 startup 즉사.
3. 모든 (layer, expert)에 대해 plan이 존재 (완전 커버, 암묵 fallback 금지).
4. COLD가 존재하면 cold_shards가 `[0, N)`을 disjoint 커버, 없으면 빈 튜플.
5. sparsity는 all-or-nothing: model-global 블록이 있으면 **모든** proj가
   `(p, λ)`를 갖고, 없으면 **어느** proj도 갖지 않는다. `0 ≤ p ≤ pmax`,
   `λ ≥ 0`, `(ng−1)·grid ≥ pmax` — 마지막 것은 격자가 pmax에 못 닿으면 idx가
   clamp되어 threshold가 조용히 포화하기 때문이다.
6. probe 주입 시 자산(calib/index) 테이블 shape 대조.

자산 로더(`index.py`)가 per-(layer, expert, proj)로 수행하는 것:

7. **순열** — `concat(hot, warm, cold)`이 `[0, K)`의 순열. 위반은 조용한
   이중계산/누락이므로 최우선.
8. **페어 무결성** — 인덱스가 페어 단위이고 gather 후에도 페어가 인접.
9. 오프셋 테이블 단조 + 합이 스토어 행 수와 일치.

**HOT/WARM 실행 계약 (2026-08-25 개정):** 두 티어의 스토어는 같은 방향
(`[Σₑ k[e], N]` K-major flat + `row_off[E+1]`)이고 같은 커널을 탄다. 인덱스는
스토어와 같은 오프셋 테이블을 공유한다. 이것이 `all_hot`과 `all_warm`의 출력이
**완전히 동일**해야 하는 이유이고, test_executor의 plan 불변성 테스트가 그
등호를 지킨다.

**Plan 파일에는 `schema_version`, `model_id`, dims가 반드시 포함된다.**
Plan 생성기는 이 코드베이스 밖이며, 여기는 스키마·파서·검증기만 소유한다.

## ② C++ partial 진입점 계약

cold 인스턴스(레이어당 1개)의 진입점 2개. 둘 다 CPUInfer TaskQueue로
submit/sync host-node 쌍을 통해 호출된다 (kt forward와 동일 기계).

```cpp
void forward_gateup_partial(int qlen, int k,
    const int64_t* expert_ids,   // [qlen × k], 두 phase가 같은 버퍼 재사용
    const void*    x,            // bf16 [qlen × hidden_FULL]  ← full-width
    ggml_bf16_t*   out,          // bf16 [qlen × k × 2·inter], slot j ↔ expert_ids[m, j]
                                 //      열 [0, inter) = gate, [inter, 2·inter) = up
    const float*   weights);     // fp32 [qlen × k] 라우터 가중. nullptr = dense
                                 //   (커널이 s → thr → 마스크를 전부 수행)

void forward_down_partial(int qlen, int k,
    const int64_t* expert_ids,
    const void*    act,          // bf16 [qlen × k × inter_FULL] ← full-width
    ggml_bf16_t*   out,          // bf16 [qlen × k × hidden]
    const float*   weights);     // fp32 [qlen × k] 라우터 가중
```

> 개정 (2026-08-20, ⑤ 개정과 연동): out은 초판의 fp32에서 **bf16**으로
> 변경 — kt의 `to_mat`(fp32 누산 → bf16 재료화) 동작을 무변경 재사용한다.
> GEMM *누산*은 여전히 fp32(AMX 타일)이며, 합산 정밀도는 GPU rejoin이
> fp32 누산으로 보장한다 (⑤).

> 정오표 (2026-08-20): `act`의 초판 표기 `[qlen × inter_FULL]`은 오기.
> act는 expert별 값이므로 slot 차원 k가 있어야 하며, 계약 ④의
> `rejoin_gateup → [M, k, inter]`와 이렇게 정합된다. "full-width"의 의미는
> inter 차원이 슬라이스되지 않았다는 뜻으로 불변.

1. **대수 경계**: 두 partial 모두 "순수 GEMM 부분합"이다. gateup partial은
   activation **이전**(pre-act), down partial은 router 가중 **이전**.
   C++는 act도, router weight 곱도, expert 합산도 하지 않는다 — 전부 GPU
   rejoin의 일.
2. **누산 의미론**: **overwrite (`=`)**. 티어 간 합산은 GPU rejoin에서 정확히
   1회. NUMA node별 N-shard는 같은 out 버퍼의 서로소 열 구간에 각자 쓴다
   (누산 아님). 인스턴스가 소유한 cold 행들의 내부 합산은 인스턴스 소관이며
   밖에서 보이지 않는다.
3. **K축 기하는 호출에 없다**: 기하는 weight 로드 시점에 인스턴스에
   구워진다 (Plan → pack). 입력이 full-width이므로 cold 행들이 비연속이든
   expert마다 개수가 다르든 호출 시그니처는 불변이고, "pack된 weight와 호출
   인자의 정합"이라는 불변식 자체가 존재하지 않는다. 2026-08-25의 밴드 →
   인덱스 전환에서 이 항이 실제로 값을 했다 — **진입점 diff가 0**인 이유다.

   기하 운반자는 kt `GeneralMOEConfig`의 중첩 구조 `config.partial`이다
   (K1에서 확정 — 이 구조가 두 저장소 간 계약이며, 변경 시 양쪽 동시 검토):

   ```cpp
   struct KIndex {                        // 한 proj의 K축 소유 행 (2026-08-25)
     std::vector<int32_t>  row_off;       // [E+1] — expert별 시작. weight와 공유
     std::vector<uint16_t> idx;           // [row_off[E]] — 원본 K행 번호
     bool contiguous;                     // 연속 퇴화형이면 gather 스킵 (밴드 등가)
   };
   struct PartialGeometry {
     bool enabled;    // false(기본) = 기존 kt와 비트 동일 동작 (침습성 상한)
     KIndex gate;     // gate/up은 **독립** (2026-08-25 — 초판의 공유 전제 폐기)
     KIndex up;
     KIndex down;     // K(intermediate) 축, global 좌표
     int n_total;     // gate/up 출력축(inter)의 full 크기 (TP shard 후에도 원본)
   };
   ```

   접근자는 per-expert다: `gate_k(e)`/`up_k(e)`/`down_k(e)` =
   `row_off[e+1] − row_off[e]`, enabled == false면 full 치수. `n_total()`/
   `down_n()`은 그대로. **인덱스 길이는 expert마다 다를 수 있다** — 스토어가
   `[E, k, N]` 균일 적층이 아니라 flat + offset이라 균일성 요구가 없다.

   **타일 올림은 밖에서 보이지 않는다**: packed 저장이 커널 타일 크기(현 AMX
   32행)로 올림되더라도 그것은 인스턴스 내부 사정이고, plan/자산이 지켜야 하는
   정렬은 페어(%2)뿐이다 (①). 커널은 패딩 행을 dense 경로에서 0으로 보고
   마스크 tail 비트를 `k_real/2` 이상에서 0으로 유지할 책임을 진다.

   gate ≠ up의 대가는 **dual-pack**이다: 지금 하나인 activation BufferA
   (`gate_up_ba_`)가 gate/up 둘로 갈리고 A 풀이 2배가 된다. 2026-08-20에
   "폐기"로 적혔던 항목이며 2026-08-25에 되살아났다.

   K2 확장 — NUMA N-shard 테이블 (같은 구조체 안):

   ```cpp
   // top-level (Plan이 주입; 비어 있으면 TP ctor가 균등 분할):
   std::vector<int> node_gateup_n_offset, node_gateup_n_rows;  // inter 축
   std::vector<int> node_down_n_offset,  node_down_n_rows;     // hidden 축
   // node-scope (TP ctor가 각 노드 config에 기록):
   KRange gateup_n;  KRange down_n;
   ```

   전 proj가 각자의 N축으로 노드 분할되므로 (down도 hidden 축 —
   GPU rejoin 왕복이 act-locality를 끊어 노드 합산이 불필요),
   partial의 out은 gateup/down 모두 **노드별 서로소 열 direct write**다.
   불균등 테이블 = warm 소켓의 cold 몫을 줄이는 비율 노브.

   down partial 진입점 (gateup과 동형):
   ```cpp
   void forward_down_partial(int qlen, int k, const int64_t* expert_ids,
       const void* act,        // bf16 [qlen × k × n_total] — rejoin#1 결과 D2H
       ggml_bf16_t* out,       // bf16 [qlen × k × hidden_FULL], 자기 shard 열만
       const float* weights);  // fp32 [qlen × k] 라우터 가중, nullptr = dense
   ```
4. **완료 계약**: sync host node가 반환한 시점에 out은 완전히 쓰여 있다.
   sync 이전의 out 내용은 undefined. 패딩 토큰 slot의 출력은 쓰레기이며
   마스킹은 GPU 소관.

---

## ③ Weight ownership

```python
@dataclass
class TierShard:                # hot/warm/cold 공용 모양 (2026-08-25)
    w_flat:  Tensor             # bf16 [Σₑ k[e], N] — hot=device / warm=pinned / cold=ckpt방향
    row_off: Tensor             # int32 [E+1] — weight와 k_index가 **공유**
    k_index: Tensor             # uint16 [Σₑ k[e]] — 원본 K행 번호
    calib:   Optional[...]      # wn/pair_dot, 같은 인덱스로 gather됨 (cold 소비)

@dataclass
class PreparedWeights:          # Stage 2의 유일한 산출물이자 lifetime owner
    hot:  HotStore              # Python 소유 device shard (없으면 멤버가 전부 None)
    warm: WarmStore             # Python 소유 pinned shard
    cold: ColdHandle            # C++ MOE 객체 핸들 — packed NUMA 메모리는 C++ 소유
```

(`ColdHandle`은 자리 표시 — K3 시점에 실물이 생긴다. 현 구현(weights.py)에서
cold는 `PendingColdTensors`(backend 접속 전 임시 소유자)다.)

- Stage 2 종료 후 full-K 텐서는 어디에도 존재하지 않는다.
- weight 수명 = PreparedWeights의 수명.
- warm의 pinned 메모리는 **GPU만 읽는다** (UVA load — DMA 엔진이 아니라 SM이
  PCIe 너머로 로드한다). C++는 warm의 존재를 모른다.
- **warm store는 GPU의 PCIe root complex와 같은 NUMA 노드에 상주해야 한다.**
  원격 소켓 배치는 UVA 읽기에 소켓 간 링크 홉을 추가해 warm 티어의 존립
  근거(PCIe 대역폭)를 무너뜨린다. 노드는 hot의 device와 같은 급의 **로더 입력**
  이고, 배치 결과는 startup에 검증된다 — 원격 배치는 수치적으로 정확하고
  느리기만 해서 다른 어떤 검사도 잡지 못한다. (`pin_memory=True`는
  `cudaHostAlloc`이라 페이지가 호출 안에서 고정되므로 정책은 할당 **전에** 걸어야
  하고 사후 마이그레이션은 불가능하다. 게다가 torch의 pinned 캐싱 할당자가 다른
  정책 아래 할당된 블록을 돌려줄 수 있어 **검증이 필수**다 — 2026-08-25 실측.)
- 인덱스와 오프셋 테이블의 소유: hot/warm은 device 상주 텐서로 shard가 들고
  (주소 고정 — 캡처가 baked), cold는 kt 인스턴스가 로드 시 **복사본**으로 갖는다
  (sparsity 점수 테이블과 달리 tiny하므로 포인터 수명 문제를 만들지 않는다).
- hot의 device 메모리는 로더가 배치한다 — 배치 device는 `prepare_layer_weights`
  의 **입력**이지 로더가 정하는 값이 아니다. hot 인덱스가 비어 있지 않은데 device가 없으면
  즉사한다(조용히 CPU에 두면 티어 의미가 사라진다).
- cold 핸들 해제 전에는 in-flight CPU task의 drain이 선행돼야 한다.

---

## ④ 실행 primitive 계약 (2026-08-25 개정)

초판의 6개 함수 목록에서 **`stage`가 사라졌다.** pinned → arena 이동은 warm이
"전송된다"는 전제의 산물이었고, 그 전제는 두 번 무너졌다: warm은 UVA 제자리
읽기가 됐고(①), 가변 per-expert K가 bmm의 연속 배치 축 요구를 불가능하게 만들어
arena를 쓸 소비자 자체가 없어졌다. `DeviceArena`, `stagers.py`, `grouping.py`,
그리고 그것들을 지키던 arena WAR 이벤트 체인이 함께 폐기된다.

```python
class GpuTier(Protocol):
    """한 (layer, proj)의 GPU 티어 하나. hot/warm의 차이를 구현이 흡수한다."""
    def run(self, x, topk_ids, out3d, out_col_off, *, x_row_is_pair: bool) -> None: ...

# 구현체 — 계약이 아니라 선택지:
#   ResidentTier(shard)      HOT  : store가 device
#   PinnedDirectTier(shard)  WARM : store가 pinned, UVA 제자리 읽기

def submit_cold(phase, x_or_act_gpu, staging, cold, stream) -> None
def sync_cold(phase, staging, stream) -> Tensor                 # bf16 [M, k, N] (GPU)
def rejoin_gateup(parts) -> Tensor          # fp32 누산 → act → bf16 [M, k, inter]
def rejoin_down(parts, router_w) -> Tensor  # fp32 누산 → 가중 expert합 → bf16 [M, hidden]
```

| primitive | sync point | buffer 소유 |
|---|---|---|
| `GpuTier.run` | 없음 — current stream에 launch만 | 출력은 호출자 소유. 티어는 스토어·인덱스만 읽는다 (로드 타임 소유) |
| `submit_cold` | 없음 — **enqueue-only, 즉시 반환** | staging은 ExecutionResources 소유, `.copy_()` in-place만 |
| `sync_cold` | sync host node (CPU 완료 블록) | 〃 |
| `rejoin_*` | 없음 (순수 GPU 연산) | 출력은 호출자 소유 |

**invariant:**

- **primitive는 영구 메모리를 절대 할당하지 않는다** — staging은
  ExecutionResources에서, 스토어·인덱스·오프셋은 로드 타임 소유자에게서 빌린다.
  이유는 주소다: CUDA graph 캡처는 커널 인자에 **포인터 값을 구워 넣고** replay가
  그 주소를 그대로 다시 쓰며, cold는 `data_ptr()`를 넘긴 뒤 host node가 **나중에**
  역참조한다. 호출마다 새로 할당하면 둘 다 조용한 오답이 된다. (일시 출력의
  신규 할당은 대상이 아니다. storage가 아닌 상태 — 카운터·이벤트 핸들 — 도 아니다.)
- `GpuTier.run`은 **인덱스를 커널 인자로 소화한다** — activation을 미리 dense로
  모으지 않는다. per-expert 인덱스에서 pre-gather는 `[M·k, k_t]` 버퍼와 launch를
  proj·티어마다 추가하는데, 인덱스 재읽기는 L2에 상주해 W 트래픽의 3% 수준이라
  거래가 성립하지 않는다 (2026-08-25 산정).
- submit 콜백은 CPU task를 enqueue만 하고 즉시 반환한다. CPU 완료 대기는
  어떤 경우에도 submit에서 하지 않는다 (CPU/GPU overlap의 존립 조건).
- **graph 경로에서 교체되는 것은 cold의 `expert_ids` 조달 하나뿐이다.**
  초판이 "구현이 교체되는 primitive는 `stage` 하나"라고 적은 자리가 비었다 —
  worklist 티어는 host 결정이 없어(pair (m,j)가 좌표이고 expert는 블록이
  `topk[pair]`에서 읽는다) eager와 graph가 **같은 호출**이다. 남은 차이는
  eager가 `ids_cpu` D2H로, graph가 device→pinned async D2H로 kt의 expert_ids를
  채운다는 것뿐이다.
- **전략 경계**: 티어의 거처·읽기 방식은 `GpuTier` 구현체가, cold 인스턴스 수명과
  Plan↔kt 번역은 `cold_backend`가
  소유한다 — 두 축은 각자의 파일 밖으로 새지 않는다 (sparsity는 cold 커널 안에
  있고 prism은 라우터 가중을 staging에 내려보낼 뿐이다). env/외부 시스템 읽기는
  전부 조립 지점(method.py)이 하고 명시 인자로 주입한다.

> **살아남은 2026-08-20 Task 8 항목** (나머지 두 항은 위 개정으로 폐기 — "교체되는
> primitive는 stage 하나"와 "최악치 고정 그룹 수"는 그룹 개념과 함께 사라졌다):
>
> 1. **staging fill의 blocking D2H는 cold 호출이 host 경로일 때만 필요하다.**
>    cold submit/sync가 current stream 경유(kt `submit_with_cuda_stream` =
>    `cudaLaunchHostFunc` host node)면 kt task가 같은 stream에서 fill 뒤에
>    실행되므로 host-측 완료 보장이 불필요하다 — fill은 `non_blocking=True`로
>    enqueue만 한다.
> 2. **graph 경로의 qlen은 bs별 전용 상수 버퍼를 쓴다.** 캡처가 baked하는 포인터를
>    eager의 버퍼와 공유하면 나중의 eager prefill 쓰기에 노출돼 replay마다 cold가
>    L토큰 분량을 계산하는 stale-share 버그가 된다 (Finding A, 2026-08-21 실측:
>    30B decode 328→56 ms/tok). bs가 다른 replay끼리도 절대 공유하지 않는다.
> 3. **host-측 즉시쓰기는 stream 순서의 보호를 받지 않는다.** `_expert_ids`의
>    eager 경로(host→host copy)와 qlen pin 쓰기는 GPU stream에 올라가지 않으므로,
>    뒤따르는 cold submit이 그 값을 읽기 전에 쓰기가 끝나 있다는 보장은 stream이
>    주는 게 아니다. eager에서는 앞선 blocking D2H가 사실상의 throttle 역할을 한다 —
>    그 D2H를 없애거나 async로 바꾸는 변경은 이 항목을 반드시 재검토해야 한다.

## ⑤ 수치 계약 (2026-08-20 개정)

원칙: **누산은 항상 fp32, 재료화(wire)는 bf16** — kt의 기존 정밀도
프로파일(GEMM 누산 fp32, 스테이지 경계마다 bf16 재료화, moe_base.hpp:44-46
/ :560 / :708)과 동일한 클래스를 유지한다.

1. **partial dtype = bf16, 예외 없음.** cold partial(C++ out, H2D 경로)은
   kt `to_mat` 무변경 재사용으로 bf16. warm partial(GPU GEMM out)도 bf16 —
   bf16 bmm이 내부 누산은 fp32(cuBLAS compute type 32F)로 하고 출력만
   1회 라운딩하므로 cold와 같은 정밀도 클래스이며, fp32 출력을 고집하면
   tensor core를 못 타 CUDA-core GEMM(수십 배 느림) + 입력 업캐스트
   복사가 붙는다 (2026-08-20 재개정: 초판 개정의 "warm은 fp32 유지(무비용)"
   판단은 계산 비용을 놓친 오류였음).
2. **모든 합산은 fp32 누산.** rejoin의 티어 합산(bf16 partial들을 upcast),
   act 계산, router 가중 expert 합까지 fp32로 수행 후 마지막에 bf16 캐스트.
   warm GEMM의 내부 누산도 fp32 — bf16 split-K 환원 허용 플래그
   (`allow_bf16_reduced_precision_reduction`)를 커널 안에서 끄고 복원한다.
3. 정밀도 등급: kt 기존이 완성값을 bf16으로 1회 라운딩하는 자리에서,
   우리는 tier별 partial이 각각 라운딩(≤ 티어 수만큼)된 뒤 fp32 합산 —
   같은 클래스, 라운딩 이벤트 수만 증가.
4. **레퍼런스**: fp64/fp32 단일-GEMM 레퍼런스 대비 tolerance 검증
   (tolerance는 bf16 재료화 횟수를 반영).
5. **plan 불변성 테스트 이원화**: bf16 라운딩이 티어 경계에 의존하므로
   일반 입력에서는 tolerance 비교. **exact 검출은 정확히 표현 가능한
   입력**(작은 정수/2의 거듭제곱 — bf16 라운딩 무손실)으로 구성한 테스트가
   담당한다: 그 입력에서 "전부 cold = 전부 warm = 전부 hot = 혼합"은 비트일치여야
   하며, 이것이 이중계산/누락의 exact 검출기다.

   **2026-08-25 개정 — 셔플 인덱스에서 성립해야 한다.** 연속 밴드에서는 좌표
   뒤섞임이 검출되지 않는다(어느 순서로 더해도 같은 행들이므로). 인덱스가
   순열이 되면서 이 테스트가 유일한 좌표 검출기가 되었고, 따라서 픽스처는
   **무작위 순열 인덱스**로 구성해야 한다. 순열 검증(①-7)과 이 비트일치, 둘이
   인덱스 시대의 방어선 전부다.
