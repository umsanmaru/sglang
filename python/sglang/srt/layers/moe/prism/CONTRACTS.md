# Prism 경계 계약 (P0 착수 전 고정본)

Prism = K-split hot/warm/cold 티어링 MoE 오프로드. 패키지명이자 프로젝트명.

구현 전에 문장/API 수준으로 고정한 5개 boundary. 여기 적힌 것을 바꾸는 변경은
"리팩터링"이 아니라 "계약 변경"이며, 의존하는 모든 층을 함께 검토해야 한다.

- 대상 스택: sglang(kvcache-ai fork, 핀 `0f36b26`) + kt-kernel
- 용어: K = 각 projection의 contraction 축, N = 출력 축. M = 토큰 수.

---

## ① Plan 최소 계약

**티어 의미:**

- `HOT` — 해당 row들이 VRAM에 상주하고 GPU가 계산한다.
- `WARM` — pinned host에 상주하고, step마다 선택된 expert의 밴드만 GPU로
  전송되어 GPU가 계산한다.
- `COLD` — pageable host(NUMA-local)에 상주하고 CPU가 계산한다.

**좌표계:**

- K band는 반개구간 `[start, end)`, 해당 proj의 contraction 축 row 인덱스.
  - `GATE`, `UP`의 K = hidden_size / `DOWN`의 K = intermediate_size
- N shard도 반개구간, 해당 proj의 출력 축.
  - `GATE`, `UP`의 N = intermediate_size / `DOWN`의 N = hidden_size
- **gate와 up은 K 밴드·N shard를 공유한다** (2026-08-20 확정 — 이
  자유도는 부하 균형 목적상 중복이라 풀 계획 없음). 실행은 cold 로드
  시 `gate == up` (bands·cold_shards)을 검증하고 위반 시 즉사한다.
  스키마가 proj별 독립 표현을 허용하는 것은 공짜 일반성으로 남겨둔
  것일 뿐 실행 계약이 아니다. down은 밴드·shard 모두 독립.
- 밴드를 같게 맞추는 padding(작은 쪽을 union으로 확장)은 **planner의 결정**
  이다. 런타임은 받은 밴드를 집행할 뿐 스스로 padding하지 않는다.

**정렬 규칙:**

- 모든 band 경계는 `ROW_GROUP = 64`의 배수. K 자체가 64로 나누어떨어져야 한다.
- 모든 shard 경계는 `COL_GROUP = 32`의 배수. N 자체가 32로 나누어떨어져야 한다.
  (AMX pack 타일 단위 — 상수 값은 pack 코드 확인 후 조정될 수 있으나
  "정렬 배수여야 한다"는 계약은 불변)

**커널 선택 위치 (co-variance 선언):**

- `gpu_warm`, `cpu_cold` 모두 **model-global** (`Plan.kernels`).
  - `gpu_warm`: warm GEMM은 선택 expert들을 한 launch에 배칭하므로 expert별
    다양성은 launch 구조와 양립 불가 → 전역.
  - `cpu_cold`: CPU 커널의 실체는 CRTP 클래스(gate/up/down GEMM 한 세트)이고,
    레이어당 C++ 인스턴스 1개를 유지하기 위해 전역으로 결정 (2026-08-20).
    expert별 커널이 필요해지는 날은 `schema_version` 범프로 처리한다.
- 커널 이름은 startup에 구현체로 resolve되며, 이후 런타임에 문자열/enum
  분기는 존재하지 않는다. **cold의 저장 형식(pack)은 커널 키가 함의한다**
  (CRTP 클래스가 자기 pack을 소유) — 별도 codec 필드 없음.

**validate_static 불변식 (전부 로드 시 hard error):**

1. bands가 disjoint이며 `[0, K)`를 완전 커버 (hot ∪ warm ∪ cold = K,
   pairwise ∅) — 위반 시 조용한 이중계산/누락이 되므로 최우선 검증.
2. 모든 경계가 정렬 배수.
3. COLD 밴드가 존재하면 cold_shards가 `[0, N)`을 disjoint 커버, 없으면 빈 튜플.
4. 커널 이름이 registry에 존재 (registry 주입 시).
5. dims가 실제 모델 config와 일치 — 다른 모델/ckpt에 Plan을 적용하는 것이
   이 시스템 최대의 silent failure이므로 startup 즉사.
6. 모든 (layer, expert)에 대해 plan이 존재 (완전 커버, 암묵 fallback 금지).

스키마는 티어당 다중 밴드(interleave)를 허용한다. P0 plan은 밴드 ≤ 3개
(hot=∅, warm=첫 10%, cold=나머지)인 퇴화형일 뿐이다.

**Plan 파일에는 `schema_version`, `model_id`, dims가 반드시 포함된다.**
Plan 생성기는 이 코드베이스 밖이며, 여기는 스키마·파서·검증기만 소유한다.

---

## ② C++ partial 진입점 계약

cold 인스턴스(레이어당 1개)의 진입점 2개. 둘 다 CPUInfer TaskQueue로
submit/sync host-node 쌍을 통해 호출된다 (kt forward와 동일 기계).

```cpp
void forward_gateup_partial(int qlen, int k,
    const int64_t* expert_ids,   // [qlen × k], 두 phase가 같은 버퍼 재사용
    const void*    x,            // bf16 [qlen × hidden_FULL]  ← full-width
    ggml_bf16_t*   out);         // bf16 [qlen × k × 2·inter], slot j ↔ expert_ids[m, j]
                                 //      열 [0, inter) = gate, [inter, 2·inter) = up

void forward_down_partial(int qlen, int k,
    const int64_t* expert_ids,
    const void*    act,          // bf16 [qlen × k × inter_FULL] ← full-width
    ggml_bf16_t*   out);         // bf16 [qlen × k × hidden]
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
   (누산 아님). 인스턴스가 cold 밴드를 여러 개 가지면 그 내부 합산은
   인스턴스 소관이며 밖에서 보이지 않는다.
3. **밴드 기하는 호출에 없다**: 기하는 weight 로드 시점에 인스턴스에
   구워진다 (Plan → pack). 입력이 full-width이므로 cold 밴드가 비연속·다중이
   되어도 호출 시그니처는 불변이고, "pack된 weight와 호출 인자의 정합"이라는
   불변식 자체가 존재하지 않는다.

   기하 운반자는 kt `GeneralMOEConfig`의 중첩 구조 `config.partial`이다
   (K1에서 확정 — 이 구조가 두 저장소 간 계약이며, 변경 시 양쪽 동시 검토):

   ```cpp
   struct KRange { int offset; int rows; };   // [offset, offset+rows)
   struct PartialGeometry {
     bool enabled;    // false(기본) = 기존 kt와 비트 동일 동작 (침습성 상한)
     KRange gateup;   // gate/up 공유 밴드 — K(hidden) 축
     KRange down;     // down 밴드 — K(intermediate) 축, global 좌표
     int n_total;     // gate/up 출력축(inter)의 full 크기 (TP shard 후에도 원본)
   };
   ```

   enabled == true이면 모든 range와 n_total은 **명시값**이다 ("0 = full"
   센티널 없음 — down이 full이면 `{0, intermediate_size}`로 적는다).
   접근자 `gateup_k()`/`down_k()`/`n_total()`/`down_n()`은
   enabled == false일 때 full 치수를 돌려준다.

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
       ggml_bf16_t* out);      // bf16 [qlen × k × hidden_FULL], 자기 shard 열만
   ```
4. **완료 계약**: sync host node가 반환한 시점에 out은 완전히 쓰여 있다.
   sync 이전의 out 내용은 undefined. 패딩 토큰 slot의 출력은 쓰레기이며
   마스킹은 GPU 소관.

---

## ③ Weight ownership

```python
@dataclass
class PreparedWeights:          # Stage 2의 유일한 산출물이자 lifetime owner
    hot:  HotWeights | None     # Python 소유 device 텐서 (P0: None)
    warm: WarmStore             # Python 소유 pinned 텐서 + (expert, proj) → offset
    cold: ColdHandle            # C++ MOE 객체 핸들 — packed NUMA 메모리는 C++ 소유
```

(`HotWeights`/`ColdHandle`은 미구현 자리 표시 — hot tier/K3 시점에 실물이
생긴다. 현 구현(weights.py)에서 hot은 `None`, cold는 `PendingColdTensors`
(backend 접속 전 임시 소유자)다.)

- Stage 2 종료 후 full-K 텐서는 어디에도 존재하지 않는다.
- weight 수명 = PreparedWeights의 수명.
- warm의 pinned 메모리는 GPU DMA만 읽는다 (C++는 warm의 존재를 모른다).
- cold 핸들 해제 전에는 in-flight CPU task의 drain이 선행돼야 한다.

---

## ④ Eager primitive 계약 (P0)

전부 함수. graph 설계는 지금 하지 않되, primitive 공유가 나중의 capture
경로와 호환되도록 아래 규칙만 지킨다.

```python
def stage(proj, topk_ids, warm_store, arena, warm_stream) -> (SlotMap, cuda.Event)
# slot: "이번 그룹의 i번째 distinct expert"라는 논리 인덱스. SlotMap은
# proj 공유이되 물리 버퍼는 proj별 — arena slot 수는 proj당 n_slots
# (기본 top_k)이고 gateup phase의 물리 상주는 gate+up = 2×n_slots.
# distinct > n_slots(M>1)이면 executor가 n_slots 단위 그룹 직렬 루프.
def run_warm(proj, x_or_act, slot_map, arena, evt_staged) -> Tensor      # bf16 [M, k, N] — ⑤
def submit_cold(phase, x_or_act_gpu, staging, cold, stream) -> None
def sync_cold(phase, staging, stream) -> Tensor                          # bf16 [M, k, N] (GPU) — ⑤ 개정
def rejoin_gateup(warm, cold, hot=None) -> Tensor    # fp32 누산(cold upcast) → act → bf16 [M, k, inter]
def rejoin_down(warm, cold, router_w, hot=None) -> Tensor  # fp32 누산 → 가중 expert합 → bf16 [M, hidden]
```

| primitive | sync point | buffer 소유 |
|---|---|---|
| `stage` | S1: topk_ids D2H 동기 (P0 유일의 host 블록) | arena는 ExecutionResources 소유 |
| `run_warm` | S2: evt_staged를 stream이 wait | 출력은 호출자 소유 (eager 신규 할당 허용) |
| `submit_cold` | 없음 — **enqueue-only, 즉시 반환** | staging은 ExecutionResources 소유, `.copy_()` in-place만 |
| `sync_cold` | S3: sync host node (CPU 완료 블록) | 〃 |
| `rejoin_*` | 없음 (순수 GPU 연산) | 출력은 호출자 소유 |

**invariant:**

- submit 콜백은 CPU task를 enqueue만 하고 즉시 반환한다. CPU 완료 대기는
  어떤 경우에도 submit에서 하지 않는다 (CPU/GPU overlap의 존립 조건).
- primitive는 영구 메모리를 절대 할당하지 않는다 — 전부 ExecutionResources
  에서 빌린다. ("Stage 4 이후 graph가 참조하는 storage identity는 바뀌지
  않는다"가 이 규칙 하나로 지켜진다.) `stage`의 host-측 결집 스크래치/인덱스
  (예: BatchedCopyStager의 `stage_scratch`/`stage_index`)도 예외 없이
  ExecutionResources 소유 — stager는 매 호출마다 빌릴 뿐 할당하지 않는다.
  `stage_scratch`는 proj당 2벌(더블버퍼) + 이벤트 가드로 소유한다 — H2D는
  enqueue-only라 언제 실제로 scratch를 읽는지 host가 모르므로, 단일 버퍼
  재사용은 "이전 H2D가 다 읽기 전에 host가 재기록"하는 WAR corruption을
  낸다(2026-08-20 Critical review finding). stager는 버퍼를 flip해 쓰고
  같은 버퍼로 돌아올 때만 그 버퍼를 마지막에 읽은 H2D 완료 이벤트를
  host-wait한다 — `stage_index`는 host index_select가 동기 소비라 예외.
- graph 경로에서 **구현이 교체되는** primitive는 `stage` 하나다 (S1이
  데이터 의존 host 분기이므로 capture 불가 → device worklist + gather
  커널). 단, 나머지가 "같은 호출"로 capture되는 것은 다음 전제 위에서다:
  1. cold 바인딩은 kt 방식(args 영구 할당, `qlen` 등 가변값은 포인터
     경유)으로 설계한다 — submit/sync가 eager와 graph에서 동일해지는
     조건 (K3 설계 제약).
  2. `run_warm`/`rejoin`은 shape-static하게 구동된다 — capture-bs별
     최악치 고정 그룹 수, 패딩 slot은 쓰레기 계산 후 rejoin에서 소거.
     데이터 의존 host 분기 금지. (동적 구동은 eager executor의 자유이고,
     고정 구동은 graph 경로 executor의 책임 — primitive 자체는 양쪽에서
     같은 callable.)
  primitive 밖에서는 executor 구동 구조(동적 루프 ↔ 고정 패스),
  capture-bs별 버퍼 등록, model_runner 접점 2줄이 graph 경로에서 추가된다.
- **전략 경계**: 그룹 구성과 (m, j) 좌표 복원은 `GroupingStrategy`(grouping.py)
  소유, staging 메커니즘(pinned → arena 이동 방식)과 그 스크래치/sel 버퍼의
  더블버퍼 flip·guard는 해당 `Stager`(stagers.py) 소유, **arena** WAR 이벤트
  체인(wait_event 발행/소비)은 executor 소유 — 세 축은 각자의 파일 밖으로
  새지 않는다. env/외부 시스템 읽기는 전부 조립 지점(method.py)이 하고
  명시 인자로 주입한다.

---

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
5. **plan 불변성 테스트 이원화**: bf16 라운딩이 밴드 경계에 의존하므로
   일반 입력에서는 tolerance 비교. **exact 검출은 정확히 표현 가능한
   입력**(작은 정수/2의 거듭제곱 — bf16 라운딩 무손실)으로 구성한 테스트가
   담당한다: 그 입력에서 "전부 cold = 전부 warm = 혼합"은 비트일치여야
   하며, 이것이 이중계산/누락의 exact 검출기다.
