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
- gate와 up은 **독립적으로 분할될 수 있다** (스키마 수준). 단 P0 실행은
  cold 로드 시 `gate.bands == up.bands`를 요구한다 — 이는 C++ dual-pack
  미구현이라는 capability gap이지 스키마 제약이 아니다. dual-pack 구현 시
  로드 검증 한 줄만 제거된다.
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
    float*         out);         // fp32 [qlen × k × 2·inter], slot j ↔ expert_ids[m, j]
                                 //      열 [0, inter) = gate, [inter, 2·inter) = up

void forward_down_partial(int qlen, int k,
    const int64_t* expert_ids,
    const void*    act,          // bf16 [qlen × inter_FULL]   ← full-width
    float*         out);         // fp32 [qlen × k × hidden]
```

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
def run_warm(proj, x_or_act, slot_map, arena, evt_staged) -> Tensor      # fp32 [M, k, N]
def submit_cold(phase, x_or_act_gpu, staging, cold, stream) -> None
def sync_cold(phase, staging, stream) -> Tensor                          # fp32 [M, k, N] (GPU)
def rejoin_gateup(warm, cold, hot=None) -> Tensor    # fp32 합 → act → bf16 [M, k, inter]
def rejoin_down(warm, cold, router_w, hot=None) -> Tensor  # 합 → 가중 expert합 → bf16 [M, hidden]
```

| primitive | sync point | buffer 소유 |
|---|---|---|
| `stage` | S1: sel D2H 동기 (P0 유일의 host 블록) | arena는 ExecutionResources 소유 |
| `run_warm` | S2: evt_staged를 stream이 wait | 출력은 호출자 소유 (eager 신규 할당 허용) |
| `submit_cold` | 없음 — **enqueue-only, 즉시 반환** | staging은 ExecutionResources 소유, `.copy_()` in-place만 |
| `sync_cold` | S3: sync host node (CPU 완료 블록) | 〃 |
| `rejoin_*` | 없음 (순수 GPU 연산) | 출력은 호출자 소유 |

**invariant:**

- submit 콜백은 CPU task를 enqueue만 하고 즉시 반환한다. CPU 완료 대기는
  어떤 경우에도 submit에서 하지 않는다 (CPU/GPU overlap의 존립 조건).
- primitive는 영구 메모리를 절대 할당하지 않는다 — 전부 ExecutionResources
  에서 빌린다. ("Stage 4 이후 graph가 참조하는 storage identity는 바뀌지
  않는다"가 이 규칙 하나로 지켜진다.)
- graph 경로에서 구현이 갈라지는 primitive는 `stage` 하나뿐이다 (S1이
  데이터 의존 host 분기이므로 capture 불가). 나머지는 문자 그대로 같은
  호출을 capture한다.

---

## ⑤ 수치 계약

1. **부분합 dtype = fp32.** 티어 partial(C++ out, warm GEMM out, H2D 경로)은
   전부 fp32이고, rejoin 합산·act·router 가중합까지 fp32로 수행 후 마지막에
   bf16 캐스트. 근거: bf16 partial은 정확도 오차가 티어 경계 위치(= plan
   내용)에 의존하게 되어 최악의 디버깅 지형을 만든다. fp32 partial이면
   "어떤 plan이든 단일 GEMM의 fp32 누산과 동등" 한 문장으로 계약된다.
2. **레퍼런스**: numpy fp32 단일-GEMM 레퍼런스에 대해, 임의 plan(경계 임의
   배치 포함)에서 rejoin 결과가 tol 이내.
3. **plan 불변성 테스트**: "hot=∅/warm=10%/cold=90%", "전부 cold", "전부
   warm" plan들이 동일 입력에서 서로 일치해야 한다 — 레퍼런스 없이도
   이중계산/누락을 잡는다.
