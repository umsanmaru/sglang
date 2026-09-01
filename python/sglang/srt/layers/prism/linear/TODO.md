# Prism dense — 인계 문서 (2026-09-01)

MoE 쪽 `moe/prism/TODO.md`의 dense 대응물. **다음 작업자에게 넘기는 것**이 목적이라
"무엇이 되어 있고 / 무엇을 실측했고 / 다음에 무엇을 왜 그렇게 하는가"를 적는다.

**경계 계약은 옆의 `CONTRACTS.md`에 있다** (2026-09-01 신설). 이 문서는 상태와
일정이고, 그 문서는 바꾸면 의존하는 모든 층을 함께 검토해야 하는 것들이다.

---

## 1. 지금까지 선 것

dense linear(qkvo·MLP)의 K-split 오프로드. **GPU 티어(hot/warm)까지 실모델에서 동작한다.**

| 모듈 | 줄 | 상태 |
|---|---|---|
| `plan.py` | 699 | `(layer, proj)` 좌표 · N축 `parts` · sparsity 블록 · calib 키 |
| `formats.py` | 319 | bf16 / blockwise fp8 · gemv/grouped 진입점 |
| `weights.py` | 392 | 절단·배치 → `PreparedLinear` |
| `calib.py` | 201 | 자산 어댑터 · **전부-0 게이트** |
| `tiers.py` | 139 | GPU 티어 (worklist ↔ grouped, E=1 퇴화) |
| `rejoin.py` | 83 | fp32 Σ → bf16 (Triton) |
| `executor.py` | 181 | 1-phase 조율 |
| `method.py` | 289 | sglang 접점 |

**실모델 실적 (Qwen3.8-27B, RTX 5090 32GB, `sglang-dense` 트리):**

```
stock                     54.6 GB → OOM (모델이 GPU에 안 들어간다)
prism hot10/warm90        10.66 GB로 기동, 384 좌표 중 256개 등록
                          "The capital of France is" → " Paris. …Berlin. …Rome. …Madrid."
                          32토큰 33.7초 = 1.05 s/tok  (warm 43.8 GB/tok ÷ PCIe 50 GB/s ≈ 0.88이 하한)
```

**미배선**: cold(kt), sparse.

---

## 2. 실측 — cold 퇴화 경로 (핵심 발견)

`scripts/qwen38/bench_cold_degenerate.py`. **결론: 돌지만 C++로 가는 게 맞다.**

### 되는 것

`MOEConfig(expert_num=1, top_k=1, hidden_size=N_proj, intermediate_size=K_proj)`에
**down 슬롯만** 채우면 `forward_down`이 정확히 `x @ W_cold.T`를 낸다 (max_err 0.124 /
|ref|max 60.0 = bf16 라운딩).

```
28스레드 · 2 NUMA · cold 75% · Qwen3.8 치수
  decode  0.39 s/tok   유효 대역 119 GB/s    ← warm-only 1.05 s/tok 대비 2.7배
  인스턴스 9 → 72개로 늘려도 슬롯당 0.74~0.77 ms로 평평 (스레드풀은 하나고
                                              인스턴스는 작업 소유자일 뿐)
```

### 안 되는/나쁜 것 — ~~C++를 파는 근거~~ **(§3에서 뒤집힘)**

> **2026-09-01 개정.** 아래 표의 "퇴화 경로" 열은 **벤치 스크립트가 슬롯마다
> 인스턴스를 하나씩 만든 결과**이지 퇴화 경로의 성질이 아니다. 슬롯을 expert
> 축으로 접으면 인스턴스가 352 → 18개(형상 그룹 9 × 2노드)가 되고 오버헤드가
> 함께 사라진다. 근거는 §3. 표는 "그때 무엇을 쟀는가"의 기록으로 남긴다.

| | 퇴화 경로 | `PartialDenseWrapper` |
|---|---|---|
| **빈 슬롯** | ❌ `moe.hpp:513` `no weight source` → gate/up에 32행 더미 필수 | 슬롯이 가변이라 문제 자체가 없음 |
| 인스턴스 수 | 슬롯당 1개 → Qwen3.8 **352개** | layer당 1개 → **64개** |
| 메모리 오버헤드 | 72 MB/개 × 352 = **+25 GB** | ≈ 5 GB |
| 생성 시간 | 222 ms/개 × 352 = **78초** | ≈ 14초 |
| 어휘 | `hidden`/`intermediate`가 실제와 무관한 값으로 쓰임 | 이름이 사실을 말함 |

더미 슬롯은 증상이고 원인은 **"슬롯이 gate/up/down 3개로 고정"**이라는 구조다.
dense는 층당 5~7슬롯(Qwen3.8), Muse-Glimmer는 8슬롯이라 3에 맞출 수가 없다.

### 이 경로를 다시 만들 때 필요한 사실 넷

퇴화 경로를 쓰든 C++를 쓰든 kt의 계약은 같다. 이번에 **실패로** 배운 것들:

1. **shard 테이블은 NUMA 노드 수(`tp_count`)만큼.** `[0]` 하나면
   `RuntimeError: partial shard table size != tp_count: gateup_n`.
2. **빈 슬롯 불가.** `config.gate_proj == nullptr`이면 `moe.hpp:513`에서 죽는다.
3. **출력 dtype은 bf16.** fp32로 잡으면 `got[j] ≈ ref[2j+1]` + 50% 0이 나온다 —
   bf16이 fp32의 상위 16비트라 짝수 두 개가 한 슬롯에 packed되기 때문이다.
   MoE `resources.py:75`가 "wire dtype = bf16"이라고 이미 적어놨다.
4. **28스레드가 최적** (물리 16코어 = 8×2소켓). 14→28은 7% 개선, **56은 9배 악화**
   (6.10 ms/슬롯) — MoE 메모의 과다구독 현상 그대로.

---

## 3. C0 답 (2026-09-01 조사) — `PartialDenseWrapper`는 **필요 없다**

§3 초판은 "`moe_base.hpp`(1904줄)를 읽고 견적을 다시 내야 한다"로 끝났다.
읽었다. **결론이 뒤집힌다: C++ `DenseConfig` + `AMX_DENSE_TP<K>`는 짓지 않는다.**

계약 전문은 옆의 `CONTRACTS.md`(신설)에 있고, 여기는 근거와 일정만 적는다.

### 3.1 왜 뒤집혔나 — 세 가지 사실

**(a) partial decode 경로에 expert 축이 "MoE적"으로 쓰이는 곳이 없다.**
`forward_down_partial(qlen=1, k=1)`은 `prepare_down_routing` →
`carve_down_decode` → `prep_down_decode` → `run_down_stage` →
`export_down_partial`인데, 여기서 expert는 **per-expert 버퍼의 인덱스**일 뿐이다.
라우팅도 토큰 gather-scatter도 합산도 없다. 이 경로는 이미 "슬롯당 dense GEMV"다.

**(b) 그래서 두 축에 dense 의미를 줄 수 있다 — 흉내가 아니라 동형이다.**

    expert 축  ↔  슬롯 신원 (layer, proj, part)   — 로드 타임 고정
    top_k 축   ↔  이 호출에서 같이 계산하는 슬롯
    인스턴스   ↔  (K, N, 커널) **형상 그룹** 하나 (전 layer 공유)

`layer_idx`는 kt에서 로그 문자열 외에 쓰이지 않으므로 layer를 expert 축으로
접는 데 장애가 없다. Qwen3.8은 형상 그룹이 9개 → **9 × 2노드 = 18개 C++ 객체**.
초판이 "슬롯당 1개 = 352개"라고 적은 것은 퇴화 경로의 성질이 아니라 **벤치
스크립트가 그렇게 만든 것**이었다.

그리고 `mlp.gate_up_proj`는 **gateup 진입점에 대가 0으로 맞는다**: gate/up이
K를 공유하고 N이 같으므로 x 복제가 없고, 출력 `[qlen, 1, 2·n_total]`이
sglang `SiluAndMul`이 기대하는 `[M, 2I]`이자 dense executor의
`out3d [M, 1, N_total]`과 **같은 레이아웃**이다. 재배치가 0이다.

**(c) 더미 슬롯의 진짜 비용은 weight가 아니라 C 버퍼 풀이다.**

    init(): gate_bc_pool_bytes_ = buffer_c(pool_count_, intermediate_size)
            up_bc_pool_bytes_   = buffer_c(pool_count_, intermediate_size)
            buffer_c = sizeof(float) × max_m × n      (amx_raw_buffers.hpp:596)
            pool_count_ = max_len·top_k + expert_num·M_STEP

down 매핑에서 `intermediate_size`는 **dense proj의 K**다. `mlp.down_proj`
(K=17408, max_len=2048)이면 풀 하나가 4 × 2080 × 17408 ≈ 145 MB, 그런 풀이 둘,
**둘 다 안 쓰인다**. §2의 "72 MB/개 × 352 = +25 GB"는 여기서 나온 것이고
(RSS 기준 — VA 예약은 더 크다) 32행 더미 weight 자체는 무해하다.

### 3.2 그래서 C++에 남는 요구는 **하나**다

> `PartialGeometry`의 각 proj에 **사용 여부** 플래그를 두고, 꺼진 proj의
> 버퍼·slab·pack·weight-source 검사를 건너뛴다. 기본값 "셋 다 사용"이라
> 기존 동작은 비트 동일 (`partial.enabled`가 이미 쓰는 침습성 상한 방식).

건드리는 곳: `AMX_MOE_BASE::init()`의 per-expert 버퍼 루프와 풀 산정,
`alloc_cold_slabs`, `moe.hpp`의 TP `load_weights` 분기(`:513`),
`experts_partial.py`의 세 텐서 필수 검사. **5개 지점, 150줄 규모의 가산적 변경**이다.
1904줄 파일의 expert 축 분리가 아니다.

없어도 갈 수 있다 (형상 그룹 병합만으로 인스턴스가 18개면 더미 풀도 18벌뿐이라
수 GB VA / 수백 MB RSS다). 그래서 **C1은 블로커가 아니라 최적화**이고, 순서를
뒤로 뺀다.

### 3.3 단계와 게이트 (개정)

| 단계 | 작업 | 게이트 |
|---|---|---|
| **D0** | `cold_backend.py` — 슬롯 테이블(형상 그룹 도출, 슬롯→(instance, expert) 매핑), Plan→`MOEConfig` 번역, 인스턴스 수명 | 형상 그룹 도출이 plan에서 결정적. 그룹 내 (K, N, 커널) 동일 검증 |
| **D1** | `resources.py` — pinned staging (x/act/out/qlen, **정적 expert_ids**) | 생성 후 재할당 없음 (계약 ④) |
| **D2** | `weights.py` — 슬롯 cold 텐서를 그룹별 flat으로 이어붙이기 + `row_off` | wrapper의 `N × Σₑ k(e)` 원소 수 불변식 통과 |
| **D3** | ~~`rejoin.py`~~ **불필요** — cold pinned out을 executor가 `out3d[:, 0, n_start:n_end]`로 H2D하면 열 매핑이 복사에 흡수된다. gateup 매핑은 out이 곧 `[M, N_total]`이라 연속 복사 1회 | — |
| **D4** | `executor.py` — submit ∥ GPU → sync → H2D → rejoin, cold 거부 해제 | **계약 ⑤-5**: 무작위 순열 인덱스에서 `all_hot == all_warm == all_cold == 혼합` 비트일치 |
| **D5** | 실모델 (Qwen3.8-27B) | 0.39 s/tok 근처 (§2 벤치가 기준선), 1.05 s/tok 대비 개선 |
| **D6** | *(선택)* kt 선택적 proj 슬롯 | 더미 풀 소멸, 기존 MoE 경로 비트 동일 |

**최대 리스크가 바뀌었다.** C0(C++ 견적)이 아니라 **D0의 형상 그룹 도출**이다:
슬롯을 잘못 묶으면(N이 다른데 같은 인스턴스에) kt가 즉사하지 않고 **엉뚱한 열에
쓴다**. `CONTRACTS.md` ②-4의 불변식 8개를 D0에서 전부 로드 타임 검사로 박는다.

### 3.4 폐기 — 초판 §3.4의 우회

"C0 견적이 나쁘면 25 GB / 78초를 감수하고 C3부터"는 폐기한다. 감수할 것이 없다.

---

## 4. 그다음 — sparse

calib 어댑터(`calib.py`)와 자산은 이미 있다. 남은 것은 `SparseSpec` 조립과 배선이다.

| | |
|---|---|
| sparse 티어 | **WARM + COLD** (HOT은 마스킹 안 함 — `tiers.SPARSE_TIERS`) |
| `lambda` | **효과 없음.** k=1이면 `slot_sparsity`(`moe_base.hpp:625`)가 `s = clip(p)`로 축약된다. 자산도 `lam0 = 0.0`으로 생성됐다 |
| 게이트 | **p=0에서 dense와 비트일치** (`gemv_*_sparse`가 "전부 keep이면 dense와 비트일치"를 보장한다) |
| 커버리지 | Qwen3.8 384슬롯 중 **256개(67%)** — `linear_attn` 3종은 calib이 없다. Muse-Glimmer는 416 중 364(87.5%) |

**대가 하나**: HOT이 마스킹되지 않으므로 세 티어 마스크의 합집합이 full-K 마스크가
아니다 — 같은 행을 warm↔hot으로 옮기면 출력이 달라진다. **sparse plan에서는 계약 ⑤의
배치 불변성이 성립하지 않는다.**

---

## 5. 이번 세션에서 게이트가 잡은 것 (설계 근거)

전부 **조용히 틀렸을** 것들이고, 각각이 그 게이트의 존재 이유다.

| 게이트 | 잡은 것 | 안 잡았다면 |
|---|---|---|
| `executor.register()` cold 거부 | plan 생성기 반올림 → 2행짜리 유령 cold 밴드 | 그 2행이 계산에서 빠진 채 정상 기동 |
| `method.check_coverage` | 포크와 upstream의 모듈 이름 차이 (`in_proj_qkv` vs `in_proj_qkvz`) | 4.0 G 파라미터(8 GB)가 오프로드에서 빠진 채 정상 응답 |
| `calib.check` 전부-0 | Qwen3.8 `linear_attn` 층의 `wn_o`/`to2l`이 0 | `imp = 0 >= thr = 0`으로 **전부 살아남아** sparsity가 0이 됨 |
| `plan.check_partition` | (내 검사가 과했음 — "일치"를 요구했다) | 정상 구성을 막음. "포함"으로 완화했다 |

**교훈**: 오프로드 계열의 결함은 대부분 "성능만 달라지고 값은 맞는" 형태다.
정확도 테스트도 서버도 다 통과하고 벤치 결론만 틀린다. 그래서 **로드 타임에 즉사시키는
검사**가 이 코드베이스의 주된 방어 수단이다. 새 검사를 넣을 때는 "안 잡으면 어떻게
조용히 틀리는가"를 docstring에 적을 것.

---

## 6. 열린 결정

| | |
|---|---|
| ~~**C0 견적**~~ | **닫힘 (§3)** — `PartialDenseWrapper`는 짓지 않는다. 남은 C++ 요구는 "선택적 proj 슬롯" 하나이고 블로커가 아니다 |
| **형상 그룹 도출** | 슬롯 → (인스턴스, expert id) 매핑을 plan에서 결정적으로 뽑는 규칙. D0의 최대 리스크 — 잘못 묶으면 즉사하지 않고 엉뚱한 열에 쓴다 |
| **cold의 prefill** | decode는 CPU가 맞지만 prefill은? MoE에는 `cold_gpu.py`(kt packed slab을 `cudaHostRegister`로 GPU가 제자리 읽기)가 있고 dense grouped 커널도 있다. 그 경로를 dense에 잇는가, prefill을 CPU에 맡기는가 |
| **`max_len`** | kt 풀이 `max_len·top_k + E·M_STEP`에 비례한다. dense는 인스턴스가 적어져 여유가 생겼지만 prefill 청크 크기와 직결된다 |
| **`check_partition` 완화** | "포함"으로 완화하면서 "조각들의 합집합 == [0, N)"이 검사되지 않는다 (CONTRACTS ①) — 선언 안 된 N 열이 조용히 빠진다 |
| **score k1 vs k2wl2** | 자산에 둘 다 있다 (`KSET: ['k1','k2wl2']`). 지금 코드는 k2wl2만 안다. k1이 페어가 아니면 `PAIR_GROUP` 제약이 풀리지만 **kt의 sparse 경로가 페어 마스크(`uint16*`)를 전제**하므로 커널 확인이 선행돼야 한다 |
| **`p` 값** | dense는 λ가 죽어 p가 곧 sparsity다. 자산의 `PMAX=0.9`가 상한 |
| **스레드 수** | 벤치는 28이 최적이지만 실서버에서는 GPU 스트림·스케줄러와 경합한다. 재측정 필요 |
| **`BandSpec` 개명** | "밴드"는 초판의 "티어당 구간 하나" 시절 이름이다. 지금은 티어당 다중 밴드가 되어 이름이 오해를 부른다(실제로 이번 세션에 유령 밴드 실수의 배경이었다). MoE 68파일이 같은 어휘를 쓰므로 일괄로만 의미가 있다 — 별도 커밋 |
| **fp8 dense** | `Fp8LinearFormat`(blockwise)은 있으나 미사용. Mistral-Medium은 **per-tensor**라 별도 포맷이 필요하고, 게다가 `Fp8LinearMethod`는 `WEIGHT_LOADER_V2_SUPPORTED`에 있어 래핑하면 로더가 조용히 v1으로 떨어진다 (`method.py:_V2_LOADER_UNSUPPORTED`가 즉사시킨다) |

---

## 7. 환경·트리 (재현에 필요)

```
sglang/         [prism-orchestration]  포크 = 개발 트리 (source of truth)
sglang-dense/   [prism-dense]          upstream/main 9e9d26a4a — dense 실행
sglang-glm/     [pr-36507]             GLM 레인

env prism-dense: torch 2.13.0+cu130, kt-kernel 0.7.0
모델: ~/models/Qwen3.8-27B (55.6 GB, 샤드 18)
자산: assets/{qwen38_27b,muse_glimer,mistral_med_128B}.pt
```

**포크에서 고칠 때마다** `DST=…/sglang-dense bash scripts/port_prism_to_upstream.sh`.
그 스크립트는 멱등이라 앵커 삽입은 건너뛰고 소스만 갱신한다.

**kt 빌드 함정**: conda의 `hwloc.pc`가 `Requires.private: libxml-2.0`인데 그 `.pc`는
`libxml2`가 아니라 **`libxml2-devel`**이 준다. 없으면 시스템 것으로 폴백해
`/usr/lib/include/libxml2`라는 없는 경로가 나오고 CMake가 죽는다
(`build_prism_dense_env.sh`에 반영됨).
