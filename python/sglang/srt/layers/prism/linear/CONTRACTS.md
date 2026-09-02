# Prism dense 경계 계약 (2026-09-01 고정본)

MoE `moe/prism/CONTRACTS.md`의 dense 대응물. **부모 문서를 대체하지 않는다** —
①~⑤의 번호와 의미는 그대로이고, 여기 적힌 것은 dense에서 **달라지는 것과
dense가 추가하는 것**뿐이다. 다시 적지 않은 항은 MoE 계약을 문자 그대로
상속한다 (특히 ④의 graph/포인터 규칙과 ⑤의 수치 규칙).

- 대상 스택: sglang 포크(`prism-orchestration`) + kt-kernel 0.7.0
- 용어: K = contraction 축, N = 출력 축, M = 토큰 수.
- 용어 (신설): **슬롯(slot)** = 하나의 `(layer, proj, part)`. dense가 kt에
  내려보내는 계산의 최소 단위이고, MoE의 "(expert, proj)"가 있던 자리다.
- 활성화: `SGLANG_PRISM_LINEAR_PLAN`. MoE의 `SGLANG_PRISM_PLAN`과 **독립**.

---

## ① Plan 최소 계약 (dense)

**티어 의미는 그대로다** — HOT=VRAM 상주, WARM=pinned host를 GPU가 UVA 제자리
읽기, COLD=pageable host를 CPU가 읽고 계산. HOT/WARM의 계산 계약이 동일하다는
것도 그대로이고, 그래서 `tiers.py`가 `pinned` 플래그 하나로만 둘을 가른다.

**좌표 (MoE와 갈리는 지점):**

- 좌표는 `(layer, proj, part)`다. `proj`는 enum이 아니라 **열린 이름**
  (`self_attn.qkv_proj`)이고, K/N은 proj마다 자기 값이다 — `dims.k_of(proj)`
  같은 파생 공식이 없다.
- `part`는 **N축 조각**이다. 분할된 linear(`gate_up_proj`, `qkv_proj`)에서
  조각마다 티어 밴드·cold N shard·calib 테이블이 **독립**이다. 자산이 gate/up을
  따로 캘리브하므로 합쳐 두면 두 절반이 같은 마스크를 강요받는다.
- K 인덱스의 합집합은 **part별로** `[0, K)`의 순열이어야 한다. 이 검사가
  이중계산/누락의 최우선 방어선인 것은 MoE와 같다.

  > **현재 완화되어 있음.** `plan.check_partition`은 조각 경계가 layer의
  > `output_partition_sizes`와 "일치"가 아니라 "포함"이면 통과한다(2026-09-01
  > 완화). 이것이 못 잡는 것: plan이 N축 일부만 조각으로 선언하면 나머지 열이
  > **어느 조각에도 속하지 않은 채** 조용히 계산에서 빠진다. 완화를 유지하려면
  > "조각들의 합집합 == [0, N)"을 별도로 요구해야 한다 (TODO).

**정렬 규칙:** `PAIR_GROUP = 2` 유지 (sparse 점수가 페어 에너지이고 cold skip
단위가 VNNI 페어라서). cold N shard는 커널 키가 정하는 배수
(`kernels.cold_n_align`; tile fp4/fp8은 256). `ROW_GROUP`은 MoE와 함께 폐기 —
packed 저장의 타일 올림(`cold_pack_tile_rows`)은 인스턴스 내부 사정이고
`LinearColdShard.real_rows`가 그 사실을 나른다.

**Plan = 정책 / 자산 = 기하**의 분리도 그대로다. 단 dense 자산은 `[L, K]` 축이라
MoE의 `[L, E, K]`와 포맷이 다르고, `calib.py`가 그 어댑터다.

---

## ② C++ partial 진입점 계약 (dense) — **핵심**

### ②-0 결론: dense는 kt에 새 진입점을 요구하지 않는다

기존 `forward_gateup_partial` / `forward_down_partial`의 두 축에 dense 의미를
부여한다. 이것은 흉내가 아니라 **동형사상**이다 — 두 축 모두 dense에서 실재하는
것을 가리킨다:

| kt 축 | MoE 의미 | dense 의미 |
|---|---|---|
| `expert` (e) | 어느 expert의 슬랩인가 (런타임 라우팅) | **슬롯 신원** `(layer, proj, part)` — 로드 타임 고정 |
| `top_k` (j) | 이 토큰이 고른 expert 목록 | **이 호출에서 같이 계산하는 슬롯 목록** |
| `gate`/`up` 슬롯 | SwiGLU의 두 행렬 | **K를 공유하고 N이 같은 두 part** |
| `down` 슬롯 | 세 번째 행렬 | **단일 part proj** |

근거는 코드다: `forward_down_partial(qlen=1, k=1)`의 경로
(`prepare_down_routing` → `carve_down_decode` → `prep_down_decode` →
`run_down_stage` → `export_down_partial`)에는 expert 축이 **per-expert 버퍼의
인덱스로만** 등장한다. 라우팅도, 토큰 gather-scatter도, 합산도 없다. 즉 이
경로는 이미 "슬롯당 dense GEMV"이며, 실측이 `x @ W_cold.T`와 일치한 것은
우연이 아니다.

### ②-1 인스턴스 단위 = **형상 그룹**, layer가 아니다

**인스턴스 하나 = (K, N, 커널) 형상 그룹 하나**이고, `expert_num`은 그 형상을
가진 슬롯의 수(= 전 layer 합)다. NUMA 노드마다 sub-instance가 생기므로 C++
객체 수 = 형상 그룹 수 × `tp_count`.

Qwen3.8-27B 실측 형상:

| 그룹 (K, N) | 슬롯 | E | 진입점 |
|---|---|---|---|
| (5120, 17408) | `mlp.gate_up_proj` gate+up | 64 | **gateup** |
| (17408, 5120) | `mlp.down_proj` | 64 | down |
| (5120, 16384) | `linear_attn.in_proj_qkv(z)` | 48 | down |
| (6144, 5120) | `linear_attn.out_proj` + `self_attn.o_proj` | 64 | down |
| (5120, 12288) | `qkv_proj` q | 16 | down |
| (5120, 1024) | `qkv_proj` k+v | 16 | **gateup** |

→ **9개 인스턴스 × 2 노드 = 18개 C++ 객체.** 퇴화 경로의 352 × 2 = 704개가
아니다. `layer_idx`는 kt에서 로그 문자열 외에 쓰이지 않으므로(`moe.hpp:270`
주석, `sft_moe.hpp`) layer를 expert 축으로 접는 데 장애가 없다.

**슬롯 신원 → expert id 매핑은 로드 타임에 고정**되고 step마다 바뀌지 않는다.
이것이 dense가 MoE보다 나은 유일한 지점이다 (④-3 참조).

### ②-2 gateup 진입점의 dense 용법

`mlp.gate_up_proj`가 여기 정확히 맞는다. 대가가 0인 유일한 매핑이다:

```
config: hidden_size = K(5120), intermediate_size = N_part(17408) [노드 shard],
        n_total = N_part(full), expert_num = 64(layer), num_experts_per_tok = 1
호출:   forward_gateup_partial(qlen, k=1, expert_ids=[layer], x, out)
        x   : bf16 [qlen, K] full-width — **복제 없음** (gate/up이 같은 x를 읽는다)
        out : bf16 [qlen, 1, 2·n_total] — [0,n_total)=gate, [n_total,2n_total)=up
```

`out`의 레이아웃이 `[M, 2I]`, 즉 **sglang `SiluAndMul`이 기대하는 것 그대로**이고
동시에 dense executor의 `out3d [M, 1, N_total]`(gate 앞 열, up 뒤 열)과 같다.
재배치가 필요 없다.

gate와 up의 K 인덱스가 같으면 kt가 pack을 공유한다(`dual_pack_ == false`).
다르면 A 풀이 2배가 되는 것이 대가이며, 계약상 허용된다.

### ②-3 down 진입점의 dense 용법

단일 part proj 전부. `config(hidden_size = N, intermediate_size = n_total = K)`.

```
호출: forward_down_partial(qlen, k, expert_ids=[slot…], act, out)
      act : bf16 [qlen, k, n_total]   ← **슬롯 축 stride가 n_total이다**
      out : bf16 [qlen, k, N]         ← 노드는 자기 열 [down_n.offset, +rows)만 쓴다
```

`prep_down_decode`가 `act + j * n_total`을 읽으므로(`act_row = config_.n_total()`),
**한 호출에 슬롯을 여럿 실으면 x를 슬롯 수만큼 복제해야 한다.** decode에서는
K×2 B의 memcpy라 무시할 만하지만 prefill에서는 M×K×2 B다. 그래서:

> **기본은 슬롯당 한 호출**(k=1)이다. top_k 축 묶음은 최적화이지 계약이 아니고,
> 묶을 때는 복제 비용을 명시적으로 지불한다.

### ②-4 불변식 (위반은 전부 조용한 오답 또는 즉사)

1. **한 호출의 `expert_ids`는 서로 달라야 한다.** 같은 값이 두 번 오면
   `m_local_num_[e]`와 per-expert 버퍼가 겹쳐 한쪽 결과가 사라진다.
2. **한 인스턴스의 모든 슬롯은 N이 같아야 한다.** `hidden_size`/`down_n()`/
   `intermediate_size`가 config 스칼라이고 expert별이 아니다 (K만 expert별이다 —
   `gate_k(e)`/`down_k(e)`).
3. **한 인스턴스의 모든 슬롯은 K도 같아야 한다.** `n_total`이 스칼라이고
   prefill의 act stride가 그 값이기 때문이다. (decode 전용이라면 K가 달라도
   되지만, 그 예외에 기대지 않는다.)
4. **shard 테이블 길이 = `tp_count`.** `[0]` 하나면
   `partial shard table size != tp_count: gateup_n`으로 죽는다.
5. **out dtype = bf16, 예외 없음.** fp32로 잡으면 `got[j] ≈ ref[2j+1]`에 50%가
   0인 값이 나온다 — bf16이 fp32의 상위 16비트라 두 슬롯이 하나에 packed된다.
6. **누산 의미론은 overwrite(`=`)**. 티어 간 합산은 GPU rejoin에서 정확히 1회.
   노드별 N shard는 같은 out 버퍼의 서로소 열에 direct write이지 누산이 아니다.
7. **주입 텐서는 CPU여야 한다.** C++가 host memcpy로 읽으므로 device 포인터는
   segfault다.
8. **K축 기하는 호출에 없다.** 로드 시점에 인스턴스에 구워진다. 입력이
   full-width이므로 인덱스가 비연속이든 슬롯마다 길이가 다르든 시그니처는 불변.

### ②-5 유일한 C++ 변경 요구 — **선택적 proj 슬롯**

지금 kt는 gate/up/down 셋을 **전부** 요구한다 (`moe.hpp:513`
`no weight source`). down만 쓰는 매핑에서 gate/up에 32행 더미를 넣어 우회할 수
있지만, 그 대가는 더미 weight가 아니라 **더미의 C 버퍼 풀**이다:

```
init(): gate_bc_pool_bytes_ = buffer_c(pool_count_, intermediate_size)
        up_bc_pool_bytes_   = buffer_c(pool_count_, intermediate_size)
        buffer_c = sizeof(float) × max_m × n           (amx_raw_buffers.hpp:596)
        pool_count_ = max_len·top_k + expert_num·M_STEP
```

down 매핑에서 `intermediate_size`는 **dense proj의 K**다. `mlp.down_proj`
(K=17408, max_len=2048)이면 풀 하나가 4 × 2080 × 17408 ≈ 145 MB이고 그런 풀이
둘, **전부 쓰이지 않는다**. 퇴화 경로의 "+25 GB"는 여기서 나온 것이며 (RSS
기준; VA 예약은 더 크다) 더미 행 자체는 무해하다.

**더미의 하한은 셋이 정한다** (2026-09-01 실측 — 각각을 실제로 눌러봤다):

| 줄이려 한 것 | 결과 |
|---|---|
| 행 0 (슬롯 소멸) | `no weight source` — 0원소 텐서의 `data_ptr()`가 0이라 `moe.hpp:465`의 `gate_proj != nullptr` 분기가 빠진다. **더미는 존재해야 한다** |
| 행 2 (타일 미만) | `per-expert rows must be a multiple of K_STEP` |
| 노드 N 2 (정렬 미만) | **SEGFAULT** — 예외가 아니라 조용한 죽음이다. kt가 안 잡으므로 `cold_backend._config`가 잡는다 |

그래서 실제로 지불하는 최소는 `2 슬롯 × E × K_STEP 행 × (align × nodes) 열`이고,
Qwen3.8 전체에서 풀 0.14 GB다. gateup 매핑에서는 더미가 `down`이고 그 N 총합이
`hidden_size`(= 실제 K)로 고정이라 아예 못 깎는다 — 그쪽이 0.13 GB의 대부분이다.

이보다 줄이는 유일한 길:

> `PartialGeometry`의 각 proj에 **사용 여부**를 두고, 꺼진 proj의
> 버퍼·slab·pack·weight-source 검사를 전부 건너뛴다. 기본값은 "셋 다 사용"이라
> 기존 동작은 비트 동일하다 (`partial.enabled`가 이미 쓰는 침습성 상한 방식).

**값어치는 0.14 GB이므로 지금은 안 한다.** 이 항목이 존재하는 이유는 메모리가
아니라 어휘다 — `hidden`/`intermediate`가 실제와 무관한 값을 갖는 상태가 남는다.

이것이 `DenseConfig` + `AMX_DENSE_TP<K>` 신설을 **대체한다**. 진입점도, 커널도,
기하도, NUMA 분할도 그대로 쓴다.

---

## ③ Weight ownership (dense)

MoE 계약 ③ 그대로. dense가 추가하는 것:

- 스토어 모양: hot/warm은 `[k_tier, N]` **K-major**(transpose 없는 정준 방향,
  GPU 커널을 MoE와 공유), cold는 `[N, k_pad]` **ckpt 방향 유지**(kt `from_mat`의
  입력 방향), 타일 배수까지 0 패딩.
- `row_off`가 없다 — expert가 없으므로. **병합 인스턴스에 주입할 때 다시
  생긴다**: 슬롯을 expert 블록으로 이어 붙인 flat이고, wrapper의 원소 수
  불변식 `N × Σₑ k(e)`가 그대로 성립한다. 이 이어붙이기가 `cold_backend`의
  일이고 `LinearColdShard`는 dense의 사실만 말한다.
- **병합의 대가: 수명 단위가 layer가 아니라 형상 그룹이다.** 한 layer만 cold에서
  내리는 것이 불가능해진다. 현재 그런 요구가 없으므로 지불한다.
- warm pinned store는 GPU의 PCIe root complex와 같은 NUMA 노드에 있어야 하고,
  배치는 `alloc_pinned_on_node`가 **할당 전에** 정책을 걸어 보장한다 (사후
  마이그레이션 불가, torch pinned 캐싱 할당자 때문에 검증 필수).

---

## ④ 실행 primitive 계약 (dense)

```python
class LinearGpuTier(Protocol):        # 이미 있음 (tiers.py)
    def run(self, x2d, ids, ones, out3d, *, masking, grouping) -> None: ...

def submit_cold(slot, x_ptr, out_ptr, staging, stream) -> None   # enqueue-only
def sync_cold(staging, stream) -> Tensor                          # bf16 [M, k, N]
def rejoin(parts) -> Tensor                                       # fp32 Σ → bf16
```

MoE ④의 invariant를 전부 상속한다. 특히:

- **primitive는 영구 메모리를 절대 할당하지 않는다.** staging은
  `LinearResources`가 소유하고 생성 후 재할당 금지, 갱신은 `.copy_()`만.
  이유는 주소다 — graph 캡처가 포인터를 굽고 kt host node가 **나중에** 역참조한다.
- **graph 경로의 `qlen`은 bs별 전용 상수 버퍼.** eager와 공유하면 replay마다
  cold가 L토큰을 계산하는 stale-share 버그가 된다 (MoE 실측 30B decode
  328→56 ms/tok).
- **host-측 즉시쓰기는 stream 순서의 보호를 받지 않는다.**

dense 고유 (**MoE보다 단순해지는 지점**):

- **`expert_ids`가 정적이다.** 슬롯 신원은 로드 타임에 고정이므로 step마다
  조달할 것이 없다. MoE ④에서 "graph 경로에서 교체되는 것은 cold의 expert_ids
  조달 하나뿐"이라고 남긴 그 하나가 dense에는 **없다** — eager와 graph가 완전히
  같은 호출이다. pinned `expert_ids` 버퍼는 로드 타임에 한 번 채우고 끝난다.
- phase가 하나다. `gate_up_proj`와 `down_proj`가 별개 `LinearBase`라 한 호출에
  하나의 GEMM만 있고, MoE의 gateup→act→down 2-phase 조율이 없다.
- 전략 경계: 티어의 거처·읽기 방식은 `tiers.py`가, 슬롯↔kt 번역과 인스턴스
  수명은 `cold_backend.py`가 소유한다. env/외부 읽기는 전부 `method.py`.

---

## ⑤ 수치 계약 (dense)

MoE ⑤ 전체를 상속한다 — **누산은 항상 fp32, 재료화(wire)는 bf16**, partial은
예외 없이 bf16, 최종 라운딩 1회, 레퍼런스는 fp32 단일-GEMM 대비 tolerance.

dense가 추가하는 것:

- rejoin이 하나다: `Σ parts (fp32) → bf16`. 활성화는 sglang `SiluAndMul`이
  `apply` **밖에서** 걸고(융합하려면 모델 파일이 prism을 알아야 한다), 라우터가
  없으니 가중합도 없다.
- cold partial은 `[M, k, N]`으로 도착하고 GPU 티어 partial은 `[M, N]`이다.
  **`j → part 열 오프셋` 매핑을 흡수하는 것이 rejoin의 일**이고, 그 매핑은
  로드 타임에 고정된 슬롯 테이블에서 온다.
- **계약 ⑤-5(plan 불변성 비트일치)는 dense에서도 유일한 좌표 검출기다.**
  정확히 표현 가능한 입력(작은 정수/2의 거듭제곱)에서
  `all_hot == all_warm == all_cold == 혼합`이 비트일치해야 하고, 픽스처는
  **무작위 순열 인덱스**여야 한다 (연속 밴드는 좌표 뒤섞임을 검출하지 못한다).
- **sparse plan에서는 이 불변성이 성립하지 않는다.** HOT은 마스킹하지 않으므로
  (`tiers.SPARSE_TIERS = {WARM}`) 세 티어 마스크의 합집합이 full-K 마스크가
  아니고, 같은 행을 warm↔hot으로 옮기면 출력이 달라진다. sparse 테스트는
  tolerance 비교만 할 수 있다.

---

## ⑥ 방어 규칙 (이 코드베이스의 성격)

오프로드 계열의 결함은 대부분 **"성능만 달라지고 값은 맞는"** 또는 **"값이
틀렸는데 그럴듯한"** 형태다. 정확도 테스트도 서버도 다 통과하고 벤치 결론만
틀린다. 그래서 이 코드베이스의 주된 방어 수단은 **로드 타임에 즉사시키는
검사**다.

새 검사를 넣을 때는 docstring에 **"안 잡으면 어떻게 조용히 틀리는가"**를 적는다.
이번 세션까지 실제로 무언가를 잡은 게이트:

| 게이트 | 잡은 것 |
|---|---|
| `executor.register()` cold 거부 | plan 반올림이 만든 2행짜리 유령 cold 밴드 |
| `method.check_coverage` | 포크/upstream 모듈명 차이 (`in_proj_qkv` vs `in_proj_qkvz`) — 8 GB가 오프로드에서 빠진 채 정상 응답 |
| `calib.check` 전부-0 | `linear_attn`의 `wn_o`/`to2l`이 0 → `imp = 0 >= thr = 0`으로 전부 살아남아 sparsity가 조용히 0 |
| `weights._index` 페어 검증 | 티어 경계가 페어를 쪼개면 점수 재구성 불가 |
