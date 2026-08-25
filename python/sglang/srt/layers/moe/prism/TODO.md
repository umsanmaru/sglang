# Prism TODO — 의도적으로 미룬 작업 대장

P0에서 결정만 해두고 구현을 미룬 항목들. 각 항목은 "왜 미뤘는지"와
"구현 시 건드릴 곳"을 함께 기록한다. (P0 범위 자체는 CONTRACTS.md와
커밋 계획 참조)

## warm 전송 ping-pong (그룹 루프 전송 노출 해소)

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

## ~~stage 일괄화 (H2D 26→3/층)~~ — ✅ 완료 (2026-08-21)

NVTX 실측(2026-08-20)에서 slot당 `copy_` 산란이 층당 H2D 26조각 + dispatch를
만들어 ~1.4ms/층의 오버헤드였다. `BatchedCopyStager`(host `index_select`로
결집 → H2D 1회/proj, 스크래치는 더블버퍼 + guard event로 WAR 방어)로 치환해
`select_stager`의 eager 기본값이 됐다. 등가성은 `PerSlotCopyStager` 대비
bitwise(`torch.equal`)로 검증 (test/prism/test_stagers.py).

## ~~graph bs=1 (GatherKernelStager)~~ — ✅ 완료 (2026-08-21)

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
- ~~C++ dual-pack (gate ≠ up K-밴드)~~ — **폐기** (2026-08-20 사용자 결정:
  gate/up은 K·N 모두 공유, 풀 계획 없음). gate==up 검증은 영구 제약으로
  cold 로드에 유지. 스키마의 독립 표현력과 로더의 처리 능력은 공짜
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
- **warm GEMM 커널 교체: torch_bmm → persistent GEMV**: torch_bmm은
  정확성 기준선용 placeholder다. 알려진 placeholder 한계 — ① x를 전
  expert에 broadcast하므로 라우팅 안 된 (토큰, expert) 쌍까지 계산
  (prefill에서 낭비 큼), ② 전역 matmul 플래그 저장/복원이 커널 안에
  있음(진짜 커널은 자체 fp32 누산이라 불필요), ③ 그룹 루프가 launch
  단위라 bs>1 graph에서 최악치 고정 필요. persistent GEMV(planir
  kernels.cu 연장)는 device worklist를 네이티브로 소화해 ①③을 없애고
  가변 k_warm도 공짜 — 교체는 registry 한 줄 (`WarmGemmFn` 계약 유지,
  단 다중 밴드 시 k_offset 인자 진화는 위 interleave 항목 참조).
- **가변 k_warm[e]**: calibration 산출 밴드. loader의 균일성 요구 제거
  (offset 테이블 store) + warm GEMM을 persistent GEMV로 교체 (위 항목).
- **pinned store NUMA 바인딩**: kt-kernel에 set_memory_to_numa 노출 추가
  후 numa.py에 연결 (현재는 first-touch 방임).
- ~~**hot tier**~~ — **완료** (2026-08-24). HotBand/HotStore + loader VRAM
  배치(`prepare_layer_weights(device=)`) + executor hot 경로(stager·arena
  없이 `index_select` → 같은 warm GEMM). 검증: 3-tier 재조립 bitexact,
  plan 불변성에 `all_hot`/`three_tier` 추가(all_hot ≡ all_warm 수치 일치),
  CUDA graph 캡처·replay bitwise 일치. plan 생성기 `--hot-frac`.
  남은 것: hot 비율 스윕 실측, `index_select` 사본 비용(GEMM 읽기량의 2배
  추가 HBM 트래픽) 측정 후 필요시 gather-free 커널.
- **N-shard rank 분산 (TP>1)**: rank별 자기 inter 샤드만 store/DMA.

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

**worklist graph vs 같은 plan eager**: bs=2에서 2.14배(GPU0 공유 상태에서
측정, 분자=31.47ms 그래프 측정값), bs=4에서 1.63배, bs=8에서 1.67배 —
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
