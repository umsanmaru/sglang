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
- **hot tier**: HotWeights 타입 신설 + loader VRAM 배치 + executor hot
  GEMM 한 줄. 스키마/rejoin은 이미 대비됨.
- **N-shard rank 분산 (TP>1)**: rank별 자기 inter 샤드만 store/DMA.

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
