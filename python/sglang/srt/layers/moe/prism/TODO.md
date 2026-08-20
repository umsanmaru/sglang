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

## partial prefill (qlen>1) — S7 전 필수

K1의 forward_gateup_partial은 decode(qlen==1) 전용 (TP 진입에서 guard).
주의: kt 기준 qlen>1은 전부 prefill 경로라 **batch>1 eager decode도 이
guard에 걸린다**. 필요 작업: ① forward_prefill의 1~4단계 절단 (decode가
아닌 prefill의 토큰 그룹핑 — m_local_pos_/m_local_input_ gather — 을
계승), ② strided from_mat 변형 (full-width x의 row stride ≠ buffer k;
SFT pack_a_transposed의 source_stride 선례), ③ m_local_pos_ 기반
(토큰, slot) scatter export. 자리: K2 직후 (down과 함께 한 번에 이식).

## 기타 미룸 항목 (합의된 것만, 요약)

- **CUDA graph 경로 — 1차 스코프는 decode bs=1** (2026-08-20 결정):
  M=1이면 distinct ≤ top_k라 그룹이 항상 1개 → 최악치 고정 그룹/패딩
  낭비 문제가 공짜로 소멸하고 run_warm/rejoin이 자명하게 shape-static.
  bs>1 decode는 eager 폴백 (`--cuda-graph-bs 1`). bs=1이어도 필요한 것:
  GatherKernelStager(device dedup+worklist — topk_ids D2H는 bs=1에도
  capture 불가), 포인터 간접 바인딩(계약 ④ 전제 ①), capture-bs 등록
  (cuda_graph_runner.py:496 옆), VRAM 예약 회계(model_runner.py:593 옆).
  bs>1 graph는 그 다음: persistent GEMV(worklist 네이티브)로 그룹 문제
  자체를 제거하는 planir 방식.
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
