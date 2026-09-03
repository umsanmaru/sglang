# GLM-5.3-Flash × Prism (fp8 3-tier) 런북 — nutella3 / RTX 5090 (sm_120)

## 0. 전제
- 트리: `~/prism-sglang/sglang-glm` = upstream PR **#36507** 워크트리(head 3992c4ce2) + prism 이식
  (`scripts/port_prism_to_upstream.sh`). 포크에 PR을 merge하지 말 것 — 포크는 upstream보다 **7,483 커밋** 뒤(병합 기점 2026-02-26).
- env: `prism-glm` (torch 2.13.0+cu130 / sglang-kernel 0.4.6.post1 / flashinfer 0.6.18 / tilelang 0.1.12 /
  transformers 5.12.1 / triton 3.7.1), kt-kernel 0.7.0 재빌드본(5개 cold 연산자 확인).
- 모델: `/home/jun/models/GLM-5.3-Flash` (로컬 NVMe, 62샤드 / 306G / 2.1 GB/s, 76,108 텐서).
  NAS 사본(`/mnt/nas/jun/checkpoints/zai-org/GLM-5.3-Flash`)도 온전하지만 **NAS는 45 MB/s**라 쓰지 말 것.
- calib: `assets/glm53_flash_calib.pt` (NL=45, E=288, k2wl2, pmax 0.9, grid 0.005, ng 201, λ0 1.4203).
- plan: `plans/glm53f/glm53f_h05_w10_sp50_fp8.json` — hot 3.125% / warm 9.375% / cold sparse p=0.5
  (fp8은 K 정렬 **128**이라 5%/10% 요청이 이렇게 내려앉는다). 실제 예산 **hot 5.91 GiB / warm 23.63 GiB(pinned, node1) /
  cold 254 GiB(127/node)**. dense 대조본은 `glm53f_h05_w10_dense_fp8.json`.

## 1. 기동
```bash
cd ~/prism-sglang
SGLANG_AUTO_NUMA_BIND=0 ./scripts/glm53/run_glm53_prism.sh \
  plans/glm53f/glm53f_h05_w10_sp50_fp8.json 30112 \
  --model-loader-extra-config '{"enable_multithread_load": false}'
```
스크립트가 넣는 것: `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`, `--dsa-{prefill,decode}-backend tilelang`,
`--kv-cache-dtype bfloat16`, `--linear-attn-backend triton`, `SGLANG_PRISM_CPUINFER_THREADS=14`,
`FLASHINFER_CUDA_ARCH_LIST=12.0a`.

### 위 두 인자가 왜 필수인가 (2026-08-31, OOM 3회로 확인)
prism fp8 `create_params`가 **42층 × 6.75 GiB = 305 GB**의 full 텐서를 CPU에 잡고,
sglang 표준 흐름상 `process_weights_after_loading`(prism 등록/해제)은 **모든 샤드 로딩이 끝난 뒤**에 돈다.
따라서 로딩 피크에 305 GB가 그대로 남는다.

| 시도 | 구성 | 결과 |
|---|---|---|
| 1 | 기본(멀티스레드 로더 + mmap, auto NUMA bind on) | shard **54/62**에서 OOM(-9) |
| 2 | `--weight-loader-disable-mmap` 추가 | shard **36/62** — **더 나쁘다**(워커마다 샤드 버퍼). mmap은 범인이 아니다 |
| 3 | `enable_multithread_load: false` (순차 로더) | shard 전량 로드 직전 OOM. 샘플러가 원인을 잡았다: `used=259 avail=244`인데 **node1 free=190 MB / node0 free=112 GB** |
| 4 | + **`SGLANG_AUTO_NUMA_BIND=0`** | node0로 폴백 가능 |

3번의 진짜 원인: `srt/managers/scheduler.py:5400`이 스케줄러를 **GPU-local NUMA 노드(node1)에 strict 바인딩**한다
(`get_numa_node_if_available` → `numa_bind_to_node`). node1 용량은 ~251 GB이므로 305 GB가 들어갈 수 없고,
node0에 112 GB가 남아 있어도 폴백하지 않고 OOM된다. DSV4가 통과한 이유는 fulls가 147 GB로 node1에 들어가기 때문.
`SGLANG_AUTO_NUMA_BIND=0`이면 `Cpus_allowed_list: 0-31`이 되고 기본 정책(local-first + fallback)으로 돌아간다.
prism/kt의 티어 배치는 자체 explicit binding이라 영향받지 않는다(warm=node1, cold=2노드 shard 유지).

여전히 부족하면: `numactl --interleave=all -- ./scripts/glm53/run_glm53_prism.sh ...`
(fulls를 두 노드에 균등 분산 → 503 GB 전부 사용). page cache 압박은
`python3 scratchpad/fadv_loop.py`(30초 주기 `posix_fadvise(DONTNEED)`)로 누른다 — 224 GiB가 즉시 회수된다.

## 2. 진행 신호
`Detected fp8 checkpoint.` → `quant_method=PrismMoEMethod` → `[prism] plan loaded (… sparsity=k2wl2)` →
`Loading safetensors checkpoint shards … 62/62` (순차, ~5.5 s/shard) → `[prism] layer N registered (…, thr=…)` ×42 →
`Capture cuda graph end` → `The server is fired up and ready to roll!`
메모리 궤적은 `~/glm53_mem.log`(10초 샘플: used/avail/cache/node0/node1/RSS).

## 3. 벤치
```bash
PATH=~/miniconda3/envs/prism-glm/bin:$PATH \
python ~/prism-sglang/scripts/dsv4/bench_dsv4.py --port 30112 --prefill 672 2048 --decode-tokens 96 --repeat 2
```

## 4. sm_120 관련 알려진 사항
- DSA 백엔드는 **tilelang 하나뿐**(trtllm=Unsupported architecture, flashmla_*=SM90,
  flashinfer_sparse_mla는 `index_kpool>1`(GLM=4)에서 불가).
- 기본 타일이 sm_120의 optin shared memory(~100 KB)를 넘겨서, 트리에 패치를 넣었다:
  `kernels/ops/attention/dsa/tilelang_kernel.py`에서 optin < 120 KB이면 `block_I=32, num_stages=1, threads=128`.
  `block_I`만 내리면 `Layout infer conflict between m_i and alpha`로 컴파일 실패한다.
- MTP/NEXTN은 끈다 — plan이 layer 0..44만 덮는다(MTP는 layer 45). 쓰려면 plan의 `num_layers`를 46으로.
- `models/clip` import 실패와 torchcodec `.so` 경고는 무해(각각 cutlass-dsl/영상 디코더 문제).
