# DeepSeek-V4-Flash × Prism (MXFP4 3-tier) 실행 런북 (2026-08-28)

## 0. 전제
- 체크아웃: `~/prism-sglang/sglang`(branch `prism-orchestration`, editable install), `~/prism-sglang/ktransformers`(같은 브랜치).
  필요 커밋: sglang ≥ `b25d0d53d5`, kt ≥ `f92f155` (`git log --oneline -1`로 확인).
- Python: `~/miniconda3/envs/ktsglang` — **셸 PATH에 `~/miniconda3/envs/ktsglang/bin`을 앞에 넣어야** jit(ninja)·kt가 잡힌다.
- kt-kernel은 위 커밋으로 이미 빌드·설치돼 있다. kt 소스를 바꾸면 재빌드(~2분):
  ```bash
  cd ~/prism-sglang/ktransformers/kt-kernel && export PATH=~/miniconda3/envs/ktsglang/bin:$PATH
  CPUINFER_CPU_INSTRUCT=AVX512 CPUINFER_ENABLE_AMX=ON CPUINFER_ENABLE_AVX512_VNNI=ON \
  CPUINFER_ENABLE_AVX512_BF16=ON CPUINFER_ENABLE_AVX512_VBMI=ON CPUINFER_USE_CUDA=1 \
  pip install --no-build-isolation --no-deps .
  ```
  확인: `python -c "from kt_kernel import kt_kernel_ext as k; print(hasattr(k.moe.TileK2MXFP4_MOE,'forward_gateup_partial'))"` → True
- 모델: `/mnt/nas/um3maru/models/DeepSeek-V4-Flash` (NAS, ~100 MB/s → 로딩 45–50분은 네트워크 하한).
- 공유 머신: **기동 전 반드시** `nvidia-smi`(GPU0 비어 있어야; GPU1 H100은 타인 vLLM), `free -g`, `ps -eo pid,user,etime,cmd | grep sglang::scheduler`(다른 세션의 서버/벤치)를 확인.

## 1. plan (이미 생성됨, `~/prism-sglang/plans/dsv4f/`)
| 파일 | 구성 | 비고 |
|---|---|---|
| `dsv4f_h125_w125_c750_tile.json` | hot 12.5 / warm 12.5 / cold 75, cold=`kt_tile_k2_mxfp4`, 2 NUMA shard | **기본** (우리 tile 커널, dense) |
| `dsv4f_h125_w125_c750.json` | 같은 비율, cold=`kt_amx_fp4`(kt 원본 fp4 커널) | 비교용 |
| `dsv4f_h125_w125_c750_tile_1node.json` | cold 전량 node 0 | **OOM** (node 128 GB에 103 GiB cold + 17 GiB warm 불가) — 쓰지 말 것 |
| `dsv4f_h125_w875_dense.json`, `dsv4f_h250_w750_dense.json` | 2-tier(hot+warm, pinned 120/103 GiB) | 호스트 pinned 과다 — 비권장 |

새 plan: `python sglang/test/prism/gen_uniform_plan.py <model_dir> <out.json> --hot-frac H --warm-frac W --numa-nodes 2 --gpu-kernel gemv_worklist_mxfp4 --cpu-kernel kt_tile_k2_mxfp4 --k-align 32`
(mxfp4는 K 정렬 32; tile 커널은 노드 N shard가 256 배수여야 한다 — DSV4 치수는 만족.)

## 2. 기동
```bash
cd ~/prism-sglang/sglang
SGLANG_PRISM_MAX_TOKENS=2048 SGLANG_PRISM_CPUINFER_THREADS=6 \
  nohup ~/prism-sglang/scripts/dsv4/run_dsv4_prism.sh \
    ~/prism-sglang/plans/dsv4f/dsv4f_h125_w125_c750_tile.json 30111 > ~/dsv4_server.log 2>&1 &
```
- 스크립트가 넣는 env: `SGLANG_DSV4_MODE=2604 SGLANG_DSV4_2604_SUBMODE=2604B`(swiglu_limit 필수 짝), `FLASHINFER_CUDA_ARCH_LIST=12.0a`, `TORCH_CUDA_ARCH_LIST=12.0+PTX`, `SGLANG_DISABLE_CUDNN_CHECK=1`, `PYTHONUNBUFFERED=1`, `SGLANG_PRISM_PLAN`.
- 인자: `--attention-backend flashinfer --cuda-graph-bs 1 --cuda-graph-max-bs 1 --chunked-prefill-size 2048 --max-running-requests 1 --disable-radix-cache --disable-shared-experts-fusion --mem-fraction-static 0.80`(`MEM_FRAC`로 변경). `--kt-method`/`--kt-weight-path`는 **넘기지 않는다**(prism이 priority 30에서 MoE를 대체).
- 스레드: `SGLANG_PRISM_CPUINFER_THREADS=N` → 2노드에 N/2씩. 소켓 하나만 쓰려면 `SGLANG_PRISM_NUMA_MAP=0` + 1-node plan이지만 c75는 메모리상 불가(위 표).
- 진행 신호(로그): `Loading safetensors … 46/46`(~8분) → 20–25분간 로그 없이 expert 텐서 물질화(NAS 읽기; 스레드 D 상태 정상) → `[prism] layer N registered … take_full/prepare/cold_load` 43줄(~15 s/층) → `Capture cuda graph end` → **`The server is fired up and ready to roll!`**. 총 45–50분.
- 실패 신호: `Traceback`, `scheduler is dead`/`Exit code: -9`(OOM — 메모리 배치 확인), `Segmentation fault`.

## 3. 벤치
```bash
export PATH=~/miniconda3/envs/ktsglang/bin:$PATH
python ~/prism-sglang/scripts/dsv4/bench_dsv4.py --port 30111 --prefill 672 2048 --decode-tokens 96 --repeat 2
```
정성 출력 1개 + decode ms/tok(스트리밍 토큰 간격 중앙값) + prefill TTFT. 기준(2026-08-28, tile, 6 thr): decode 92–95 ms/tok, TTFT 672/2048 tok 2.57/4.12 s.
층 단위 벤치(합성 가중치, 서버 없이): `python ~/prism-sglang/scripts/dsv4/bench_layer_mxfp4.py <plan.json> --m 1 2048`.

## 4. 종료
```bash
~/prism-sglang/scripts/dsv4/stop_dsv4.sh 30111
```
(`pkill -f "port 30111"`처럼 자기 명령줄에 패턴이 들어가면 자기 셸을 죽인다 — 스크립트는 awk로 pid를 뽑는다.)

## 5. 알려진 사항
- sparsity(k2wl2)는 **미적용(dense)** — DSV4용 calib 자산이 없다. 기계는 준비됨(GPU sparse GEMV, tile 커널 Seam B). calib이 생기면 `gen_uniform_plan.py --calib <pt> --target-p 0.5`.
- 테스트: `CUDA_VISIBLE_DEVICES=0 SGLANG_PRISM_CPUINFER_THREADS=4 python -m pytest test/prism -q -p no:cacheprovider` (sglang 디렉터리; kt_kernel 필요).
- 결과/원인 기록: `~/prism-sglang/docs/superpowers/plans/2026-08-28-dsv4-flash-mxfp4-results.md`, 계획: `…/2026-08-27-dsv4-flash-mxfp4-port.md`.
