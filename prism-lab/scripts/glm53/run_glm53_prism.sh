#!/usr/bin/env bash
# GLM-5.3-Flash (fp8 e4m3 blockwise) + Prism 3-tier 서버 기동. nutella3 / RTX 5090(sm_120) 전용.
# 사용: run_glm53_prism.sh <plan.json> [port] [extra sglang args...]
#
# sm_120에서 필요한 세 가지 (issue #37105 해결 코멘트 + PR #36507 코멘트에서 확인된 것):
#   1) SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 — sm120은 DeepGEMM이 강제 off인데 mhc_pre가
#      무조건 deep_gemm 분기를 타서 NameError로 죽는다.
#   2) --dsa-{prefill,decode}-backend tilelang — sm120에서 되는 DSA 백엔드는 tilelang 하나뿐이다
#      (trtllm=Unsupported architecture, flashmla_*=SM90, flashinfer_sparse_mla는 index_kpool>1 불가).
#   3) tilelang 타일 재튜닝 — 트리 쪽 패치로 처리했다 (block_I 32 / num_stages 1 / threads 128,
#      optin shared memory < 120 KB인 장치에서만).
# MTP/NEXTN은 끈다: prism plan이 layer 0..44만 덮는다 (MTP layer 45는 커버리지 밖).
set -euo pipefail
PLAN=${1:?plan json}; PORT=${2:-30112}; shift 2 || true
CONDA_ENV=${CONDA_ENV:-prism-glm}
TREE=${TREE:-/home/um3maru/prism-sglang/sglang-glm}
export PATH=$HOME/miniconda3/envs/$CONDA_ENV/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# tilelang JIT의 CUDA 툴킷: pip 휠 조합이 깨져 있다 (nvidia-cuda-nvcc 13.3 + nvidia-cuda-runtime 13.0
# → CCCL 가드가 "CUDA compiler and CUDA toolkit headers are incompatible"로 컴파일을 막는다).
# tilelang env.py는 CUDA_HOME을 먼저 보므로 자기 정합인 시스템 13.1을 가리킨다 (격리 검증 완료).
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.1}
export PATH=$CUDA_HOME/bin:$PATH
# 스케줄러의 GPU-local NUMA strict 바인딩을 끈다 — prism의 CPU full 텐서 305 GB가 node1(251 GB)에 안 들어간다.
export SGLANG_AUTO_NUMA_BIND=${SGLANG_AUTO_NUMA_BIND:-0}
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0a
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export SGLANG_PRISM_PLAN=$PLAN
export PYTHONUNBUFFERED=1
export SGLANG_PRISM_MAX_TOKENS=${SGLANG_PRISM_MAX_TOKENS:-2048}
export SGLANG_PRISM_CPUINFER_THREADS=${SGLANG_PRISM_CPUINFER_THREADS:-14}
MODEL=${MODEL:-/home/jun/models/GLM-5.3-Flash}
# fulls를 두 NUMA 노드에 균등 분산 (node1에 warm 23.6 GiB 자리를 남긴다) +
# 순차 로더 (멀티스레드 로더는 샤드 버퍼로 피크를 밀어올린다).
exec numactl --interleave=all -- python -m sglang.launch_server \
  --model-loader-extra-config '{"enable_multithread_load": false}' \
  --host 127.0.0.1 --port "$PORT" \
  --model-path "$MODEL" \
  --tensor-parallel-size 1 \
  --context-length 16384 \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --kv-cache-dtype bfloat16 \
  --linear-attn-backend triton \
  --mem-fraction-static "${MEM_FRAC:-0.85}" \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
  --max-running-requests 1 \
  --watchdog-timeout 3600 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --cuda-graph-bs 1 \
  --cuda-graph-max-bs 1 \
  --disable-radix-cache \
  --skip-server-warmup \
  --log-level info \
  "$@"
