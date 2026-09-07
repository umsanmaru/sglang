#!/usr/bin/env bash
# Kimi K3 (12층 절단 config, dummy 가중치) + Prism mxfp4 3-tier — nutella3 / RTX 5090(sm_120) 경로 검증용.
# 체크포인트 없이 K3 모델 코드(KDA/MLA/latent MoE/situ) → FusedMoE → PrismMoEMethod 경로가 e2e로 도는지,
# 그리고 층당 decode 시간을 재기 위한 하니스다. 정확도는 의미 없다(랜덤 가중치).
# 사용: run_k3_dummy_prism.sh <plan.json> [port] [extra sglang args...]
set -euo pipefail
PLAN=${1:?plan json}; PORT=${2:-30113}; shift 2 || true
CONDA_ENV=${CONDA_ENV:-prism-glm}
TREE=${TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export PATH=$HOME/miniconda3/envs/$CONDA_ENV/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# prism-glm env의 editable sglang은 sglang-glm을 가리킨다 — PYTHONPATH가 PathFinder에서 먼저 잡힌다.
export PYTHONPATH=$TREE/python${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.1}
export PATH=$CUDA_HOME/bin:$PATH
export SGLANG_AUTO_NUMA_BIND=${SGLANG_AUTO_NUMA_BIND:-0}
export FLASHINFER_CUDA_ARCH_LIST=12.0a
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export SGLANG_PRISM_PLAN=$PLAN
export PYTHONUNBUFFERED=1
export SGLANG_PRISM_MAX_TOKENS=${SGLANG_PRISM_MAX_TOKENS:-2048}
export SGLANG_PRISM_CPUINFER_THREADS=${SGLANG_PRISM_CPUINFER_THREADS:-14}
MODEL=${MODEL:-/home/um3maru/prism-sglang/modelcfg/Kimi-K3-12L}
exec numactl --interleave=all -- python -m sglang.launch_server \
  --host 127.0.0.1 --port "$PORT" \
  --model-path "$MODEL" \
  --load-format dummy \
  --tensor-parallel-size "${TP:-1}" \
  --context-length 8192 \
  --attention-backend "${ATTN_BACKEND:-flashinfer}" \
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
