#!/usr/bin/env bash
# DeepSeek-V4-Flash + 순정 kt (CPU experts, prism 미개입) 기동 — 로딩 시간 비교용.
# prism과 달리 SGLANG_PRISM_PLAN을 설정하지 않는다.
set -euo pipefail
PORT=${1:-30112}; shift || true
CONDA_ENV=${CONDA_ENV:-prism-e2e}
export PATH=$HOME/miniconda3/envs/$CONDA_ENV/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_2604_SUBMODE=2604B
export FLASHINFER_CUDA_ARCH_LIST=12.0a
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export SGLANG_DISABLE_CUDNN_CHECK=1
export PYTHONUNBUFFERED=1
MODEL=${MODEL:-/home/um3maru/models/DeepSeek-V4-Flash}
exec python -m sglang.launch_server \
  --host 127.0.0.1 --port "$PORT" \
  --model-path "$MODEL" \
  --tensor-parallel-size 1 \
  --context-length 16384 \
  --attention-backend flashinfer \
  --mem-fraction-static "${MEM_FRAC:-0.80}" \
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
  --kt-method MXFP4 \
  --kt-weight-path "$MODEL" \
  --kt-cpuinfer "${KT_CPUINFER:-14}" \
  --kt-threadpool-count 2 \
  "$@"
