#!/usr/bin/env bash
# DeepSeek-V4-Flash + Prism(mxfp4 hot/warm GPU 스트리밍) 서버 기동.
# 사용: run_dsv4_prism.sh <plan.json> [port] [extra sglang args...]
set -euo pipefail
PLAN=${1:?plan json}; PORT=${2:-30111}; shift 2 || true
# env/모델은 머신마다 다르므로 덮어쓸 수 있게 둔다 (기본값은 원래 머신 그대로).
#   nutella3(RTX 5090, NAS 대신 로컬 NVMe): CONDA_ENV=prism-e2e MODEL=~/models/DeepSeek-V4-Flash
CONDA_ENV=${CONDA_ENV:-ktsglang}
export PATH=$HOME/miniconda3/envs/$CONDA_ENV/bin:$PATH
# kt_kernel 확장이 conda의 libhwloc/libnuma에 링크돼 있어 런타임 경로가 필요하다.
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_2604_SUBMODE=2604B
export FLASHINFER_CUDA_ARCH_LIST=12.0a
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_PRISM_PLAN=$PLAN
export PYTHONUNBUFFERED=1
export SGLANG_PRISM_MAX_TOKENS=${SGLANG_PRISM_MAX_TOKENS:-4096}
MODEL=${MODEL:-/mnt/nas/um3maru/models/DeepSeek-V4-Flash}
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
  "$@"
