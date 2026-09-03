#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 (DSpark 변종) + Prism mxfp4 3-tier 서버 기동. nutella3 전용.
# 사용: run_dsv4_0731_prism.sh <plan.json> [port] [extra sglang args...]
#
# 기존 run_dsv4_prism.sh와 다른 점 셋:
#   1) MODEL 기본값이 로컬 NVMe의 0731 체크포인트
#   2) SGLANG_APPLY_CONFIG_BACKUP=none — 포크가 패키징한 config_backup_small.json은
#      0731과 맞지 않는다 (compress_ratios 44 vs 46, dspark_* 필드 없음, expert_dtype 누락).
#      백업을 적용하면 조용히 다른 모델을 만든다.
#   3) CPUINFER_THREADS 기본 14 (소켓당 7 — 물리 8코어 중 1개는 GPU 구동/스핀용으로 남긴다)
set -euo pipefail
PLAN=${1:?plan json}; PORT=${2:-30111}; shift 2 || true
CONDA_ENV=${CONDA_ENV:-prism-e2e}
export PATH=$HOME/miniconda3/envs/$CONDA_ENV/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SGLANG_APPLY_CONFIG_BACKUP=${SGLANG_APPLY_CONFIG_BACKUP:-none}
export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_2604_SUBMODE=2604B
export FLASHINFER_CUDA_ARCH_LIST=12.0a
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_PRISM_PLAN=$PLAN
export PYTHONUNBUFFERED=1
export SGLANG_PRISM_MAX_TOKENS=${SGLANG_PRISM_MAX_TOKENS:-2048}
export SGLANG_PRISM_CPUINFER_THREADS=${SGLANG_PRISM_CPUINFER_THREADS:-14}
MODEL=${MODEL:-$HOME/models/DeepSeek-V4-Flash-0731}
exec python -m sglang.launch_server \
  --host 127.0.0.1 --port "$PORT" \
  --model-path "$MODEL" \
  --tensor-parallel-size 1 \
  --context-length 16384 \
  --attention-backend flashinfer \
  --mem-fraction-static "${MEM_FRAC:-0.92}" \
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
