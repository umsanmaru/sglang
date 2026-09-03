#!/usr/bin/env bash
# Qwen3.8-27B 서버 기동 — stock 또는 Prism dense.
#
#   PLAN 없이  →  stock (prism 미개입). Phase 1의 기준선이다.
#   PLAN 지정  →  dense prism. env `SGLANG_PRISM_LINEAR_PLAN`으로만 켜진다
#                 (MoE의 SGLANG_PRISM_PLAN과 **독립** — 이 모델은 dense라 MoE는 안 쓴다).
#
# 사용:
#   scripts/qwen38/run_qwen38.sh                                   # stock
#   scripts/qwen38/run_qwen38.sh plans/qwen38/dense_h125_w125.json # prism dense
#   PORT=30222 CPUINFER=14 scripts/qwen38/run_qwen38.sh <plan>     # cold 배선 후
#
# 트리/env: dense 레인은 `sglang-dense`(upstream/main) + `prism-dense` env다.
# 포크의 editable 설치가 경로를 하드코딩하므로 env를 나누지 않으면 파이썬이
# 계속 포크를 본다 (build_prism_dense_env.sh 주석 참조).
set -euo pipefail
PLAN=${1:-}
PORT=${PORT:-30140}
CONDA_ENV=${CONDA_ENV:-prism-dense}
export PATH=$HOME/miniconda3/envs/$CONDA_ENV/bin:$PATH
# kt_kernel 확장이 conda의 libhwloc/libnuma에 링크돼 있어 런타임 경로가 필요하다.
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# RTX 5090 (sm_120). DSV4 스크립트와 같은 값.
export FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-12.0a}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"12.0+PTX"}
export SGLANG_DISABLE_CUDNN_CHECK=1
export PYTHONUNBUFFERED=1
MODEL=${MODEL:-$HOME/models/Qwen3.8-27B}

if [[ -n "$PLAN" ]]; then
  [[ -f "$PLAN" ]] || { echo "plan 없음: $PLAN" >&2; exit 1; }
  export SGLANG_PRISM_LINEAR_PLAN=$(readlink -f "$PLAN")
  export SGLANG_PRISM_LINEAR_MAX_TOKENS=${SGLANG_PRISM_LINEAR_MAX_TOKENS:-4096}
  # cold(kt)가 배선되면 쓴다. 과다구독은 submit/sync 고정비를 폭증시키므로
  # 물리 코어 수를 넘기지 않는다 (MoE 실측: 16코어에 60스레드 → sync 1.85 ms).
  [[ -n "${CPUINFER:-}" ]] && export SGLANG_PRISM_CPUINFER_THREADS=$CPUINFER
  echo "== prism dense: $SGLANG_PRISM_LINEAR_PLAN"
else
  echo "== stock (prism 미개입)"
fi

# hot 밴드가 VRAM을 먹으므로 static 몫을 낮춘다. stock은 기본값이면 된다.
DEFAULT_MEM_FRAC=0.85; [[ -n "$PLAN" ]] && DEFAULT_MEM_FRAC=0.70
exec python -m sglang.launch_server \
  --host 127.0.0.1 --port "$PORT" \
  --model-path "$MODEL" \
  --tensor-parallel-size 1 \
  --context-length "${CTX:-8192}" \
  --mem-fraction-static "${MEM_FRAC:-$DEFAULT_MEM_FRAC}" \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
  --max-running-requests 1 \
  --watchdog-timeout 3600 \
  --trust-remote-code \
  --cuda-graph-bs 1 \
  --cuda-graph-max-bs 1 \
  --disable-radix-cache \
  --skip-server-warmup \
  --log-level info \
  "${@:2}"
