#!/usr/bin/env bash
# Kimi K3 + Prism — H100 ×2 (sm_90) 기동. nutella3용 run_k3_dummy_prism.sh의 H100 판.
#
# nutella3(5090/sm_120)와 다른 점:
#   - TORCH_CUDA_ARCH_LIST / FLASHINFER_CUDA_ARCH_LIST = 9.0 (sm_120 → sm_90)
#   - TP 기본 = 보이는 GPU 수. prism은 MoE-TP owner(rank 0)만 routed expert를 갖고 나머지는 0을 낸다.
#   - tilelang/DSA 관련 sm120 우회 없음 (K3는 DSA를 안 쓴다 — KDA + MLA)
#
# 사용:
#   MODEL=/path/to/Kimi-K3 run_k3_h100.sh <plan.json> [port] [extra args...]
#   DUMMY=1 MODEL=/path/to/Kimi-K3-12L run_k3_h100.sh <plan.json>     # 가중치 없이 경로/속도만
#   TP=2 CUDA_VISIBLE_DEVICES=0,1 ... run_k3_h100.sh <plan.json>       # GPU를 좁힐 때
set -euo pipefail
PLAN=${1:?plan json}; PORT=${2:-30113}; shift 2 || true

CONDA_ENV=${CONDA_ENV:-prism-k3}
# 이 스크립트는 <repo>/scripts/prism_k3/ 에 있다 → TREE 는 그 두 단계 위 (체크아웃 이름 무관).
TREE=${TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export PATH=$HOME/miniconda3/envs/$CONDA_ENV/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# editable install이 다른 트리를 가리켜도 이 트리가 이긴다 (PathFinder 순서).
export PYTHONPATH=$TREE/python${PYTHONPATH:+:$PYTHONPATH}

# TP 기본 = 이 프로세스에 보이는 GPU 수. prism은 rank 0이 routed expert를 전부 소유하므로
# TP가 크면 dense가 rank당 얇아져 rank 0의 hot 예산이 커진다 (런북 머리말 표). 좁히려면
# CUDA_VISIBLE_DEVICES=0,1 처럼 주거나 TP=2 로 직접 지정한다.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  _NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
  _NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  [ "${_NGPU:-0}" -ge 1 ] || _NGPU=1
fi
TP=${TP:-$_NGPU}
echo "[run_k3_h100] TP=$TP (visible GPUs: $_NGPU)" >&2
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"9.0+PTX"}
export FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-9.0}

# 스케줄러의 GPU-local NUMA strict 바인딩을 끈다 — prism의 CPU full 텐서가 한 노드에 안 들어간다.
export SGLANG_AUTO_NUMA_BIND=${SGLANG_AUTO_NUMA_BIND:-0}
export SGLANG_PRISM_PLAN=$PLAN
export PYTHONUNBUFFERED=1
export SGLANG_PRISM_MAX_TOKENS=${SGLANG_PRISM_MAX_TOKENS:-2048}
# 물리 코어 − 2 가 기본. 과다구독은 kt submit/sync 고정비를 폭증시킨다 (실측: 16코어에 60스레드 → 1.85 ms/회).
export SGLANG_PRISM_CPUINFER_THREADS=${SGLANG_PRISM_CPUINFER_THREADS:-$(( $(nproc) / 2 - 2 ))}

MODEL=${MODEL:?MODEL=<checkpoint dir> 를 지정할 것}
EXTRA=()
if [ "${DUMMY:-0}" = 1 ]; then
  EXTRA+=(--load-format dummy)
else
  # prism은 층당 full expert 텐서를 CPU에 잡고, 해제는 모든 샤드 로딩이 끝난 뒤에 층별로 일어난다.
  # K3는 그 피크가 ~1.45 TB라 멀티스레드 로더의 샤드 버퍼가 얹히면 OOM killer가 온다 (GLM에서 실측).
  EXTRA+=(--model-loader-extra-config '{"enable_multithread_load": false}')
fi

exec numactl --interleave=all -- python -m sglang.launch_server \
  --host 127.0.0.1 --port "$PORT" \
  --model-path "$MODEL" \
  --tensor-parallel-size "$TP" \
  --context-length "${CTX:-8192}" \
  --attention-backend "${ATTN_BACKEND:-flashinfer}" \
  --linear-attn-backend triton \
  --mem-fraction-static "${MEM_FRAC:-0.90}" \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
  --max-running-requests 1 \
  --watchdog-timeout 7200 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --cuda-graph-bs 1 \
  --cuda-graph-max-bs 1 \
  --disable-radix-cache \
  --skip-server-warmup \
  --log-level info \
  "${EXTRA[@]}" \
  "$@"
