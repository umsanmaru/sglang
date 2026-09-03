#!/usr/bin/env bash
# Qwen3.8-Flash-Next (bf16 MoE) + Prism 3-tier 서버 기동. nutella3 / RTX 5090(sm_120) 전용.
# 사용: run_qwen38fn_prism.sh <plan.json|stock> [port] [extra sglang args...]
#
# 레인 B: 트리는 `sglang-qwen`(브랜치 prism-qwen = PR #36787 base + prism 이식)이고
# env는 **GLM 레인의 prism-glm을 그대로 쓴다** — 스택 삼중항이 같고(torch 2.13.0+cu130,
# kt-kernel 0.7.0), PYTHONPATH가 editable finder보다 우선하므로 pip 재설치가 필요 없다.
#
# 이 모델에서 자동으로 켜지는 것들 (arg_groups/overrides.py::_qwen4_exp_overrides):
#   - ple_offload_embedding=True (bf16 + CUDA) → PLE 임베딩 테이블이 host로 내려간다.
#     2소켓 호스트라 한 노드가 마르지 않게 SGLANG_PLE_OFFLOAD_NUMA_INTERLEAVE=1을 준다.
#   - page_size=64 (compressed QSA는 full_slot//ratio 주소지정이라 page 정렬이 필수)
#     → MambaRadixCache가 page>1을 mamba extra-buffer 전략에서만 받으므로 radix cache를 끈다.
#   - QSA decode/MQA 백엔드는 `auto`가 sm_120에서 Triton을 고른다 (env로 덮을 수 있다).
set -euo pipefail
PLAN=${1:?plan json 또는 'stock'}; PORT=${2:-30150}; shift 2 || true
# plan 경로는 cd 전에 절대화한다 (아래에서 트리로 cd 하므로 상대 경로가 깨진다).
if [[ "$PLAN" != "stock" ]]; then
  [[ -f "$PLAN" ]] || { echo "plan 없음: $PLAN" >&2; exit 1; }
  PLAN=$(readlink -f "$PLAN")
fi
CONDA_ENV=${CONDA_ENV:-prism-glm}
TREE=${TREE:-/home/um3maru/prism-sglang/sglang-qwen}
export PATH=$HOME/miniconda3/envs/$CONDA_ENV/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TREE/python${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.1}
export PATH=$CUDA_HOME/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=12.0a
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export PYTHONUNBUFFERED=1
# 스케줄러의 GPU-local NUMA strict 바인딩을 끈다 — prism의 CPU full 텐서(층당 5.03 GiB × 48)가
# 한 노드(251 GiB)에 안 들어간다. PLE 테이블도 같은 이유로 노드 간 interleave.
export SGLANG_AUTO_NUMA_BIND=${SGLANG_AUTO_NUMA_BIND:-0}
export SGLANG_PLE_OFFLOAD_NUMA_INTERLEAVE=${SGLANG_PLE_OFFLOAD_NUMA_INTERLEAVE:-1}
MODEL=${MODEL:-$HOME/models/Qwen3.8-Flash-Next}

# cwd 방어: ~/prism-sglang 에는 `sglang/`(포크 워크트리) 디렉터리가 있어서 그 자리에서
# python을 띄우면 `import sglang`이 namespace package로 잡히고 sglang/__init__.py가
# 실행되지 않는다 (서브모듈은 우연히 import돼 더 위험하다). 트리 안으로 들어가서 띄운다.
cd "$TREE"

if [[ "$PLAN" != "stock" ]]; then
  export SGLANG_PRISM_PLAN=$PLAN
  export SGLANG_PRISM_MAX_TOKENS=${SGLANG_PRISM_MAX_TOKENS:-2048}
  export SGLANG_PRISM_CPUINFER_THREADS=${SGLANG_PRISM_CPUINFER_THREADS:-14}
  echo "== prism MoE: $SGLANG_PRISM_PLAN"
else
  echo "== stock (prism 미개입)"
fi
python -c "import sglang; print('== tree:', sglang.__file__)"

# 순차 로더: 멀티스레드 로더는 샤드 버퍼가 스레드마다 잡혀 host 피크를 밀어올린다
# (GLM 실측: 멀티스레드 54/62에서 OOM kill, 순차는 통과). mmap은 끄지 말 것.
exec numactl --interleave=all -- python -m sglang.launch_server \
  --model-loader-extra-config '{"enable_multithread_load": false}' \
  --host 127.0.0.1 --port "$PORT" \
  --model-path "$MODEL" \
  --tensor-parallel-size 1 \
  --context-length "${CTX:-8192}" \
  --mem-fraction-static "${MEM_FRAC:-0.85}" \
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
  "$@"
