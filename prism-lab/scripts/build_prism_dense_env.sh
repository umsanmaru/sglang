#!/usr/bin/env bash
# dense 레인 env: upstream main 핀. `build_prism_glm_env.sh`의 복제이고 ENV/TREE만 다르다.
#
# 트리를 나눈 이유(2026-09-01): dense는 Muse-Glimmer-30B가 타깃인데 그 모델 파일이
# upstream main에만 있다(포크는 7,496커밋 뒤처져 있다). GLM 레인은 pr-36507 기준이고
# 거기엔 GLM 전용 tilelang 패치가 섞여 있어 dense에는 불필요하므로 main으로 새로 판다.
#
#   git worktree add /home/um3maru/prism-sglang/sglang-dense -b prism-dense upstream/main
#   DST=/home/um3maru/prism-sglang/sglang-dense bash scripts/port_prism_to_upstream.sh
#   bash scripts/build_prism_dense_env.sh          # ← 이 파일
#
# env를 따로 파는 이유: 포크의 editable 설치가 경로를 하드코딩한다
# (site-packages의 __editable__ finder가 /home/um3maru/prism-sglang/sglang/python/sglang).
# 같은 env를 쓰면 worktree를 파도 파이썬은 계속 포크를 본다.
set -euo pipefail
CONDA=$HOME/miniconda3
ENV=prism-dense
TREE=/home/um3maru/prism-sglang/sglang-dense
$CONDA/bin/conda create -y -n $ENV python=3.12 || true
export PATH=$CONDA/envs/$ENV/bin:$PATH
# kt-kernel이 링크하는 hwloc/numa (kt-kernel-build-env 메모와 동일)
# libxml2-devel이 필요한 이유: conda의 `hwloc.pc`가 `Requires.private: libxml-2.0`인데
# 그 `.pc`는 libxml2 본체가 아니라 devel 패키지가 준다. 없으면 pkg-config가 **시스템**
# libxml-2.0.pc로 폴백하고, 그쪽 `prefix=/usr`가 conda의 것과 섞여
# `/usr/lib/include/libxml2`라는 없는 경로가 나와 kt-kernel의 CMake가 죽는다
# (2026-09-01 실측: "Imported target PkgConfig::HWLOC includes non-existent path").
$CONDA/bin/conda install -y -n $ENV -c conda-forge libhwloc libnuma numactl pkg-config ninja cmake libxml2-devel
pip install -U pip wheel setuptools
# upstream 트리의 핀을 그대로 받는다
# Rust 확장(gRPC/mm 게이트웨이)은 필요 없다 — cargo 없이 설치한다.
SGLANG_BUILD_RUST_EXTS=none pip install -e "$TREE/python"
python -c "import torch;print('torch',torch.__version__,'archs',torch.cuda.get_arch_list())"
pip list | grep -iE "^(torch|triton|flashinfer|tilelang|transformers|sglang|apache-tvm-ffi|nvidia-cutlass)" || true

# ── 게이트 (Phase 0.4) ──────────────────────────────────────────────────────
# 이식이 성립했는지 여기서 확인한다. 하나라도 실패하면 이후 단계가 무의미하다.
echo "== 게이트 =="
python -c "import sglang.srt.models.muse_glimmer; print('  muse_glimmer import OK')"
python -c "
from sglang.srt.configs.muse_glimmer import MuseGlimmerConfig
import inspect
d = inspect.signature(MuseGlimmerConfig.__init__).parameters['use_attn_output_gate'].default
print(f'  use_attn_output_gate 기본값 = {d}')"
cd "$TREE" && python -m pytest test/prism -q \
  --ignore=test/prism/bench_cold_cpu.py --ignore=test/prism/bench_full_layer.py \
  --ignore=test/prism/bench_gpu_dense_gemv.py --ignore=test/prism/bench_warm_cold_sparse.py \
  2>&1 | tail -3
