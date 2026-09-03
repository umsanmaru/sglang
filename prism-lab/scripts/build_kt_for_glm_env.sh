#!/usr/bin/env bash
# kt-kernel을 prism-glm(torch 2.13/cu130)용으로 재빌드. 소스는 포크와 동일 커밋을 쓴다.
set -euo pipefail
ENV=${ENV:-prism-glm}
export PATH=$HOME/miniconda3/envs/$ENV/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CONDA_PREFIX=$HOME/miniconda3/envs/$ENV
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=$CUDA_HOME/bin:$PATH
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_LIBRARY_PATH=$CONDA_PREFIX/lib -DCMAKE_INCLUDE_PATH=$CONDA_PREFIX/include"
cd /home/um3maru/prism-sglang/ktransformers/kt-kernel
git log --oneline -1
CPUINFER_CPU_INSTRUCT=AVX512 CPUINFER_ENABLE_AMX=ON CPUINFER_ENABLE_AVX512_VNNI=ON \
CPUINFER_ENABLE_AVX512_BF16=ON CPUINFER_ENABLE_AVX512_VBMI=ON CPUINFER_USE_CUDA=1 \
  pip install --no-build-isolation --no-deps .
python - <<'PY'
from kt_kernel import kt_kernel_ext as k
names=[n for n in dir(k.moe) if n.endswith('_MOE')]
need=['AMXBF16_MOE','TileK2BF16_MOE','AMXFP4_KGroup_MOE','TileK2MXFP4_MOE','TileK2FP8B128_MOE']
print("cold 연산자:", {n: hasattr(getattr(k.moe,n),'forward_gateup_partial') for n in need if n in names})
PY
