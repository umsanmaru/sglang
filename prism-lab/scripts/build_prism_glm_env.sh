#!/usr/bin/env bash
# 레인 B/C 용 env: upstream main 핀(torch 2.13 / sglang-kernel 0.4.6.post1 / flashinfer 0.6.18 / tilelang 0.1.12)
set -euo pipefail
CONDA=$HOME/miniconda3
ENV=prism-glm
TREE=/home/um3maru/prism-sglang/sglang-glm
$CONDA/bin/conda create -y -n $ENV python=3.12 || true
export PATH=$CONDA/envs/$ENV/bin:$PATH
# kt-kernel이 링크하는 hwloc/numa (kt-kernel-build-env 메모와 동일)
$CONDA/bin/conda install -y -n $ENV -c conda-forge libhwloc libnuma numactl pkg-config ninja cmake
pip install -U pip wheel setuptools
# upstream 트리의 핀을 그대로 받는다
# Rust 확장(gRPC/mm 게이트웨이)은 필요 없다 — cargo 없이 설치한다.
SGLANG_BUILD_RUST_EXTS=none pip install -e "$TREE/python"
python -c "import torch;print('torch',torch.__version__,'archs',torch.cuda.get_arch_list())"
pip list | grep -iE "^(torch|triton|flashinfer|tilelang|transformers|sglang|apache-tvm-ffi|nvidia-cutlass)" || true
