#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
command -v conda >/dev/null || { echo 'conda is required'; exit 1; }
export CONDA_PKGS_DIRS="$EXT_ROOT/cache/conda_pkgs"
ENV="$EXT_ROOT/envs/kgmodels"
if [[ ! -x "$ENV/bin/python" ]]; then
  conda create -y -p "$ENV" python=3.10 pip
fi
# The system CUDA 11.5 compiler cannot compile reliably for Ada / torch cu118.
# Install compiler/runtime headers only, not the multi-GB profiler/GUI toolkit.
# PyTorch's cu118 wheel supplies its own numerical runtime libraries.
conda install -y -p "$ENV" -c nvidia/label/cuda-11.8.0 cuda-nvcc=11.8 cuda-cudart-dev=11.8 cuda-cccl=11.8
# torch 2.1 cpp_extension imports pkg_resources.packaging; keep its provider.
"$ENV/bin/python" -m pip install 'setuptools==69.5.1' 'numpy==1.26.4'
"$ENV/bin/python" -m pip install 'torch==2.1.0' --index-url https://download.pytorch.org/whl/cu118
"$ENV/bin/python" -m pip install 'torch-scatter==2.1.2' 'torch-sparse==0.6.18' --only-binary=:all: -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
"$ENV/bin/python" -m pip install 'torch-geometric==2.4.0' 'transformers==4.38.2' 'tokenizers==0.15.2' 'huggingface-hub==0.23.5' 'safetensors==0.4.5' 'easydict==1.13' ninja pyyaml jinja2 scipy scikit-learn tqdm packaging
"$ENV/bin/python" -m pip freeze > "$EXT_ROOT/logs/server_pip_freeze.txt"
echo "Environment ready. Next: bash scripts/preflight.sh"
