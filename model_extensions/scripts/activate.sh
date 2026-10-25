#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="$EXT_ROOT/envs/kgmodels"
[[ -x "$ENV/bin/python" ]] || { echo 'Run bash scripts/setup_server.sh first'; exit 1; }
export PATH="$ENV/bin:$PATH"
export CUDA_HOME="$ENV"
# CUDAContext includes numerical-library headers shipped with CUDA PyTorch wheels.
# Support both monolithic torch wheels and split nvidia/* wheels.
WHEEL_INCLUDES="$("$ENV/bin/python" -c 'import pathlib,site; paths=[]; [(paths.extend([p / "torch" / "include"]), paths.extend((p / "nvidia").glob("*/include"))) for p in map(pathlib.Path,site.getsitepackages())]; print(":".join(str(p) for p in paths if p.is_dir()))')"
export CPATH="$ENV/include:$WHEEL_INCLUDES${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$ENV/lib:$ENV/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
