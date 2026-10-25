#!/usr/bin/env bash
source "$(dirname "$0")/activate.sh"
exec > >(tee "$EXT_ROOT/logs/server_preflight.log") 2>&1
nvidia-smi
nvcc --version
python -c 'from pkg_resources import packaging; from torch.utils import cpp_extension; print("C++ extension Python dependencies OK")'
python -c 'import torch,transformers,torch_geometric,torch_scatter; print(torch.__version__,torch.version.cuda,transformers.__version__,torch_geometric.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name()); assert torch.version.cuda == "11.8"'
python "$EXT_ROOT/scripts/smoke_models.py" --model compounde
python "$EXT_ROOT/scripts/smoke_models.py" --model simkgc
python "$EXT_ROOT/scripts/smoke_models.py" --model ultra
