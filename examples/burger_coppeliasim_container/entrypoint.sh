#!/usr/bin/env bash
set -e

export PYTHONPATH="/app:${PYTHONPATH}"

echo "== GPU (nvidia-smi) =="
nvidia-smi || echo "nvidia-smi not available inside the container"

python3 -c "import torch; print('torch CUDA available:', torch.cuda.is_available(), \
'| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

exec python3 /app/examples/burger_coppeliasim/rl_side_burger_coppeliasim.py "$@"
