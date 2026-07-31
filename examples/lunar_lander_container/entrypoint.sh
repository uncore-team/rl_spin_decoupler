#!/usr/bin/env bash
set -e

echo "== GPU (nvidia-smi) =="
nvidia-smi || echo "nvidia-smi not available inside the container"

python3 -c "import torch; print('torch CUDA available:', torch.cuda.is_available(), \
'| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# The RL server always binds to 0.0.0.0 (every interface) by design, which
# works unmodified whether it runs in a container or on a bare host. Extra
# args are forwarded to the RL script, e.g. --timesteps, --device, --port.
exec python3 /app/examples/lunar_lander/rl_side_lunarlander.py "$@"
