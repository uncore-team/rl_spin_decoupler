#!/usr/bin/env bash
# Container entry point: report the GPU, then run the RL side. Arguments are
# forwarded to rl_side_burger_coppeliasim.py (e.g. --timesteps, --device, --port).
set -euo pipefail

# The repository root: /app in the image, but not hard-coded so that the script
# also works from a checkout.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$APP_DIR:${PYTHONPATH:-}"

echo "== GPU (nvidia-smi) =="
nvidia-smi || echo "nvidia-smi not available (no GPU passed to the container)"

python3 -c "import torch; print('torch CUDA available:', torch.cuda.is_available(), \
'| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# The RL server always binds to 0.0.0.0 (every interface) by design, which works
# unchanged in a container and on a bare host.
exec python3 "$APP_DIR/examples/burger_coppeliasim/rl_side_burger_coppeliasim.py" "$@"
