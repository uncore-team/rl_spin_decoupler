# RL side (server) of the decoupled LunarLander example, on a GPU-capable
# Ubuntu 22.04 container.
#
# The RL side only needs Stable-Baselines3 + PyTorch (plus gymnasium for its
# spaces/Env base class). LunarLander's Box2D physics runs in the AGENT process
# on the local host, so no swig / gymnasium[box2d] is required in this image.
#
# The image is network-agnostic: the profile (LAN vs SSH tunnel) is chosen at
# `docker run` time by run_rl_container.sh, not here. Build with build_image.sh
# from any directory; the build context is the repository root.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so this layer is cached until the requirements change.
# PyTorch is the build matching the CUDA 12.4 base image.
COPY examples/lunar_lander_container/requirements-rl.txt /tmp/requirements-rl.txt
RUN pip3 install --no-cache-dir torch==2.5.1+cu124 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir -r /tmp/requirements-rl.txt

# The library (whose core is pure standard library) and the example sources.
COPY pyproject.toml setup.py README.md LICENSE __init__.py spindecoupler.py spindecoupler_gym.py ./
COPY socketcomms ./socketcomms
COPY examples/lunar_lander ./examples/lunar_lander
COPY examples/lunar_lander_container ./examples/lunar_lander_container
RUN pip3 install --no-cache-dir -e . \
    && chmod +x examples/lunar_lander_container/entrypoint.sh

ENTRYPOINT ["/app/examples/lunar_lander_container/entrypoint.sh"]
