# RL side (server) of the decoupled LunarLander example, on a GPU-capable
# Ubuntu 22.04 container.
#
# The RL side only needs Stable-Baselines3 + PyTorch (plus gymnasium for its
# spaces/Env base class). LunarLander's Box2D physics runs in the AGENT process
# on the local host, so no swig / gymnasium[box2d] is required in this image.
#
# This image is network-agnostic: it works for both deployment profiles. The
# profile (LAN vs SSH tunnel) is chosen at `docker run` time, not here.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# PyTorch build matching the CUDA 12.4 base image, then the RL-side deps and the
# rl_spin_decoupler package itself (whose core is pure standard library).
RUN pip3 install --no-cache-dir "stable-baselines3>=2.0" "gymnasium>=0.29" \
    && pip3 install --no-cache-dir .

RUN chmod +x /app/examples/lunar_lander_container/entrypoint.sh
ENTRYPOINT ["/app/examples/lunar_lander_container/entrypoint.sh"]
