# RL side of the decoupled CoppeliaSim Burger example.
# CoppeliaSim runs with the agent on the local host, so this image needs only
# the RL environment and the communication package.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip3 install --no-cache-dir torch==2.5.1+cu124 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir -r examples/burger_coppeliasim_container/requirements-rl.txt \
    && pip3 install --no-cache-dir -e .

RUN chmod +x /app/examples/burger_coppeliasim_container/entrypoint.sh
ENTRYPOINT ["/app/examples/burger_coppeliasim_container/entrypoint.sh"]
