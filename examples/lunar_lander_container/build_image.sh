#!/usr/bin/env bash
set -e
# Build a RL-side image with GPU.
#
# Usage:
#   ./build_image.sh
#
# Extra args are forwarded to docker, e.g.:
#   ./build_image.sh --no-cache

# Build context is the REPO ROOT so the image can COPY spindecoupler + example.
# Run this script from the repository root.
docker build "$@" -f examples/lunar_lander_container/Dockerfile -t lunarlander-rl .
