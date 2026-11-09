#!/usr/bin/env bash
# Build the RL-side image. Run it from anywhere; the build context is the
# repository root. Extra arguments go to `docker build`, e.g. --no-cache.
#
# Usage:   ./build_image.sh [docker build options]
# Environment variables: IMAGE (image name, default lunarlander-rl)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repository root

exec docker build "$@" -f examples/lunar_lander_container/Dockerfile \
  -t "${IMAGE:-lunarlander-rl}" .
