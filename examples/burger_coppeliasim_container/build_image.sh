#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

exec docker build -f "$SCRIPT_DIR/Dockerfile" -t burger-coppelia-rl "$REPO_ROOT"
