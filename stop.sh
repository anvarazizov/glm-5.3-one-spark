#!/usr/bin/env bash
set -euo pipefail
CONTAINER="${CONTAINER:-glm53-one-spark}"
docker rm -f "$CONTAINER"
