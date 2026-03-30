#!/usr/bin/env bash
set -euo pipefail

docker compose \
  --profile sentiment \
  --profile topic \
  --profile linux \
  stop "$@"
