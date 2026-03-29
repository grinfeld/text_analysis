#!/usr/bin/env bash
set -euo pipefail

# ── Build model-server image ───────────────────────────────────────────────────
echo "Building text-analysis/model-server:latest..."
docker build -t text-analysis/model-server:latest ./model_server
echo ""

# ── OS ────────────────────────────────────────────────────────────────────────
echo "OS:"
echo "  1) macOS"
echo "  2) Linux"
read -rp "Choose [1-2]: " os_choice

case "$os_choice" in
  1) OS=macos ;;
  2) OS=linux ;;
  *) echo "Invalid choice: $os_choice"; exit 1 ;;
esac

# ── Models ────────────────────────────────────────────────────────────────────
echo ""
echo "Models:"
echo "  1) Sentiment only"
echo "  2) Topic only"
echo "  3) Both"
read -rp "Choose [1-3]: " model_choice

case "$model_choice" in
  1) PROFILES="sentiment" ;;
  2) PROFILES="topic" ;;
  3) PROFILES="sentiment topic" ;;
  *) echo "Invalid choice: $model_choice"; exit 1 ;;
esac

if [ "$OS" = "linux" ]; then
  PROFILES="$PROFILES linux"
fi

# ── LLM URL ───────────────────────────────────────────────────────────────────
if [ "$OS" = "macos" ]; then
  export LLM_URL="${LLM_URL:-http://host.docker.internal:8900}"
fi

# ── Build profile flags ───────────────────────────────────────────────────────
PROFILE_FLAGS=""
for p in $PROFILES; do
  PROFILE_FLAGS="$PROFILE_FLAGS --profile $p"
done

echo ""
echo "Starting: profiles=[${PROFILES}]"
[ "$OS" = "macos" ] && echo "LLM_URL=${LLM_URL}"
echo ""

# shellcheck disable=SC2086
docker compose $PROFILE_FLAGS up -d --build
