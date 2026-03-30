#!/usr/bin/env bash
set -euo pipefail

if ! command -v gum &>/dev/null; then
  echo "Error: gum is not installed."
  echo ""
  echo "Install it with:"
  echo "  macOS:  brew install gum"
  echo "  Linux:  https://github.com/charmbracelet/gum#installation"
  exit 1
fi

# ── Build model-server image ───────────────────────────────────────────────────
gum spin --spinner dot --title "Building text-analysis/model-server:latest..." -- \
  docker build -t text-analysis/model-server:latest ./model_server

# ── OS ────────────────────────────────────────────────────────────────────────
OS=$(gum choose --header "OS:" --selected "macOS" "macOS" "Linux" "Custom")

# ── Profiles ──────────────────────────────────────────────────────────────────
PROFILE_CHOICE=$(gum choose --header "Profiles:" --selected "Both" \
  "Sentiment only" "Topic only" "Both")

case "$PROFILE_CHOICE" in
  "Sentiment only") PROFILES="sentiment" ;;
  "Topic only")     PROFILES="topic" ;;
  "Both")           PROFILES="sentiment topic" ;;
esac

if [ "$OS" = "Linux" ] || [ "$OS" = "Custom" ]; then
  PROFILES="$PROFILES linux"
fi

# ── Select individual models ───────────────────────────────────────────────────
MODEL_LIST=()
while IFS= read -r line; do
  MODEL_LIST+=("$line")
done < <(python3 -c "
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
for m in cfg['models']:
    if m.get('type') != 'llm':
        print('{} ({})'.format(m['name'], m.get('for', 'sentiment')))
" 2>/dev/null)

if [ "${#MODEL_LIST[@]}" -gt 0 ]; then
  CHOSEN=()
  while IFS= read -r line; do
    CHOSEN+=("$line")
  done < <(gum choose --no-limit \
    --header "Models (LLM always included, Space to toggle, Enter to confirm):" \
    --selected "$(IFS=$'\n'; echo "${MODEL_LIST[*]}")" \
    "${MODEL_LIST[@]}")

  if [ "${#CHOSEN[@]}" -eq 0 ] || [ "${#CHOSEN[@]}" -eq "${#MODEL_LIST[@]}" ]; then
    export ENABLED_MODELS="*"
  else
    # Strip " (for)" suffix to get plain model names
    NAMES=()
    for item in "${CHOSEN[@]}"; do
      NAMES+=("${item% (*}")
    done
    joined=$(IFS=,; echo "${NAMES[*]}")
    export ENABLED_MODELS="$joined"
  fi
else
  export ENABLED_MODELS="*"
fi

# ── LLM URL ───────────────────────────────────────────────────────────────────
if [ "$OS" = "macOS" ]; then
  export LLM_URL="${LLM_URL:-http://host.docker.internal:8900}"
elif [ "$OS" = "Custom" ]; then
  export LLM_URL=$(gum input --header "LLM URL:" --placeholder "http://..." --value "${LLM_URL:-}")
fi

# ── Build profile flags ───────────────────────────────────────────────────────
PROFILE_FLAGS=""
for p in $PROFILES; do
  PROFILE_FLAGS="$PROFILE_FLAGS --profile $p"
done

gum style --bold "Starting:"
echo "  Profiles : ${PROFILES}"
echo "  Models   : ${ENABLED_MODELS}"
{ [ "$OS" = "macOS" ] || [ "$OS" = "Custom" ]; } && echo "  LLM URL  : ${LLM_URL}"
echo ""

# shellcheck disable=SC2086
docker compose $PROFILE_FLAGS up -d --build --force-recreate
