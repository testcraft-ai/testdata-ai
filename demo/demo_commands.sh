#!/usr/bin/env bash
# Commands executed during the asciinema recording.
#
# Uses demo/mock_cli.py instead of the real `testdata-ai` CLI so that:
#   - The spinner is visible (each AI "call" takes exactly 0.8 s)
#   - No real API call or Ollama instance is needed
#   - The GIF shows only meaningful output, not loading waits

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOCK="python3 $REPO_ROOT/demo/mock_cli.py"

# Fake a human typing the command, then run it
_cmd() {
    printf '\033[1;32m$\033[0m \033[1m%s\033[0m\n' "$*"
    sleep 0.6
    eval "$*"
}

clear
sleep 1.0

# 1. List available contexts — instant, no AI call
_cmd "$MOCK list-contexts"
sleep 2.0

# 2. Generate 3 records as pretty JSON — output scrolls, shows data richness
_cmd "$MOCK generate --context ecommerce_customer --count 3"
sleep 2.5

# 3. Batch generation: 9 records in batches of 3 — records appear progressively as compact JSONL
_cmd "$MOCK generate --context ecommerce_customer --count 9 --batch-size 3 -o jsonl"
sleep 2.0

# 4. Show schema for a different context — instant, no AI call
_cmd "$MOCK show-context banking_user"
sleep 1.5
