#!/usr/bin/env bash
# Record the testdata-ai demo with asciinema and convert it to a GIF.
#
# Prerequisites
# -------------
#   pip install asciinema          # recorder
#   cargo install agg              # asciinema → GIF converter (https://github.com/asciinema/agg)
#
#   No AI provider needed — the demo uses pre-canned fixture data (demo/mock_cli.py).
#
# Usage
# -----
#   bash demo/record_demo.sh
#
# Output
# ------
#   demo/demo.cast  — raw recording
#   demo/demo.gif   — final GIF (commit this to the repo)
#
# After committing demo/demo.gif the README will show the animation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAST="$REPO_ROOT/demo/demo.cast"
GIF="$REPO_ROOT/demo/demo.gif"

echo "==> Recording to $CAST ..."
asciinema rec "$CAST" \
    --overwrite \
    --cols 96 \
    --rows 36 \
    --title "testdata-ai demo" \
    --command "bash $REPO_ROOT/demo/demo_commands.sh"

echo ""
echo "==> Converting to GIF ($GIF) ..."
# The demo spinner uses directional arrows (←↖↑↗→↘↓↙) which are in
# JetBrains Mono — agg's default font. No --font-dir/--font-family needed.
agg \
    --theme monokai \
    --font-size 14 \
    --speed 1.2 \
    --last-frame-duration 4 \
    "$CAST" "$GIF"

echo ""
echo "Done! Commit demo/demo.gif and push — the README will show the animation."
