#!/bin/bash
set -e

# Install uv (installs Python 3.14 automatically via .python-version)
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

uv sync

echo ""
echo "Done. Next steps:"
echo "  1. ./download_data.sh     # fetch COCO + DTD (~19 GB)"
echo "  2. export WANDB_API_KEY=<your-key>"
echo "  3. ./train.sh"
