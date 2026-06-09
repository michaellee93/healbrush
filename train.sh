#!/bin/bash
set -e

if [ -z "$WANDB_API_KEY" ]; then
    echo "error: WANDB_API_KEY not set"
    exit 1
fi

mkdir -p outputs/4layer
uv run python main.py
