#!/usr/bin/env bash
# One-time setup on a fresh NVIDIA VM (Ubuntu). Run from anywhere.
set -euo pipefail

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$(dirname "$0")/.."
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"
uv run pytest -q
echo "VM ready."
