#!/usr/bin/env bash
set -euo pipefail

model_dir="${QWEN_MODEL_PATH:-models/Qwen2.5-0.5B-Instruct}"
if [[ ! -f "$model_dir/config.json" ]]; then
  echo "Model not found at $model_dir. Run: bash scripts/download_model.sh" >&2
  exit 2
fi

export QWEN_MODEL_PATH="$model_dir"
python -m neologism_embedding.experiment2 --mode quick --stage all

