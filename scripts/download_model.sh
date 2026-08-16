#!/usr/bin/env bash
set -euo pipefail

mkdir -p models/Qwen2.5-0.5B-Instruct
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir models/Qwen2.5-0.5B-Instruct

