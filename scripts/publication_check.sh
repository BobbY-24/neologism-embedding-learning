#!/usr/bin/env bash
set -euo pipefail

if git grep -nE '/Users/[^/]+/|/home/[^/]+/' -- ':!scripts/publication_check.sh'; then
  echo "Absolute local path found in tracked content." >&2
  exit 1
fi

if git grep -nE '(gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})'; then
  echo "Credential-like string found in tracked content." >&2
  exit 1
fi

if git ls-files | grep -E '(^|/)(models|\.venv|__pycache__)/|\.(safetensors|gguf|ckpt|pth|pt|zip)$'; then
  echo "Model, environment, cache, checkpoint, or archive is tracked." >&2
  exit 1
fi

large_files="$(find . -path ./.git -prune -o -type f -size +10M -print)"
if [[ -n "$large_files" ]]; then
  echo "Files larger than 10 MiB found:" >&2
  echo "$large_files" >&2
  exit 1
fi

echo "Publication checks passed."
