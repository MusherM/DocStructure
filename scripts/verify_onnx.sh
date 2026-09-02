#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <onnx-output-dir> <image> [expected-prefix]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=$1
image_path=$2
expected_prefix=${3:-已处理}
log_file="${model_dir%/}/verify_onnx.log"

cd "$repo_root"
uv run python -m model_tools.verify_onnx \
  --model-dir "$model_dir" \
  --image "$image_path" \
  --expected-prefix "$expected_prefix" 2>&1 | tee "$log_file"

