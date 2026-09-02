#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <OpenOCR-root> <unirec-model-dir> <output-dir>" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
openocr_root=$1
model_dir=$2
output_dir=$3
log_file="${output_dir%/}/export_onnx.log"

mkdir -p "$output_dir"
cd "$repo_root"
PYTHONPATH="$openocr_root" uv run python -m model_tools.export_onnx \
  --openocr-root "$openocr_root" \
  --model-dir "$model_dir" \
  --output-dir "$output_dir" 2>&1 | tee "$log_file"

