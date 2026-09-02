#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: SOC_VERSION=<exact-target-soc> ./scripts/convert_to_om.sh <onnx-dir> <om-dir>

Required environment:
  SOC_VERSION   Exact SoC accepted by the CANN Kit toolchain for the target device.

Optional environment:
  ATC_BIN       ATC executable (default: atc)
  PRECISION     allow_fp32_to_fp16 (default) or a toolchain-supported precision mode
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi
if [[ -z "${SOC_VERSION:-}" ]]; then
  echo "SOC_VERSION is required; never compile a phone OM with a guessed server SoC." >&2
  usage
  exit 2
fi

onnx_dir=$1
om_dir=$2
atc_bin=${ATC_BIN:-atc}
precision=${PRECISION:-allow_fp32_to_fp16}
encoder_onnx="${onnx_dir%/}/unirec_encoder.onnx"
decoder_onnx="${onnx_dir%/}/unirec_decoder.onnx"

for required in "$encoder_onnx" "$decoder_onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required ONNX: $required" >&2
    exit 2
  fi
done
if ! command -v "$atc_bin" >/dev/null 2>&1; then
  echo "ATC executable not found: $atc_bin" >&2
  exit 127
fi

mkdir -p "$om_dir"

"$atc_bin" \
  --framework=5 \
  --model="$encoder_onnx" \
  --output="${om_dir%/}/unirec_encoder" \
  --soc_version="$SOC_VERSION" \
  --input_format=ND \
  --input_shape="pixel_values:1,3,-1,-1" \
  --dynamic_dims="512,384;768,576;1024,768;1408,960" \
  --input_fp16_nodes="pixel_values" \
  --output_type=FP16 \
  --precision_mode="$precision" 2>&1 | tee "${om_dir%/}/atc_encoder.log"

decoder_shape="input_ids:1,1;position_ids:1,1;self_attention_mask:1,1,1,2049"
decoder_shape+=";cross_k:6,1,6,-1,128;cross_v:6,1,6,-1,128"
decoder_shape+=";past_keys:6,1,6,2048,128;past_values:6,1,6,2048,128"

"$atc_bin" \
  --framework=5 \
  --model="$decoder_onnx" \
  --output="${om_dir%/}/unirec_decoder" \
  --soc_version="$SOC_VERSION" \
  --input_format=ND \
  --input_shape="$decoder_shape" \
  --dynamic_dims="192,192;432,432;768,768;1320,1320" \
  --input_fp16_nodes="self_attention_mask;cross_k;cross_v;past_keys;past_values" \
  --output_type=FP16 \
  --precision_mode="$precision" 2>&1 | tee "${om_dir%/}/atc_decoder.log"

test -s "${om_dir%/}/unirec_encoder.om"
test -s "${om_dir%/}/unirec_decoder.om"
sha256sum "${om_dir%/}/unirec_encoder.om" "${om_dir%/}/unirec_decoder.om" \
  | tee "${om_dir%/}/SHA256SUMS"

echo "OM conversion complete: $om_dir"
