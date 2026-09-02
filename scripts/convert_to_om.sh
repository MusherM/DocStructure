#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: PLATFORM=<exact-target-platform> ./scripts/convert_to_om.sh <onnx-dir> <om-dir>

Required environment:
  PLATFORM      Exact platform accepted by CANN Kit OMG for the target device.

Optional environment:
  OMG_BIN       OMG executable (default: omg)
  FLOAT_TYPE    FP16 (default) or FP32 for floating-point model I/O and weights
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi
if [[ -z "${PLATFORM:-}" ]]; then
  echo "PLATFORM is required; use the exact target listed by the CANN Kit OMG toolchain." >&2
  usage
  exit 2
fi

onnx_dir=$1
om_dir=$2
omg_bin=${OMG_BIN:-omg}
float_type=${FLOAT_TYPE:-FP16}
encoder_onnx="${onnx_dir%/}/unirec_encoder.onnx"
decoder_onnx="${onnx_dir%/}/unirec_decoder.onnx"

if [[ "$float_type" != FP16 && "$float_type" != FP32 ]]; then
  echo "FLOAT_TYPE must be FP16 or FP32: $float_type" >&2
  exit 2
fi

for required in "$encoder_onnx" "$decoder_onnx"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required ONNX: $required" >&2
    exit 2
  fi
done
if ! command -v "$omg_bin" >/dev/null 2>&1; then
  echo "OMG executable not found: $omg_bin" >&2
  exit 127
fi

mkdir -p "$om_dir"

"$omg_bin" \
  --framework=5 \
  --model="$encoder_onnx" \
  --output="${om_dir%/}/unirec_encoder" \
  --platform="$PLATFORM" \
  --input_shape="pixel_values:1,3,-1,-1" \
  --dynamic_dims="512,384;768,576;1024,768;1408,960" \
  --input_type="pixel_values:$float_type" \
  --output_type="cross_k:$float_type;cross_v:$float_type" \
  --weight_data_type="$float_type" 2>&1 | tee "${om_dir%/}/omg_encoder.log"

decoder_shape="input_ids:1,1;position_ids:1,1;self_attention_mask:1,1,1,2049"
decoder_shape+=";cross_k:6,1,6,-1,128;cross_v:6,1,6,-1,128"
decoder_shape+=";past_keys:6,1,6,2048,128;past_values:6,1,6,2048,128"
decoder_input_type="input_ids:INT32;position_ids:INT32;self_attention_mask:$float_type"
decoder_input_type+=";cross_k:$float_type;cross_v:$float_type"
decoder_input_type+=";past_keys:$float_type;past_values:$float_type"

"$omg_bin" \
  --framework=5 \
  --model="$decoder_onnx" \
  --output="${om_dir%/}/unirec_decoder" \
  --platform="$PLATFORM" \
  --input_shape="$decoder_shape" \
  --dynamic_dims="192,192;432,432;768,768;1320,1320" \
  --input_type="$decoder_input_type" \
  --output_type="logits:$float_type;new_keys:$float_type;new_values:$float_type" \
  --weight_data_type="$float_type" 2>&1 | tee "${om_dir%/}/omg_decoder.log"

test -s "${om_dir%/}/unirec_encoder.om"
test -s "${om_dir%/}/unirec_decoder.om"
sha256sum "${om_dir%/}/unirec_encoder.om" "${om_dir%/}/unirec_decoder.om" \
  | tee "${om_dir%/}/SHA256SUMS"

echo "OM conversion complete: $om_dir"
