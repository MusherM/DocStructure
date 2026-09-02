# UniRec-0.1B Dynamic-Gear OM Demo

中文：[README_zh.md](README_zh.md)

This repository contains two deliverables:

1. A reproducible UniRec-0.1B export pipeline that produces one dynamic-gear encoder ONNX and one dynamic-gear decoder ONNX, verifies all four gears, and provides a Linux OMG script for producing one dynamic-gear encoder OM and one dynamic-gear decoder OM.
2. A native HarmonyOS app that loads those two OM files through CANN Kit/NNRt, selects a gear for the input image, runs encoder + autoregressive decoder inference, and decodes tokens locally.

OM conversion is intentionally not claimed as completed here: it must be run in your Linux CANN Kit/toolchain environment for the exact target phone SoC. A successful OMG conversion is not proof that the OM is compatible with a phone or fully executable on its NPU.

See the rendered [delivery and verification report](dev/delivery_report.html) for the measured four-gear results and the remaining device-side validation boundary.

## Model contract

The requested sizes are interpreted as **width × height**. Runtime tensors use NCHW, so the shapes are:

| Gear | Encoder input | Visual tokens | Encoder `cross_k/cross_v` |
|---|---:|---:|---:|
| 384×512 | `[1,3,512,384]` | 192 | `[6,6,192,128]` |
| 576×768 | `[1,3,768,576]` | 432 | `[6,6,432,128]` |
| 768×1024 | `[1,3,1024,768]` | 768 | `[6,6,768,128]` |
| 960×1408 | `[1,3,1408,960]` | 1320 | `[6,6,1320,128]` |

There is exactly one encoder model and one decoder model. The decoder has four linked visual-sequence gears. Its self-attention cache capacity is fixed at 2048 tokens:

```text
past_keys / past_values: [6,6,2048,128]
new_keys  / new_values:  [6,6,1,128]
logits:                    [1,56371]
```

All public ONNX tensors have rank four or less for OMG compatibility. Because batch size is
fixed at one, the wrappers omit the singleton batch axis from public K/V tensors and restore it
inside the decoder without changing attention semantics or contiguous element order.

The fixed-capacity cache avoids a new dynamic shape at every generated token. The app updates only the current cache slot. The trade-off is that self-attention computes against a masked 2048-slot cache each step; this is more OM-friendly but can be slower than a truly variable-length cache.

Images are resized with bicubic interpolation while preserving aspect ratio, aligned at the top-left, and padded with white on the right/bottom. RGB values are normalized with `(x / 255 - 0.5) / 0.5`. Top-left alignment is required for the supplied wide verification image; centered padding produced incorrect leading tokens in three gears.

## Part 1: export ONNX and convert it to OM

### 1. Prerequisites

- Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/).
- The official [Topdu/OpenOCR](https://github.com/Topdu/OpenOCR) repository. This demo was verified against commit `0d522801ec6dc1df852c6b6d4ed6a08f5127ed97`.
- The official [topdu/unirec-0.1b](https://huggingface.co/topdu/unirec-0.1b) files in one local directory:

```text
unirec-0.1b/
├── config.json
├── generation_config.json
├── model.pth
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json
```

The exporter reads `model.pth` with `torch.load(..., weights_only=True)` and rejects missing or unexpected state-dict keys.

Install Python dependencies once:

```bash
uv sync
```

Example model/source setup:

```bash
git clone https://github.com/Topdu/OpenOCR.git third_party/OpenOCR
git -C third_party/OpenOCR checkout 0d522801ec6dc1df852c6b6d4ed6a08f5127ed97
uv run hf download topdu/unirec-0.1b \
  model.pth config.json generation_config.json tokenizer.json \
  tokenizer_config.json special_tokens_map.json \
  --local-dir models/unirec-0.1b
```

### 2. Export the modified ONNX models

```bash
./scripts/export_onnx.sh \
  third_party/OpenOCR \
  models/unirec-0.1b \
  model_tools/output
```

Generated files:

```text
model_tools/output/
├── unirec_encoder.onnx
├── unirec_decoder.onnx
├── unirec_tokenizer.bin
├── model_contract.json
└── export_onnx.log
```

The encoder precomputes the six decoder layers' cross-attention K/V. The decoder combines twelve separate variable-length cache inputs into two fixed-capacity tensors and emits only the one-step K/V delta. This reduces dynamic-shape pressure and host/device copies without changing the trained weights.

### 3. Verify all four ONNX gears

The supplied verification image is [model_tools/assets/verify.png](model_tools/assets/verify.png). Success requires every selected gear to produce text beginning with `已处理`.

```bash
./scripts/verify_onnx.sh \
  model_tools/output \
  model_tools/assets/verify.png \
  已处理
```

The verifier checks both models with ONNX Checker, rejects any public tensor above rank four,
checks every K/V shape, runs encoder + fixed-cache decoder in ONNX Runtime, and writes
`verify_report.json`. The local verified result was:

| Gear | Result | Token IDs |
|---|---|---|
| 384×512 | `已处理` | `0, 23709, 11536` |
| 576×768 | `已处理` | `0, 23709, 11536` |
| 768×1024 | `已处理` | `0, 23709, 11536` |
| 960×1408 | `已处理` | `0, 23709, 11536` |

### 4. Convert to a single dynamic-gear OM pair on Linux

Use the CANN Kit OMG conversion toolchain that matches the target HarmonyOS device. Do not guess `PLATFORM`, and do not substitute another platform merely because OMG accepts it. Source the toolchain environment first, then run:

```bash
source /path/to/cann/set_env.sh
PLATFORM='<exact target platform from your CANN Kit package/device documentation>' \
  ./scripts/convert_to_om.sh model_tools/output om_output
```

The script creates exactly:

```text
om_output/
├── unirec_encoder.om      # four H/W gears
├── unirec_decoder.om      # four linked visual-token gears
├── omg_encoder.log
├── omg_decoder.log
└── SHA256SUMS
```

Encoder dynamic dimensions are `512,384;768,576;1024,768;1408,960`. Decoder dynamic dimension pairs are `192,192;432,432;768,768;1320,1320` for `cross_k` and `cross_v`. Float model inputs/outputs are compiled as FP16 to reduce memory traffic, while token and position inputs remain INT32.

The script invokes `omg` by default. Set `OMG_BIN` if the executable is not on `PATH`. It uses OMG's ONNX-aware `--input_type` and `--output_type` mappings; `--input_fp16_nodes` is intentionally not used because it has no effect for ONNX. An OM produced by a generic Ascend Toolkit is not automatically a mobile CANN Kit OM.

## Part 2: native HarmonyOS app

### Required files outside this repository

Before opening/building `harmony_app`, copy the two Linux-generated models to these exact paths and names:

```text
harmony_app/entry/src/main/resources/rawfile/models/unirec_encoder.om
harmony_app/entry/src/main/resources/rawfile/models/unirec_decoder.om
```

The repository already contains:

```text
harmony_app/entry/src/main/resources/rawfile/models/unirec_tokenizer.bin
harmony_app/entry/src/main/resources/rawfile/images/verify.png
```

If you regenerate ONNX/tokenizer artifacts, replace the app tokenizer with the matching file:

```bash
cp model_tools/output/unirec_tokenizer.bin \
  harmony_app/entry/src/main/resources/rawfile/models/unirec_tokenizer.bin
```

Do not mix an encoder, decoder, and tokenizer from different exports. The native app enforces input/output counts, all relevant shapes, dtypes, vocabulary size, and special-token IDs.

### Build and run

The project follows Huawei's official [CANN Kit C++ sample](https://gitee.com/harmonyos_samples/cannkit-samplecode-clientdemo-cpp) integration path and links `hiai_foundation`, `libneural_network_core.so`, N-API, HiLog, and RawFile.

Recommended baseline:

- DevEco Studio 6.0.0 Release or newer.
- HarmonyOS SDK 6.0.0 Release or newer.
- HarmonyOS 5.1.0 Release or newer on a Huawei phone/tablet exposing the `HIAI_F` device.
- API 18 build target (the NNRt/CANN APIs used here start earlier, but the official current sample uses this baseline).

Open `harmony_app` in DevEco Studio, configure signing, build the `entry` HAP for `arm64-v8a`, install it on the target device, and tap either **Run bundled verification image** or **Choose image**.

The app uses one dynamic OM pair. It chooses the smallest gear by source pixel area, preserves aspect ratio, selects that gear at `OH_NNCompilation_Build` using `HMS_HiAIOptions_SetInputTensorShapes`, and rebuilds the executor only if the selected gear changes.

### NPU-only policy

The app explicitly sets:

```text
HMS_HiAIOptions_SetModelDeviceOrder(... NPU ...)
HMS_HiAIOptions_SetFallbackMode(... DISABLED)
```

There is no silent CPU fallback. Unsupported operators, incompatible OM files, missing NPU devices, build failures, and I/O mismatches are fatal. A memory-reuse-high plan is enabled because the two models plus K/V buffers are large.

### Asynchronous diagnostics and fatal-log termination

Inference is exposed to ArkTS as a native asynchronous Promise, so model building and synchronous NNRt execution do not block the UI event handler. Native logs use tag `UniRecOM` and JSON-like records with monotonic `seq`, `stage`, `event`, and `detail` fields.

Stages include:

```text
BOOT → TOKENIZER → ASSET → MODEL_COMPAT → DEVICE → MODEL_BUILD
→ IO_CONTRACT → PREPROCESS → ENCODER → DECODER → POSTPROCESS → DONE
```

Decoder logging is rate-limited to the first four steps, every 25th step, and EOS. Tensor contents are never dumped. On the first fatal error, exactly one record is emitted with `"terminal":true`; an atomic latch then suppresses all later native business logs for the process. Cleanup still runs silently. Restart the app before the next asynchronous validation attempt.

Example collection command:

```bash
hdc shell hilog | grep UniRecOM
```

Keep the complete log from process start through either `DONE/SUCCESS` or the single terminal `FATAL` record.

## Important limits

- This repository proves the modified ONNX models on CPU ONNX Runtime. It does not claim that OM conversion or phone execution has succeeded; those require your Linux CANN toolchain and physical target.
- OMG supports at most 16 dynamic shape gears, and successful conversion still depends on the graph and target compiler. This project uses four gears.
- OMG success does not prove phone compatibility. The app calls `HMS_HiAICompatibility_CheckFromBuffer`, requires `HIAI_F`, disables fallback, then requires `OH_NNCompilation_Build` and all I/O contract checks to pass.
- OM files are hardware/toolchain-specific. A model compiled for an Ascend server/card generally cannot be assumed to run on a Kirin phone NPU.
- FP16 compilation may change borderline token logits. Re-run the supplied image on-device and compare the prefix. If necessary, set `FLOAT_TYPE=FP32`, accepting higher memory and possible NPU coverage loss.
- The combined model and runtime memory footprint is large. Packaging limits, device RAM, contiguous shared-memory allocation, and thermal constraints can still prevent execution.
- NNRt/CANN Kit supports only the operators implemented by the target driver. A graph can convert but fail compatibility/build, and an OM can build but still fail at runtime.
- Dynamic multi-gear execution is not unrestricted dynamic shape. Only the four declared gears are accepted.

## Repository layout

```text
model_tools/                 Python export, preprocessing, tokenizer, verification
scripts/export_onnx.sh       Reproducible ONNX export wrapper
scripts/verify_onnx.sh       Four-gear ONNX correctness check
scripts/convert_to_om.sh     Manual Linux OMG dynamic-gear conversion
harmony_app/                 Native HarmonyOS CANN Kit/NNRt app
dev/                         Delivery and verification report
```

## References

- [UniRec-0.1B model and paper links](https://github.com/Topdu/OpenOCR/blob/main/docs/unirec.md)
- [Official UniRec ONNX inference implementation](https://github.com/Topdu/OpenOCR/blob/main/tools/infer_unirec_onnx.py)
- [HarmonyOS Neural Network Runtime introduction](https://gitee.com/openharmony/docs/blob/master/en/application-dev/ai/nnrt/Neural-Network-Runtime-Kit-Introduction.md)
- [Huawei CANN Kit model conversion](https://developer.huawei.com/consumer/en/doc/harmonyos-guides/cannkit-model-conversion)
- [OMG parameter documentation](https://developer.huawei.com/consumer/en/doc/hiai-guides/overall-parameter-0000001052966900)
