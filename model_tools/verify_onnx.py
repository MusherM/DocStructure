"""Verify every dynamic image gear and the fixed-cache decoder with ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

from model_tools.contract import (
    BOS_TOKEN_ID,
    DECODER_HEADS,
    DECODER_LAYERS,
    EOS_TOKEN_ID,
    GEARS,
    HEAD_DIM,
    MAX_DECODE_LENGTH,
    PAD_TOKEN_ID,
)
from model_tools.image_processing import letterbox_image
from model_tools.tokenizer_binary import clean_decoded_text, read_tokenizer_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("model_tools/output"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--expected-prefix", default="已处理")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--report", type=Path, default=Path("model_tools/output/verify_report.json"))
    parser.add_argument("--gear", choices=[gear.name for gear in GEARS], action="append")
    return parser.parse_args()


def describe_model(path: Path) -> dict[str, object]:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    return {
        "path": str(path),
        "opset": model.opset_import[0].version,
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
        "nodes": len(model.graph.node),
    }


def assert_public_tensor_ranks(path: Path, maximum: int = 4) -> None:
    model = onnx.load(path, load_external_data=False)
    for value in (*model.graph.input, *model.graph.output):
        rank = len(value.type.tensor_type.shape.dim)
        if rank > maximum:
            raise RuntimeError(
                f"{path}: public tensor {value.name!r} has rank {rank}, max {maximum}"
            )


def decode_until_prefix_decidable(
    decoder: ort.InferenceSession,
    cross_k: np.ndarray,
    cross_v: np.ndarray,
    tokens: list[str],
    expected_prefix: str,
    max_new_tokens: int,
) -> tuple[str, list[int], float]:
    past_keys = np.zeros(
        (DECODER_LAYERS, DECODER_HEADS, MAX_DECODE_LENGTH, HEAD_DIM),
        dtype=np.float32,
    )
    past_values = np.zeros_like(past_keys)
    mask = np.full((1, 1, 1, MAX_DECODE_LENGTH + 1), -10_000.0, dtype=np.float32)
    mask[..., MAX_DECODE_LENGTH] = 0.0
    generated = [BOS_TOKEN_ID]
    started = time.perf_counter()

    for step in range(min(max_new_tokens, MAX_DECODE_LENGTH - 1)):
        outputs = decoder.run(
            None,
            {
                "input_ids": np.asarray([[generated[-1]]], dtype=np.int32),
                "position_ids": np.asarray([[PAD_TOKEN_ID + 1 + step]], dtype=np.int32),
                "self_attention_mask": mask,
                "cross_k": cross_k,
                "cross_v": cross_v,
                "past_keys": past_keys,
                "past_values": past_values,
            },
        )
        logits, new_keys, new_values = outputs
        next_token = int(np.argmax(logits[0]))
        generated.append(next_token)
        past_keys[:, :, step : step + 1, :] = new_keys
        past_values[:, :, step : step + 1, :] = new_values
        mask[..., step] = 0.0

        text = clean_decoded_text("".join(tokens[token] for token in generated))
        if next_token == EOS_TOKEN_ID:
            break
        if len(text) >= len(expected_prefix):
            break

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    text = clean_decoded_text("".join(tokens[token] for token in generated))
    return text, generated, elapsed_ms


def main() -> None:
    args = parse_args()
    encoder_path = args.model_dir / "unirec_encoder.onnx"
    decoder_path = args.model_dir / "unirec_decoder.onnx"
    tokenizer_path = args.model_dir / "unirec_tokenizer.bin"
    assert_public_tensor_ranks(encoder_path)
    assert_public_tensor_ranks(decoder_path)
    tokens, special = read_tokenizer_binary(tokenizer_path)
    if special != {"bos": BOS_TOKEN_ID, "eos": EOS_TOKEN_ID, "pad": PAD_TOKEN_ID}:
        raise RuntimeError(f"Unexpected special token IDs: {special}")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    encoder = ort.InferenceSession(
        encoder_path, sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    decoder = ort.InferenceSession(
        decoder_path, sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    image = Image.open(args.image)
    selected = [gear for gear in GEARS if not args.gear or gear.name in args.gear]

    results: list[dict[str, object]] = []
    for gear in selected:
        pixels = letterbox_image(image, gear)
        started = time.perf_counter()
        cross_k, cross_v = encoder.run(None, {"pixel_values": pixels})
        encoder_ms = (time.perf_counter() - started) * 1000.0
        expected_shape = (
            DECODER_LAYERS,
            DECODER_HEADS,
            gear.visual_tokens,
            HEAD_DIM,
        )
        if cross_k.shape != expected_shape or cross_v.shape != expected_shape:
            raise RuntimeError(
                f"Gear {gear.name} produced K/V {cross_k.shape}/{cross_v.shape}, "
                f"expected {expected_shape}"
            )
        text, generated, decoder_ms = decode_until_prefix_decidable(
            decoder,
            cross_k,
            cross_v,
            tokens,
            args.expected_prefix,
            args.max_new_tokens,
        )
        passed = text.startswith(args.expected_prefix)
        results.append(
            {
                "gear": gear.name,
                "pixel_values_shape": list(pixels.shape),
                "cross_k_shape": list(cross_k.shape),
                "text": text,
                "token_ids": generated,
                "encoder_ms": round(encoder_ms, 3),
                "decoder_ms": round(decoder_ms, 3),
                "passed": passed,
            }
        )
        print(f"[{gear.name}] passed={passed} text={text!r}")

    report = {
        "image": str(args.image.resolve()),
        "expected_prefix": args.expected_prefix,
        "encoder": describe_model(encoder_path),
        "decoder": describe_model(decoder_path),
        "results": results,
        "passed": all(bool(result["passed"]) for result in results),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(f"Verification failed; see {args.report}")


if __name__ == "__main__":
    main()
