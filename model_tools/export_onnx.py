"""Export UniRec-0.1B into ATC-oriented dynamic-gear ONNX models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx
import torch

from model_tools.contract import (
    DECODER_HEADS,
    DECODER_LAYERS,
    GEARS,
    HEAD_DIM,
    MAX_DECODE_LENGTH,
)
from model_tools.tokenizer_binary import export_tokenizer_binary
from model_tools.wrappers import DecoderStepForOm, EncoderForOm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("model_tools/output"))
    parser.add_argument("--opset", type=int, default=14)
    return parser.parse_args()


def load_model(openocr_root: Path, model_dir: Path) -> torch.nn.Module:
    sys.path.insert(0, str(openocr_root.resolve()))
    from openrec.modeling.unirec_modeling.configuration_unirec import UniRecConfig
    from openrec.modeling.unirec_modeling.modeling_unirec import (
        UniRecForConditionalGenerationNew,
    )

    config = UniRecConfig.from_pretrained(model_dir)
    config._attn_implementation = "eager"
    model = UniRecForConditionalGenerationNew(config=config)
    checkpoint = torch.load(model_dir / "model.pth", map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model


def export_encoder(model: torch.nn.Module, path: Path, opset: int) -> None:
    wrapper = EncoderForOm(model).eval()
    gear = GEARS[0]
    sample = torch.zeros(gear.image_shape, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (sample,),
        path,
        input_names=["pixel_values"],
        output_names=["cross_k", "cross_v"],
        dynamic_axes={
            "pixel_values": {2: "height", 3: "width"},
            "cross_k": {3: "encoder_sequence_length"},
            "cross_v": {3: "encoder_sequence_length"},
        },
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )


def export_decoder(model: torch.nn.Module, path: Path, opset: int) -> None:
    wrapper = DecoderStepForOm(model).eval()
    visual_tokens = GEARS[0].visual_tokens
    sample_inputs = (
        torch.zeros((1, 1), dtype=torch.int32),
        torch.full((1, 1), 2, dtype=torch.int32),
        torch.zeros((1, 1, 1, MAX_DECODE_LENGTH + 1), dtype=torch.float32),
        torch.zeros(
            (DECODER_LAYERS, 1, DECODER_HEADS, visual_tokens, HEAD_DIM),
            dtype=torch.float32,
        ),
        torch.zeros(
            (DECODER_LAYERS, 1, DECODER_HEADS, visual_tokens, HEAD_DIM),
            dtype=torch.float32,
        ),
        torch.zeros(
            (DECODER_LAYERS, 1, DECODER_HEADS, MAX_DECODE_LENGTH, HEAD_DIM),
            dtype=torch.float32,
        ),
        torch.zeros(
            (DECODER_LAYERS, 1, DECODER_HEADS, MAX_DECODE_LENGTH, HEAD_DIM),
            dtype=torch.float32,
        ),
    )
    torch.onnx.export(
        wrapper,
        sample_inputs,
        path,
        input_names=[
            "input_ids",
            "position_ids",
            "self_attention_mask",
            "cross_k",
            "cross_v",
            "past_keys",
            "past_values",
        ],
        output_names=["logits", "new_keys", "new_values"],
        dynamic_axes={
            "cross_k": {3: "encoder_sequence_length"},
            "cross_v": {3: "encoder_sequence_length"},
        },
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )


def write_contract(path: Path, opset: int) -> None:
    contract = {
        "version": 1,
        "opset": opset,
        "layout": "NCHW",
        "image_normalization": "RGB float32, (x/255 - 0.5) / 0.5",
        "resize": "aspect-preserving bicubic, top-left aligned, white pad right/bottom",
        "gears": [
            {
                "name": gear.name,
                "width": gear.width,
                "height": gear.height,
                "visual_tokens": gear.visual_tokens,
            }
            for gear in GEARS
        ],
        "decoder": {
            "layers": DECODER_LAYERS,
            "heads": DECODER_HEADS,
            "head_dim": HEAD_DIM,
            "cache_capacity": MAX_DECODE_LENGTH,
        },
    }
    path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.openocr_root, args.model_dir)
    encoder_path = args.output_dir / "unirec_encoder.onnx"
    decoder_path = args.output_dir / "unirec_decoder.onnx"
    export_encoder(model, encoder_path, args.opset)
    export_decoder(model, decoder_path, args.opset)
    export_tokenizer_binary(args.model_dir, args.output_dir / "unirec_tokenizer.bin")
    write_contract(args.output_dir / "model_contract.json", args.opset)
    for path in (encoder_path, decoder_path):
        onnx.checker.check_model(onnx.load(path, load_external_data=False))
    print(f"Export complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
