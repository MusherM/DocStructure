"""Compact tokenizer artifact readable without a JSON dependency on Harmony."""

from __future__ import annotations

import json
import struct
from pathlib import Path

MAGIC = b"URTK"
VERSION = 1


def export_tokenizer_binary(model_dir: Path, output_path: Path) -> None:
    tokenizer_json = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocabulary: dict[str, int] = tokenizer_json["model"]["vocab"]
    added_tokens: list[dict[str, object]] = tokenizer_json.get("added_tokens", [])
    maximum_id = max(
        max(int(token_id) for token_id in vocabulary.values()),
        max((int(item["id"]) for item in added_tokens), default=-1),
    )
    id_to_token = [""] * (maximum_id + 1)
    for token, token_id in vocabulary.items():
        id_to_token[int(token_id)] = token
    for item in added_tokens:
        id_to_token[int(item["id"])] = str(item["content"])
    missing = [token_id for token_id, token in enumerate(id_to_token) if not token]
    if missing:
        raise ValueError(f"Tokenizer has unmapped IDs, first entries: {missing[:8]}")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(
            struct.pack(
                "<IIIII",
                VERSION,
                len(id_to_token),
                int(config["bos_token_id"]),
                int(config["eos_token_id"]),
                int(config["pad_token_id"]),
            )
        )
        for token in id_to_token:
            encoded = token.encode("utf-8")
            stream.write(struct.pack("<I", len(encoded)))
            stream.write(encoded)


def read_tokenizer_binary(path: Path) -> tuple[list[str], dict[str, int]]:
    with path.open("rb") as stream:
        if stream.read(4) != MAGIC:
            raise ValueError("Invalid UniRec tokenizer magic")
        version, size, bos, eos, pad = struct.unpack("<IIIII", stream.read(20))
        if version != VERSION:
            raise ValueError(f"Unsupported tokenizer version: {version}")
        tokens = []
        for _ in range(size):
            (length,) = struct.unpack("<I", stream.read(4))
            tokens.append(stream.read(length).decode("utf-8"))
    return tokens, {"bos": bos, "eos": eos, "pad": pad}


def clean_decoded_text(text: str) -> str:
    replacements = (
        ("Ġ", " "),
        ("Ċ", "\n"),
        ("<|bos|>", ""),
        ("<|eos|>", ""),
        ("<|pad|>", ""),
        ("<|unk|>", ""),
        ("<|sn|>", " "),
        ("<s>", ""),
        ("</s>", ""),
        ("\uffff", ""),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text.strip()
