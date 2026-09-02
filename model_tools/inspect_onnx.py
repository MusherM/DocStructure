"""Print ONNX I/O contracts without loading model weights into a runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def shape_of(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        result.append(dimension.dim_value or dimension.dim_param or "?")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.models:
        model = onnx.load(path, load_external_data=False)
        print(path)
        for label, values in (("input", model.graph.input), ("output", model.graph.output)):
            for value in values:
                print(f"  {label:6} {value.name:24} {shape_of(value)}")


if __name__ == "__main__":
    main()
