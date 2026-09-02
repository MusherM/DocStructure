"""Image preprocessing shared by ONNX verification and documented for Harmony."""

from __future__ import annotations

import numpy as np
from PIL import Image

from model_tools.contract import Gear


def letterbox_image(image: Image.Image, gear: Gear) -> np.ndarray:
    """Resize with aspect ratio intact, pad right/bottom white, normalize to NCHW [-1, 1]."""
    rgb = image.convert("RGB")
    scale = min(gear.width / rgb.width, gear.height / rgb.height)
    resized_width = max(1, min(gear.width, round(rgb.width * scale)))
    resized_height = max(1, min(gear.height, round(rgb.height * scale)))
    resized = rgb.resize((resized_width, resized_height), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (gear.width, gear.height), color=(255, 255, 255))
    # UniRec is trained with native sizes whose content starts at the origin.
    # Keeping that origin is materially more accurate for extreme aspect ratios.
    canvas.paste(resized, (0, 0))

    array = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)
    array = (array - np.float32(0.5)) / np.float32(0.5)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])
