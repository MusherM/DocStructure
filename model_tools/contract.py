"""Shared model contract used by export, verification, and the Harmony app."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gear:
    name: str
    width: int
    height: int

    @property
    def image_shape(self) -> tuple[int, int, int, int]:
        return (1, 3, self.height, self.width)

    @property
    def visual_tokens(self) -> int:
        # FocalSVTR downsamples both spatial axes by 32 before flattening.
        return (self.height // 32) * (self.width // 32)


GEARS: tuple[Gear, ...] = (
    Gear("384x512", 384, 512),
    Gear("576x768", 576, 768),
    Gear("768x1024", 768, 1024),
    Gear("960x1408", 960, 1408),
)

BATCH_SIZE = 1
CHANNELS = 3
DECODER_LAYERS = 6
DECODER_HEADS = 6
HEAD_DIM = 128
MODEL_DIM = DECODER_HEADS * HEAD_DIM
VOCAB_SIZE = 56_371
MAX_DECODE_LENGTH = 2048
PAD_TOKEN_ID = 1
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 2


def gear_for_image(width: int, height: int) -> Gear:
    """Return the smallest gear that can contain an aspect-preserving resize."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")

    # All images can be letterboxed into every gear; choose by source pixel count
    # relative to the gear's usable area, then clamp to the largest gear.
    source_area = width * height
    for gear in GEARS:
        if source_area <= gear.width * gear.height:
            return gear
    return GEARS[-1]
