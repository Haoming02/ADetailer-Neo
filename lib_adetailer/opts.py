import math

import numpy as np


def dynamic_denoise_strength(
    denoise_power: float,
    denoise_strength: float,
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> float:
    assert len(bbox) == 4

    if np.isclose(denoise_power, 0.0) or len(bbox) != 4:
        return denoise_strength

    width, height = image_size

    image_pixels = width * height
    bbox_pixels = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    normalized_area = bbox_pixels / image_pixels
    denoise_modifier = (1.0 - normalized_area) ** denoise_power

    return denoise_strength * denoise_modifier


class OptimalCropSize:

    @staticmethod
    def _round(v: int) -> int:
        return round(v / 64.0) * 64

    @staticmethod
    def strict(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        assert len(bbox) == 4

        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_aspect_ratio = bbox_width / bbox_height

        target = 1024 * 1024
        w = math.sqrt(target * bbox_aspect_ratio)
        h = w / bbox_aspect_ratio

        return OptimalCropSize._round(w), OptimalCropSize._round(h)

    @staticmethod
    def free(
        inpaint_width: int, inpaint_height: int, bbox: tuple[int, int, int, int]
    ) -> tuple[int, int]:
        assert len(bbox) == 4

        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_aspect_ratio = bbox_width / bbox_height

        target = inpaint_width * inpaint_height
        w = math.sqrt(target * bbox_aspect_ratio)
        h = w / bbox_aspect_ratio

        return OptimalCropSize._round(w), OptimalCropSize._round(h)
