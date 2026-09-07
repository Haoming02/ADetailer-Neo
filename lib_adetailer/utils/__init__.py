from typing import TYPE_CHECKING, TypeVar, Union

if TYPE_CHECKING:
    import numpy as np
    import torch

import rich
from PIL import Image
from torchvision.transforms.functional import to_pil_image

NUM = TypeVar("NUM", int, float)


def print(msg: str):
    rich.print(f"[ADetailer]: {msg}")


def ensure_pil_image(
    image: Union[Image.Image, "np.ndarray", "torch.Tensor"],
    mode: str = "RGB",
) -> Image.Image:
    if not isinstance(image, Image.Image):
        image: Image.Image = to_pil_image(image)
    if image.mode != mode:
        image = image.convert(mode)
    return image
