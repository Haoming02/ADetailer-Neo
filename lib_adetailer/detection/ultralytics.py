import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import cv2
from PIL import Image

from modules.shared import opts

from ..utils import ensure_pil_image
from .common import PredictOutput, create_mask_from_bbox


@contextmanager
def screw_yolo():
    # why the actual #### do I need to do this?

    orig1 = os.makedirs
    orig2 = os.mkdir

    os.makedirs = lambda *args, **kwargs: None
    os.mkdir = lambda *args, **kwargs: None

    yield

    os.makedirs = orig1
    os.mkdir = orig2


def ultralytics_predict(
    model_path: os.PathLike,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
    classes: str = "",
) -> PredictOutput:
    from ultralytics import YOLO

    model = YOLO(model_path)

    if bool(classes) and "-world" in str(model_path):
        if parsed := [c.strip() for c in classes.split(",") if c.strip()]:
            model.set_classes(parsed)

    with screw_yolo():
        pred = model(
            image,
            conf=confidence,
            device=device,
            imgsz=1024 if getattr(opts, "ad_hd_yolo", False) else 640,
        )

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return PredictOutput()

    bboxes = bboxes.tolist()

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)

    confidences = pred[0].boxes.conf.cpu().numpy().tolist()

    preview = pred[0].plot()
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)

    return PredictOutput(
        bboxes=bboxes,
        masks=masks,
        confidences=confidences,
        preview=preview,
    )


def mask_to_pil(masks: "torch.Tensor", shape: tuple[int, int]) -> list[Image.Image]:
    masks = masks.float()
    n = masks.shape[0]
    return [ensure_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]
