import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Optional, TypeVar

from PIL import Image, ImageDraw
from torch.hub import download_url_to_file

from modules.shared import cmd_opts

from ..utils import NUM, print

URL = TypeVar("URL", bound=str)

no_huggingface: bool = getattr(cmd_opts, "ad_no_huggingface", False)


@dataclass
class PredictOutput:
    bboxes: list[tuple[NUM]] = field(default_factory=tuple)
    masks: list[Image.Image] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    preview: Optional[Image.Image] = None


def _scan_models(path: Path) -> list[Path]:
    return [
        obj
        for obj in path.rglob("*")
        if (obj.is_file() and obj.suffix in (".pt", ".tflite", ".task"))
    ]


def _download_model(url: URL, filename: os.PathLike):
    if not os.path.isfile(filename):
        download_url_to_file(url=url, dst=filename, progress=False)


def _download(folder: os.PathLike, names: dict[str, URL]):

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures: list[Future] = [
            executor.submit(
                _download_model,
                url=url,
                filename=os.path.join(folder, file),
            )
            for file, url in names.items()
        ]

    for file, future in zip(names.keys(), futures):
        try:
            future.result()
        except Exception:
            print(f'Failed to download "{file}"')


def get_models(ad_dir: str, *extra_dirs: str) -> dict[str, os.PathLike]:

    TO_DOWNLOAD: Final[dict[str, URL]] = {
        # https://huggingface.co/Bingsu/adetailer
        "face_yolov8n.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt?download=true",
        "face_yolov8s.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt?download=true",
        "hand_yolov8n.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt?download=true",
        "hand_yolov8s.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt?download=true",
        "person_yolov8n-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8n-seg.pt?download=true",
        "person_yolov8s-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8s-seg.pt?download=true",
        # https://github.com/ultralytics/assets
        "yolov8x-worldv2.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-worldv2.pt",
        # https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector#models
        "mediapipe_face_short.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        "mediapipe_face_full.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite",
        # https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    }

    if not no_huggingface:
        print("Loading Models...")
        _download(ad_dir, TO_DOWNLOAD)

    models: dict[str, os.PathLike] = {}
    model_paths: list[Path] = []

    for _dir in (ad_dir, *extra_dirs):
        if not os.path.isdir(_dir):
            continue
        model_paths.extend(_scan_models(Path(_dir)))

    for path in model_paths:
        if path.name in models:
            continue
        models[path.name] = str(path)

    return models


# region BBox / Mask


def create_mask_from_bbox(
    bboxes: list[tuple[int, int, int, int]], shape: tuple[int, int]
) -> list[Image.Image]:
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def create_bbox_from_mask(
    masks: list[Image.Image], shape: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    bboxes = []
    for mask in masks:
        mask = mask.resize(shape)
        bbox = mask.getbbox()
        if bbox is not None:
            bboxes.append(list(bbox))
    return bboxes
