import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Optional

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

from modules.shared import cmd_opts

from ..utils import NUM, print

no_huggingface: bool = getattr(cmd_opts, "ad_no_huggingface", False)


@dataclass
class PredictOutput(Generic[NUM]):
    bboxes: list[list[NUM]] = field(default_factory=list)
    masks: list[Image.Image] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    preview: Optional[Image.Image] = None


def _scan_models(path: Path) -> list[Path]:
    return [p for p in path.rglob("*") if p.is_file() and p.suffix == ".pt"]


def _hf_download(file: str, repo_id: str, remote: bool = True) -> os.PathLike:
    if remote:
        with suppress(Exception):
            return hf_hub_download(repo_id, file)
        with suppress(Exception):
            return hf_hub_download(repo_id, file, endpoint="https://hf-mirror.com")

    with suppress(Exception):
        return hf_hub_download(repo_id, file, local_files_only=True)

    if remote:
        print(f'Failed to download model "{file}"')

    return "INVALID"


def _download(names: list[str]) -> dict[str, os.PathLike]:
    models = {}

    with ThreadPoolExecutor() as executor:
        for name in names:
            if "-world" in name:
                models[name] = executor.submit(
                    _hf_download,
                    name,
                    repo_id="Bingsu/yolo-world-mirror",
                    remote=not no_huggingface,
                )
            else:
                models[name] = executor.submit(
                    _hf_download,
                    name,
                    repo_id="Bingsu/adetailer",
                    remote=not no_huggingface,
                )

    return {name: future.result() for name, future in models.items()}


def get_models(*dirs: str) -> dict[str, os.PathLike]:
    model_paths: list[Path] = []

    for _dir in dirs:
        if not os.path.isdir(_dir):
            continue
        model_paths.extend(_scan_models(Path(_dir)))

    models: dict[str, os.PathLike] = {}

    to_download = (
        "face_yolov8n.pt",
        "face_yolov8s.pt",
        "hand_yolov8n.pt",
        "person_yolov8n-seg.pt",
        "person_yolov8s-seg.pt",
        "yolov8x-worldv2.pt",
    )

    models.update(_download(to_download))

    models.update(
        {
            "mediapipe_face_full": "mediapipe_face_full",
            "mediapipe_face_short": "mediapipe_face_short",
            "mediapipe_face_mesh": "mediapipe_face_mesh",
            "mediapipe_face_mesh_eyes_only": "mediapipe_face_mesh_eyes_only",
        }
    )

    models = {k: v for k, v in models.items() if v != "INVALID"}

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
