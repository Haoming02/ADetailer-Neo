import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .common import PredictOutput, create_bbox_from_mask, create_mask_from_bbox


def mediapipe_predict(
    model_path: os.PathLike, image: Image.Image, confidence: float = 0.3
) -> PredictOutput:

    if "landmarker" in model_path:
        return _mediapipe_face_mesh(model_path, image, confidence)
    elif "face" in model_path:
        return _mediapipe_face_detection(model_path, image, confidence)
    else:
        raise RuntimeError(f'[ADetailer]: Invalid MediaPipe Model: "{model_path}"')


def _draw_preview(
    preview: Image.Image,
    bboxes: list[tuple[int, int, int, int]],
    masks: list[Image.Image],
) -> Image.Image:
    red = Image.new("RGB", preview.size, "red")
    for mask in masks:
        masked = Image.composite(red, preview, mask)
        preview = Image.blend(preview, masked, 0.25)

    draw = ImageDraw.Draw(preview)
    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=2)

    return preview


def _mediapipe_face_mesh(
    model_path: str,
    image: Image.Image,
    confidence: float = 0.3,
) -> PredictOutput:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    w, h = image.size
    img_array = np.array(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=20,
        min_face_detection_confidence=confidence,
    )

    with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
        pred = face_landmarker.detect(mp_image)

    if not pred.face_landmarks:
        return PredictOutput()

    preview = image.copy()
    masks = []
    confidences = []

    for landmarks in pred.face_landmarks:
        points = np.array([[int(land.x * w), int(land.y * h)] for land in landmarks])

        mask = Image.new("L", image.size, "black")
        draw = ImageDraw.Draw(mask)

        outline = cv2.convexHull(points).reshape(-1).tolist()
        draw.polygon(outline, fill="white")

        masks.append(mask)
        confidences.append(1.0)

    bboxes = create_bbox_from_mask(masks, image.size)

    preview = _draw_preview(preview, bboxes, masks)

    return PredictOutput(
        bboxes=bboxes,
        masks=masks,
        confidences=confidences,
        preview=preview,
    )


def _mediapipe_face_detection(
    model_path: str, image: Image.Image, confidence: float = 0.3
) -> PredictOutput:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    img_array = np.array(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_options, min_detection_confidence=confidence
    )

    with vision.FaceDetector.create_from_options(options) as detector:
        pred = detector.detect(mp_image)

    if not pred.detections:
        return PredictOutput()

    preview_array = img_array.copy()
    bboxes = []
    confidences = []

    for detection in pred.detections:
        bbox = detection.bounding_box
        x1 = int(bbox.origin_x)
        y1 = int(bbox.origin_y)
        x2 = int(x1 + bbox.width)
        y2 = int(y1 + bbox.height)

        confidences.append(detection.categories[0].score)
        bboxes.append([x1, y1, x2, y2])

        cv2.rectangle(preview_array, (x1, y1), (x2, y2), (255, 0, 0), 2)

    masks = create_mask_from_bbox(bboxes, image.size)
    preview = Image.fromarray(preview_array)

    return PredictOutput(
        bboxes=bboxes,
        masks=masks,
        confidences=confidences,
        preview=preview,
    )
