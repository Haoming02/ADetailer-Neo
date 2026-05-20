from .args import ALL_ARGS, ADetailerArgs
from .detection.common import PredictOutput, get_models
from .detection.mediapipe import mediapipe_predict
from .detection.ultralytics import ultralytics_predict

__version__ = "Neo"

__all__ = [
    "__version__",
    "ALL_ARGS",
    "ADetailerArgs",
    "PredictOutput",
    "get_models",
    "mediapipe_predict",
    "ultralytics_predict",
]
