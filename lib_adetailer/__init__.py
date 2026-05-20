from .args import ALL_ARGS, ADetailerArgs
from .common import PredictOutput, get_models
from .mediapipe import mediapipe_predict
from .ultralytics import ultralytics_predict

__version__ = "Neo"

ADETAILER = "ADetailer"

__all__ = [
    "__version__",
    "ADETAILER",
    "ALL_ARGS",
    "ADetailerArgs",
    "PredictOutput",
    "get_models",
    "mediapipe_predict",
    "ultralytics_predict",
]
