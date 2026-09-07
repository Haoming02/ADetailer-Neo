from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing


from modules.shared import opts


@contextmanager
def pause_total_tqdm():
    with patch.dict(opts.data, {"multiple_tqdm": False}, clear=False):
        yield


@contextmanager
def preserve_prompts(p: "StableDiffusionProcessing"):
    try:
        all_pt = p.all_prompts.copy()
        all_np = p.all_negative_prompts.copy()

        yield

    finally:
        p.all_prompts = all_pt
        p.all_negative_prompts = all_np


def copy_extra_params(extra_params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in extra_params.items() if not callable(v)}
