from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing


def need_call_process(p: "StableDiffusionProcessing") -> bool:
    if p.scripts is None:
        return False
    i: int = p.batch_index
    bs: int = p.batch_size
    return i == bs - 1


def need_call_postprocess(p: "StableDiffusionProcessing") -> bool:
    if p.scripts is None:
        return False
    return p.batch_index == 0


def is_img2img_inpaint(p: "StableDiffusionProcessing") -> bool:
    return getattr(p, "image_mask", None) is not None


def is_inpaint_only_masked(p: "StableDiffusionProcessing") -> bool:
    return getattr(p, "inpaint_full_res", False)


def is_skip_img2img(p: "StableDiffusionProcessing") -> bool:
    return getattr(p, "_ad_skip_img2img", False)


def get_i(p: "StableDiffusionProcessing") -> int:
    it: int = p.iteration
    bs: int = p.batch_size
    i: int = p.batch_index
    return it * bs + i
