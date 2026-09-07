import copy
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessingImg2Img

import numpy as np

try:
    from lib_controlnet import external_code, global_state
    from lib_controlnet.external_code import ControlNetUnit
except ImportError:
    CNET_AVAILABLE = False
else:
    CNET_AVAILABLE = True

from modules import scripts

SUPPORTED_TYPES: Final[tuple[str]] = (
    "Inpaint",
    "Scribble",
    "Lineart",
    "OpenPose",
    "Tile",
    "Depth",
)


def find_script(p: "StableDiffusionProcessingImg2Img", script: str) -> scripts.Script:
    script = next((s for s in p.scripts.scripts if s.title() == script), None)
    assert script is not None
    return script


class ControlNetExt:
    def __init__(self):
        self._cnet_script: scripts.Script = None

    def update_scripts_args(
        self,
        p: "StableDiffusionProcessingImg2Img",
        model: str,
        module: str,
        weight: float,
        guidance_start: float,
        guidance_end: float,
    ):
        if model == "None":
            return

        image = np.asarray(p.init_images[0])
        mask = np.full_like(image, fill_value=255)

        cnet_image = {"image": image, "mask": mask}

        pres = external_code.pixel_perfect_resolution(
            image,
            target_H=p.height,
            target_W=p.width,
            resize_mode=external_code.resize_mode_from_value(p.resize_mode),
        )

        self.add_forge_script_to_adetailer_run(
            p,
            "ControlNet",
            [
                ControlNetUnit(
                    enabled=True,
                    image=cnet_image,
                    model=model,
                    module=module,
                    weight=weight,
                    guidance_start=guidance_start,
                    guidance_end=guidance_end,
                    processor_res=pres,
                )
            ],
        )

    def add_forge_script_to_adetailer_run(
        self,
        p: "StableDiffusionProcessingImg2Img",
        script_title: str,
        script_args: list,
    ):
        p.scripts = copy.copy(scripts.scripts_img2img)
        p.scripts.alwayson_scripts = []
        p.script_args_value = []

        if self._cnet_script is None:
            self._cnet_script = copy.copy(find_script(p, script_title))

        self._cnet_script.args_from = len(p.script_args_value)
        self._cnet_script.args_to = len(p.script_args_value) + len(script_args)
        p.scripts.alwayson_scripts.append(self._cnet_script)
        p.script_args_value.extend(script_args)


if CNET_AVAILABLE:

    def get_cn_models() -> list[str]:
        models = set()
        for mod in SUPPORTED_TYPES:
            models.update(global_state.get_filtered_controlnet_names(mod))
        models.remove("None")

        return ["None", "Passthrough", *sorted(models)]

    def get_cn_modules() -> list[str]:
        modules = set()
        for mod in SUPPORTED_TYPES:
            modules.update(global_state.get_filtered_preprocessors(mod))
        modules.remove("None")

        return ["None", *sorted(modules)]

else:

    def get_cn_models() -> list[str]:
        return ["None"]

    def get_cn_modules() -> list[str]:
        return ["None"]
