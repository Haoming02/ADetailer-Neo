from __future__ import annotations

from collections import UserList
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, partial
from typing import Any, Literal, NamedTuple, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
)


@dataclass
class SkipImg2ImgOrig:
    steps: int
    sampler_name: str
    width: int
    height: int


class Arg(NamedTuple):
    attr: str
    name: str


class ArgsList(UserList):
    @cached_property
    def attrs(self) -> tuple[str, ...]:
        return tuple(attr for attr, _ in self)

    @cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(name for _, name in self)


class ADetailerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ad_model: str = "None"
    ad_model_classes: str = ""
    ad_tab_enable: bool = True
    ad_prompt: str = ""
    ad_negative_prompt: str = ""
    ad_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    ad_mask_filter_method: Literal["Area", "Confidence"] = "Area"
    ad_mask_k: NonNegativeInt = 0
    ad_mask_min_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    ad_mask_max_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    ad_dilate_erode: int = 4
    ad_x_offset: int = 0
    ad_y_offset: int = 0
    ad_mask_merge_invert: Literal["None", "Merge", "Merge and Invert"] = "None"
    ad_mask_blur: NonNegativeInt = 4
    ad_denoising_strength: float = Field(default=0.4, ge=0.0, le=1.0)
    ad_inpaint_only_masked: bool = True
    ad_inpaint_only_masked_padding: NonNegativeInt = 32
    ad_use_inpaint_width_height: bool = False
    ad_inpaint_width: PositiveInt = 512
    ad_inpaint_height: PositiveInt = 512
    ad_use_steps: bool = False
    ad_steps: PositiveInt = Field(default=20, ge=1, le=150)
    ad_use_cfg_scale: bool = False
    ad_cfg_scale: PositiveFloat = Field(default=4.0, ge=1.0, le=24.0)
    ad_use_checkpoint: bool = False
    ad_checkpoint: Optional[str] = None
    ad_use_vae: bool = False
    ad_vae: Optional[str] = None
    ad_use_sampler: bool = False
    ad_sampler: str = "Use same sampler"
    ad_scheduler: str = "Use same scheduler"
    ad_use_noise_multiplier: bool = False
    ad_noise_multiplier: float = Field(default=1.0, ge=0.5, le=1.5)
    ad_restore_face: bool = False
    ad_controlnet_model: str = "None"
    ad_controlnet_module: str = "None"
    ad_controlnet_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    ad_controlnet_guidance_start_end: tuple[float, float] = Field(default=(0.0, 1.0))
    is_api: bool = True

    @field_validator("is_api", mode="before")
    @classmethod
    def is_api_validator(cls, v: Any):
        return type(v) is not tuple

    @staticmethod
    def pop_param(
        key: str,
        pops: list[str] = None,
        condition: Any = None,
        *,
        params: dict[str, Any],
    ):
        if key not in params:
            return

        value = params[key]
        condition = (not bool(value)) if condition is None else value == condition

        if condition:
            if pops is None:
                pops = [key]
            for k in pops:
                params.pop(k, None)

    def extra_params(self, suffix: str = "") -> dict[str, Any]:
        if self.need_skip():
            return {}

        p = {name: getattr(self, attr) for attr, name in ALL_ARGS}
        p.pop("ADetailer tab enable", None)

        ppop = partial(self.pop_param, params=p)
        _CNET_GSE = "ADetailer ControlNet guidance start/end"

        ppop("ADetailer model classes")
        ppop("ADetailer prompt")
        ppop("ADetailer negative prompt")
        ppop(
            "ADetailer mask only top k",
            [
                "ADetailer mask only top k",
                "ADetailer method to decide top k masks",
            ],
            condition=0,
        )
        ppop("ADetailer mask min ratio", condition=0.0)
        ppop("ADetailer mask max ratio", condition=1.0)
        ppop("ADetailer x offset", condition=0)
        ppop("ADetailer y offset", condition=0)
        ppop("ADetailer mask merge invert", condition="None")
        ppop(
            "ADetailer inpaint only masked",
            ["ADetailer inpaint padding"],
        )
        ppop(
            "ADetailer use inpaint width height",
            [
                "ADetailer use inpaint width height",
                "ADetailer inpaint width",
                "ADetailer inpaint height",
            ],
        )
        ppop(
            "ADetailer use separate steps",
            [
                "ADetailer use separate steps",
                "ADetailer steps",
            ],
        )
        ppop(
            "ADetailer use separate CFG scale",
            [
                "ADetailer use separate CFG scale",
                "ADetailer CFG scale",
            ],
        )
        ppop(
            "ADetailer use separate checkpoint",
            [
                "ADetailer use separate checkpoint",
                "ADetailer checkpoint",
            ],
        )
        ppop(
            "ADetailer use separate VAE",
            [
                "ADetailer use separate VAE",
                "ADetailer VAE",
            ],
        )
        ppop(
            "ADetailer use separate sampler",
            [
                "ADetailer use separate sampler",
                "ADetailer sampler",
                "ADetailer scheduler",
            ],
        )
        ppop("ADetailer sampler", condition="Use same sampler")
        ppop("ADetailer scheduler", condition="Use same scheduler")
        ppop(
            "ADetailer use separate noise multiplier",
            [
                "ADetailer use separate noise multiplier",
                "ADetailer noise multiplier",
            ],
        )
        ppop("ADetailer restore face")
        ppop(
            "ADetailer ControlNet model",
            [
                "ADetailer ControlNet model",
                "ADetailer ControlNet module",
                "ADetailer ControlNet weight",
                _CNET_GSE,
            ],
            condition="None",
        )
        ppop("ADetailer ControlNet module", condition="None")
        ppop("ADetailer ControlNet weight", condition=1.0)
        ppop(_CNET_GSE, condition=(0.0, 1.0))

        if _CNET_GSE in p:
            p[_CNET_GSE] = str(p.pop(_CNET_GSE))

        if suffix:
            p = {k + suffix: v for k, v in p.items()}

        return p

    def is_mediapipe(self) -> bool:
        return not self.ad_model.lower().endswith(".pt")

    def need_skip(self) -> bool:
        return self.ad_model == "None" or self.ad_tab_enable is False


_all_args = [
    ("ad_model", "ADetailer model"),
    ("ad_model_classes", "ADetailer model classes"),
    ("ad_tab_enable", "ADetailer tab enable"),
    ("ad_prompt", "ADetailer prompt"),
    ("ad_negative_prompt", "ADetailer negative prompt"),
    ("ad_confidence", "ADetailer confidence"),
    ("ad_mask_filter_method", "ADetailer method to decide top k masks"),
    ("ad_mask_k", "ADetailer mask only top k"),
    ("ad_mask_min_ratio", "ADetailer mask min ratio"),
    ("ad_mask_max_ratio", "ADetailer mask max ratio"),
    ("ad_x_offset", "ADetailer x offset"),
    ("ad_y_offset", "ADetailer y offset"),
    ("ad_dilate_erode", "ADetailer dilate erode"),
    ("ad_mask_merge_invert", "ADetailer mask merge invert"),
    ("ad_mask_blur", "ADetailer mask blur"),
    ("ad_denoising_strength", "ADetailer denoising strength"),
    ("ad_inpaint_only_masked", "ADetailer inpaint only masked"),
    ("ad_inpaint_only_masked_padding", "ADetailer inpaint padding"),
    ("ad_use_inpaint_width_height", "ADetailer use inpaint width height"),
    ("ad_inpaint_width", "ADetailer inpaint width"),
    ("ad_inpaint_height", "ADetailer inpaint height"),
    ("ad_use_steps", "ADetailer use separate steps"),
    ("ad_steps", "ADetailer steps"),
    ("ad_use_cfg_scale", "ADetailer use separate CFG scale"),
    ("ad_cfg_scale", "ADetailer CFG scale"),
    ("ad_use_checkpoint", "ADetailer use separate checkpoint"),
    ("ad_checkpoint", "ADetailer checkpoint"),
    ("ad_use_vae", "ADetailer use separate VAE"),
    ("ad_vae", "ADetailer VAE"),
    ("ad_use_sampler", "ADetailer use separate sampler"),
    ("ad_sampler", "ADetailer sampler"),
    ("ad_scheduler", "ADetailer scheduler"),
    ("ad_use_noise_multiplier", "ADetailer use separate noise multiplier"),
    ("ad_noise_multiplier", "ADetailer noise multiplier"),
    ("ad_restore_face", "ADetailer restore face"),
    ("ad_controlnet_model", "ADetailer ControlNet model"),
    ("ad_controlnet_module", "ADetailer ControlNet module"),
    ("ad_controlnet_weight", "ADetailer ControlNet weight"),
    ("ad_controlnet_guidance_start_end", "ADetailer ControlNet guidance start/end"),
]

_args = [Arg(*args) for args in _all_args]
ALL_ARGS = ArgsList(_args)

BBOX_SORTBY = [
    "None",
    "Position (left to right)",
    "Position (center to edge)",
    "Area (large to small)",
]

MASK_MERGE_INVERT = ["None", "Merge", "Merge and Invert"]

_script_default = (
    "dynamic_prompting",
    "dynamic_thresholding",
    "wildcard_recursive",
    "wildcards",
    "lora_block_weight",
    "negpip",
)
SCRIPT_DEFAULT = ",".join(sorted(_script_default))

_builtin_script = ("soft_inpainting", "hypertile_script")
BUILTIN_SCRIPT = ",".join(sorted(_builtin_script))


class InpaintBBoxMatchMode(Enum):
    OFF = "Off"
    STRICT = "Strict"
    FREE = "Free"


INPAINT_BBOX_MATCH_MODES = [
    InpaintBBoxMatchMode.OFF.value,
    InpaintBBoxMatchMode.STRICT.value,
    InpaintBBoxMatchMode.FREE.value,
]
