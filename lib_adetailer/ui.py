from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

import gradio as gr
from gradio_rangeslider import RangeSlider

from modules.launch_utils import is_installed
from modules.ui_components import FormColumn, FormRow, InputAccordion

from . import __version__
from .args import ALL_ARGS, MASK_MERGE_INVERT
from .controlnet import get_cn_models, get_cn_modules


class Widgets(SimpleNamespace):
    def tolist(self):
        return [getattr(self, attr) for attr in ALL_ARGS.attrs]


@dataclass
class WebuiInfo:
    ad_model_list: list[str]
    sampler_names: list[str]
    scheduler_names: list[str]
    t2i_button: gr.Button
    i2i_button: gr.Button
    checkpoints_list: list[str]
    vae_list: list[str]


# region Utils


def gr_interactive(value: bool = True):
    return gr.update(interactive=value)


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


def suffix(n: int, c: str = " ") -> str:
    return "" if n == 0 else c + ordinal(n + 1)


def on_widget_change(state: dict, value: Any, *, attr: str):
    if "is_api" in state:
        state = state.copy()
        state.pop("is_api")
    state[attr] = value
    return state


def on_generate(state: dict, *values: Any):
    for attr, value in zip(ALL_ARGS.attrs, values):
        state[attr] = value
    state["is_api"] = ()
    return state


def elem_id(item_id: str, n: int, is_img2img: bool) -> str:
    tab = "img2img" if is_img2img else "txt2img"
    suf = "" if n is None else suffix(n, "_")
    return f"script_{tab}_adetailer_{item_id}{suf}"


def state_init(w: Widgets) -> dict[str, Any]:
    return {attr: getattr(w, attr).value for attr in ALL_ARGS.attrs}


# region UI


def adui(num_models: int, is_img2img: bool, webui_info: WebuiInfo):
    states = []
    infotext_fields = []
    eid = partial(elem_id, n=None, is_img2img=is_img2img)

    with InputAccordion(
        value=False, label="ADetailer", elem_id=eid("ad_main_accordion")
    ) as ad_enable:
        with gr.Row():
            with gr.Column(scale=10):
                ad_skip_img2img = gr.Checkbox(
                    value=False,
                    label="Skip img2img",
                    visible=is_img2img,
                    elem_id=eid("ad_skip_img2img"),
                )
            with gr.Column(scale=1, min_width=80):
                gr.Markdown(
                    f"ver. {__version__}",
                    elem_id=eid("ad_version"),
                )

        infotext_fields.append((ad_enable, "ADetailer enable"))
        infotext_fields.append((ad_skip_img2img, "ADetailer skip img2img"))

        with gr.Tabs():
            for n in range(num_models):
                with gr.Tab(label=ordinal(n + 1)):
                    state, infofields = _ui_group(
                        n=n,
                        is_img2img=is_img2img,
                        webui_info=webui_info,
                    )

                states.append(state)
                infotext_fields.extend(infofields)

    components = [ad_enable, ad_skip_img2img, *states]
    return components, infotext_fields


def _ui_group(n: int, is_img2img: bool, webui_info: WebuiInfo):
    w = Widgets()
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    w.ad_tab_enable = gr.Checkbox(
        value=(n == 0),
        label=f"Enable {ordinal(n + 1)} Tab",
        elem_id=eid("ad_tab_enable"),
    )

    with FormColumn():
        w.ad_model = gr.Dropdown(
            label="ADetailer Detector" + suffix(n),
            choices=["None", *webui_info.ad_model_list],
            value="None",
            type="value",
            elem_id=eid("ad_model"),
        )
        w.ad_model_classes = gr.Textbox(
            label="Detector Classes" + suffix(n),
            value="",
            visible=False,
            interactive=is_installed("clip"),
            placeholder=(
                'Comma-separated class names to detect (e.g. "person,cat" ; default: COCO 80 classes)'
                if is_installed("clip")
                else "Detector Classes require Clip to be installed (https://github.com/ultralytics/CLIP.git)"
            ),
            lines=1,
            max_lines=1,
            elem_id=eid("ad_model_classes"),
        )

        w.ad_model.change(
            fn=lambda model: gr.update(visible="-world" in model),
            inputs=[w.ad_model],
            outputs=[w.ad_model_classes],
            queue=False,
            show_progress=False,
        )

        with FormColumn():
            with gr.Row(elem_id=eid("ad_toprow_prompt")):
                w.ad_prompt = gr.Textbox(
                    value="",
                    show_label=False,
                    lines=3,
                    max_lines=6,
                    placeholder=f"ADetailer Prompt{suffix(n)}"
                    + "\n(if blank, the original prompt is used)",
                    elem_id=eid("ad_prompt"),
                )
            with gr.Row(elem_id=eid("ad_toprow_negative_prompt")):
                w.ad_negative_prompt = gr.Textbox(
                    value="",
                    show_label=False,
                    lines=3,
                    max_lines=6,
                    placeholder=f"ADetailer Negative Prompt{suffix(n)}"
                    + "\n(if blank, the original negative prompt is used)",
                    elem_id=eid("ad_negative_prompt"),
                )

    with gr.Column(variant="compact"):
        with gr.Accordion("Detection", open=False):
            detection(w, n, is_img2img)
        with gr.Accordion("Mask Preprocessing", open=False):
            mask_preprocessing(w, n, is_img2img)
        with gr.Accordion("Inpaint Parameters", open=False):
            inpainting(w, n, is_img2img, webui_info)
        with gr.Accordion("ControlNet", open=False):
            controlnet(w, n, is_img2img)

    state = gr.State(lambda: state_init(w))

    for attr in ALL_ARGS.attrs:
        widget = getattr(w, attr)
        on_change = partial(on_widget_change, attr=attr)
        widget.change(fn=on_change, inputs=[state, widget], outputs=state, queue=False)

    all_inputs = [state, *w.tolist()]
    target_button = webui_info.i2i_button if is_img2img else webui_info.t2i_button
    target_button.click(fn=on_generate, inputs=all_inputs, outputs=state, queue=False)

    infotext_fields = [(getattr(w, attr), name + suffix(n)) for attr, name in ALL_ARGS]
    return state, infotext_fields


# region SubGroups


def detection(w: Widgets, n: int, is_img2img: bool):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    with FormColumn():
        w.ad_confidence = gr.Slider(
            label="Confidence Threshold",
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            value=0.3,
            elem_id=eid("ad_confidence"),
        )

        with FormRow():
            w.ad_mask_min_ratio = gr.Slider(
                label="Min Area Ratio",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.0,
                elem_id=eid("ad_mask_min_ratio"),
            )
            w.ad_mask_max_ratio = gr.Slider(
                label="Max Area Ratio",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=1.0,
                elem_id=eid("ad_mask_max_ratio"),
            )

        with FormRow():
            w.ad_mask_k = gr.Slider(
                label="Keep only the top n masks",
                info="0 for unlimited",
                minimum=0,
                maximum=10,
                step=1,
                value=0,
                elem_id=eid("ad_mask_k"),
            )
            w.ad_mask_filter_method = gr.Radio(
                label="Filter Method",
                choices=("Area", "Confidence"),
                value="Area",
                elem_id=eid("ad_mask_filter_method"),
            )


def mask_preprocessing(w: Widgets, n: int, is_img2img: bool):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    with FormColumn():
        with FormRow():
            w.ad_x_offset = gr.Slider(
                label="Offset: Right (+) / Left(-)",
                minimum=-256,
                maximum=256,
                step=4,
                value=0,
                elem_id=eid("ad_x_offset"),
            )
            w.ad_y_offset = gr.Slider(
                label="Offset: Up (+) / Down (-)",
                minimum=-256,
                maximum=256,
                step=4,
                value=0,
                elem_id=eid("ad_y_offset"),
            )
            w.ad_dilate_erode = gr.Slider(
                label="Mask: Dilation (+) / Erosion (-)",
                minimum=-256,
                maximum=256,
                step=4,
                value=4,
                elem_id=eid("ad_dilate_erode"),
            )

        w.ad_mask_merge_invert = gr.Radio(
            label="Masks Merge Mode",
            choices=MASK_MERGE_INVERT,
            value="None",
            elem_id=eid("ad_mask_merge_invert"),
        )


# TODO
def inpainting(w: Widgets, n: int, is_img2img: bool, webui_info: WebuiInfo):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    with gr.Group():
        with gr.Row():
            w.ad_mask_blur = gr.Slider(
                label="Inpaint mask blur" + suffix(n),
                minimum=0,
                maximum=64,
                step=1,
                value=4,
                visible=True,
                elem_id=eid("ad_mask_blur"),
            )

            w.ad_denoising_strength = gr.Slider(
                label="Inpaint denoising strength" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.4,
                visible=True,
                elem_id=eid("ad_denoising_strength"),
            )

        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_inpaint_only_masked = gr.Checkbox(
                    label="Inpaint only masked" + suffix(n),
                    value=True,
                    visible=True,
                    elem_id=eid("ad_inpaint_only_masked"),
                )
                w.ad_inpaint_only_masked_padding = gr.Slider(
                    label="Inpaint only masked padding, pixels" + suffix(n),
                    minimum=0,
                    maximum=256,
                    step=4,
                    value=32,
                    visible=True,
                    elem_id=eid("ad_inpaint_only_masked_padding"),
                )

                w.ad_inpaint_only_masked.change(
                    gr_interactive,
                    inputs=w.ad_inpaint_only_masked,
                    outputs=w.ad_inpaint_only_masked_padding,
                    queue=False,
                )

            with gr.Column(variant="compact"):
                w.ad_use_inpaint_width_height = gr.Checkbox(
                    label="Use separate width/height" + suffix(n),
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_inpaint_width_height"),
                )

                w.ad_inpaint_width = gr.Slider(
                    label="inpaint width" + suffix(n),
                    minimum=64,
                    maximum=2048,
                    step=4,
                    value=512,
                    visible=True,
                    elem_id=eid("ad_inpaint_width"),
                )

                w.ad_inpaint_height = gr.Slider(
                    label="inpaint height" + suffix(n),
                    minimum=64,
                    maximum=2048,
                    step=4,
                    value=512,
                    visible=True,
                    elem_id=eid("ad_inpaint_height"),
                )

                w.ad_use_inpaint_width_height.change(
                    lambda value: (gr_interactive(value), gr_interactive(value)),
                    inputs=w.ad_use_inpaint_width_height,
                    outputs=[w.ad_inpaint_width, w.ad_inpaint_height],
                    queue=False,
                )

        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_use_steps = gr.Checkbox(
                    label="Use separate steps" + suffix(n),
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_steps"),
                )

                w.ad_steps = gr.Slider(
                    label="ADetailer steps" + suffix(n),
                    minimum=1,
                    maximum=150,
                    step=1,
                    value=28,
                    visible=True,
                    elem_id=eid("ad_steps"),
                )

                w.ad_use_steps.change(
                    gr_interactive,
                    inputs=w.ad_use_steps,
                    outputs=w.ad_steps,
                    queue=False,
                )

            with gr.Column(variant="compact"):
                w.ad_use_cfg_scale = gr.Checkbox(
                    label="Use separate CFG scale" + suffix(n),
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_cfg_scale"),
                )

                w.ad_cfg_scale = gr.Slider(
                    label="ADetailer CFG scale" + suffix(n),
                    minimum=0.0,
                    maximum=30.0,
                    step=0.5,
                    value=7.0,
                    visible=True,
                    elem_id=eid("ad_cfg_scale"),
                )

                w.ad_use_cfg_scale.change(
                    gr_interactive,
                    inputs=w.ad_use_cfg_scale,
                    outputs=w.ad_cfg_scale,
                    queue=False,
                )

        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_use_checkpoint = gr.Checkbox(
                    label="Use separate checkpoint" + suffix(n),
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_checkpoint"),
                )

                ckpts = ["Use same checkpoint", *webui_info.checkpoints_list]

                w.ad_checkpoint = gr.Dropdown(
                    label="ADetailer checkpoint" + suffix(n),
                    choices=ckpts,
                    value=ckpts[0],
                    visible=True,
                    elem_id=eid("ad_checkpoint"),
                )

            with gr.Column(variant="compact"):
                w.ad_use_vae = gr.Checkbox(
                    label="Use separate VAE" + suffix(n),
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_vae"),
                )

                vaes = ["Use same VAE", *webui_info.vae_list]

                w.ad_vae = gr.Dropdown(
                    label="ADetailer VAE" + suffix(n),
                    choices=vaes,
                    value=vaes[0],
                    visible=True,
                    elem_id=eid("ad_vae"),
                )

        with gr.Row(), gr.Column(variant="compact"):
            w.ad_use_sampler = gr.Checkbox(
                label="Use separate sampler" + suffix(n),
                value=False,
                visible=True,
                elem_id=eid("ad_use_sampler"),
            )

            sampler_names = [
                "Use same sampler",
                *webui_info.sampler_names,
            ]

            with gr.Row():
                w.ad_sampler = gr.Dropdown(
                    label="ADetailer sampler" + suffix(n),
                    choices=sampler_names,
                    value=sampler_names[1],
                    visible=True,
                    elem_id=eid("ad_sampler"),
                )

                scheduler_names = [
                    "Use same scheduler",
                    *webui_info.scheduler_names,
                ]
                w.ad_scheduler = gr.Dropdown(
                    label="ADetailer scheduler" + suffix(n),
                    choices=scheduler_names,
                    value=scheduler_names[0],
                    visible=len(scheduler_names) > 1,
                    elem_id=eid("ad_scheduler"),
                )

                w.ad_use_sampler.change(
                    lambda value: (gr_interactive(value), gr_interactive(value)),
                    inputs=w.ad_use_sampler,
                    outputs=[w.ad_sampler, w.ad_scheduler],
                    queue=False,
                )

        with gr.Row():
            with gr.Column(variant="compact"):
                w.ad_use_noise_multiplier = gr.Checkbox(
                    label="Use separate noise multiplier" + suffix(n),
                    value=False,
                    visible=True,
                    elem_id=eid("ad_use_noise_multiplier"),
                )

                w.ad_noise_multiplier = gr.Slider(
                    label="Noise multiplier for img2img" + suffix(n),
                    minimum=0.5,
                    maximum=1.5,
                    step=0.01,
                    value=1.0,
                    visible=True,
                    elem_id=eid("ad_noise_multiplier"),
                )

                w.ad_use_noise_multiplier.change(
                    gr_interactive,
                    inputs=w.ad_use_noise_multiplier,
                    outputs=w.ad_noise_multiplier,
                    queue=False,
                )

        with gr.Row(), gr.Column(variant="compact"):
            w.ad_restore_face = gr.Checkbox(
                label="Restore faces after ADetailer" + suffix(n),
                value=False,
                elem_id=eid("ad_restore_face"),
            )


def controlnet(w: Widgets, n: int, is_img2img: bool):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    with FormColumn():
        with FormRow():
            w.ad_controlnet_module = gr.Dropdown(
                label="ControlNet Module",
                choices=get_cn_modules(),
                value="None",
                type="value",
                elem_id=eid("ad_controlnet_module"),
            )
            w.ad_controlnet_model = gr.Dropdown(
                label="ControlNet Model",
                choices=get_cn_models(),
                value="None",
                type="value",
                elem_id=eid("ad_controlnet_model"),
            )

        with FormRow():
            w.ad_controlnet_weight = gr.Slider(
                label="ControlNet Weight",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=1.0,
                elem_id=eid("ad_controlnet_weight"),
            )
            w.ad_controlnet_guidance_start_end = RangeSlider(
                label="ControlNet Guidance Start/End",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=(0.0, 1.0),
                elem_id=eid("ad_controlnet_guidance_start_end"),
            )
