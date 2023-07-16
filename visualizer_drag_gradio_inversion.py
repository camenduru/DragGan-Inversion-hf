# https://huggingface.co/DragGan/DragGan-Models
# https://arxiv.org/abs/2305.10973
import os
import os.path as osp
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
import time
import tempfile
import psutil

import gradio as gr
import numpy as np
import torch
from PIL import Image
import uuid

import dnnlib
from gradio_utils import (
    ImageMask,
    draw_mask_on_image,
    draw_points_on_image,
    get_latest_points_pair,
    get_valid_mask,
    on_change_single_global_state,
)
from viz.renderer import Renderer, add_watermark_np
from torch_utils.pti import run_PTI, export_updated_pickle

# download models from Hugging Face hub
from huggingface_hub import snapshot_download

model_dir = Path("./checkpoints")
snapshot_download("DragGan/DragGan-Models", repo_type="model", local_dir=model_dir)

# parser = ArgumentParser()
# parser.add_argument('--share', action='store_true')
# parser.add_argument('--cache-dir', type=str, default='./checkpoints')
# args = parser.parse_args()

cache_dir = model_dir

device = "cuda"
IS_SPACE = "DragGan/DragGan" in os.environ.get("SPACE_ID", "")
TIMEOUT = 80


def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points


def clear_state(global_state, target=None):
    """Clear target history state from global_state
    If target is not defined, points and mask will be both removed.
    1. set global_state['points'] as empty dict
    2. set global_state['mask'] as full-one mask.
    """
    if target is None:
        target = ["point", "mask"]
    if not isinstance(target, list):
        target = [target]
    if "point" in target:
        global_state["points"] = dict()
        print("Clear Points State!")
    if "mask" in target:
        image_raw = global_state["images"]["image_raw"]
        global_state["mask"] = np.ones(
            (image_raw.size[1], image_raw.size[0]), dtype=np.uint8
        )
        print("Clear mask State!")

    return global_state


def init_images(global_state):
    """This function is called only ones with Gradio App is started.
    0. pre-process global_state, unpack value from global_state of need
    1. Re-init renderer
    2. run `renderer._render_drag_impl` with `is_drag=False` to generate
       new image
    3. Assign images to global state and re-generate mask
    """

    if isinstance(global_state, gr.State):
        state = global_state.value
    else:
        state = global_state

    state["renderer"].init_network(
        state["generator_params"],  # res
        state["pretrained_weight"],  # pkl
        state["params"]["seed"],  # w0_seed,
        state["w_pivot"],  # w_load
        state["params"]["latent_space"] == "w+",  # w_plus
        "const",
        state["params"]["trunc_psi"],  # trunc_psi,
        state["params"]["trunc_cutoff"],  # trunc_cutoff,
        None,  # input_transform
        state["params"]["lr"],  # lr,
    )

    state["renderer"]._render_drag_impl(
        state["generator_params"], is_drag=False, to_pil=True
    )

    init_image = state["generator_params"].image
    state["images"]["image_orig"] = init_image
    state["images"]["image_raw"] = init_image
    state["images"]["image_show"] = Image.fromarray(
        add_watermark_np(np.array(init_image))
    )
    state["mask"] = np.ones((init_image.size[1], init_image.size[0]), dtype=np.uint8)
    return global_state


def update_image_draw(image, points, mask, show_mask, global_state=None):
    image_draw = draw_points_on_image(image, points)
    if (
        show_mask
        and mask is not None
        and not (mask == 0).all()
        and not (mask == 1).all()
    ):
        image_draw = draw_mask_on_image(image_draw, mask)

    image_draw = Image.fromarray(add_watermark_np(np.array(image_draw)))
    if global_state is not None:
        global_state["images"]["image_show"] = image_draw
    return image_draw


def preprocess_mask_info(global_state, image):
    """Function to handle mask information.
    1. last_mask is None: Do not need to change mask, return mask
    2. last_mask is not None:
        2.1 global_state is remove_mask:
        2.2 global_state is add_mask:
    """
    if isinstance(image, dict):
        last_mask = get_valid_mask(image["mask"])
    else:
        last_mask = None
    mask = global_state["mask"]

    # mask in global state is a placeholder with all 1.
    if (mask == 1).all():
        mask = last_mask

    # last_mask = global_state['last_mask']
    editing_mode = global_state["editing_state"]

    if last_mask is None:
        return global_state

    if editing_mode == "remove_mask":
        updated_mask = np.clip(mask - last_mask, 0, 1)
        print(f"Last editing_state is {editing_mode}, do remove.")
    elif editing_mode == "add_mask":
        updated_mask = np.clip(mask + last_mask, 0, 1)
        print(f"Last editing_state is {editing_mode}, do add.")
    else:
        updated_mask = mask
        print(f"Last editing_state is {editing_mode}, " "do nothing to mask.")

    global_state["mask"] = updated_mask
    # global_state['last_mask'] = None  # clear buffer
    return global_state


def print_memory_usage():
    # Print system memory usage
    print(f"System memory usage: {psutil.virtual_memory().percent}%")

    # Print GPU memory usage
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1e9} GB")
        print(f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9} GB")
        device_properties = torch.cuda.get_device_properties(device)
        available_memory = (
            device_properties.total_memory - torch.cuda.max_memory_allocated()
        )
        print(f"Available GPU memory: {available_memory / 1e9} GB")
    else:
        print("No GPU available")


# filter large models running on SPAC

css = """
#output-image {
    width: 100% !important;
    aspect-ratio: 1 / 1 !important;
    height: auto !important;
}
#output-image canvas {
    width: 100% !important;
    aspect-ratio: 1 / 1 !important;
    height: auto !important;
}
"""
with gr.Blocks(css=css) as app:
    gr.Markdown(
        """
# DragGAN - Drag Your GAN - Face Inversion
                
## Interactive Point-based Manipulation on the Generative Image Manifold
### Unofficial Gradio Demo

**Due to high demand, only one model can be run at a time, or you can duplicate the space and run your own copy.**

<a href="https://huggingface.co/spaces/DragGan/DragGan-Inversion?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
<img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a> for no queue on your own hardware.</p>

* Official Repo: [XingangPan](https://github.com/XingangPan/DragGAN)
* Gradio Demo by: [LeoXing1996](https://github.com/LeoXing1996) © [OpenMMLab MMagic](https://github.com/open-mmlab/mmagic)
* Inversion Code: [ProgrammingHut](https://www.youtube.com/watch?v=viWiOC1Mikw), [EthanZhangCN](https://github.com/EthanZhangCN) 
"""
    )

    # renderer = Renderer()
    global_state = gr.State(
        {
            "images": {
                # image_orig: the original image, change with seed/model is changed
                # image_raw: image with mask and points, change durning optimization
                # image_show: image showed on screen
            },
            "temporal_params": {
                # stop
            },
            "w_pivot": None,
            "mask": None,  # mask for visualization, 1 for editing and 0 for unchange
            "last_mask": None,  # last edited mask
            "show_mask": True,  # add button
            "generator_params": dnnlib.EasyDict(),
            "params": {
                "seed": int(np.random.randint(0, 2**32 - 1)),
                "motion_lambda": 20,
                "r1_in_pixels": 3,
                "r2_in_pixels": 12,
                "magnitude_direction_in_pixels": 1.0,
                "latent_space": "w+",
                "trunc_psi": 0.7,
                "trunc_cutoff": None,
                "lr": 0.01,
            },
            "device": device,
            "draw_interval": 1,
            "renderer": Renderer(disable_timing=True),
            "points": {},
            "curr_point": None,
            "curr_type_point": "start",
            "editing_state": "add_points",
            "pretrained_weight": str(model_dir / "stylegan2-ffhq1024x1024.pkl"),
        }
    )

    # init image
    global_state = init_images(global_state)
    with gr.Row():
        with gr.Row():
            # Left --> tools
            with gr.Column():
                # Latent
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value="Latent", show_label=False)

                    with gr.Column(scale=4, min_width=10):
                        form_seed_number = gr.Slider(
                            mininium=0,
                            maximum=2**32 - 1,
                            step=1,
                            value=global_state.value["params"]["seed"],
                            interactive=True,
                            # randomize=True,
                            label="Seed",
                        )
                        form_lr_number = gr.Number(
                            value=global_state.value["params"]["lr"],
                            precision=5,
                            interactive=True,
                            label="Step Size",
                        )

                        with gr.Row():
                            with gr.Column(scale=2, min_width=10):
                                form_reset_image = gr.Button("Reset Image")
                            with gr.Column(scale=3, min_width=10):
                                form_latent_space = gr.Radio(
                                    ["w", "w+"],
                                    value=global_state.value["params"]["latent_space"],
                                    interactive=True,
                                    label="Latent space to optimize",
                                    show_label=False,
                                )
                        with gr.Row():
                            with gr.Column(scale=3, min_width=10):
                                form_custom_image = gr.Image(
                                    type="filepath", label="Custom Image", height=100
                                )
                            with gr.Column(scale=3, min_width=10):
                                form_reset_custom_image = gr.Button(
                                    "Remove Custom Image", interactive=False
                                )

                # Drag
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value="Drag", show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                enable_add_points = gr.Button("Add Points")
                            with gr.Column(scale=1, min_width=10):
                                undo_points = gr.Button("Reset Points")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                form_start_btn = gr.Button("Start")
                            with gr.Column(scale=1, min_width=10):
                                form_stop_btn = gr.Button("Stop")

                        form_steps_number = gr.Number(
                            value=0, label="Steps", interactive=False
                        )

                # Mask
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value="Mask", show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        enable_add_mask = gr.Button("Edit Flexible Area")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                form_reset_mask_btn = gr.Button("Reset mask")
                            with gr.Column(scale=1, min_width=10):
                                show_mask = gr.Checkbox(
                                    label="Show Mask",
                                    value=global_state.value["show_mask"],
                                    show_label=False,
                                )

                        with gr.Row():
                            form_lambda_number = gr.Number(
                                value=global_state.value["params"]["motion_lambda"],
                                interactive=True,
                                label="Lambda",
                            )

                form_draw_interval_number = gr.Number(
                    value=global_state.value["draw_interval"],
                    label="Draw Interval (steps)",
                    interactive=True,
                    visible=False,
                )

            # Right --> Image
            with gr.Column(scale=2):
                form_image = ImageMask(
                    value=global_state.value["images"]["image_show"],
                    brush_radius=100,
                    elem_id="output-image",
                )
    gr.Markdown(
        """
        ## Quick Start

        1. Select desired `Pretrained Model` and adjust `Seed` to generate an
           initial image.
        2. Click on image to add control points.
        3. Click `Start` and enjoy it!

        ## Advance Usage

        1. Change `Step Size` to adjust learning rate in drag optimization.
        2. Select `w` or `w+` to change latent space to optimize:
        * Optimize on `w` space may cause greater influence to the image.
        * Optimize on `w+` space may work slower than `w`, but usually achieve
          better results.
        * Note that changing the latent space will reset the image, points and
          mask (this has the same effect as `Reset Image` button).
        3. Click `Edit Flexible Area` to create a mask and constrain the
           unmasked region to remain unchanged.

        
        """
    )
    gr.HTML(
        """
        <style>
            .container {
                position: absolute;
                height: 50px;
                text-align: center;
                line-height: 50px;
                width: 100%;
            }
        </style>
        <div class="container">
        Gradio demo supported by
        <img src="https://avatars.githubusercontent.com/u/10245193?s=200&v=4" height="20" width="20" style="display:inline;">
        <a href="https://github.com/open-mmlab/mmagic">OpenMMLab MMagic</a>
        </div>
        """
    )
    # Network & latents tab listeners

    def on_click_reset_image(global_state):
        """Reset image to the original one and clear all states
        1. Re-init images
        2. Clear all states
        """

        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state["images"]["image_show"]

    def on_click_reset_custom_image(global_state):
        """Reset image to the original one and clear all states
        1. Re-init images
        2. Clear all states
        """
        Path(global_state["pretrained_weight"]).unlink(missing_ok=True)
        global_state["w_pivot"] = None
        global_state["pretrained_weight"] = str(
            model_dir / "stylegan2-ffhq1024x1024.pkl"
        )

        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state["images"]["image_show"]

    def on_image_change(
        custom_image, global_state, progress=gr.Progress(track_tqdm=True)
    ):
        new_img = Image.open(custom_image)
        new_img = new_img.convert("RGB")
        from PTI.configs import paths_config

        paths_config.stylegan2_ada_ffhq = global_state["pretrained_weight"]
        paths_config.dlib = (model_dir / "align.dat").as_posix()
        run_name = str(uuid.uuid4())
        new_G, w_pivot = run_PTI(new_img, run_name)

        out_path = Path(f"checkpoints/stylegan2-{run_name}.pkl")
        print(f"Exporting to {out_path}")
        export_updated_pickle(new_G, out_path, run_name)
        global_state["w_pivot"] = w_pivot
        global_state["pretrained_weight"] = str(out_path)
        init_images(global_state)
        clear_state(global_state)

        return (
            global_state,
            global_state["images"]["image_show"],
            gr.Image.update(interactive=True),
        )

    form_custom_image.upload(
        on_image_change,
        [form_custom_image, global_state],
        [global_state, form_image, form_reset_custom_image],
    )

    form_reset_custom_image.click(
        on_click_reset_custom_image, [global_state], [global_state, form_image]
    )

    form_reset_image.click(
        on_click_reset_image,
        inputs=[global_state],
        outputs=[global_state, form_image],
        queue=False,
        show_progress=True,
    )

    # Update parameters
    def on_change_update_image_seed(seed, global_state):
        """Function to handle generation seed change.
        1. Set seed to global_state
        2. Re-init images and clear all states
        """

        global_state["params"]["seed"] = int(seed)
        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state["images"]["image_show"]

    form_seed_number.change(
        on_change_update_image_seed,
        inputs=[form_seed_number, global_state],
        outputs=[global_state, form_image],
    )

    def on_click_latent_space(latent_space, global_state):
        """Function to reset latent space to optimize.
        NOTE: this function we reset the image and all controls
        1. Set latent-space to global_state
        2. Re-init images and clear all state
        """

        global_state["params"]["latent_space"] = latent_space
        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state["images"]["image_show"]

    form_latent_space.change(
        on_click_latent_space,
        inputs=[form_latent_space, global_state],
        outputs=[global_state, form_image],
    )

    # ==== Params
    form_lambda_number.change(
        partial(on_change_single_global_state, ["params", "motion_lambda"]),
        inputs=[form_lambda_number, global_state],
        outputs=[global_state],
    )

    def on_change_lr(lr, global_state):
        if lr == 0:
            print("lr is 0, do nothing.")
            return global_state
        else:
            global_state["params"]["lr"] = lr
            renderer = global_state["renderer"]
            renderer.update_lr(lr)
            print("New optimizer: ")
            print(renderer.w_optim)
        return global_state

    form_lr_number.change(
        on_change_lr,
        inputs=[form_lr_number, global_state],
        outputs=[global_state],
        queue=False,
        show_progress=True,
    )

    def on_click_start(global_state, image):
        p_in_pixels = []
        t_in_pixels = []
        valid_points = []

        # handle of start drag in mask editing mode
        global_state = preprocess_mask_info(global_state, image)

        # Prepare the points for the inference
        if len(global_state["points"]) == 0:
            # yield on_click_start_wo_points(global_state, image)
            image_raw = global_state["images"]["image_raw"]
            update_image_draw(
                image_raw,
                global_state["points"],
                global_state["mask"],
                global_state["show_mask"],
                global_state,
            )

            yield (
                global_state,  # global_state
                0,  # form_steps_number,
                global_state["images"]["image_show"],  # form image
                gr.Button.update(interactive=True),  # form_reset_image
                gr.Button.update(interactive=True),  # add points button
                gr.Button.update(interactive=True),  # enable mask button
                gr.Button.update(interactive=True),  # undo points button
                gr.Button.update(interactive=True),  # reset mask button
                gr.Radio.update(interactive=True),  # latent space
                gr.Button.update(interactive=True),  # start button
                gr.Button.update(interactive=False),  # stop button
                gr.Number.update(interactive=True),  # form_seed_number
                gr.Number.update(interactive=True),  # form_lr_number
                gr.Checkbox.update(interactive=True),  # show_mask
                gr.Number.update(interactive=True),  # form_lambda_number
                gr.Button.update(interactive=True),  # form_reset_custom_image
            )
        else:
            # Transform the points into torch tensors
            for key_point, point in global_state["points"].items():
                try:
                    p_start = point.get("start_temp", point["start"])
                    p_end = point["target"]

                    if p_start is None or p_end is None:
                        continue

                except KeyError:
                    continue

                p_in_pixels.append(p_start)
                t_in_pixels.append(p_end)
                valid_points.append(key_point)

            mask = torch.tensor(global_state["mask"]).float()
            drag_mask = 1 - mask

            renderer: Renderer = global_state["renderer"]
            global_state["temporal_params"]["stop"] = False
            global_state["editing_state"] = "running"

            # reverse points order
            p_to_opt = reverse_point_pairs(p_in_pixels)
            t_to_opt = reverse_point_pairs(t_in_pixels)
            print("Running with:")
            print(f"    Source: {p_in_pixels}")
            print(f"    Target: {t_in_pixels}")
            step_idx = 0
            last_time = time.time()
            while True:
                print_memory_usage()
                # add a TIMEOUT break
                print(f"Running time: {time.time() - last_time}")
                if IS_SPACE and time.time() - last_time > TIMEOUT:
                    print("Timeout break!")
                    break
                if (
                    global_state["temporal_params"]["stop"]
                    or global_state["generator_params"]["stop"]
                ):
                    break

                # do drage here!
                renderer._render_drag_impl(
                    global_state["generator_params"],
                    p_to_opt,  # point
                    t_to_opt,  # target
                    drag_mask,  # mask,
                    global_state["params"]["motion_lambda"],  # lambda_mask
                    reg=0,
                    feature_idx=5,  # NOTE: do not support change for now
                    r1=global_state["params"]["r1_in_pixels"],  # r1
                    r2=global_state["params"]["r2_in_pixels"],  # r2
                    # random_seed     = 0,
                    # noise_mode      = 'const',
                    trunc_psi=global_state["params"]["trunc_psi"],
                    # force_fp32      = False,
                    # layer_name      = None,
                    # sel_channels    = 3,
                    # base_channel    = 0,
                    # img_scale_db    = 0,
                    # img_normalize   = False,
                    # untransform     = False,
                    is_drag=True,
                    to_pil=True,
                )

                if step_idx % global_state["draw_interval"] == 0:
                    print("Current Source:")
                    for key_point, p_i, t_i in zip(valid_points, p_to_opt, t_to_opt):
                        global_state["points"][key_point]["start_temp"] = [
                            p_i[1],
                            p_i[0],
                        ]
                        global_state["points"][key_point]["target"] = [
                            t_i[1],
                            t_i[0],
                        ]
                        start_temp = global_state["points"][key_point]["start_temp"]
                        print(f"    {start_temp}")

                    image_result = global_state["generator_params"]["image"]
                    image_draw = update_image_draw(
                        image_result,
                        global_state["points"],
                        global_state["mask"],
                        global_state["show_mask"],
                        global_state,
                    )
                    global_state["images"]["image_raw"] = image_result

                yield (
                    global_state,  # global_state
                    step_idx,  # form_steps_number,
                    global_state["images"]["image_show"],  # form image
                    # gr.File.update(visible=False),
                    gr.Button.update(interactive=False),  # form_reset_image
                    gr.Button.update(interactive=False),  # add points button
                    gr.Button.update(interactive=False),  # enable mask button
                    gr.Button.update(interactive=False),  # undo points button
                    gr.Button.update(interactive=False),  # reset mask button
                    # latent space
                    gr.Radio.update(interactive=False),  # latent space
                    gr.Button.update(interactive=False),  # start button
                    # enable stop button in loop
                    gr.Button.update(interactive=True),  # stop button
                    # update other comps
                    gr.Number.update(interactive=False),  # form_seed_number
                    gr.Number.update(interactive=False),  # form_lr_number
                    gr.Checkbox.update(interactive=False),  # show_mask
                    gr.Number.update(interactive=False),  # form_lambda_number
                    gr.Button.update(interactive=False),  # form_reset_custom_image
                )

                # increate step
                step_idx += 1

            image_result = global_state["generator_params"]["image"]
            global_state["images"]["image_raw"] = image_result
            image_draw = update_image_draw(
                image_result,
                global_state["points"],
                global_state["mask"],
                global_state["show_mask"],
                global_state,
            )

            # fp = NamedTemporaryFile(suffix=".png", delete=False)
            # image_result.save(fp, "PNG")

            global_state["editing_state"] = "add_points"

            yield (
                global_state,  # global_state
                0,  # reset step to 0 after stop. # form_steps_number,
                global_state["images"]["image_show"],  # form image
                gr.Button.update(interactive=True),  # form_reset_image
                gr.Button.update(interactive=True),  # add points button
                gr.Button.update(interactive=True),  # enable mask button
                gr.Button.update(interactive=True),  # undo points button
                gr.Button.update(interactive=True),  # reset mask button
                gr.Radio.update(interactive=True),  # latent space
                gr.Button.update(interactive=True),  # start button
                gr.Button.update(interactive=False),  # stop button
                gr.Number.update(interactive=True),  # form_seed_number
                gr.Number.update(interactive=True),  # form_lr_number
                gr.Checkbox.update(interactive=True),  # show_mask
                gr.Number.update(interactive=True),  # form_lambda_number
                gr.Button.update(interactive=True),  # form_reset_custom_image
            )

    form_start_btn.click(
        on_click_start,
        inputs=[global_state, form_image],
        outputs=[
            global_state,
            form_steps_number,
            form_image,
            # form_download_result_file,
            # >>> buttons
            form_reset_image,
            enable_add_points,
            enable_add_mask,
            undo_points,
            form_reset_mask_btn,
            form_latent_space,
            form_start_btn,
            form_stop_btn,
            # <<< buttonm
            # >>> inputs comps
            form_seed_number,
            form_lr_number,
            show_mask,
            form_lambda_number,
            form_reset_custom_image,
        ],
    )

    def on_click_stop(global_state):
        """Function to handle stop button is clicked.
        1. send a stop signal by set global_state["temporal_params"]["stop"] as True
        2. Disable Stop button
        """
        global_state["temporal_params"]["stop"] = True

        return global_state, gr.Button.update(interactive=False)

    form_stop_btn.click(
        on_click_stop,
        inputs=[global_state],
        outputs=[global_state, form_stop_btn],
        queue=False,
        show_progress=True,
    )

    form_draw_interval_number.change(
        partial(
            on_change_single_global_state,
            "draw_interval",
            map_transform=lambda x: int(x),
        ),
        inputs=[form_draw_interval_number, global_state],
        outputs=[global_state],
        queue=False,
        show_progress=True,
    )

    def on_click_remove_point(global_state):
        choice = global_state["curr_point"]
        del global_state["points"][choice]

        choices = list(global_state["points"].keys())

        if len(choices) > 0:
            global_state["curr_point"] = choices[0]

        return (
            gr.Dropdown.update(choices=choices, value=choices[0]),
            global_state,
        )

    # Mask
    def on_click_reset_mask(global_state):
        global_state["mask"] = np.ones(
            (
                global_state["images"]["image_raw"].size[1],
                global_state["images"]["image_raw"].size[0],
            ),
            dtype=np.uint8,
        )
        image_draw = update_image_draw(
            global_state["images"]["image_raw"],
            global_state["points"],
            global_state["mask"],
            global_state["show_mask"],
            global_state,
        )
        return global_state, gr.Image.update(value=image_draw, interactive=False)

    form_reset_mask_btn.click(
        on_click_reset_mask,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    # Image
    def on_click_enable_draw(global_state, image):
        """Function to start add mask mode.
        1. Preprocess mask info from last state
        2. Change editing state to add_mask
        3. Set curr image with points and mask
        """
        global_state = preprocess_mask_info(global_state, image)
        global_state["editing_state"] = "add_mask"
        image_raw = global_state["images"]["image_raw"]
        image_draw = update_image_draw(
            image_raw, global_state["points"], global_state["mask"], True, global_state
        )
        return (
            global_state,
            gr.Image.update(value=image_draw, interactive=True),
        )

    def on_click_remove_draw(global_state, image):
        """Function to start remove mask mode.
        1. Preprocess mask info from last state
        2. Change editing state to remove_mask
        3. Set curr image with points and mask
        """
        global_state = preprocess_mask_info(global_state, image)
        global_state["edinting_state"] = "remove_mask"
        image_raw = global_state["images"]["image_raw"]
        image_draw = update_image_draw(
            image_raw, global_state["points"], global_state["mask"], True, global_state
        )
        return (
            global_state,
            gr.Image.update(value=image_draw, interactive=True),
        )

    enable_add_mask.click(
        on_click_enable_draw,
        inputs=[global_state, form_image],
        outputs=[
            global_state,
            form_image,
        ],
        queue=False,
        show_progress=True,
    )

    def on_click_add_point(global_state, image: dict):
        """Function switch from add mask mode to add points mode.
        1. Updaste mask buffer if need
        2. Change global_state['editing_state'] to 'add_points'
        3. Set current image with mask
        """

        global_state = preprocess_mask_info(global_state, image)
        global_state["editing_state"] = "add_points"
        mask = global_state["mask"]
        image_raw = global_state["images"]["image_raw"]
        image_draw = update_image_draw(
            image_raw,
            global_state["points"],
            mask,
            global_state["show_mask"],
            global_state,
        )

        return (
            global_state,
            gr.Image.update(value=image_draw, interactive=False),
        )

    enable_add_points.click(
        on_click_add_point,
        inputs=[global_state, form_image],
        outputs=[global_state, form_image],
        queue=False,
        show_progress=True,
    )

    def on_click_image(global_state, evt: gr.SelectData):
        """This function only support click for point selection"""
        xy = evt.index
        if global_state["editing_state"] != "add_points":
            print(f'In {global_state["editing_state"]} state. ' "Do not add points.")

            return global_state, global_state["images"]["image_show"]

        points = global_state["points"]

        point_idx = get_latest_points_pair(points)
        if point_idx is None:
            points[0] = {"start": xy, "target": None}
            print(f"Click Image - Start - {xy}")
        elif points[point_idx].get("target", None) is None:
            points[point_idx]["target"] = xy
            print(f"Click Image - Target - {xy}")
        else:
            points[point_idx + 1] = {"start": xy, "target": None}
            print(f"Click Image - Start - {xy}")

        image_raw = global_state["images"]["image_raw"]
        image_draw = update_image_draw(
            image_raw,
            global_state["points"],
            global_state["mask"],
            global_state["show_mask"],
            global_state,
        )

        return global_state, image_draw

    form_image.select(
        on_click_image,
        inputs=[global_state],
        outputs=[global_state, form_image],
        queue=False,
        show_progress=True,
    )

    def on_click_clear_points(global_state):
        """Function to handle clear all control points
        1. clear global_state['points'] (clear_state)
        2. re-init network
        2. re-draw image
        """
        clear_state(global_state, target="point")

        renderer: Renderer = global_state["renderer"]
        renderer.feat_refs = None

        image_raw = global_state["images"]["image_raw"]
        image_draw = update_image_draw(
            image_raw, {}, global_state["mask"], global_state["show_mask"], global_state
        )
        return global_state, image_draw

    undo_points.click(
        on_click_clear_points,
        inputs=[global_state],
        outputs=[global_state, form_image],
        queue=False,
        show_progress=True,
    )

    def on_click_show_mask(global_state, show_mask):
        """Function to control whether show mask on image."""
        global_state["show_mask"] = show_mask

        image_raw = global_state["images"]["image_raw"]
        image_draw = update_image_draw(
            image_raw,
            global_state["points"],
            global_state["mask"],
            global_state["show_mask"],
            global_state,
        )
        return global_state, image_draw

    show_mask.change(
        on_click_show_mask,
        inputs=[global_state, show_mask],
        outputs=[global_state, form_image],
        queue=False,
        show_progress=True,
    )

# print("SHAReD: Start app", parser.parse_args())
gr.close_all()
app.queue(concurrency_count=1, max_size=200, api_open=False)
app.launch(show_api=False, share=True)
