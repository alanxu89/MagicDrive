from typing import Tuple, List, Union
import logging
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image, to_tensor
from einops import rearrange

from diffusers import UniPCMultistepScheduler
from diffusers.pipelines.controlnet import MultiControlNetModel
from accelerate.tracking import GeneralTracker

from magicdrive.runner.utils import (
    visualize_map,
    img_m11_to_01,
    concat_6_views,
    concat_m_by_n_views,
)
from magicdrive.misc.common import move_to
from magicdrive.misc.test_utils import draw_box_on_imgs
from magicdrive.pipeline.pipeline_fpv_controlnet import FPVStableDiffusionPipelineOutput
from magicdrive.dataset.utils import collate_fn_v2
from magicdrive.networks.unet_addon_rawbox import BEVControlNetModel


def format_ori_with_gen(ori_img, gen_img_list):
    formatted_images = []

    # first image is input, followed by generations.
    formatted_images.append(np.asarray(ori_img))

    for image in gen_img_list:
        formatted_images.append(np.asarray(image))

    # formatted_images = np.stack(formatted_images)
    # 0-255 np -> 0-1 tensor -> grid -> 0-255 pil -> np
    formatted_images = torchvision.utils.make_grid(
        [to_tensor(im) for im in formatted_images], nrow=1)
    formatted_images = np.asarray(to_pil_image(formatted_images))
    return formatted_images


class BaseValidator:

    def __init__(self, cfg, val_dataset, pipe_cls, pipe_param) -> None:
        self.cfg = cfg
        self.n_cam = len(self.cfg.dataset.view_order)
        self.n_frame = self.cfg.dataset.num_frames
        self.val_dataset = val_dataset
        self.pipe_cls = pipe_cls
        self.pipe_param = pipe_param
        logging.info(
            f"[BaseValidator] Validator use model_param: {pipe_param.keys()}")

    def validate(self, controlnet: Union[BEVControlNetModel,
                                         MultiControlNetModel], unet,
                 trackers: Tuple[GeneralTracker,
                                 ...], step, weight_dtype, device):
        logging.info("[BaseValidator] Running validation... ")
        controlnet.eval()  # important !!!
        unet.eval()

        pipeline = self.pipe_cls.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            **self.pipe_param,
            unet=unet,
            controlnet=controlnet,
            safety_checker=None,
            feature_extractor=
            None,  # since v1.5 has default, we need to override
            torch_dtype=weight_dtype,
        )
        # NOTE: this scheduler does not take generator as kwargs.
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config)
        pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)

        if self.cfg.runner.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        image_logs = []
        progress_bar = tqdm(
            range(
                0,
                len(self.cfg.runner.validation_index) *
                self.cfg.runner.validation_times,
            ),
            desc="Val Steps",
        )

        for validation_i in self.cfg.runner.validation_index:
            raw_data = self.val_dataset[validation_i]  # cannot index loader
            val_input = collate_fn_v2(
                [raw_data],
                self.cfg.dataset.template,
                tokenizer=None,
                is_train=False,
            )
            # camera_emb = self._embed_camera(val_input["camera_param"])

            # let different prompts have the same random seed
            if self.cfg.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device).manual_seed(
                    self.cfg.seed)

            # for each input param, we generate several times to check variance.
            gen_list, gen_wb_list = [], []
            for _ in range(self.cfg.runner.validation_times):
                with torch.autocast("cuda"):
                    image: FPVStableDiffusionPipelineOutput = pipeline(
                        prompt=val_input["captions"],
                        image=[val_input["depth"], val_input["semantic_map"]],
                        height=self.cfg.dataset.image_size[0],
                        width=self.cfg.dataset.image_size[1],
                        generator=generator,
                        bev_controlnet_kwargs=val_input["kwargs"],
                        **self.cfg.runner.pipeline_param,
                    )
                    assert len(image.images) == 1
                    image: List[Image.Image] = image.images[0]

                gen_img = concat_m_by_n_views(image, self.n_frame, self.n_cam)
                gen_list.append(gen_img)
                # if self.cfg.runner.validation_show_box:
                #     image_with_box = draw_box_on_imgs(
                #         self.cfg, 0, val_input, image)
                #     gen_wb_list.append(concat_6_views(image_with_box))

                progress_bar.update(1)

            # make image for 6 views and save to dict
            val_input["pixel_values"] = rearrange(
                val_input["pixel_values"], "b c f n h w -> b (f n) c h w")
            val_input["depth"] = rearrange(val_input["depth"],
                                           "b c f n h w -> b (f n) c h w")
            val_input["semantic_map"] = rearrange(
                val_input["semantic_map"], "b c f n h w -> b (f n) c h w")

            N = self.n_frame * self.n_cam
            ori_imgs = [
                to_pil_image(img_m11_to_01(val_input["pixel_values"][0][i]))
                for i in range(N)
            ]
            ori_img = concat_m_by_n_views(ori_imgs, self.n_frame, self.n_cam)
            # make image for 6 views and save to dict
            ori_depths = [
                to_pil_image(img_m11_to_01(val_input["depth"][0][i]))
                for i in range(N)
            ]
            ori_depth = concat_m_by_n_views(ori_depths, self.n_frame,
                                            self.n_cam)
            ori_maps = [
                to_pil_image(img_m11_to_01(val_input["semantic_map"][0][i]))
                for i in range(N)
            ]
            ori_map = concat_m_by_n_views(ori_maps, self.n_frame, self.n_cam)
            # ori_img_wb = concat_6_views(
            #     draw_box_on_imgs(self.cfg, 0, val_input, ori_imgs))
            # map_img_np = visualize_map(
            #     self.cfg, val_input["bev_map_with_aux"][0])
            image_logs.append({
                # "map_img_np": map_img_np,  # condition
                "gen_img_list": gen_list,  # output
                # "gen_img_wb_list": gen_wb_list,  # output
                "ori_img": ori_img,  # condition
                "ori_depth": ori_depth,  # condition
                "ori_semantic_map": ori_map,
                "validation_prompt": val_input["captions"][0],
            })

        for tracker in trackers:
            if tracker.name == "tensorboard":
                for log in image_logs:
                    # map_img_np = log["map_img_np"]
                    validation_prompt = log["validation_prompt"]

                    formatted_images = format_ori_with_gen(
                        log["ori_img"], log["gen_img_list"])
                    formatted_images1 = format_ori_with_gen(
                        log["ori_depth"], [log["ori_semantic_map"]])

                    final_image = concat_6_views([
                        to_pil_image(formatted_images),
                        to_pil_image(formatted_images1)
                    ],
                                                 oneline=True)
                    final_image = np.asarray(final_image)
                    tracker.writer.add_image(validation_prompt,
                                             final_image,
                                             step,
                                             dataformats="HWC")

                    # formatted_images = format_ori_with_gen(
                    #     log["ori_img_wb"], log["gen_img_wb_list"])
                    # tracker.writer.add_image(
                    #     validation_prompt + "(with box)", formatted_images,
                    #     step, dataformats="HWC")

                    # tracker.writer.add_image(
                    #     "map: " + validation_prompt, map_img_np, step,
                    #     dataformats="HWC")
            elif tracker.name == "wandb":
                raise NotImplementedError("Do not use wandb.")
                formatted_images = []

                for log in image_logs:
                    images = log["images"]
                    validation_prompt = log["validation_prompt"]
                    validation_image = log["validation_image"]

                    formatted_images.append(
                        wandb.Image(
                            validation_image,
                            caption="Controlnet conditioning"))

                    for image in images:
                        image = wandb.Image(image, caption=validation_prompt)
                        formatted_images.append(image)

                tracker.log({"validation": formatted_images})
            else:
                logging.warn(
                    f"image logging not implemented for {tracker.name}")

        return image_logs
