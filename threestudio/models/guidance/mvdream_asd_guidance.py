import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from mvdream.camera_utils import convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *

from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available


@threestudio.register("mvdream-multiview-diffusion-guidance")
class MultiviewDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        model_name: str = (
            "sd-v2.1-base-4view"  # check mvdream.model_zoo.PRETRAINED_MODELS
        )
        ckpt_path: Optional[
            str
        ] = None  # path to local checkpoint (None for loading from url)
        guidance_scale: float = 7.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_view: int = 4
        image_size: int = 256

        pretrained_model_name_or_path_lora: str = "stabilityai/stable-diffusion-2-base"
        guidance_scale_lora: float = 1.0
        lora_cfg_training: bool = True
        lora_n_timestamp_samples: int = 1


    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Diffusion ...")

        self.model = build_model(self.cfg.model_name, ckpt_path=self.cfg.ckpt_path).to(
            self.device
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Multiview Diffusion!")

        pipe_lora_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        pipe_lora = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora,
            **pipe_lora_kwargs,
        ).to(self.device)
        del pipe_lora.vae
        del self.pipe_lora.text_encoder
        cleanup()

        @dataclass
        class SubModules:
            pipe: StableDiffusionPipeline
            pipe_lora: StableDiffusionPipeline

        self.submodules = SubModules(pipe=None, pipe_lora=pipe_lora)

    def get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(
                f"Unknown camera_condition_type={self.cfg.camera_condition_type}"
            )
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs)
        )
        return latents  # [B, 4, 32, 32] Latent space image

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        camera = c2w

        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = (
                    F.interpolate(
                        rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
                    )
                    * 2
                    - 1
                )
            else:
                # interp to 256x256 to be fed into vae.
                pred_rgb = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size, self.cfg.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        # sample timestep
        if timestep is None:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=latents.device,
            )
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
        t_expand = t.repeat(text_embeddings.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(camera, fovy)
                camera = camera.repeat(2, 1).to(text_embeddings)
                context = {
                    "context": text_embeddings,
                    "camera": camera,
                    "num_frames": self.cfg.n_view,
                }
            else:
                context = {"context": text_embeddings}
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(
            2
        )  # Note: flipped compared to stable-dreamfusion
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # Original SDS
        # w(t), sigma_t^2
        w = 1 - self.alphas_cumprod[t]
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        return {
            "loss_sds": loss,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)