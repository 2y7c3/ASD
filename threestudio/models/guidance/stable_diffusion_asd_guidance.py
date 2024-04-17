from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, AutoencoderKL, schedulers
from tqdm import tqdm
from contextlib import contextmanager
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule, BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component, perp_arrange
from threestudio.utils.typing import *

from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def pred_original(
        self,
        model_output: torch.FloatTensor,
        timesteps: int,
        sample: torch.FloatTensor,
    ):
        if isinstance(self, DDPMScheduler) or isinstance(self, DDIMScheduler):
            # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
            alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
            timesteps = timesteps.to(sample.device)

            # 1. compute alphas, betas
            alpha_prod_t = alphas_cumprod[timesteps]
            while len(alpha_prod_t.shape) < len(sample.shape):
                alpha_prod_t = alpha_prod_t.unsqueeze(-1)

            beta_prod_t = 1 - alpha_prod_t

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if self.config.prediction_type == "epsilon":
                pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif self.config.prediction_type == "sample":
                pred_original_sample = model_output
            elif self.config.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                    " `v_prediction` for the DDPMScheduler."
                )

            # 3. Clip or threshold "predicted x_0"
            if self.config.thresholding:
                pred_original_sample = self._threshold_sample(pred_original_sample)
            elif self.config.clip_sample:
                pred_original_sample = pred_original_sample.clamp(
                    -self.config.clip_sample_range, self.config.clip_sample_range
                )
        elif isinstance(self, EulerAncestralDiscreteScheduler) or isinstance(self, EulerDiscreteScheduler):
            timestep = timesteps.to(self.timesteps.device)

            step_index = (self.timesteps == timestep).nonzero().item()
            sigma = self.sigmas[step_index].to(device=sample.device, dtype=sample.dtype)

            # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            if self.config.prediction_type == "epsilon":
                pred_original_sample = sample - sigma * model_output
            elif self.config.prediction_type == "v_prediction":
                # * c_out + input * c_skip
                pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
            elif self.config.prediction_type == "sample":
                raise NotImplementedError("prediction_type not implemented yet: sample")
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
                )
        else:
            raise NotImplementedError

        return pred_original_sample

class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)

@threestudio.register("stable-diffusion-asd-guidance")
class StableDiffusionGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-base"
        pretrained_model_name_or_path_lora: str = "stabilityai/stable-diffusion-2-base"
        guidance_scale_lora: float = 1.0
        lora_cfg_training: bool = True
        lora_n_timestamp_samples: int = 1
        camera_condition_type: str = "extrinsics"

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        single_model: bool = False

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None

        weighting_strategy: str = "sds"

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

        guidance_rescale: float = 0.0

        gamma: float = -1

        LoRA_path: Any = None
        ckpt_path: Any = None

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion Model ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        @dataclass
        class SubModules:
            pipe: StableDiffusionPipeline
            pipe_lora: StableDiffusionPipeline

        # Create model
        if self.cfg.ckpt_path:
            pipe = StableDiffusionPipeline.from_ckpt(
                self.cfg.ckpt_path,
                **pipe_kwargs,
            ).to(self.device)
        
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                **pipe_kwargs,
            ).to(self.device)

        if self.cfg.LoRA_path is not None:
            
            
            print("load lora in:.{}".format(self.cfg.LoRA_path))
            try:
                from threestudio.utils.lora import tune_lora_scale, patch_pipe
                patch_pipe(
                    pipe,
                    self.cfg.LoRA_path,
                    patch_text=False,
                    patch_ti=False,
                    patch_unet=True,
                )
                tune_lora_scale(pipe.unet, 1.00)
                #tune_lora_scale(pipe.text_encoder, 1.00)
            except:
                from threestudio.utils.lora import load_lora_weights
                pipe = load_lora_weights(
                    pipe, self.cfg.LoRA_path, self.device, self.weights_dtype, 
                    multiplier=1.00, patch_unet=True, patch_text=False
                )

        if (
            self.cfg.pretrained_model_name_or_path
            == self.cfg.pretrained_model_name_or_path_lora
        ):
            self.single_model = True
            pipe_lora = pipe
        
        else:
            self.single_model = False
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
            cleanup()
            pipe_lora.vae = pipe.vae
        self.submodules = SubModules(pipe=pipe, pipe_lora=pipe_lora)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()
                self.pipe_lora.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe_lora.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)
            self.pipe_lora.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe_lora.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        if not self.single_model:
            del self.pipe_lora.text_encoder
        cleanup()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)

        # FIXME: hard-coded dims
        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280), self.weights_dtype
        ).to(self.device)
        self.unet_lora.class_embedding = self.camera_embedding

        # set up LoRA layers
        lora_attn_procs = {}
        for name in self.unet_lora.attn_processors.keys():
            #print(name)
            
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet_lora.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet_lora.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_lora.config.block_out_channels[block_id]
            
            
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
        
        self.unet_lora.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors).to(
            self.device
        )
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()
        
        for param in self.lora_layers.parameters():
            param.requires_grad_(True)

        self.scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_lora = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.pipe.scheduler = self.scheduler
        self.pipe_lora.scheduler = self.scheduler_lora

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        print(self.scheduler.prediction_type, self.scheduler_lora.prediction_type)
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
    
        self.grad_clip_val: Optional[float] = None

        self.gamma = self.cfg.gamma
        threestudio.info(f"gamma: " + str(self.gamma))
        threestudio.info(f"Loaded Stable Diffusion Model !")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
    
    @torch.cuda.amp.autocast(enabled=False)
    def set_gamma(self, gamma):
        self.gamma = gamma
        print(self.gamma)

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def pipe_lora(self):
        return self.submodules.pipe_lora

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def unet_lora(self):
        return self.submodules.pipe_lora.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @property
    def vae_lora(self):
        return self.submodules.pipe_lora.vae

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
    
    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding
    
    def compute_grad_asd_perpneg(
        self,
        #control_image: Float[Tensor, "B C 512 512"],
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_condition,
        res: Int = 512,
    ):  
    
        batch_size = elevation.shape[0]
        B = latents.shape[0]
        (
            text_embeddings,
            neg_guidance_weights,
        ) = prompt_utils.get_text_embeddings_perp_neg(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        text_embeddings_nvd = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = perp_arrange(latents_noisy, N = neg_guidance_weights.shape[-1])
            t_tmp = perp_arrange(t, N = neg_guidance_weights.shape[-1])

            with self.disable_unet_class_embedding(self.unet) as unet:
                #cross_attention_kwargs = None
                cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                noise_pred_pretrain = self.forward_unet(
                    unet,
                    latent_model_input,
                    #torch.cat([t] * 4),
                    t_tmp,
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            
            text_embeddings_cond, _ = text_embeddings_nvd.chunk(2)
            latent_lora_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred_est = self.forward_unet(
                self.unet_lora,
                latent_lora_input,
                torch.cat([t] * 2),
                encoder_hidden_states=torch.cat([text_embeddings_cond] * 2),
                class_labels=torch.cat(
                    [
                        camera_condition.view(B, -1),
                        torch.zeros_like(camera_condition.view(B, -1)),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
            )

        (
            noise_pred_est_camera,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)
        noise_pred_est = noise_pred_est_uncond + self.cfg.guidance_scale_lora * (
            noise_pred_est_camera - noise_pred_est_uncond
        )
        noise_pred_est_tmp = noise_pred_est

        if self.scheduler.config.prediction_type != "v_prediction"\
            and self.scheduler_lora.config.prediction_type == "v_prediction":
            
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latents_noisy * sigma_t.view(
                -1, 1, 1, 1
            ) + noise_pred_est * alpha_t.view(-1, 1, 1, 1)
        
        noise_pred_text = noise_pred_pretrain[:batch_size]
        noise_pred_uncond = noise_pred_pretrain[batch_size : batch_size * 2]
        noise_pred_neg = noise_pred_pretrain[batch_size * 2 :] 

        e_pos = noise_pred_text - noise_pred_uncond
        accum_grad = torch.zeros_like(e_pos)
        n_negative_prompts = neg_guidance_weights.shape[-1]

        for i in range(n_negative_prompts):
            e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
            idx_non_zero = torch.abs(neg_guidance_weights[:, i]) > 1e-4
            if sum(idx_non_zero) == 0:
                continue

            accum_grad[idx_non_zero] += neg_guidance_weights[idx_non_zero, i].view(
                -1, 1, 1, 1
            ) * perpendicular_component(e_i_neg[idx_non_zero], e_pos[idx_non_zero])

        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                        e_pos + accum_grad
                    )

        if self.cfg.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.cfg.guidance_rescale)

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "asd":
            w = (((1 - self.alphas[t])/self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise_pred_est)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights":  neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "text_embeddings_nvd": text_embeddings_nvd,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
            "noise_pred_lora": noise_pred_est_tmp,
            "camera_condition": camera_condition,
            "grad": grad
        }

        return grad, guidance_eval_utils

    def compute_grad_asd(
        self,
        #control_image: Float[Tensor, "B C 512 512"],
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_condition,
        res: Int = 512,
        # guidance_rescale: Float = 0.7,
    ):
        batch_size = elevation.shape[0]
        B = latents.shape[0]

        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        text_embeddings_nvd = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

            with self.disable_unet_class_embedding(self.unet) as unet:
                #cross_attention_kwargs = None
                cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                noise_pred_pretrain = self.forward_unet(
                    unet,
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            
            text_embeddings_cond, _ = text_embeddings_nvd.chunk(2)
            noise_pred_est = self.forward_unet(
                self.unet_lora,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=torch.cat([text_embeddings_cond] * 2),
                class_labels=torch.cat(
                    [
                        camera_condition.view(B, -1),
                        torch.zeros_like(camera_condition.view(B, -1)),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
            )
        
        (
            noise_pred_est_camera,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)
        noise_pred_est = noise_pred_est_uncond + self.cfg.guidance_scale_lora * (
            noise_pred_est_camera - noise_pred_est_uncond
        )
        noise_pred_est_tmp = noise_pred_est

        if self.scheduler.config.prediction_type != "v_prediction"\
            and self.scheduler_lora.config.prediction_type == "v_prediction":
            
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latents_noisy * sigma_t.view(
                -1, 1, 1, 1
            ) + noise_pred_est * alpha_t.view(-1, 1, 1, 1)

        noise_pred_text, noise_pred_uncond = noise_pred_pretrain.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.cfg.guidance_rescale)

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "asd":
            w = (((1 - self.alphas[t])/self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        # grad = w * (noise_pred - noise)
        grad = w * (noise_pred - noise_pred_est)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": None,
            "text_embeddings": text_embeddings,
            "text_embeddings_nvd": text_embeddings_nvd,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
            "noise_pred_lora": noise_pred_est_tmp,
            "camera_condition": camera_condition,
            "grad": grad
        }

        return grad, guidance_eval_utils

    def train_lora(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        #t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_condition,
    ):
        batch_size = elevation.shape[0]
        B = latents.shape[0]
        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                #weights
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

            text_embeddings = torch.cat([text_embeddings[:B], text_embeddings[B*2:]], dim=0)
            text_embeddings_nvd = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, view_dependent_prompting=False
            )
        else:
            text_embeddings_nvd = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, view_dependent_prompting=False
            )

            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            text_embeddings, _ = text_embeddings.chunk(2)


        latents = latents.detach().repeat(self.cfg.lora_n_timestamp_samples, 1, 1, 1)
    
        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B * self.cfg.lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )
        
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler_lora.add_noise(latents, noise, t)

        with torch.no_grad():
            if prompt_utils.use_perp_neg:
                latent_model_input = perp_arrange(latents_noisy, N = neg_guidance_weights.shape[-1], C = 1)
                t_model = perp_arrange(t, N = neg_guidance_weights.shape[-1], C=1)
            else:
                latent_model_input = latents_noisy
                t_model = t

            with self.disable_unet_class_embedding(self.unet) as unet:
                cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                noise_pred_pretrain = self.forward_unet(
                    unet,
                    latent_model_input,
                    t_model,
                    encoder_hidden_states=text_embeddings.repeat(
                        self.cfg.lora_n_timestamp_samples, 1, 1
                    ),
                    cross_attention_kwargs=cross_attention_kwargs,
                )

        if prompt_utils.use_perp_neg:
            noise_pred_text = noise_pred_pretrain[:batch_size]
            #noise_pred_uncond = noise_pred_pretrain[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred_pretrain[batch_size:] 

            e_pos = noise_pred_text #- noise_pred_uncond
            accum_grad = torch.zeros_like(e_pos)
            n_negative_prompts = neg_guidance_weights.shape[-1]

            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] #- noise_pred_uncond
                
                idx_non_zero = torch.abs(neg_guidance_weights[:, i]) > 1e-4
                if sum(idx_non_zero) == 0:
                    continue
                accum_grad[idx_non_zero] += neg_guidance_weights[idx_non_zero, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg[idx_non_zero], e_pos[idx_non_zero])
            
            noise_pred_text =  e_pos + accum_grad
        else:
            noise_pred_text = noise_pred_pretrain
        
        import random
        text_embeddings_cond, _ = text_embeddings_nvd.chunk(2)
        if self.cfg.lora_cfg_training and random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        
        if self.scheduler_lora.config.prediction_type == "epsilon":
            noise = noise
            target_text = noise_pred_text.detach()
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            noise = self.scheduler_lora.get_velocity(latents, noise, t)
            target_text = self.scheduler.get_velocity(
                latents, noise_pred_text, t).detach()
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        
        noise_pred = self.forward_unet(
            self.unet_lora,
            latents_noisy.detach(),
            t,
            encoder_hidden_states=text_embeddings_cond.repeat(
                self.cfg.lora_n_timestamp_samples, 1, 1
            ),
            class_labels=camera_condition.view(B, -1).repeat(
                self.cfg.lora_n_timestamp_samples, 1
            ),
            cross_attention_kwargs={"scale": 1.0},
        )
        
        return F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") \
                + (
                    (self.gamma) * F.mse_loss(noise_pred.float(), 
                    target_text.float(), reduction="mean")
                )

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        guidance_eval=False,
        # guidance_rescale=0.7,
        same_t = False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        
        #control_images = control_images.permute(0, 3, 1, 2)
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512.to(self.weights_dtype))
        
        if same_t:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=self.device,
            )
            t = t.repeat(batch_size)
        
        else:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        if prompt_utils.use_perp_neg:
            grad, guidance_eval_utils = self.compute_grad_asd_perpneg(
            latents, t, prompt_utils, elevation, azimuth, camera_distances, camera_condition, res=rgb_BCHW.shape[-1],
        )
        else:
            grad, guidance_eval_utils = self.compute_grad_asd(
                latents, t, prompt_utils, elevation, azimuth, camera_distances, camera_condition, res=rgb_BCHW.shape[-1],
            )

        grad = torch.nan_to_num(grad)

        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_asd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        
        loss_lora = self.train_lora(            
            latents, prompt_utils, elevation, azimuth, camera_distances, camera_condition,
        )

        guidance_out = {
            "loss_asd": loss_asd,
            "loss_lora": loss_lora,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils, res=rgb_BCHW.shape[-1])
            texts = []
            for n, e, a, c in zip(
                t, elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out


    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        text_embeddings_nvd,
        latents_noisy,
        noise_pred,
        noise_pred_lora,
        camera_condition,
        grad,
        res=1024, 
        use_perp_neg=False,
        neg_guidance_weights=None,
        # guidance_rescale=0.7,
    ):
        imgs_noisy = self.decode_latents(latents_noisy).permute(0, 2, 3, 1)
        #self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(
        #    self.device
        #)
        #self.scheduler_lora.alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
        #    self.device
        #)
        #print(t_orig)

        pred_x0 = pred_original(#self.scheduler.step(
            self.scheduler, noise_pred, t_orig, latents_noisy
        )#.pred_original_sample

        pred_x0_lora = pred_original(#self.scheduler_lora.step(
            self.scheduler_lora, noise_pred_lora, t_orig, latents_noisy
        )#.pred_original_sample

        pred_x0 = self.decode_latents(pred_x0).permute(0, 2, 3, 1)
        pred_x0_lora = self.decode_latents(pred_x0_lora).permute(0, 2, 3, 1)

        grad_abs = torch.abs(grad.detach())
        norm_grad  = F.interpolate(
            (grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), 
            (res, res), mode='bilinear', align_corners=False
        ).repeat(1,3,1,1).permute(0, 2, 3, 1)

        return {
            "imgs_noisy": imgs_noisy,
            "grad": norm_grad,
            "imgs_1orig": pred_x0,
            "imgs_1orig_lora": pred_x0_lora,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
