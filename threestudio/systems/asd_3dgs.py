import math
from dataclasses import dataclass

import numpy as np
import threestudio
import torch
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.typing import *
from torch.cuda.amp import autocast

import torch.nn.functional as F
from ..models.geometry.gaussian_base import BasicPointCloud, Camera

from threestudio.utils.misc import tv_loss

@threestudio.register("gaussian-splatting-system-asd")
class GaussianSplatting(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        guidance_eval: bool = False

        warmup_iter: int = 1500

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

        ## debug
        torch.autograd.set_detect_anomaly(True)
    
    @property
    def global_step(self) -> int:
        return (super().global_step / 2)

    def configure_optimizers(self, merge_optim=False):
        optim = self.geometry.optimizer
        if merge_optim:
            pass
        else:
            def get_parameters(model):
                return [param for param in model.parameters() if param.requires_grad]

            params = get_parameters(self.guidance)
            optim_d = torch.optim.AdamW(params, lr=0.0001, betas=[0.9,0.99], eps=1.e-15)
            total_params = sum(p.numel() for group in optim_d.param_groups for p in group['params'])
            print(f'Total number of parameters in the model: {total_params}')
        #if hasattr(self, "merged_optimizer"):
        #    return [optim]
        #if hasattr(self.cfg.optimizer, "name"):
        #    net_optim = parse_optimizer(self.cfg.optimizer, self)
        #    optim = self.geometry.merge_optimizer(net_optim)
        #    self.merged_optimizer = True
        #else:
        #    self.merged_optimizer = False
        return [optim, optim_d]

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def guidance_update(self):
        warm_up_rate = 1. - min(self.global_step / self.cfg.warmup_iter, 1.)
        tmp_max_step = self.guidance.cfg.max_step_percent
        self.guidance.set_min_max_steps(max_step_percent=tmp_max_step + (0.98 - tmp_max_step) * warm_up_rate)
        #print("update max t to: ", tmp_max_step + (0.98 - tmp_max_step) * warm_up_rate)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        out = self(batch)

        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]
        # import pdb; pdb.set_trace()
        viewspace_point_tensor = out["viewspace_points"]
        guidance_eval = self.cfg.guidance_eval and (self.global_step % 200 == 0)


        self.guidance_update()
        guidance_out = self.guidance(
            guidance_inp, self.prompt_utils, **batch, 
            rgb_as_latents=False, guidance_eval=guidance_eval
        )

        loss_asd = 0.0
        loss_lora = 0.0
        #loss = 0.0

        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for name, value in guidance_out.items():
            if name == 'eval': continue
            self.log(f"train/{name}", value)
            if name == "loss_lora":
                loss_lora += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )
                continue
            if name.startswith("loss_"):
                loss_asd += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )
        #print(self.geometry.get_opacity.max(dim=1).values.max(), self.geometry.get_scaling.max(dim=1).values.max())
        xyz_mean = None
        if self.cfg.loss["lambda_position"] > 0.0:
            xyz_mean = self.geometry.get_xyz.norm(dim=-1)
            loss_position = xyz_mean.mean()
            self.log(f"train/loss_position", loss_position)
            loss_asd += self.C(self.cfg.loss["lambda_position"]) * loss_position

        if self.cfg.loss["lambda_opacity"] > 0.0:
            scaling = self.geometry.get_scaling.norm(dim=-1)
            loss_opacity = (
                scaling.detach().unsqueeze(-1) * self.geometry.get_opacity
            ).sum()
            self.log(f"train/loss_opacity", loss_opacity)
            loss_asd += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

        if self.cfg.loss["lambda_scales"] > 0.0:
            #scale_sum = torch.sum(self.geometry.get_scaling)
            scale_sum = torch.mean(self.geometry.get_scaling,dim=-1).mean()
            self.log(f"train/scales", scale_sum)
            loss_asd += self.C(self.cfg.loss["lambda_scales"]) * scale_sum

        if self.cfg.loss["lambda_tv"] > 0.0:
            depth = out["comp_depth"]
            self.log(f"train/tv", scale_sum)
            loss_asd += self.C(self.cfg.loss["lambda_tv"]) * (tv_loss(depth) + tv_loss(guidance_inp))

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        loss_asd.backward(retain_graph=True)
        loss_lora.backward()

        iteration = self.global_step
        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
        )
        #if loss > 0:
        #    loss.backward()
        opt_g.step()
        opt_g.zero_grad(set_to_none=True)

        opt_d.step()
        opt_d.zero_grad()
        
        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach(),
                guidance_out["eval"],
            )
        
        return {"loss": loss_asd}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        #print(f"gaussians num: {self.geometry.get_xyz.shape[0]}")
        #print(out["comp_rgb"].shape)
        # import pdb; pdb.set_trace()
        self.save_image_grid(
            f"it{self.global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="test_step",
            step=self.global_step,
        )
        if batch["index"][0] == 0:
            save_path = self.get_save_path("point_cloud.ply")
            self.geometry.save_ply(save_path)

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.global_step}-test",
            f"it{self.global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.global_step,
        )

    def on_load_checkpoint(self, ckpt_dict) -> None:
        num_pts = ckpt_dict["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(
            points=np.zeros((num_pts, 3)),
            colors=np.zeros((num_pts, 3)),
            normals=np.zeros((num_pts, 3)),
        )
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        super().on_load_checkpoint(ckpt_dict)

    @torch.no_grad()
    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        #print("num of points: ", (int(self.gaussian.get_xyz.shape[0])))
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.global_step}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["grad"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )

            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig_lora"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.global_step,
            texts=guidance_eval_out["texts"],
        )