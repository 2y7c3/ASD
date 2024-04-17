from threestudio.utils.ops import get_cam_info_gaussian
from torch.cuda.amp import autocast

from ..geometry.gaussian_base import BasicPointCloud, Camera

import math
from dataclasses import dataclass

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *

class GaussianBatchRenderer:
    def batch_forward(self, batch):
        bs = batch["c2w"].shape[0]
        renders = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        normals = []
        pred_normals = []
        depths = []
        masks = []
        for batch_idx in range(bs):
            batch["batch_idx"] = batch_idx
            fovy = batch["fovy"][batch_idx]
            w2c, proj, cam_p = get_cam_info_gaussian(
                c2w=batch["c2w"][batch_idx], fovx=fovy, fovy=fovy, znear=0.1, zfar=100
            )

            # import pdb; pdb.set_trace()
            viewpoint_cam = Camera(
                FoVx=fovy,
                FoVy=fovy,
                image_width=batch["width"],
                image_height=batch["height"],
                world_view_transform=w2c,
                full_proj_transform=proj,
                camera_center=cam_p,
            )

            with autocast(enabled=False):
                render_pkg = self.forward(
                    viewpoint_cam, self.background_tensor, **batch
                )
                renders.append(render_pkg["render"])
                viewspace_points.append(render_pkg["viewspace_points"])
                visibility_filters.append(render_pkg["visibility_filter"])
                radiis.append(render_pkg["radii"])
                if render_pkg.__contains__("normal"):
                    normals.append(render_pkg["normal"])
                if (
                    render_pkg.__contains__("pred_normal")
                    and render_pkg["pred_normal"] is not None
                ):
                    pred_normals.append(render_pkg["pred_normal"])
                if render_pkg.__contains__("depth"):
                    depths.append(render_pkg["depth"])
                if render_pkg.__contains__("mask"):
                    masks.append(render_pkg["mask"])

        outputs = {
            "comp_rgb": torch.stack(renders, dim=0).permute(0, 2, 3, 1),
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filters,
            "radii": radiis,
        }
        if len(normals) > 0:
            outputs.update(
                {
                    "comp_normal": torch.stack(normals, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(pred_normals) > 0:
            outputs.update(
                {
                    "comp_pred_normal": torch.stack(pred_normals, dim=0).permute(
                        0, 2, 3, 1
                    ),
                }
            )
        if len(depths) > 0:
            outputs.update(
                {
                    "comp_depth": torch.stack(depths, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(masks) > 0:
            outputs.update(
                {
                    "comp_mask": torch.stack(masks, dim=0).permute(0, 2, 3, 1),
                }
            )
        return outputs


@threestudio.register("diff-gaussian-rasterizer-advanced")
class DiffGaussian(Rasterizer, GaussianBatchRenderer):
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False
        invert_bg_prob: float = 1.0
        back_ground_color: Tuple[float, float, float] = (0.2, 0.2, 0.2)
        bg_aug_ratio: float = 0.66
    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        threestudio.info(
            "[Note] Gaussian Splatting doesn't support material and background now."
        )
        super().configure(geometry, material, background)
        self.background_tensor = torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32, device="cuda"
        )

    def forward(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        if np.random.rand() < self.cfg.bg_aug_ratio and self.training:
            if np.random.rand() < 0.5:
                bg_color = torch.rand_like(bg_color)
            else:
                bg_color = torch.zeros_like(bg_color)
        
        if self.training:
            invert_bg_color = np.random.rand() < self.cfg.invert_bg_prob
        else:
            invert_bg_color = False

        bg_color = bg_color if not invert_bg_color else (1.0 - bg_color)

        pc = self.geometry
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            #debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = pc.get_scaling
        rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        rendered_depth, rendered_alpha = torch.chunk(rendered_depth_alpha, 2)
        # Retain gradients of the 2D (screen-space) means for batch dim
        if self.training:
            screenspace_points.retain_grad()
        
        focal = 1 / (2 * math.tan(viewpoint_camera.FoVx / 2))  
        disp = focal / (rendered_depth + (rendered_alpha * 10) + 1e-5)

        try:
            min_d = disp[rendered_alpha <= 0.1].min()
        except Exception:
            min_d = disp.min()

        disp = torch.clamp((disp - min_d) / (disp.max() - min_d), 0.0, 1.0)
    
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "depth": disp,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }