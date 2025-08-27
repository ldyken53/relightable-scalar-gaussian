
# This is script for 3D Gaussian Splatting rendering

import math
import torch
import torch.nn.functional as F
from arguments import OptimizationParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim
from utils.image_utils import psnr
from .diff_rasterization import GaussianRasterizationSettings, GaussianRasterizer 

class LUTColour(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, lut):                    # s shape (N,1), lut (100,3)
        """
        s assumed in [0,1]; lut fixed length 100.
        Returns RGB (N,3) by linear interpolation between neighbours.
        """
        # scale to [0,99]
        idx = (s.clamp(0, 1) * 99.0).squeeze(-1)          # (N,)
        idx_floor = torch.floor(idx).long()                # (N,)
        idx_ceil  = torch.clamp(idx_floor + 1, max=99)    # (N,)
        t = (idx - idx_floor.float()).unsqueeze(-1)        # (N,1)

        c0 = lut[idx_floor]    # (N,3)  -> constant wrt s inside segment
        c1 = lut[idx_ceil]     # (N,3)
        rgb = (1.0 - t) * c0 + t * c1

        # save what we need for backward
        ctx.save_for_backward(c0, c1)
        return rgb                                                # (N,3)

    @staticmethod
    def backward(ctx, grad_rgb):
        c0, c1 = ctx.saved_tensors        # neither depends on s
        slope = (c1 - c0) * 99.0         # (N,3)  dRGB/ds inside segment

        # chain rule:  dL/ds = <dL/dRGB , dRGB/ds>
        grad_s = (grad_rgb * slope).sum(dim=-1, keepdim=True)     # (N,1)

        # lut is constant → no grad (return None)
        return grad_s, None


def scalar2rgb(s, lut):
    """
    Wrapper that exposes an nn.Module-like interface.
    """
    return LUTColour.apply(s, lut)


class LUTOpacity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, lut):                    # s shape (N,1), lut (100,1)
        """
        s assumed in [0,1]; lut fixed length 100.
        Returns opacity (N,1) by linear interpolation between neighbours.
        """
        # scale to [0,99]
        idx = (s.clamp(0, 1) * 99.0).squeeze(-1)          # (N,)
        idx_floor = torch.floor(idx).long()                # (N,)
        idx_ceil  = torch.clamp(idx_floor + 1, max=99)    # (N,)
        t = (idx - idx_floor.float()).unsqueeze(-1)        # (N,1)

        c0 = lut[idx_floor]    # (N,1)  -> constant wrt s inside segment
        c1 = lut[idx_ceil]     # (N,1)
        opacity = (1.0 - t) * c0 + t * c1

        # save what we need for backward
        ctx.save_for_backward(c0, c1)
        return opacity                                                # (N,1)

    @staticmethod
    def backward(ctx, grad_opac):
        c0, c1 = ctx.saved_tensors        # neither depends on s
        slope = (c1 - c0) * 99.0         # (N,1)  dO/ds inside segment

        # chain rule:  dL/ds = <dL/dO , dO/ds>
        grad_s = grad_opac * slope    # (N,1)

        # lut is constant → no grad (return None)
        return grad_s, None


def scalar2opac(s, lut):
    """
    Wrapper that exposes an nn.Module-like interface.
    """
    return LUTOpacity.apply(s, lut)


def render_view(camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
                scaling_modifier, override_color, computer_pseudo_normal=True):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    intrinsic = camera.intrinsics
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=computer_pseudo_normal,
        debug=pipe.debug
    )
    

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points #* this is zero tensor now, will be updated by rasterizer
    # opacity = pc.get_opacity
    opacity = scalar2opac(pc.get_scalar, camera.opac_map)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.compute_SHs_python:
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - camera.camera_center.repeat(pc.get_shs.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_shs
    else:
        colors_precomp = override_color

    scalar_color = scalar2rgb(pc.get_scalar2, camera.colormap)
    normal = pc.get_normal
    features = torch.cat([scalar_color, normal], dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )
    # ic("Program exits after rasterizer being called")
    # exit()
    rendered_scalar_color, rendered_normal = rendered_feature.split([3, 3], dim=0)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results = {"render": rendered_scalar_color,
               "opacity": rendered_opacity,
               "depth": rendered_depth,
               "normal": rendered_normal,
               "pseudo_normal": rendered_pseudo_normal,
               "surface_xyz": rendered_surface_xyz,
               "color_render": rendered_image,
               "viewspace_points": screenspace_points,
               "visibility_filter": radii > 0,
               "radii": radii,
               "num_rendered": num_rendered,
               "num_contrib": num_contrib}
    
    return results

def calculate_loss(viewpoint_camera, pc, render_pkg, opt):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = render_pkg["render"]
    rendered_opacity = render_pkg["opacity"]
    rendered_depth = render_pkg["depth"]
    rendered_normal = render_pkg["normal"]
    gt_image = viewpoint_camera.original_image.cuda()
    image_mask = viewpoint_camera.image_mask.cuda()

    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    # ic(opt.lambda_depth, opt.lambda_mask_entropy, opt.lambda_normal_render_depth, opt.lambda_normal_mvs_depth)
    # exit()
    if opt.lambda_opacity > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        image_mask = viewpoint_camera.image_mask.cuda()
        Ll1_opacity = F.l1_loss(o, image_mask).item()
        ssim_val_opacity = ssim(o, image_mask)
        loss_opacity = (1.0 - opt.lambda_dssim) * Ll1_opacity + opt.lambda_dssim * (1.0 - ssim_val_opacity)
        tb_dict["loss_mask_entropy"] = loss_opacity.item()
        loss = loss + opt.lambda_opacity * loss_opacity

    if opt.lambda_normal_render_depth > 0: # 0.01
        normal_pseudo = render_pkg['pseudo_normal']
        loss_normal_render_depth = F.mse_loss(
            rendered_normal * image_mask, normal_pseudo.detach() * image_mask)
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth

    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict

def render_scalar(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
           scaling_modifier=1.0,override_color=None, opt: OptimizationParams = None, 
           is_training=False, dict_params=None):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color,
                          computer_pseudo_normal=True if opt is not None and opt.lambda_normal_render_depth>0 else False)

    results["hdr"] = viewpoint_camera.hdr

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt)
        results["tb_dict"] = tb_dict
        results["loss"] = loss
    
    return results
