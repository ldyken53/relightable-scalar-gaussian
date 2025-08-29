import math
import torch
import numpy as np
import torch.nn.functional as F
from arguments import OptimizationParams

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.sh_utils import eval_sh, eval_sh_coef
from utils.loss_utils import ssim, bilateral_smooth_loss, contentrate_loss, sparsity_loss
from utils.image_utils import psnr
from utils.graphics_utils import fibonacci_sphere_sampling
from .diff_rasterization import GaussianRasterizationSettings, GaussianRasterizer, RenderEquation, \
    RenderEquation_complex


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
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None):
    gamma_transform = dict_params.get("gamma")
    opacity_transforms = dict_params.get("opacity_factors")
    light_transform = dict_params.get("light_transform")

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # ic(camera)
    # exit()
    
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
        sh_degree=0,
        campos=camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
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
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Just use white as debug color
    shs = None
    colors_precomp = torch.ones(
        (means3D.shape[0], 3),
        device="cuda", dtype=means3D.dtype)

    scalar_color = scalar2rgb(pc.get_scalar2, camera.colormap)
    diffuse_factor = pc.get_diffuse_factor
    shininess = pc.get_shininess
    ambient_factor = pc.get_ambient_factor
    specular_factor = pc.get_specular_factor 
    normal = pc.get_normal
    num_GSs_TF = pc.get_num_GSs_TF

    viewdirs = F.normalize(camera.camera_center - means3D, dim=-1)

    # exit()
    light_pos = light_transform.get_light_dir()
    if light_pos is None:
        incidents_dirs = viewdirs.detach().contiguous() #* now we are using headlights
    else:
        incidents_dirs = F.normalize(light_pos, dim=-1).detach().contiguous() #* orbital light with directional lighting
    # ic(incidents.shape) # [N, 16, 3]

     
    brdf_color, extra_results, opacity = rendering_equation_BlinnPhong_python(
        opacity_transforms, light_transform, opacity, scalar_color, diffuse_factor, shininess, ambient_factor, specular_factor, normal, viewdirs, incidents_dirs, num_GSs_TF)


    if is_training:
        features = torch.cat([brdf_color, normal, scalar_color, diffuse_factor, shininess, ambient_factor, specular_factor], dim=-1)
    else:
        features = torch.cat([brdf_color, normal, scalar_color, diffuse_factor, shininess, ambient_factor, specular_factor,
                              extra_results['diffuse_render'].mean(dim=1),
                              extra_results['specular_render'].mean(dim=1),
                              extra_results['ambient_render'].mean(dim=1)], dim=-1)
        
    # ic(shs.shape) #* this one can not be empty
    # ic(colors_precomp.shape) #* this one is OK to be empty

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

    feature_dict = {}
    if is_training:
        rendered_phong, rendered_normal, rendered_scalar, \
            rendered_diffuse_factor, rendered_shininess, \
            rendered_amibent_factor, rendered_specular_factor, \
            = rendered_feature.split([3, 3, 3, 1, 1, 1, 1], dim=0)
        feature_dict.update({"diffuse_factor": rendered_diffuse_factor,
                             "shininess": rendered_shininess,
                             "ambient_factor": rendered_amibent_factor,
                             "specular_factor": rendered_specular_factor,
                             #* below are used for calculating distribution loss
                             "ambient_values": ambient_factor,
                             })
    else:
        rendered_phong, rendered_normal, rendered_scalar, \
            rendered_diffuse_factor, rendered_shininess, \
            rendered_amibent_factor, rendered_specular_factor, \
            rendered_diffuse_term, rendered_specular_term, rendered_ambient_term \
            = rendered_feature.split([3, 3, 3, 1, 1, 1, 1, 3, 3, 3], dim=0)

        feature_dict.update({"diffuse_factor": rendered_diffuse_factor,
                             "shininess": rendered_shininess,
                             "ambient_factor": rendered_amibent_factor,
                             "specular_factor": rendered_specular_factor,
                             "diffuse_term": rendered_diffuse_term,
                             "specular_term": rendered_specular_term,
                             "ambient_term": rendered_ambient_term,
                             })

    phong = rendered_phong
    rendered_phong = phong + (1 - rendered_opacity) * bg_color[:, None, None]

    val_gamma = 0
    if gamma_transform is not None:
        rendered_phong = gamma_transform.hdr2ldr(rendered_phong)
        val_gamma = gamma_transform.gamma.item()
    # ic(gamma_transform.gamma)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results = {"render": rendered_scalar, 
               "phong": rendered_phong,
                "color_render": rendered_image,
               "normal": rendered_normal, #* normal
               "pseudo_normal": rendered_pseudo_normal, #* pseudo normal
               "surface_xyz": rendered_surface_xyz, #* surface xyz
               "opacity": rendered_opacity, #* opacity
               "depth": rendered_depth, #* depth
               "viewspace_points": screenspace_points, #* viewspace points
               "visibility_filter": radii > 0, #* visibility filter
               "radii": radii, #* radii
               "num_rendered": num_rendered, #* num rendered
               "num_contrib": num_contrib, #* num contrib
               }

    results.update(feature_dict)
    results["hdr"] = camera.hdr
    results["val_gamma"] = val_gamma

    return results


def calculate_loss(camera, pc, results, opt):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    rendered_image = results["render"]
    rendered_depth = results["depth"]
    rendered_normal = results["normal"]
    rendered_phong = results["phong"] #* pbr -> phong
    rendered_opacity = results["opacity"]
    rendered_diffuse_factor = results["diffuse_factor"]
    rendered_ambient_intensity = results["ambient_factor"]
    rendered_specular_intensity = results["specular_factor"]
    cur_ambient_factors = results["ambient_values"]
    gt_image = camera.original_image.cuda()
    loss = 0.0
    #* OK
    if opt.lambda_render > 0:
        Ll1 = F.l1_loss(rendered_image, gt_image)
        ssim_val = ssim(rendered_image, gt_image)
        tb_dict["l1"] = Ll1.item()
        tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
        tb_dict["ssim"] = ssim_val.item()
        loss = loss + (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    
    #* OK
    if opt.lambda_phong > 0:
        Ll1_pbr = F.l1_loss(rendered_phong, gt_image)
        ssim_val_pbr = ssim(rendered_phong, gt_image)
        tb_dict["l1_pbr"] = Ll1_pbr.item()
        tb_dict["ssim_pbr"] = ssim_val_pbr.item()
        tb_dict["psnr_pbr"] = psnr(rendered_phong, gt_image).mean().item()
        loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
        loss = loss + opt.lambda_phong * loss_pbr

    #* necessary? I do not think we have depth gt
    if opt.lambda_depth > 0:
        gt_depth = camera.depth.cuda()
        image_mask = camera.image_mask.cuda().bool()
        depth_mask = gt_depth > 0
        sur_mask = torch.logical_xor(image_mask, depth_mask)

        loss_depth = F.l1_loss(rendered_depth[~sur_mask], gt_depth[~sur_mask])
        tb_dict["loss_depth"] = loss_depth.item()
        loss = loss + opt.lambda_depth * loss_depth

    if opt.lambda_opacity > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        image_mask = camera.image_mask.cuda()
        Ll1_opacity = F.l1_loss(o, image_mask).item()
        ssim_val_opacity = ssim(o, image_mask)
        # loss_mask_entropy = -(image_mask * torch.log(o) + (1 - image_mask) * torch.log(1 - o)).mean()
        loss_opacity = (1.0 - opt.lambda_dssim) * Ll1_opacity + opt.lambda_dssim * (1.0 - ssim_val_opacity)
        tb_dict["loss_mask_entropy"] = loss_opacity.item()
        loss = loss + opt.lambda_opacity * loss_opacity
        
    if opt.lambda_normal_render_depth > 0:
        normal_pseudo = results['pseudo_normal']
        image_mask = camera.image_mask.cuda()
        loss_normal_render_depth = F.mse_loss(
            rendered_normal * image_mask, normal_pseudo.detach() * image_mask)
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth

    if opt.lambda_normal_mvs_depth > 0:
        gt_depth = camera.depth.cuda()
        depth_mask = (gt_depth > 0).float()
        mvs_normal = camera.normal.cuda()

        # depth to normal, if there is a gt depth but not a MVS normal map
        if torch.allclose(mvs_normal, torch.zeros_like(mvs_normal)):
            from kornia.geometry import depth_to_normals
            normal_pseudo_cam = -depth_to_normals(gt_depth[None], camera.intrinsics[None])[0]
            c2w = camera.world_view_transform.T.inverse()
            R = c2w[:3, :3]
            _, H, W = normal_pseudo_cam.shape
            mvs_normal = (R @ normal_pseudo_cam.reshape(3, -1)).reshape(3, H, W)
            camera.normal = mvs_normal.cpu()

        loss_normal_mvs_depth = F.mse_loss(
            rendered_normal * depth_mask, mvs_normal * depth_mask)
        tb_dict["loss_normal_mvs_depth"] = loss_normal_mvs_depth.item()
        loss = loss + opt.lambda_normal_mvs_depth * loss_normal_mvs_depth
    
    if opt.lambda_diffuse_factor_smooth > 0:
        image_mask = camera.image_mask.cuda()
        loss_diffuse_factor_smooth = bilateral_smooth_loss(rendered_diffuse_factor, gt_image, image_mask)
        tb_dict["lambda_diffuse_factor_smooth"] = loss_diffuse_factor_smooth.item()
        loss = loss + opt.lambda_diffuse_factor_smooth * loss_diffuse_factor_smooth
        
    if opt.lambda_specular_factor_smooth > 0:
        image_mask = camera.image_mask.cuda()
        loss_specular_smooth = bilateral_smooth_loss(rendered_specular_intensity, gt_image, image_mask)
        tb_dict["lambda_specular_factor_smooth"] = loss_specular_smooth.item()
        loss = loss + opt.lambda_specular_factor_smooth * loss_specular_smooth
    
    #* ambient smooth
    if opt.lambda_ambient_factor_smooth > 0:
        image_mask = camera.image_mask.cuda()
        loss_ambient_intensity_smooth = bilateral_smooth_loss(rendered_ambient_intensity, gt_image, image_mask)
        tb_dict["lambda_ambient_factor_smooth"] = loss_ambient_intensity_smooth.item()
        loss = loss + opt.lambda_ambient_factor_smooth * loss_ambient_intensity_smooth
    
    #*normal smooth
    if opt.lambda_normal_smooth > 0:
        image_mask = camera.image_mask.cuda()
        loss_normal_smooth = bilateral_smooth_loss(rendered_normal, gt_image, image_mask)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        loss = loss + opt.lambda_normal_smooth * loss_normal_smooth


    tb_dict["loss"] = loss.item()

    return loss, tb_dict


def render_neilf_scalar(camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                 scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                 is_training=False, dict_params=None, light_pos=None):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(camera, pc, pipe, bg_color,
                          scaling_modifier, override_color, is_training, dict_params)

    if is_training:
        loss, tb_dict = calculate_loss(camera, pc, results, opt)
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results


def rendering_equation_BlinnPhong_python(opacity_transforms, light_transform, opacity, color, diffuse_factor, shininess, ambient_factor, specular_factor, normals, viewdirs,
                                         incidents_dirs, num_GSs_TFs):
    spcular_multi, diffuse_factor_multi, ambient_multi, shininess_multi,\
        specular_offset, diffuse_factor_offset, ambient_offset, shininess_offset= light_transform.get_light_transform()
    color = color.unsqueeze(-2).contiguous()
    diffuse_factor = diffuse_factor.unsqueeze(-2).contiguous()*diffuse_factor_multi+diffuse_factor_offset # ambient white light intensity
    shininess = shininess.unsqueeze(-2).contiguous()*shininess_multi+shininess_offset # specular white light intensity
    ambient_factor = ambient_factor.unsqueeze(-2).contiguous()*ambient_multi+ambient_offset # ambient white light intensity
    specular_factor = specular_factor.unsqueeze(-2).contiguous()*spcular_multi+specular_offset # specular white light intensity
    normals = normals.unsqueeze(-2).contiguous()
    viewdirs = viewdirs.unsqueeze(-2).contiguous()
    incident_dirs = incidents_dirs.unsqueeze(-2).contiguous()
    diffuse_color = color.clone()
    
    start_GS = 0
    for i in range(len(num_GSs_TFs)):
        end_GS = num_GSs_TFs[i] + start_GS
        if opacity_transforms is not None:
            opacity[start_GS:end_GS,:] = opacity[start_GS:end_GS,:] * opacity_transforms[i].opacity_factor
        start_GS = end_GS

    #* light_intesity = kd
    #* diffuse color 
    cos_l = (normals * incident_dirs).sum(dim=-1, keepdim=True)
    
    diffuse_intensity = diffuse_factor * torch.abs(cos_l)
    diffuse_color = diffuse_color * (ambient_factor + diffuse_intensity).repeat(1, 1, 3)
    diffuse_term = diffuse_intensity.repeat(1, 1, 3)*diffuse_color
    ambient_term = torch.clamp(ambient_factor.repeat(1, 1, 3)*diffuse_color,0.,1.)
    #* specular color
    h = F.normalize(incident_dirs + viewdirs, dim=-1) # bisector h
    cos_h = (normals * h).sum(dim=-1, keepdim=True)
    # specular_intensity = specular_factor*diffuse_factor*torch.where(cos_l != 0, (torch.abs(cos_h)).pow(shininess), 0.0)
    specular_intensity = specular_factor * torch.where(cos_l != 0, (torch.abs(cos_h)).pow(shininess), 0.0)
    specular_color = specular_intensity.repeat(1, 1, 3)
    
    pbr = diffuse_color.squeeze() + specular_color.squeeze() #* seems better
    pbr = torch.clamp(pbr, 0., 1.)

    extra_results = {
        "diffuse_render": diffuse_term,
        "specular_render": specular_color,
        "ambient_render": ambient_term,
    }

    return pbr, extra_results, opacity

