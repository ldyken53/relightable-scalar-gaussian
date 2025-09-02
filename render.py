from icecream import install
install()
ic.configureOutput(includeContext=True) # type: ignore


import json
import os
import cv2
from gaussian_renderer import render_fn_dict
import numpy as np
import torch
import re
from scene import GaussianModel, ScalarGaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, fov2focal
from utils.system_utils import searchForMaxIteration
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.image_utils import psnr
from utils.loss_utils import ssim
from scene.palette_color import LearningPaletteColor
from scene.opacity_trans import LearningOpacityTransform
from scene.light_trans import LearningLightTransform
import glob
from lpipsPyTorch import lpips
from time import time_ns

def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict

def load_ckpts_paths(source_dir):
    ckpts_transforms = {}
    one_TF_json = {'path': None, 'palette':None, 'transform': [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
    ckpt_dir = os.path.join(source_dir, "point_cloud")
    max_iters = searchForMaxIteration(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, f"iteration_{max_iters}", "point_cloud.ply")
    palette_path = os.path.join(ckpt_dir, f"iteration_{max_iters}", "palette_colors_chkpnt.pth")
    one_TF_json['path'] = ckpt_path
    one_TF_json['palette'] = palette_path
    ckpts_transforms['0'] = one_TF_json
    return ckpts_transforms

def scene_composition(scene_dict: dict, dataset: ModelParams, is_scalar = False):
    gaussians_list = []
    for scene in scene_dict:
        if is_scalar:
            gaussians = ScalarGaussianModel(render_type="phong")
        else:
            gaussians = GaussianModel(dataset.sh_degree, render_type="phong")
        print("Compose scene from GS path:", scene_dict[scene]["path"])
        gaussians.my_load_ply(scene_dict[scene]["path"], quantised=True, half_float=True)
        # gaussians.my_load_ply(scene_dict[scene]["path"], quantised=False, half_float=False)
        
        torch_transform = torch.tensor(scene_dict[scene]["transform"], device="cuda").reshape(4, 4)
        gaussians.set_transform(transform=torch_transform)

        gaussians_list.append(gaussians)
    if is_scalar:
        gaussians_composite = ScalarGaussianModel.create_from_gaussians(gaussians_list, dataset)
    else:
        gaussians_composite = GaussianModel.create_from_gaussians(gaussians_list, dataset)
    n = gaussians_composite.get_xyz.shape[0]
    print(f"Totally {n} points loaded.")

    return gaussians_composite


def render_points(camera, gaussians):
    intrinsic = camera.get_intrinsics()
    w2c = camera.world_view_transform.transpose(0, 1)

    xyz = gaussians.get_xyz
    color = gaussians.get_base_color
    xyz_homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
    xyz_cam = (xyz_homo @ w2c.T)[:, :3]
    z = xyz_cam[:, 2]
    uv_homo = xyz_cam @ intrinsic.T
    uv = uv_homo[:, :2] / uv_homo[:, 2:]
    uv = uv.long()

    valid_point = torch.logical_and(torch.logical_and(uv[:, 0] >= 0, uv[:, 0] < W),
                                    torch.logical_and(uv[:, 1] >= 0, uv[:, 1] < H))
    uv = uv[valid_point]
    z = z[valid_point]
    color = color[valid_point]

    depth_buffer = torch.full_like(render_pkg['render'][0], 10000)
    rgb_buffer = torch.full_like(render_pkg['render'], bg)
    while True:
        mask = depth_buffer[uv[:, 1], uv[:, 0]] > z
        if mask.sum() == 0:
            break
        uv_mask = uv[mask]
        depth_buffer[uv_mask[:, 1], uv_mask[:, 0]] = z[mask]
        rgb_buffer[:, uv_mask[:, 1], uv_mask[:, 0]] = color[mask].transpose(-1, -2)

    return rgb_buffer

def numpy_to_tensor(img):
    """Convert numpy image (H,W,C) to tensor (C,H,W) and normalize to [0,1]"""
    img = img.astype(np.float32) / 255.0  # Convert to float and normalize
    img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
    return img


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('-vo', '--view_config', default=None, required=True, help="the config root")
    parser.add_argument('-so', '--source_dir', default=None, required=True, help="the source ckpts dir")
    parser.add_argument('-bg', "--background_color", type=float, default=None,
                        help="If set, use it as background color")
    parser.add_argument('--video', action='store_true', default=False, help="If True, output video as well.")
    parser.add_argument('--output', default="./capture_trace", help="Output dir.")
    parser.add_argument('--capture_list',
                        default="phong",
                        help="what should be rendered for output.")
    parser.add_argument('--useHeadlight', action='store_true', help="If True, use head light.")
    parser.add_argument('--evaluation', action='store_false', help="If True, eval mode.")
    parser.add_argument('--EvalTime', action='store_true', help="If True, eval time, not save images.")
    parser.add_argument('--is_scalar', action='store_true', default=False)
    parser.add_argument('--TFnums', type=int, default=10)
    args = parser.parse_args()
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    

    # load configs
    view_config_file = f"{args.view_config}/cameras_test.json"
    final_folder = os.path.basename(os.path.normpath(args.view_config))
    TFidx = int(re.search(r'\d+$', final_folder).group()) - 1
    print(f"TFidx is: {TFidx}")
    scene_dict = load_ckpts_paths(args.source_dir)
    TFs_names = list(scene_dict.keys())
    TFs_nums = len(TFs_names)

    view_dict = load_json_config(view_config_file)
    
    # load palette
    palette_color_transforms = []
    opacity_transforms = []

    palette_color_transform = LearningPaletteColor()
    palette_color_transform.create_from_ckpt(f"{scene_dict['0']['palette']}")
    palette_color_transforms.append(palette_color_transform)
    # ic(palette_color_transform.palette_color)
    
    opacity_transform = LearningOpacityTransform()
    opacity_transforms.append(opacity_transform)
        
    light_transform = LearningLightTransform(theta=180, phi=0)
    # load gaussians
    gaussians_composite = scene_composition(scene_dict, dataset, args.is_scalar)

    # rendering
    capture_dir = args.output
    os.makedirs(capture_dir, exist_ok=True)
    capture_list = [str.strip() for str in args.capture_list.split(",")]
    for capture_type in capture_list:
        capture_type_dir = os.path.join(capture_dir, capture_type)
        os.makedirs(capture_type_dir, exist_ok=True)
    

    bg = args.background_color
    if bg is None:
        bg = 1 if dataset.white_background else 0
    background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")
    render_fn = render_fn_dict[f"{'scalar_' if args.is_scalar else ''}phong"]

    render_kwargs = {
        "pc": gaussians_composite,
        "pipe": pipe,
        "bg_color": background,
        "is_training": False,
        "dict_params": {
            "sample_num": args.sample_num,
            "palette_colors": palette_color_transforms,
            "light_transform": light_transform
        }
    }
    if not args.useHeadlight:
        render_kwargs["dict_params"]["light_transform"].useHeadLight = False

    H = 800
    W = 800
    fovx = 30 * np.pi / 180
    fovy = focal2fov(fov2focal(fovx, W), H)

    progress_bar = tqdm(view_dict, desc="Rendering")
    time_log = []
    for idx, cam_info in enumerate(progress_bar):
        tic = time_ns()
        # cam_info = traject_dict["trajectory"].items()
        # ic(cam_info)
        # cam_pose = cam_info['transform_matrix']
        light_angle = cam_info.get('light_angle', None)
        # # exit()

        # c2w = np.array(cam_pose, dtype=np.float32).reshape(4, 4) 
        
        # c2w[:3, 1:3] *= -1
        # w2c = np.linalg.inv(c2w)
        # R = w2c[:3, :3].T # R is stored transposed due to 'glm' in CUDA code
        # T = w2c[:3, 3]
        R_w2c = np.array(cam_info["R"])      # as written in JSON
        C      = np.array(cam_info["T"])     # camera centre

        R = R_w2c.T                       # convert to C2W
        T = -R_w2c @ C                    # *use the ORIGINAL W2C* to build t

        fovx = cam_info["FovX"]
        fovy = cam_info["FovY"]
        custom_cam = Camera(colmap_id=0, R=R, T=T,
                            FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
                            image=torch.zeros(3, H, W), image_name=None, uid=0, colormap=cam_info.get("colormap"))
        if not args.is_scalar and cam_info.get("colormap"):
            cmap = plt.cm.get_cmap(cam_info["colormap"])
            # Calculate the interval for this TF
            interval_start = TFidx / args.TFnums
            interval_end = (TFidx + 1) / args.TFnums
            midpoint = (interval_start + interval_end) / 2.0
            # Get the color from the colormap at the midpoint
            rgba_color = cmap(midpoint)  # Returns (r, g, b, a) in [0, 1]
            rgb_color = rgba_color[:3]  # Take only RGB, ignore alpha
            # Convert to tensor and assign to palette color
            with torch.no_grad():
                render_kwargs["dict_params"]["palette_colors"][0].palette_color = torch.tensor(
                    rgb_color, dtype=torch.float32, device="cuda"
                )
      
        
        if light_angle is not None:
            render_kwargs["dict_params"]["light_transform"].set_light_theta_phi(light_angle[0], light_angle[1])
        
        
        with torch.no_grad():
            render_pkg = render_fn(camera=custom_cam, **render_kwargs)
        
     
        if not args.EvalTime:
            for capture_type in capture_list:
                if capture_type == "points":
                    render_pkg[capture_type] = render_points(custom_cam, gaussians_composite)
                elif capture_type == "normal":
                    render_pkg[capture_type] = render_pkg[capture_type] * 0.5 + 0.5
                    render_pkg[capture_type] = render_pkg[capture_type] + (1 - render_pkg['opacity']) * bg
                elif capture_type in ["base_color", "roughness", "metallic", "visibility"]:
                    render_pkg[capture_type] = render_pkg[capture_type] + (1 - render_pkg['opacity']) * bg
            
                save_image(render_pkg[capture_type], f"{capture_dir}/{capture_type}/frame_{int(idx):04d}.png")
        toc = time_ns()
        time_log.append((toc - tic)/1e9)
    time_log = np.array(time_log)
    print(f"Averaged Rendering time: {time_log.mean()*1e3} ms")
    print(f"FPS: {1/time_log.mean()}")
    
    if (args.evaluation and not args.EvalTime):
        GTImgPaths = sorted(glob.glob(f"{args.view_config}/test/*.png"))
        evalImgPaths = sorted(glob.glob(f"{capture_dir}/phong/*.png"))
        psnr_test = 0
        psnr2_test = 0
        lpips_test = 0
        ssim_test = 0
        
        for idx, evalImgPath in enumerate(evalImgPaths):
            evalImg = cv2.imread(evalImgPath) 
            GTImg = cv2.imread(GTImgPaths[idx], cv2.IMREAD_UNCHANGED) 

            if GTImg.shape[2] == 4:
                alpha = GTImg[:, :, 3] / 255.0  
                white_bg = np.zeros_like(GTImg[:, :, :3]) * 255  
                GTImg = (GTImg[:, :, :3] * alpha[:, :, None] + white_bg * (1 - alpha[:, :, None])).astype(np.uint8)
            else:
                GTImg = GTImg[:, :, :3]
            GTtensor = numpy_to_tensor(GTImg)
            evaltensor = numpy_to_tensor(evalImg)
            psnr_test += psnr(GTtensor, evaltensor).mean().double()
            ssim_test += ssim(GTtensor, evaltensor).mean().double()
            psnr2_test += cv2.PSNR(GTImg, evalImg)
            # lpips_test += lpips(GTtensor, evaltensor, net_type='vgg').mean().double()
            diff = cv2.absdiff(GTImg, evalImg)
            cv2.imwrite(f"{capture_dir}/diff{idx}.png", diff)
            # print(f"{idx}")
        psnr_test /= len(evalImgPaths)
        lpips_test /= len(evalImgPaths)
        ssim_test /= len(evalImgPaths)
        psnr2_test /= len(evalImgPaths)
        print(f"Tensor PSNR: {psnr_test}, LPIPS: {lpips_test}, SSIM: {ssim_test}, PSNR: {psnr2_test}")


    # output as video
    if (args.video and not args.EvalTime):
        progress_bar = tqdm(capture_list, desc="Outputting video")
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        for capture_type in progress_bar:
            video_path = f"{capture_dir}/{capture_type}.mp4"
            image_names = [os.path.join(capture_dir, capture_type, f"frame_{int(j):04d}.png") for j in
                           range(len(view_dict))]
            media_writer = cv2.VideoWriter(video_path, fourcc, 60, (W, H))

            for image_name in image_names:
                img = cv2.imread(image_name)
                media_writer.write(img)
            media_writer.release()