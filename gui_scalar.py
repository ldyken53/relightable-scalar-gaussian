from icecream import install
install()
ic.configureOutput(includeContext=True) #type: ignore

import glob
import json
import os
import torchvision.transforms
import dearpygui.dearpygui as dpg
from dearpygui_ext.themes import create_theme_imgui_light
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from gaussian_renderer import render_fn_dict
from scene import ScalarGaussianModel
from utils.general_utils import safe_state
from utils.camera_utils import Camera, JSON_to_camera
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov,ThetaPhi2xyz,fov2focal
from scene.palette_color import LearningPaletteColor
from scene.opacity_trans import LearningOpacityTransform
from scene.light_trans import LearningLightTransform
from pyquaternion import Quaternion
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class OpacityMapEditor:
    def __init__(self, width=400, height=200, max_opacity=3.0):
        self.width = width
        self.height = height
        self.max_opacity = max_opacity
        self.control_points = [(0.0, 1.0), (1.0, 1.0)]  # (x, opacity) pairs
        self.opacity_curve = None
        self.update_curve()
        
    def update_curve(self):
        """Update the interpolated opacity curve from control points"""
        if len(self.control_points) < 2:
            return
            
        sorted_points = sorted(self.control_points, key=lambda p: p[0])
        x_vals = [p[0] for p in sorted_points]
        y_vals = [p[1] for p in sorted_points]
        
        self.interp_func = interp1d(x_vals, y_vals, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
        
        x_smooth = np.linspace(0, 1, 100)
        y_smooth = np.clip(self.interp_func(x_smooth), 0, self.max_opacity)
        self.opacity_curve = list(zip(x_smooth, y_smooth))
    
    def data_to_screen(self, data_x, data_y):
        """Convert data coordinates to screen coordinates"""
        screen_x = data_x * self.width
        screen_y = self.height - (data_y / self.max_opacity * self.height)
        return screen_x, screen_y
    
    def add_control_point(self, position, opacity):
        """Add a new control point at normalized position (0-1) with given opacity"""
        position = np.clip(position, 0, 1)
        opacity = np.clip(opacity, 0, self.max_opacity)
        
        # Don't add points too close to existing ones
        for existing_x, existing_y in self.control_points:
            if abs(position - existing_x) < 0.05:  # 5% of range
                return False
                
        self.control_points.append((position, opacity))
        self.control_points.sort(key=lambda p: p[0])  # Keep sorted by position
        self.update_curve()
        return True
    
    def remove_control_point(self, index):
        """Remove a control point (but keep at least 2 points)"""
        if len(self.control_points) > 2 and 0 <= index < len(self.control_points):
            del self.control_points[index]
            self.update_curve()
            return True
        return False
    
    def update_control_point(self, index, position, opacity):
        """Update an existing control point"""
        if 0 <= index < len(self.control_points):
            position = np.clip(position, 0, 1)
            opacity = np.clip(opacity, 0, self.max_opacity)
            self.control_points[index] = (position, opacity)
            self.control_points.sort(key=lambda p: p[0])  # Keep sorted by position
            self.update_curve()
    
    def get_opacity_at_position(self, position):
        """Get opacity value at a specific position (0-1)"""
        if self.interp_func is None:
            return 1.0
        return float(np.clip(self.interp_func(position), 0, self.max_opacity))
    
    def reset_to_default(self):
        """Reset to default linear opacity map"""
        self.control_points = [(0.0, 1.0), (1.0, 1.0)]
        self.update_curve()

def create_opacity_map_widget(tag, width=400, height=200, callback=None):
    """Create an opacity map editor with individual sliders for each control point"""
    
    editor = OpacityMapEditor(width, height)
    
    def update_point_from_slider(sender, app_data):
        """Update a control point from its slider values"""
        # Extract point index from slider tag - need to get the last part after splitting
        if "_pos_" in sender:
            point_idx = int(sender.split("_pos_")[-1])  # Get last part
            position = app_data / 100.0  # Convert from percentage
            opacity = dpg.get_value(f"{tag}_opacity_{point_idx}")
        elif "_opacity_" in sender:
            point_idx = int(sender.split("_opacity_")[-1])  # Get last part
            opacity = app_data
            position = dpg.get_value(f"{tag}_pos_{point_idx}") / 100.0
        else:
            return
            
        if 0 <= point_idx < len(editor.control_points):
            editor.update_control_point(point_idx, position, opacity)
            update_opacity_display(tag, editor)
            if callback:
                callback(editor)
    
    def add_point_callback(sender, app_data):
        """Add a new control point"""
        position = dpg.get_value(f"{tag}_new_position") / 100.0
        opacity = dpg.get_value(f"{tag}_new_opacity")
        
        success = editor.add_control_point(position, opacity)
        if success:
            update_opacity_display(tag, editor)
            refresh_point_sliders(tag, editor, callback)
            if callback:
                callback(editor)
            print(f"Added point at position {position:.2f}, opacity {opacity:.2f}")
        else:
            print("Could not add point - too close to existing point")
    
    def remove_point_callback(sender, app_data):
        """Remove a control point"""
        # Extract point index from button tag - get the last part after splitting
        point_idx = int(sender.split("_remove_")[-1])
        
        success = editor.remove_control_point(point_idx)
        if success:
            update_opacity_display(tag, editor)
            refresh_point_sliders(tag, editor, callback)
            if callback:
                callback(editor)
            print(f"Removed point {point_idx}")
    
    def reset_callback(sender, app_data):
        """Reset to default"""
        editor.reset_to_default()
        update_opacity_display(tag, editor)
        refresh_point_sliders(tag, editor, callback)
        if callback:
            callback(editor)
    
    def refresh_point_sliders(tag, editor, callback):
        """Refresh the slider controls for all points"""
        # Clear existing point controls
        if dpg.does_item_exist(f"{tag}_points_group"):
            dpg.delete_item(f"{tag}_points_group")
        
        # Recreate point controls
        with dpg.group(tag=f"{tag}_points_group", parent=f"{tag}_main_group"):
            dpg.add_text("Control Points:")
            
            for i, (pos, opacity) in enumerate(editor.control_points):
                with dpg.group(horizontal=True):
                    dpg.add_text(f"Point {i}:")
                    
                    dpg.add_text("Pos:")
                    dpg.add_slider_float(
                        tag=f"{tag}_pos_{i}",
                        min_value=0, max_value=100,
                        default_value=pos * 100,
                        callback=update_point_from_slider,
                        width=80, format="%.0f%%"
                    )
                    
                    dpg.add_text("Opacity:")
                    dpg.add_slider_float(
                        tag=f"{tag}_opacity_{i}",
                        min_value=0, max_value=editor.max_opacity,
                        default_value=opacity,
                        callback=update_point_from_slider,
                        width=80, format="%.2f"
                    )
                    
                    # Only show remove button if we have more than 2 points
                    if len(editor.control_points) > 2:
                        dpg.add_button(
                            label="Remove",
                            tag=f"{tag}_remove_{i}",
                            callback=remove_point_callback,
                            width=60
                        )
    
    # Create the main widget structure
    with dpg.group(tag=tag):
        dpg.add_text("Opacity Map Editor")
        
        # Visual display
        drawlist_tag = f"{tag}_drawlist"
        with dpg.drawlist(width=width, height=height, tag=drawlist_tag):
            pass
        
        dpg.add_separator()
        
        # Main controls group
        with dpg.group(tag=f"{tag}_main_group"):
            # Add new point controls
            with dpg.group():
                dpg.add_text("Add New Point:")
                with dpg.group(horizontal=True):
                    dpg.add_text("Position (%):")
                    dpg.add_slider_float(
                        tag=f"{tag}_new_position",
                        min_value=0, max_value=100,
                        default_value=50,
                        width=100, format="%.0f%%"
                    )
                    
                    dpg.add_text("Opacity:")
                    dpg.add_slider_float(
                        tag=f"{tag}_new_opacity",
                        min_value=0, max_value=editor.max_opacity,
                        default_value=1.5,
                        width=100, format="%.2f"
                    )
                    
                    dpg.add_button(label="Add Point", callback=add_point_callback)
            
            dpg.add_separator()
        
        dpg.add_button(label="Reset to Linear", callback=reset_callback)
    
    # Initialize display and controls
    update_opacity_display(tag, editor)
    refresh_point_sliders(tag, editor, callback)
    
    print(f"Created opacity widget with sliders: {tag}")
    
    return editor

def update_opacity_display(tag, editor):
    """Update the visual display of the opacity map"""
    drawlist_tag = f"{tag}_drawlist"
    dpg.delete_item(drawlist_tag, children_only=True)
    
    # Draw background
    dpg.draw_rectangle((0, 0), (editor.width, editor.height), 
                      color=(60, 60, 60, 255), fill=(40, 40, 40, 255), 
                      parent=drawlist_tag)
    
    # Draw grid lines
    grid_color = (100, 100, 100, 255)
    # Vertical lines (every 20% = 0.2 * width)
    for i in range(6):  # 0%, 20%, 40%, 60%, 80%, 100%
        x = i * editor.width // 5
        dpg.draw_line((x, 0), (x, editor.height), color=grid_color, 
                     thickness=1, parent=drawlist_tag)
        # Add percentage labels
        dpg.draw_text((x + 2, 5), f"{i*20}%", 
                     color=(200, 200, 200, 255), size=10, parent=drawlist_tag)
    
    # Horizontal lines (opacity levels)
    for i in range(4):  # 0, 1, 2, 3 opacity
        y = editor.height - (i * editor.height // 3)
        dpg.draw_line((0, y), (editor.width, y), color=grid_color, 
                     thickness=1, parent=drawlist_tag)
        # Add opacity labels
        dpg.draw_text((5, y - 15), f"{i:.1f}", 
                     color=(200, 200, 200, 255), size=10, parent=drawlist_tag)
    
    # Draw opacity curve
    if editor.opacity_curve and len(editor.opacity_curve) > 1:
        curve_points = []
        for data_x, data_y in editor.opacity_curve:
            screen_x, screen_y = editor.data_to_screen(data_x, data_y)
            curve_points.append([screen_x, screen_y])
        
        # Draw curve as connected lines
        for i in range(len(curve_points) - 1):
            dpg.draw_line(curve_points[i], curve_points[i + 1], 
                         color=(120, 255, 120, 255), thickness=3, parent=drawlist_tag)
    
    # Draw control points
    for i, (data_x, data_y) in enumerate(editor.control_points):
        screen_x, screen_y = editor.data_to_screen(data_x, data_y)
        
        # Draw point with outline
        dpg.draw_circle((screen_x, screen_y), 8, 
                       color=(255, 255, 255, 255), thickness=2, parent=drawlist_tag)
        dpg.draw_circle((screen_x, screen_y), 6, 
                       color=(120, 150, 255, 255), fill=(120, 150, 255, 255), 
                       parent=drawlist_tag)
        
        # Draw point number
        dpg.draw_text((screen_x - 5, screen_y - 5), str(i), 
                     color=(255, 255, 255, 255), size=12, parent=drawlist_tag)

def screen_to_arcball(p:np.ndarray):
    dist = np.dot(p, p)
    if dist < 1.:
        return np.array([*p, np.sqrt(1.-dist)])
    else:
        return np.array([*normalize_vec(p), 0.])

def normalize_vec(v: np.ndarray):
    if v is None:
        print("None")
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.all(norm == np.zeros_like(norm)):
        return np.zeros_like(v)
    else:
        return v/norm

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def load_ckpts_paths(source_dir):
    TFs_folders = sorted(glob.glob(f"{source_dir}/TF*"))
    TFs_names = sorted([os.path.basename(folder) for folder in TFs_folders])

    ckpts_transforms = {}
    for idx, TF_folder in enumerate(TFs_folders):
        one_TF_json = {'path': None, 'transform': [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
        ckpt_dir = os.path.join(TF_folder,"neilf","point_cloud")
        max_iters = searchForMaxIteration(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, f"iteration_{max_iters}", "point_cloud.ply")
        one_TF_json['path'] = ckpt_path
        ckpts_transforms[TFs_names[idx]] = one_TF_json

    return ckpts_transforms

def scene_composition(scene_dict: dict, dataset: ModelParams):
    gaussians_list = []
    for scene in scene_dict:
        gaussians = ScalarGaussianModel(render_type="phong")
        print("Compose scene from GS path:", scene_dict[scene]["path"])
        gaussians.my_load_ply(scene_dict[scene]["path"], quantised=True, half_float=True)
        
        torch_transform = torch.tensor(scene_dict[scene]["transform"], device="cuda").reshape(4, 4)
        gaussians.set_transform(transform=torch_transform)

        gaussians_list.append(gaussians)

    gaussians_composite = ScalarGaussianModel.create_from_gaussians(gaussians_list, dataset)
    n = gaussians_composite.get_xyz.shape[0]
    print(f"Totally {n} points loaded.")

    return gaussians_composite

class ArcBallCamera:
    def __init__(self, W, H, fovy=60, near=0.1, far=10, rot=None, translate=None, center=None):
        self.W = W
        self.H = H
        if translate is None:
            self.radius = 1
            self.original_radius = 1
        else:
            self.radius = np.linalg.norm(translate)
            self.original_radius = np.linalg.norm(translate)
            
        # self.radius *= 2
        self.radius *= 2
        self.fovy = fovy  # in degree
        self.near = near
        self.far = far

        if center is None:
            self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        else:
            self.center = center

        if rot is None:
            self.rot = R.from_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))  # looking back to z axis
            self.original_rot = R.from_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
        else:
            self.rot = R.from_matrix(rot)
            self.original_rot = R.from_matrix(rot)

        # self.up = np.array([0, -1, 0], dtype=np.float32)  # need to be normalized!
        self.up = -self.rot.as_matrix()[:3, 1]

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)
    
    def reset_view(self):
        self.rot = self.original_rot
        self.radius = self.original_radius
        self.radius *= 2

    def orbit(self, lastX, lastY, X, Y):
        def vec_angle(v0: np.ndarray, v1: np.ndarray):
            return np.arccos(np.clip(np.dot(v0, v1)/(np.linalg.norm(v0)*np.linalg.norm(v1)), -1., 1.))
        ball_start = screen_to_arcball(np.array([lastX+1e-6, lastY+1e-6]))
        ball_curr = screen_to_arcball(np.array([X, Y]))
        rot_radians = vec_angle(ball_start, ball_curr)
        rot_axis = normalize_vec(np.cross(ball_start, ball_curr))
        q = Quaternion(axis=rot_axis, radians=rot_radians)
        self.rot = self.rot * R.from_matrix(q.inverse.rotation_matrix)
    
    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])

def replace_color_to_contrast(color):
    return (1 - color) * 0.7

class GUI:
    def __init__(self, H, W, fovy, c2w, center, render_fn, render_kwargs, TFnums,
                 mode="phong", debug=True):
        """
        If the image is hdr, set use_hdr2ldr = True for LDR visualization. [0, 1]
        If the image is hdr, set use_hdr2ldr = False, the range of the image is not [0,1].
        """
        self.ctrlW = 500 #475
        self.widget_indent = 75
        self.widget_top = 150
        self.imgW = W
        self.imgH = H
        self.debug = debug
        rot = c2w[:3, :3]
        translate = c2w[:3, 3] - center
        self.TFnums = TFnums
        self.render_fn = render_fn
        self.render_kwargs = render_kwargs
        self.selected_colormap = "rainbow"
        
        #* in case if you wish to use StyleRF-VolVis Camera Control
        # self.cam = OrbitCamera(self.imgW, self.imgH, fovy=fovy * 180 / np.pi, rot=rot, translate=translate, center=center)
        self.cam = ArcBallCamera(self.imgW, self.imgH, fovy=fovy * 180 / np.pi, rot=rot, translate=translate, center=center)

        self.render_buffer = np.zeros((self.imgW, self.imgH, 3), dtype=np.float32)
        self.resize_fn = torchvision.transforms.Resize((self.imgH, self.imgW), antialias=True)
        self.downsample = 1
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.prev_mouseX = None
        self.prev_mouseY = None
        self.rotating = False
        
        self.light_elevation = 0
        self.light_angle = 180
        self.useHeadlight = True
        
        self.menu = None
        self.mode = None
        self.step()
        self.mode = "phong"
        dpg.create_context()
        
        self.setup_font_theme()
        # dpg.bind_item_font(your_item, default_font)
        
        light_theme = create_theme_imgui_light()
        dpg.bind_theme(light_theme)
        self.register_dpg()

    def __del__(self):
        dpg.destroy_context()
    
    def setup_font_theme(self):
        with dpg.font_registry():
            default_font = dpg.add_font("./assets/font/Helvetica.ttf", 16)
        with dpg.theme() as theme_button:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (161, 238, 189)) #(139, 205, 162)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (174, 255, 204))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (205, 250, 219)) #(174, 255, 203)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
        dpg.bind_font(default_font)
        self.theme_button = theme_button

    def get_buffer(self, render_results, mode=None):
        if render_results is None or mode is None:
            output = torch.ones(self.imgH, self.imgW, 3, dtype=torch.float32, device='cuda').detach().cpu().numpy()
        else:
            output = render_results[mode]
            
            if mode == "depth":
                output = (output - output.min()) / (output.max() - output.min())
            elif mode == "num_contrib":
                output = output.clamp_max(1000) / 1000

            if len(output.shape) == 2:
                output = output[None]
            if output.shape[0] == 1:
                output = output.repeat(3, 1, 1)
            if "normal" in mode:
                opacity = render_results["opacity"]
                output = output * 0.5 + 0.5 * opacity
                output = output + (1 - opacity)
            elif mode in ["diffuse_term", "specular_term", "ambient_term"]:
                opacity = render_results["opacity"]
                output = output + (1 - opacity)
            if (self.imgH, self.imgW) != tuple(output.shape[1:]):
                output = self.resize_fn(output)

            output = output.permute(1, 2, 0).contiguous().detach().cpu().numpy()
        return output

    @property
    def custom_cam(self):
        w2c = self.cam.view
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        down = self.downsample
        H, W = self.imgH // down, self.imgW // down
        fovy = self.cam.fovy * np.pi / 180
        fovx = fovy * W / H
        custom_cam = Camera(colmap_id=0, R=R, T=-T,
                            FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
                            image=torch.zeros(3, H, W), image_name=None, uid=0, 
                            colormap=self.selected_colormap)
        return custom_cam

    @torch.no_grad()
    def render(self):
        self.step()
        dpg.render_dearpygui_frame()

    def step(self):
        self.start.record()
        render_pkg = self.render_fn(camera=self.custom_cam, **self.render_kwargs)
        self.end.record()
        torch.cuda.synchronize()
        t = self.start.elapsed_time(self.end)

        buffer1 = self.get_buffer(render_pkg, self.mode)
        self.render_buffer = buffer1

        if t == 0:
            fps = 0
        else:
            fps = int(1000 / t)

        if self.menu is None:
            #* Forgive me for this ugly fix for menu
            self.menu_map = {"phong": "Blinn-Phong", "normal": "Normal", "diffuse_term": "Diffuse",
                             "specular_term": "Specular", "ambient_term": "Ambient"}
            self.inv_menu_map = {v: k for k, v in self.menu_map.items()}
            # self.menu = [self.menu_map[k] for k, v in render_pkg.items() if
            #              k not in ["pseudo_normal","render", "num_contrib", "surface_xyz", "diffuse_factor","depth", "shininess", "ambient_factor", "specular_factor", "offset_color", "opacity", "color_render"] and isinstance(v, torch.Tensor) and np.array(v.shape).prod() % (self.imgH * self.imgW) == 0]
            self.menu = ["Blinn-Phong", "Ambient", "Diffuse", "Specular", "Normal"]
            
        else:
            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({fps} FPS)')
            dpg.set_value("_texture", self.render_buffer)
    
    def add_oneTFSlider(self, TFidx):
        def callback_TF_slider(sender, app_data):
            TFidx = int(sender.replace("_slider_TF", "")) - 1
            with torch.no_grad():
                self.render_kwargs["dict_params"]["opacity_factors"][TFidx].opacity_factor = torch.tensor(app_data, dtype=torch.float32, device="cuda")
        
        slider_tag = "_slider_TF" + str(TFidx+1)
        
        # indent = self.widget_indent if TFidx == 0 else 0
        indent = 0
        slider_width = (self.ctrlW-10)//self.TFnums # leave some space (10 pixels) at right
      
        with dpg.group():
            dpg.add_text(f"TF{TFidx+1}",indent=indent+slider_width//4)
            dpg.add_slider_float(
                tag=slider_tag,
                label='',
                default_value=1.0,
                min_value=0,
                max_value=3.0,
                height=150,
                # format="",
                callback=callback_TF_slider,
                vertical=True,
                width=slider_width, 
                indent=indent
            )
    
    def add_opacity_map_to_gui(self):
        """Add opacity map editor to the GUI"""
        
        def opacity_map_callback(editor):
            """Callback when opacity map changes"""
            # Apply opacity values to each TF based on their interval midpoints
            for TFidx in range(self.TFnums):
                interval_start = TFidx / self.TFnums
                interval_end = (TFidx + 1) / self.TFnums
                midpoint = (interval_start + interval_end) / 2.0
                
                opacity_value = editor.get_opacity_at_position(midpoint)
                
                with torch.no_grad():
                    self.render_kwargs["dict_params"]["opacity_factors"][TFidx].opacity_factor = torch.tensor(
                        opacity_value, dtype=torch.float32, device="cuda"
                    )
                
                # Update slider UI
                dpg.set_value(f"_slider_TF{TFidx+1}", opacity_value)
            
        
        # Create the opacity map widget
        self.opacity_editor = create_opacity_map_widget(
            tag="_opacity_map", 
            width=self.ctrlW - 20, 
            height=100,
            callback=opacity_map_callback
        )

    def get_colormap_options(self):
        """Return list of 10 popular matplotlib colormap names"""
        return ["rainbow", "rainbow_r", "viridis", "plasma", "inferno", "Blues", "Purples", 
                "jet", "coolwarm", "coolwarm_r", "RdYlBu"]

    def register_dpg(self):

        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.imgW, self.imgH, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.imgW, height=self.imgH):
            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=self.ctrlW, height=self.imgH+160, pos=(self.imgW, 0),
                        no_resize=True, no_move=True, no_title_bar=True, no_background=True):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            with dpg.group(horizontal=True):
                dpg.add_text("Inference Time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True, leaf=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = self.inv_menu_map[app_data]
                with dpg.group(horizontal=True):
                    dpg.add_text("Mode")
                    dpg.add_combo(self.menu, indent=self.widget_top, label='', default_value="Blinn-Phong", callback=callback_change_mode)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    
                with dpg.group(horizontal=True):
                    dpg.add_text("Field of View")
                    dpg.add_slider_int(label="",indent=self.widget_top, min_value=1, max_value=120, format="%d deg",
                                   default_value=self.cam.fovy, callback=callback_set_fovy)
                    
                def callback_set_BG_color(sender, app_data):
                    bg_color = app_data[:3]
                    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    self.render_kwargs["bg_color"] = bg_color
                  
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Background Color")
                    dpg.add_color_edit(label="", no_alpha=True, default_value=[255, 255, 255],
                                       indent=self.widget_top, callback=callback_set_BG_color) 
                
                def callback_reset_view(sender, app_data):
                    self.cam.reset_view()
                
                def callback_save_image(sender, app_data):
                    rendered_img = self.render_buffer
                    rendered_img = (rendered_img*255).astype(np.uint8)[...,[2,1,0]]
                    cv2.imwrite(os.path.join("./GUI_results", f'rendered_img.png'), rendered_img)
                    print("Image Saved")
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Reset View", tag="_button_reset_view", width=self.ctrlW//2, callback=callback_reset_view)
                    dpg.add_button(label="Save Image", tag="_button_save_image",width=self.ctrlW//2, callback=callback_save_image)
                    dpg.bind_item_theme("_button_reset_view", self.theme_button)
                    dpg.bind_item_theme("_button_save_image", self.theme_button)
                                    
            # color & opacity editing
            with dpg.collapsing_header(label="Color & Opacity Editing", default_open=True, leaf=True):
                self.add_opacity_map_to_gui()
                with dpg.group(horizontal=True, horizontal_spacing=0):
                    for i in range(self.TFnums):
                        self.add_oneTFSlider(i)
                def callback_reset_color_opacity(sender, app_data):
                    with torch.no_grad():
                        for TFidx in range(self.TFnums):
                            self.render_kwargs["dict_params"]["opacity_factors"][TFidx].opacity_factor = torch.tensor(1.0, dtype=torch.float32, device="cuda")
                            dpg.set_value(f"_slider_TF{TFidx+1}", 1)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Reset Color & Opacity", tag="_button_reset_color_opacity",width=self.ctrlW-15, callback=callback_reset_color_opacity)
                    dpg.bind_item_theme("_button_reset_color_opacity", self.theme_button)
                    # dpg.bind_item_theme("_button_save_color_opacity", self.theme_button)
                def callback_change_colormap(sender, app_data):
                    self.selected_colormap = app_data

                with dpg.group(horizontal=True):
                    dpg.add_text("Colormap")
                    dpg.add_combo(self.get_colormap_options(), indent=self.widget_top, 
                                label='', default_value="rainbow", 
                                callback=callback_change_colormap)
                            
            # light editing
            with dpg.collapsing_header(label="Light Editing", default_open=True, leaf=True):
                #* Use Headlight button
                def callback_headlight(sender, app_data):
                    if app_data == False:
                        self.useHeadlight = app_data
                        self.render_kwargs["dict_params"]["light_transform"].set_light_theta_phi(self.light_angle, self.light_elevation)
                    else:
                        self.useHeadlight = app_data
                    self.render_kwargs["dict_params"]["light_transform"].useHeadLight = self.useHeadlight
                with dpg.group(horizontal=True):
                    dpg.add_text("Headlight")
                    dpg.add_checkbox(label="", tag="_checkbox_headlight", callback=callback_headlight, default_value=self.useHeadlight)
                
                #* Light angle and elevation sliders
                def callback_light_angle(sender, app_data):
                    if self.useHeadlight:
                        return
                    if sender == "_slider_light_angle":
                        self.light_angle = app_data
                    else:
                        self.light_elevation = app_data
                    
                    self.render_kwargs["dict_params"]["light_transform"].set_light_theta_phi(self.light_angle, self.light_elevation)
                with dpg.group(horizontal=True):
                    dpg.add_text("Azimuthal")
                    dpg.add_slider_int(label="", tag="_slider_light_angle", indent=self.widget_indent,
                                       default_value=self.light_angle, min_value=-180, max_value=180, callback=callback_light_angle)
                with dpg.group(horizontal=True):
                    dpg.add_text("Polar")
                    dpg.add_slider_int(label="", tag="_slider_light_elevation", indent=self.widget_indent,
                                       default_value=self.light_elevation, min_value=-90, max_value=90, callback=callback_light_angle)
                
                #* ambient sldiers
                def callback_light_multi(sender, app_data):
                    if sender == "_slider_ambient_multi":
                        self.render_kwargs["dict_params"]["light_transform"].ambient_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    elif sender == "_slider_light_intensity_multi":
                        self.render_kwargs["dict_params"]["light_transform"].light_intensity_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    elif sender == "_slider_specular_multi":
                        self.render_kwargs["dict_params"]["light_transform"].specular_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    elif sender == "_slider_shininess_multi":
                        self.render_kwargs["dict_params"]["light_transform"].shininess_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Ambient")
                    dpg.add_slider_float(label="", tag="_slider_ambient_multi", indent=self.widget_indent, default_value=1, min_value=0, max_value=5, callback=callback_light_multi)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Diffuse")
                    dpg.add_slider_float(label="", tag="_slider_light_intensity_multi", indent=self.widget_indent, default_value=1, min_value=0, max_value=5, callback=callback_light_multi)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Specular")
                    dpg.add_slider_float(label="", tag="_slider_specular_multi", indent=self.widget_indent, default_value=1, min_value=0, max_value=5, callback=callback_light_multi)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Shininess")
                    dpg.add_slider_float(label="", tag="_slider_shininess_multi", indent=self.widget_indent, default_value=1, min_value=1, max_value=5, callback=callback_light_multi)
            
            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")

        ### register camera handler
        def callback_camera_start_rotate(sender, app_data):
            self.rotating = True

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return
            MouseX, MouseY = dpg.get_mouse_pos()
            x = -(MouseX/ self.imgW - 0.5) * 2
            y = -(MouseY/ self.imgH - 0.5) * 2

            # self.cam.orbit(dx, dy)
            if(self.prev_mouseX is None or self.prev_mouseY is None):
                self.prev_mouseX = x
                self.prev_mouseY = y
                return
            if (self.rotating):
                self.cam.orbit(self.prev_mouseX, self.prev_mouseY, x, y)
                self.prev_mouseX = x
                self.prev_mouseY = y
            
            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))
                
        
        def callback_camera_end_rotate(sender, app_data):
            self.rotating = False
            self.prev_mouseX = None
            self.prev_mouseY = None
        
        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))
        
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_start_rotate)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)

            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_end_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_drag_pan)

        dpg.create_viewport(title='iVR-GS', width=self.imgW+self.ctrlW, height=self.imgH+160, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()
        dpg.show_viewport()

def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict

if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('-vo', '--view_config', default=None, required=False, help="the config root")
    parser.add_argument('-so', '--source_dir', default=None, required=True, help="the source ckpts dir")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument('-t', '--type', choices=['inverse','phong'], default='phong')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="resume from checkpoint")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--gui_debug", action="store_true", help="show debug info in GUI")

    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    
    
    pbr_kwargs = dict()
    scene_dict = load_ckpts_paths(args.source_dir)
    TFs_names = list(scene_dict.keys())
    TFs_nums = len(TFs_names)
    opacity_transforms = []
    for TFs_name in TFs_names:
        opacity_factor = 1.0
        opacity_transform = LearningOpacityTransform(opacity_factor=opacity_factor)
        opacity_transforms.append(opacity_transform)
        
    light_transform = LearningLightTransform(theta=180, phi=0)
    # load gaussians
    gaussians_composite = scene_composition(scene_dict, dataset)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    render_kwargs = {
        "pc": gaussians_composite,
        "pipe": pipe,
        "bg_color": background,
        "is_training": False,
        "dict_params": {
            "opacity_factors": opacity_transforms,
            "light_transform": light_transform
        }
    }
    
    # ic(scene_dict)
    # ic(checkpoints)
        
    render_fn = render_fn_dict[f"scalar_{args.type}"]
    
    #* remove this if remove --vo argument
    
    
    H, W = 800, 800
    fovx = 30 * np.pi / 180
    fovy = focal2fov(fov2focal(fovx, W), H)
    # fovy = 30.5 * np.pi / 180
    if args.view_config is None:
        c2w = np.array([
            [0.0, 0.0, -1.0, 2.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    else:
        view_config_file = f"{args.view_config}/transforms_test.json"
        view_dict = load_json_config(view_config_file)
        all_views = view_dict["frames"]
        c2w = np.array(all_views[0]["transform_matrix"]).reshape(4, 4) 
        c2w /= 2
        # c2w = np.array([
        #         [
        #             -0.5,
        #             -0.2241438627243042,
        #             0.8365163207054138,
        #             1.67303257#3.3720808029174805
        #         ],
        #         [
        #             1.2145874528357581e-08,
        #             0.9659258723258972,
        #             0.258819043636322,
        #             0.51763808#1.043325424194336
        #         ],
        #         [
        #             -0.866025447845459,
        #             0.1294095367193222,
        #             -0.4829629063606262,
        #             -0.96592575#-1.9468716382980347
        #         ],
        #         [
        #             9.99999993922529e-09,
        #             9.99999993922529e-09,
        #             9.99999993922529e-09,
        #             1.0
        #         ]
        #     ])
        c2w[:3, 1:3] *= -1
    
    windows = GUI(H, W, fovy,
                  c2w=c2w, center=np.array([0.5,0.5,3.5]),
                  render_fn=render_fn, render_kwargs=render_kwargs, TFnums=TFs_nums,
                  mode=args.type, debug=args.gui_debug)
    
    while dpg.is_dearpygui_running():
        windows.render()