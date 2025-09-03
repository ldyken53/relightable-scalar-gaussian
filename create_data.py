#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import argparse
import sys
import os
import shutil
from typing import NamedTuple
import pyvista as pv
from vtk import vtkMatrix3x3, vtkMatrix4x4
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from utils.graphics_utils import focal2fov, fov2focal

from numba import njit
import time


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    colormap: str
    opac_map: int


def caminfo_to_dict(cam: CameraInfo) -> dict:
    return {
        "uid": cam.uid,
        "R": cam.R.tolist(),
        "T": cam.T.tolist(),
        "FovY": float(cam.FovY),
        "FovX": float(cam.FovX),
        "image_path": cam.image_path,
        "image_name": cam.image_name,
        "width": cam.width,
        "height": cam.height,
        "colormap": cam.colormap,
        "opac_map": cam.opac_map
    }


def arrayFromVTKMatrix(vmatrix):
    if isinstance(vmatrix, vtkMatrix4x4):
        matrixSize = 4
    elif isinstance(vmatrix, vtkMatrix3x3):
        matrixSize = 3
    else:
        raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    narray = np.eye(matrixSize, dtype=np.float32)
    vmatrix.DeepCopy(narray.ravel(), vmatrix)
    return narray


@njit(cache=True)
def is_image_blank_alpha(image: np.ndarray,
                         alpha_threshold: int = 1,
                         min_fraction: float = 0.01) -> bool:
    """
    Return True when fewer than `min_fraction` of the pixels have alpha >
    `alpha_threshold`.  Faster and simpler than checking RGB.

    Works only if `image` has 4 channels; if not, falls back to an RGB-based test.
    """
    h, w, c = image.shape
    total = h * w

    if c < 4:                       # RGB screenshot – no alpha to test
        return np.max(image) < alpha_threshold

    lit = 0
    for y in range(h):
        for x in range(w):
            if image[y, x, 3] > alpha_threshold:  # α channel
                lit += 1

    return lit < min_fraction * total


def buildRawDataset(
    out_path,
    raw_file,
    num_maps,
    triangular,
    shade,
    random_colormaps
):
    print(shade)
    start_time = time.time()
    num_points = 100
    slope = 1
    opacs = []
    if not triangular:
        indices = np.arange(num_points)
        bins = np.linspace(0, num_points, num_maps+1).astype(int)
        for arr in [((indices >= start) & (indices < end)).astype(np.float32) for start, end in zip(bins[:-1], bins[1:])]:
            arr = arr * 0.5
            opacs.append(arr)
    else: 
        indices = np.linspace(0, 1, num_points)
        step_size = 1.0 / num_maps
        eps = 1e-4
        for step in range(num_maps):
            center = step * step_size + step_size / 2 + eps
            arr = np.zeros(num_points, dtype=np.float32)
            
            for i, x in enumerate(indices):
                dist = abs(x - center)
                arr[i] = max(0, 1 - (dist * 2 * slope * (num_maps / 2)))
            opacs.append(arr)
    if random_colormaps:
        cmaps = ["rainbow_r", "rainbow",
                "coolwarm", "coolwarm_r", "RdYlBu"]
    else:
        cmaps = ["rainbow"]

    # Window setup
    width = 800
    height = 800
    pl = pv.Plotter(off_screen=True, lighting="none")
    headlight = pv.Light(light_type='headlight')
    headlight.intensity = 1.0
    pl.add_light(headlight)
    pl.window_size = [width, height]

    if raw_file.endswith(".vtk"):
        dir = os.path.join(out_path, os.path.basename(raw_file).rsplit(".", 1)[0])
        os.makedirs(dir, exist_ok=True)
    
        mesh = pv.read(raw_file)
        values = mesh.get_array("volume_scalars").reshape(-1, 1)
        values_min = values.min()
        values_max = values.max()
        values = (values - values_min) / (values_max - values_min)
        mesh.get_array("volume_scalars")[:] = values.ravel()

        # Scale mesh to the unit cube
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        global_min = min(xmin, ymin, zmin)
        global_max = max(xmax, ymax, zmax)
        mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
        mesh.scale(1.0/(global_max - global_min), inplace=True)
        scalars = "volume_scalars"
        # mesh.translate(np.array([0.01,0.01,0.01]), inplace=True)
    else:
        # Parse the filename
        filename = os.path.basename(raw_file).rsplit(".", 1)[0]
        parts = filename.split("_")
        dimensions = tuple(map(int, parts[-2].split("x")))
        base_name = parts[0]

        # Directory setup
        dir = os.path.join(out_path, base_name)
        os.makedirs(dir, exist_ok=True)

        dtype_map = {
            "uint8": np.uint8,
            "int8": np.int8,
            "uint16": np.uint16,
            "int16": np.int16,
            "uint32": np.uint32,
            "int32": np.int32,
            "float32": np.float32,
            "float64": np.float64,
        }
        data_type = dtype_map[parts[-1]]

        # Load the raw data
        values = np.fromfile(raw_file, dtype=data_type)

        # Ensure the size matches the dimensions
        if values.size != dimensions[0] * dimensions[1] * dimensions[2]:
            raise ValueError("Data size does not match the specified dimensions.")

        mesh = pv.ImageData(dimensions=dimensions)
        mesh.point_data["value"] = values

        # Point scaling
        points_min = np.array(mesh.origin)
        points_max = points_min + (np.array(mesh.dimensions) - 1) * np.array(mesh.spacing)
        extents = points_max - points_min
        max_extent = np.max(extents)
        scale_factor = 1.0 / max_extent
        mesh.spacing = tuple(np.array(mesh.spacing) * scale_factor)
        scalars = "value"

    # Get the focal point so that we can translate the mesh to the origin
    offset = list(pl.camera.focal_point)

    # However, the renderer has a bug(s) if the the camera's z-position is too close to 0, this works around it
    offset[2] -= 3
    offset = [-x for x in offset]
    mesh.origin = offset

    print(f"Time taken to load the dataset: {time.time() - start_time:.2f} seconds")

    camera = pl.camera
    camera.clipping_range = (0.001, 1000.0)
    print(mesh.bounds)

    # Controls the camera orbit and capture frequency
    azimuth_steps = 32 if not random_colormaps else 16
    elevation_steps = 10 if not random_colormaps else 5
    azimuth_range = np.linspace(0, 360, azimuth_steps, endpoint=False)
    # elevation is intentionally limited to avoid a render bug(s) that occurs when elevation is outside of [-35, 35]
    elevation_range = np.linspace(-35, 35, elevation_steps, endpoint=True)
    for i, opac in enumerate(opacs):
        train_cams = []
        test_cams = []
        opac_dir = os.path.join(dir, f"{'R' if not triangular else ''}{'NS' if not shade else ''}{'CC' if random_colormaps else ''}TF{(i+1):02d}")
        if os.path.exists(opac_dir):
            shutil.rmtree(opac_dir)
        os.makedirs(opac_dir)
        train_dir = os.path.join(opac_dir, f"train")
        os.makedirs(train_dir)
        test_dir = os.path.join(opac_dir, f"test")
        os.makedirs(test_dir)
        im_count = 0

        for cmap in cmaps:
            start_time = time.time()
            print(f"Updating the volume to colormap {cmap}")

            pl.add_volume(
                mesh,
                name="volume_actor",
                show_scalar_bar=False,
                scalars=scalars,
                cmap=cmap,
                opacity=opac * 255,
                shade=shade,
                render=False,
            )
            pl.view_xy(render=False)
            print(
                f"Time taken to update the volume: {time.time() - start_time:.2f} seconds"
            )
            start_time = time.time()
            for elevation in elevation_range:
                for azimuth in azimuth_range:
                    # Set new azimuth and elevation
                    camera.elevation = elevation
                    camera.azimuth = azimuth

                    # Produce a new render at the new camera position
                    pl.render()

                    img = pl.screenshot(None, transparent_background=True, return_img=True)

                    # if is_image_blank_alpha(img):
                    #     print(f"SKIP HAPPENED, FIX SOMETHING")
                    #     continue
                    
                    # Save the render as a new image
                    image_name = (
                        f"r_{(im_count // 2):04d}.png"
                    )
                    if im_count % 2 == 0:
                        if random_colormaps:
                            im_count += 1
                            continue
                        image_path = os.path.join(train_dir, image_name)
                    else:
                        image_path = os.path.join(test_dir, image_name)
                    plt.imsave(image_path, img)

                    # Convert 4x4 VTK matrix to NumPy and invert
                    mvt_matrix = np.linalg.inv(
                        arrayFromVTKMatrix(camera.GetModelViewTransformMatrix())
                    )

                    # Y/Z flip (likely due to coordinate system handedness)
                    mvt_matrix[:3, 1:3] *= -1

                    # Extract rotation and translation
                    R = mvt_matrix[:3, :3].T  # transpose to match camera convention
                    T = mvt_matrix[:3, 3]

                    # FOV conversions
                    FovY = np.radians(camera.view_angle)
                    FovX = focal2fov(fov2focal(FovY, height), width)

                    cam_info = CameraInfo(
                        uid=im_count // 2,
                        R=R,
                        T=T,
                        FovY=FovY,
                        FovX=FovX,
                        image_path=image_path.replace(opac_dir, "./", 1),
                        image_name=image_name,
                        width=width,
                        height=height,
                        colormap=cmap,
                        opac_map=i
                    )
                    if im_count % 2 == 0:
                        train_cams.append(cam_info)
                    else:
                        test_cams.append(cam_info)
                    im_count += 1
        with open(
            os.path.join(opac_dir, "cameras_train.json"), "w"
        ) as file:
            json.dump([caminfo_to_dict(c) for c in train_cams], file, indent=2)
        with open(
            os.path.join(opac_dir, "cameras_val.json"), "w"
        ) as file:
            json.dump([caminfo_to_dict(c) for c in train_cams], file, indent=2)
        with open(
            os.path.join(opac_dir, "cameras_test.json"), "w"
        ) as file:
            json.dump([caminfo_to_dict(c) for c in test_cams], file, indent=2)
        print(
            f"Time taken to generate images: {time.time() - start_time:.2f} seconds"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="buildRawDataset",
        description="Generate multi-TF training/validation/test "
                    "image sets from a raw volume."
    )
    parser.add_argument(
        "path",
        help="Folder to output to."
    )
    parser.add_argument(
        "file",
        help="Volume filename (including extension)."
    )
    parser.add_argument(
        "num_maps",
        type=int,
        help="Number of opacity maps / transfer-function bins to generate."
    )
    parser.add_argument(
        "--rectangular",
        action="store_true"
    )
    parser.add_argument(
        "--noshade",
        action="store_true"
    )
    parser.add_argument(
        "--randomcolormaps",
        action="store_true"
    )
    args = parser.parse_args(sys.argv[1:])
    buildRawDataset(args.path, args.file, args.num_maps, (not args.rectangular), (not args.noshade), args.randomcolormaps)