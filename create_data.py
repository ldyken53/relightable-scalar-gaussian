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


@njit
def is_image_too_dark_numba(image, threshold=3):
    return np.max(image) < threshold


def buildRawDataset(
    out_path,
    raw_file,
    num_maps
):
    start_time = time.time()

    num_steps = 256
    indices = np.arange(num_steps)
    bins = np.linspace(0, num_steps, num_maps+1).astype(int)
    opacs = [((indices >= start - 1) & (indices < end + 1)).astype(np.float32) for start, end in zip(bins[:-1], bins[1:])]
    opacs = [opac * 0.5 for opac in opacs]
    cmap = "rainbow"

    # Window setup
    width = 800
    height = 800
    ratio = width / height
    pl = pv.Plotter(off_screen=True)
    pl.window_size = [width, height]

    # Parse the filename
    filename = os.path.basename(raw_file).rsplit(".", 1)[0]
    parts = filename.split("_")
    dimensions = tuple(map(int, parts[-2].split("x")))
    base_name = parts[0]

    # Directory setup
    dir = os.path.join(out_path, base_name)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

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
    azimuth_steps = 36
    elevation_steps = 10
    azimuth_range = np.linspace(0, 360, azimuth_steps, endpoint=False)
    # elevation is intentionally limited to avoid a render bug(s) that occurs when elevation is outside of [-35, 35]
    elevation_range = np.linspace(-35, 35, elevation_steps, endpoint=True)

    for i, opac in enumerate(opacs):
        train_cams = []
        test_cams = []
        opac_dir = os.path.join(dir, f"TF{(i+1):02d}")
        os.makedirs(opac_dir)
        train_dir = os.path.join(opac_dir, f"train")
        os.makedirs(train_dir)
        test_dir = os.path.join(opac_dir, f"test")
        os.makedirs(test_dir)

        start_time = time.time()
        print("Updating the volume")

        pl.add_volume(
            mesh,
            name="volume_actor",
            show_scalar_bar=False,
            scalars="value",
            cmap=cmap,
            opacity=opac * 255,
            # shade=True,
            render=False,
        )
        headlight = pv.Light(light_type='headlight')
        headlight.intensity = 1.0
        pl.add_light(headlight)
        pl.view_xy(render=False)
        print(
            f"Time taken to update the volume: {time.time() - start_time:.2f} seconds"
        )

        im_count = 0
        start_time = time.time()
        for elevation in elevation_range:
            for azimuth in azimuth_range:
                # Set new azimuth and elevation
                camera.elevation = elevation
                camera.azimuth = azimuth

                # Produce a new render at the new camera position
                pl.render()

                img = pl.screenshot(None, transparent_background=True, return_img=True)

                if is_image_too_dark_numba(img):
                    print(f"SKIP HAPPENED, FIX SOMETHING")
                    skip_counter += 1
                    continue
                
                # Save the render as a new image
                image_name = (
                    f"r_{(im_count // 2):04d}.png"
                )
                if im_count % 2 == 0:
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

                # # Projection matrix conversion and adjustment
                proj_matrix = arrayFromVTKMatrix(
                    camera.GetCompositeProjectionTransformMatrix(
                        ratio, 0.001, 1000.0
                    )
                )
                # Y and Z flip
                proj_matrix[1:3, :] *= -1

                # Fix up modelview matrix if camera is flipped
                if camera.position[1] < 0:
                    mvt_matrix[2, 1] *= -1
                mvt_matrix[2, 3] = abs(mvt_matrix[2, 3])

                # Get camera center in world space
                center = mvt_matrix[:3, 3]

                cam_info = CameraInfo(
                    uid=im_count // 2,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image_path=image_path.replace(opac_dir, "./", 1),
                    image_name=image_name,
                    width=width,
                    height=height
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
        "raw_file",
        help="Raw volume filename (including extension)."
    )
    parser.add_argument(
        "num_maps",
        type=int,
        help="Number of opacity maps / transfer-function bins to generate."
    )
    args = parser.parse_args(sys.argv[1:])
    buildRawDataset(args.path, args.raw_file, args.num_maps)