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
import random
from plyfile import PlyData, PlyElement
import torch
import math

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


def is_image_blank(image: np.ndarray,
                         alpha_threshold: int = 0,
                         min_fraction: float = 0.005) -> bool:
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


def storePly(path, xyz, values):
    values = values.reshape(-1, 1)
    if xyz.shape[0] != values.shape[0]:
        raise ValueError(
            f"Mismatch in number of points: mesh has {xyz.shape[0]} points, "
            f"but values has {values.shape[0]} entries."
        )
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("value", "f4"),
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, values), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def random_dropout_exact(mesh, num_particles_to_keep, scalars):
    num_points = mesh.n_points

    # if num_particles_to_keep > num_points:
    #     num_particles_to_keep = num_points

    # # Randomly select indices without replacement
    selected_indices = np.random.choice(
        num_points, size=num_particles_to_keep, replace=False
    )

    # Extract selected points and associated values
    new_points = mesh.points[selected_indices]
    new_values = mesh.point_data[scalars][selected_indices]

    # idx = torch.randint(mesh.n_points, num_particles_to_keep)
    # nx, ny, nz = mesh.dimensions
    # ox, oy, oz = mesh.origin
    # sx, sy, sz = mesh.spacing
    # nxny = nx * ny
    # k, r = np.divmod(idx, nxny)
    # j, i = np.divmod(r, nx)
    # x = ox + i * sx
    # y = oy + j * sy
    # z = oz + k * sz
    # new_points = np.stack((x, y, z), axis=-1)
    # new_values = mesh.point_data[scalars][idx]

    return new_points, new_values


def buildRawDataset(
    out_path,
    raw_file,
    num_maps,
    triangular,
    shade,
    random_colormaps,
    dropout,
    narrow,
    broad,
    white,
    zoom
):
    if narrow:
        num_maps = num_maps * 2
    if broad:
        num_maps = num_maps // 2
    start_time = time.time()
    num_points = 100
    slope = 1
    opacs = []
    if not triangular:
        indices = np.arange(num_points)
        bins = np.linspace(0, num_points, num_maps+1).astype(int)
        for arr in [((indices >= start) & (indices < end)).astype(np.float32) for start, end in zip(bins[:-1], bins[1:])]:
            arr = arr * 0.1
            opacs.append(arr)
        # opacs = [opacs[9] + opacs[7] + opacs[5], opacs[8] + opacs[6] + opacs[4], opacs[9] + opacs[6], opacs[8] + opacs[5], opacs[4] + opacs[7], opacs[9] + opacs[4]]
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
        opac = 0.05 * (opacs[1] + opacs[2] + opacs[3] + opacs[4])
        # opac[(opac > 0) & (opac < 0.03)] = 0.03
        opacs = [opac]
        # control_x = np.array([0.0, 0.1, 1.0])
        # control_y = np.array([0.0, 0.0, 1.0])
        # control_x = np.array([0.0, 0.2, 0.2, 0.3, 0.3, 1.0])
        # control_y = np.array([0.025, 0.025, 0.5, 0.5, 0.0, 0.0])

        # indices = np.linspace(0, 1, num_points)
        # opacs.append(0.1 * np.interp(indices, control_x, control_y).astype(np.float32))
    if random_colormaps:
        cmaps = ["rainbow_r", "rainbow",
                "coolwarm", "coolwarm_r", "RdYlBu"]
    else:
        cmaps = ["viridis"]
        
    # Window setup
    width = 1200
    height = 1200
    pl = pv.Plotter(off_screen=True, lighting="none")
    headlight = pv.Light(light_type='headlight')
    headlight.intensity = 1.0
    pl.add_light(headlight)
    pl.window_size = [width, height]
    if white:
       pl.background_color = 'white'

    if raw_file.endswith(".vtu"):
        dir = os.path.join(out_path, os.path.basename(raw_file).rsplit(".", 1)[0])
        os.makedirs(dir, exist_ok=True)
        unstructured_mesh = pv.read(raw_file)
        
        print("Converting unstructured mesh to structured grid...")
        conversion_start = time.time()
        
        bounds = unstructured_mesh.bounds
        # xmin, xmax, ymin, ymax, zmin, zmax = bounds
        # global_min = min(xmin, ymin, zmin)
        # global_max = max(xmax, ymax, zmax)
        # unstructured_mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
        # unstructured_mesh.scale(1.0/(global_max - global_min), inplace=True)
        resolution = 512
        
        structured_grid = pv.ImageData(
            dimensions=[resolution, resolution, resolution],
            spacing=[
                (bounds[1] - bounds[0]) / (resolution - 1),
                (bounds[3] - bounds[2]) / (resolution - 1), 
                (bounds[5] - bounds[4]) / (resolution - 1)
            ],
            origin=[bounds[0], bounds[2], bounds[4]]
        )
        
        mesh = structured_grid.sample(unstructured_mesh)
        if "value" not in mesh.point_data:
            array_names = list(mesh.point_data.keys())
            if array_names:
                first_array = array_names[0]
                mesh.point_data["value"] = mesh.point_data[first_array]
                print(f"Using array '{first_array}' as 'value'")
            else:
                raise ValueError("No scalar data found in the unstructured mesh")
        
        print(f"Conversion completed in {time.time() - conversion_start:.2f} seconds")
        print(f"Structured grid dimensions: {mesh.dimensions}")
        
        values = mesh.get_array("value").reshape(-1, 1)
        
        # Only normalize values that are not NaN (i.e., inside the original mesh)
        valid_mask = ~np.isnan(values.ravel())
        if np.sum(valid_mask) == 0:
            raise ValueError("No valid data points found in the sampled mesh")
            
        values_min = values[valid_mask].min()
        values_max = values[valid_mask].max()
        
        # Normalize only the valid values, keep NaN values as NaN
        normalized_values = values.copy()
        normalized_values[valid_mask] = (values[valid_mask] - values_min) / (values_max - values_min)
        
        # Set NaN values to a value that will result in zero opacity
        normalized_values[~valid_mask] = 0.0
        
        mesh.get_array("value")[:] = normalized_values.ravel()

        # Scale mesh to the unit cube
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        global_min = min(xmin, ymin, zmin)
        global_max = max(xmax, ymax, zmax)
        mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
        mesh.scale(1.0/(global_max - global_min), inplace=True)
        scalars = "value"
        # mesh.translate(np.array([0.01,0.01,0.01]), inplace=True)
    elif raw_file.endswith(".vtk") or raw_file.endswith(".vti"):
        dir = os.path.join(out_path, os.path.basename(raw_file).rsplit(".", 1)[0])
        os.makedirs(dir, exist_ok=True)

        mesh = pv.read(raw_file)
        if mesh.point_data.keys():
            scalars = list(mesh.point_data.keys())[0]
        elif mesh.cell_data.keys():
            scalars = list(mesh.cell_data.keys())[0]
            # Convert cell data to point data for volume rendering
            mesh = mesh.cell_data_to_point_data()
        else:
            raise ValueError("No scalar data found in the mesh")
        
        mesh.point_data[scalars] = mesh.point_data[scalars].astype(np.float64)
        values = mesh.get_array(scalars).reshape(-1, 1)
        values_min = values.min()
        values_max = values.max()
        values = (values - values_min) / (values_max - values_min)
        mesh.get_array(scalars)[:] = values.ravel()

        # Scale mesh to the unit cube
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        global_min = min(xmin, ymin, zmin)
        global_max = max(xmax, ymax, zmax)
        mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
        mesh.scale(1.0/(global_max - global_min), inplace=True)
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
        mesh.point_data["value"] = (values - values.min()) / (values.max() - values.min())

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

    if dropout:
        start_time = time.time()
        points_dropout, values_dropout = random_dropout_exact(
            mesh,
            500000,
            scalars
        )
        print(f"Time taken to perform dropout: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        storePly(
            os.path.join(dir, "points3d.ply"),
            points_dropout,
            values_dropout,
        )
        print(f"Time taken to store points3d.ply: {time.time() - start_time:.2f} seconds")
        return

    # Controls the camera orbit and capture frequency
    zooms = [1] if not zoom else [2, 2, 3, 3]
    azimuth_steps = 20 if not random_colormaps and not narrow and not zoom else 20
    elevation_steps = 8 if not random_colormaps and not narrow and not zoom else 8
    azimuth_range = np.linspace(0, 360, azimuth_steps, endpoint=False)
    # elevation is intentionally limited to avoid a render bug(s) that occurs when elevation is outside of [-35, 35]
    elevation_range = np.linspace(-60, 60, elevation_steps, endpoint=True)
    for i, opac in enumerate(opacs):
        train_cams = []
        test_cams = []
        opac_dir = os.path.join(dir, 
            f"{'Z' if zoom else ''}{'WT' if white else ''}{'R' if not triangular else ''}{'NS' if not shade else ''}{'CC' if random_colormaps else ''}{'NA' if narrow else ''}{'B' if broad else ''}TF{(i+1):02d}")
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
                opacity=opac,
                shade=shade,
                render=False,
                # ambient=0.1,
                # opacity_unit_distance=(1.0 / 512.0)
            )
            pl.view_xy(render=False)
            initial_view_angle = camera.view_angle

            print(
                f"Time taken to update the volume: {time.time() - start_time:.2f} seconds"
            )
            skip_count = 0
            start_time = time.time()
            for z in zooms:
                camera.view_angle = initial_view_angle
                pl.camera.zoom(z)
                for elevation in elevation_range:
                    for azimuth in azimuth_range:
                        # Set new azimuth and elevation
                        camera.elevation = elevation
                        camera.azimuth = azimuth

                        # Produce a new render at the new camera position
                        pl.render()

                        img = pl.screenshot(None, transparent_background=(not white), return_img=True)

                        if is_image_blank(img):
                            skip_count += 1
                            im_count += 1
                            continue
                        
                        # Save the render as a new image
                        image_name = (
                            f"r_{(im_count // 2):04d}.png"
                        )
                        if im_count % 2 == 0:
                            if random_colormaps or narrow or broad or zoom:
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
            print(f"Number of skips: {skip_count}")

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
    parser.add_argument(
        "--dropout",
        action="store_true"
    )
    parser.add_argument(
        "--narrow",
        action="store_true"
    )
    parser.add_argument(
        "--broad",
        action="store_true"
    )
    parser.add_argument(
        "--white",
        action="store_true"
    )
    parser.add_argument(
        "--zoom",
        action="store_true"
    )
    args = parser.parse_args(sys.argv[1:])
    buildRawDataset(args.path, args.file, args.num_maps, (not args.rectangular), (not args.noshade), args.randomcolormaps, args.dropout, args.narrow, args.broad, args.white, args.zoom)