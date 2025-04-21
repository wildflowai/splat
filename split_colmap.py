#!/usr/bin/env python3
"""
Split a COLMAP reconstruction into cells based on stats.yaml file.
The script reads cell information and z-range from the stats file and creates
separate COLMAP reconstructions for each significant cell.
"""

import os
from tqdm import tqdm
import yaml
import pycolmap
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.cell import Cell


def load_stats(stats_file):
    """Load statistics from the YAML file."""
    with open(stats_file, 'r') as f:
        return yaml.safe_load(f)


def process_cell(args):
    """Process a single cell with the given model and parameters."""
    cell_data, model, min_z, max_z, config = args
    cell_id = cell_data['cell_id']
    points = cell_data['points']
    cameras = cell_data['cameras']

    # Create Cell object from cell_id with correct grid parameters
    cell = Cell(cell_id,
                x_step=config.plan.grid.x_step,
                y_step=config.plan.grid.y_step,
                x_offset=config.plan.grid.x_offset,
                y_offset=config.plan.grid.y_offset)

    # Get cell bounds with margin
    min_x, min_y, max_x, max_y = cell.bbox(margin=config.split.margin)

    # Create output path for this cell
    cell_output_dir = os.path.join(config.split.out_cells_dir, cell_id, "sparse", "0")
    os.makedirs(cell_output_dir, exist_ok=True)

    # Filter images by camera position
    kept_images = {
        img_id: img for img_id, img in model.images.items()
        if (min_x <= (pos := img.projection_center())[0] <= max_x and
            min_y <= pos[1] <= max_y and
            min_z <= pos[2] <= max_z)
    }

    # Filter 3D points by position
    kept_points = {
        pt_id: pt for pt_id, pt in model.points3D.items()
        if (min_x <= (xyz := pt.xyz)[0] <= max_x and
            min_y <= xyz[1] <= max_y and
            min_z <= xyz[2] <= max_z)
    }

    # Initialize new reconstruction
    new_model = pycolmap.Reconstruction()

    # Add relevant cameras
    for camera_id in {img.camera_id for img in kept_images.values()}:
        new_model.add_camera(model.cameras[camera_id])

    # Process images with filtered 2D-3D correspondences
    idx_map = {}
    invalid_id = 18446744073709551615  # COLMAP's invalid point3D_id
    
    for img_id, img in kept_images.items():
        points2D = img.points2D
        kept_points2D = [
            pycolmap.Point2D(pt.xy.reshape(2, 1), invalid_id)
            for pt in points2D if pt.point3D_id in kept_points
        ]
        original_indices = [i for i, pt in enumerate(
            points2D) if pt.point3D_id in kept_points]
        idx_map[img_id] = {orig_i: new_i for new_i,
                          orig_i in enumerate(original_indices)}

        new_img = pycolmap.Image(
            name=img.name, camera_id=img.camera_id, image_id=img_id)
        new_img.cam_from_world = pycolmap.Rigid3d(
            rotation=pycolmap.Rotation3d(img.cam_from_world.rotation.quat),
            translation=img.cam_from_world.translation
        )
        new_img.points2D = kept_points2D
        new_model.add_image(new_img)

    # Process 3D points with updated tracks
    for pt_id, pt in kept_points.items():
        track = pycolmap.Track()
        for el in pt.track.elements:
            if el.image_id in kept_images and (new_idx := idx_map[el.image_id].get(el.point2D_idx)) is not None:
                track.add_element(el.image_id, new_idx)
        if track.elements:
            new_model.add_point3D(pt.xyz, track, color=pt.color)

    # Write output
    new_model.write(cell_output_dir)
    
    return f"✓ Cell {cell_id}: {len(kept_points)} points, {len(kept_images)} cameras"


def split_colmap(config):
    """
    Split COLMAP reconstruction into cells based on stats.yaml.
    
    Args:
        config (addict.Dict): Configuration object containing:
            - split.inp_stats_file: Input statistics YAML file
            - split.inp_colmap_dir: Input COLMAP reconstruction directory
            - split.out_cells_dir: Output directory for cell reconstructions
            - split.margin: Margin to add around each cell (in meters)
            - plan.grid.x_step: X grid step size
            - plan.grid.y_step: Y grid step size
            - plan.grid.x_offset: X grid offset
            - plan.grid.y_offset: Y grid offset
    """
    # Validate required parameters
    required_params = ['inp_stats_file', 'inp_colmap_dir', 'out_cells_dir', 'margin']
    for param in required_params:
        assert hasattr(config.split, param), f"Missing required parameter in config: split.{param}"

    # Load statistics file
    stats = load_stats(config.split.inp_stats_file)
    
    # Get z-range from stats
    min_z, max_z = stats['z_crop_range']
    
    # Create output directory
    os.makedirs(config.split.out_cells_dir, exist_ok=True)
    
    # Load COLMAP model once
    print("\n📥 Loading COLMAP model...")
    model = pycolmap.Reconstruction(config.split.inp_colmap_dir)
    
    # Process each cell in parallel
    print(f"📊 Processing {len(stats['cells'])} cells")
    print(f"   Z-range: [{min_z}, {max_z}]")
    
    # Prepare arguments for parallel processing
    process_args = [(cell_data, model, min_z, max_z, config) 
                   for cell_data in stats['cells']]
    
    # Process cells in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_cell, args) for args in process_args]
        
        # Show progress bar and collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing cells"):
            print(future.result())
    
    print(f"\n🟢 All cells processed and saved to {config.split.out_cells_dir}") 