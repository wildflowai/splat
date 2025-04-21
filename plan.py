import pycolmap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from typing import List, Dict, Tuple, NamedTuple
from core.cell import Cell
import yaml
import os
import math
import re

_DEPTH_BELOW_LOWEST = 4.0  # Meters below lowest camera where no coral points are expected
_HEIGHT_ABOVE_HIGHEST = 0.5  # Meters above highest camera where no coral points can exist


class Point3D(NamedTuple):
    x: float
    y: float
    z: float
    name: str  # point_id or image_name


def read_point_cloud(model_path: str) -> List[Point3D]:
    """Extract 3D points from COLMAP reconstruction"""
    model = pycolmap.Reconstruction(model_path)
    return [Point3D(point.xyz[0], point.xyz[1], point.xyz[2], str(point_id))
            for point_id, point in model.points3D.items()]


def read_camera_positions(model_path: str) -> List[Point3D]:
    """Extract camera positions from COLMAP reconstruction"""
    model = pycolmap.Reconstruction(model_path)
    return [Point3D(img.projection_center()[0], img.projection_center()[1], img.projection_center()[2], img_name)
            for img_name, img in model.images.items()]


def compute_grid_stats(points: List[Point3D], cameras: List[Point3D],
                      x_step: int = 5, y_step: int = 5,
                      x_offset: int = 0, y_offset: int = 0) -> Tuple[Dict[Cell, Dict[str, any]], Tuple[float, float, float, float], Dict[str, float]]:
    """Compute statistics for each grid cell and overall z-coordinate stats"""
    all_points = points + cameras
    x_coords = [p.x for p in all_points]
    y_coords = [p.y for p in all_points]
    bounds = (min(x_coords), max(x_coords), min(y_coords), max(y_coords))

    # Initialize grid cells with extended statistics
    grid_cells = defaultdict(lambda: {
        "point_count": 0,
        "camera_count": 0,
        "bbox": None,
        "points_z": [],
        "cameras_z": []
    })

    # Collect points statistics
    for point in points:
        cell = Cell.fromPoint(point.x, point.y, x_step, y_step, x_offset, y_offset)
        grid_cells[cell]["point_count"] += 1
        grid_cells[cell]["points_z"].append(point.z)
        if grid_cells[cell]["bbox"] is None:
            grid_cells[cell]["bbox"] = cell.bbox()

    # Collect camera statistics
    for camera in cameras:
        cell = Cell.fromPoint(camera.x, camera.y, x_step, y_step, x_offset, y_offset)
        grid_cells[cell]["camera_count"] += 1
        grid_cells[cell]["cameras_z"].append(camera.z)
        if grid_cells[cell]["bbox"] is None:
            grid_cells[cell]["bbox"] = cell.bbox()

    # Calculate site-wide z-coordinate statistics
    camera_z = [c.z for c in cameras]
    point_z = [p.z for p in points]
    
    z_stats = {
        "highest_camera_z": float(max(camera_z)),  # Convert to float
        "lowest_camera_z": float(min(camera_z)),   # Convert to float
        "min_z": float(math.floor(min(camera_z) - _DEPTH_BELOW_LOWEST)),  # Convert to float
        "max_z": float(math.ceil(max(camera_z) + _HEIGHT_ABOVE_HIGHEST)), # Convert to float
        "median_point_z": float(np.median(point_z)),  # Convert to float
        "min_point_z": float(min(point_z)),           # Convert to float
        "max_point_z": float(max(point_z))            # Convert to float
    }

    return grid_cells, bounds, z_stats


def world_to_image_coords(x: float, y: float, grid_size: float, min_cell_x: int, min_cell_y: int, cell_px: int) -> Tuple[int, int]:
    """Convert world coordinates to image pixel coordinates."""
    cell_x = int(x // grid_size) - min_cell_x
    cell_y = int(y // grid_size) - min_cell_y
    offset_x = (x % grid_size) / grid_size
    offset_y = (y % grid_size) / grid_size
    px = int(cell_x * cell_px + offset_x * cell_px)
    py = int(cell_y * cell_px + offset_y * cell_px)
    return px, py


def create_visualization(grid_stats: Dict[Cell, Dict[str, any]],
                        points: List[Point3D],
                        cameras: List[Point3D],
                        bounds: Tuple[float, float, float, float],
                        grid_config: any,
                        cell_width: int = 300,
                        min_points_per_cell: int = None) -> Image.Image:
    """Create visualization of the point cloud and grid"""
    x_min, x_max, y_min, y_max = bounds

    if min_points_per_cell is None:
        significant_cells = grid_stats
    else:
        significant_cells = {cell: stats for cell, stats in grid_stats.items()
                           if stats["point_count"] >= min_points_per_cell}

    if not significant_cells:
        raise ValueError("No cells with sufficient points found")

    cell_coords = [(cell.x, cell.y) for cell in significant_cells.keys()]
    min_cell_x = min(c[0] for c in cell_coords)
    max_cell_x = max(c[0] for c in cell_coords)
    min_cell_y = min(c[1] for c in cell_coords)
    max_cell_y = max(c[1] for c in cell_coords)

    # Calculate cell dimensions maintaining aspect ratio
    aspect_ratio = grid_config.y_step / grid_config.x_step
    cell_height = int(cell_width * aspect_ratio)

    img_width = (max_cell_x - min_cell_x + 1) * cell_width
    img_height = (max_cell_y - min_cell_y + 1) * cell_height
    image = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(image)

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "arial.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]

    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 40)
            break
        except:
            continue

    if font is None:
        font = ImageFont.load_default()
        print("Warning: Could not load TrueType font, using default font")

    # Draw cells
    for cell, stats in significant_cells.items():
        # Convert to image coordinates (flip y-axis)
        cell_img_x = (cell.x - min_cell_x) * cell_width
        cell_img_y = img_height - ((cell.y - min_cell_y + 1) * cell_height)

        # Draw cell rectangle
        draw.rectangle([
            cell_img_x, cell_img_y,
            cell_img_x + cell_width, cell_img_y + cell_height
        ], outline='red')

        # Draw cell text
        cell_text = f"{cell.x} {cell.y}\n{stats['point_count']}\n{stats['camera_count']}"
        text_bbox = draw.textbbox((0, 0), cell_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = cell_img_x + (cell_width - text_width) // 2
        text_y = cell_img_y + (cell_height - text_height) // 2

        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((text_x + dx, text_y + dy), cell_text, font=font, fill='black')
        draw.text((text_x, text_y), cell_text, font=font, fill='green')

    # Draw points
    for point in points:
        cell = Cell.fromPoint(point.x, point.y, grid_config.x_step, grid_config.y_step,
                            grid_config.x_offset, grid_config.y_offset)
        if cell in significant_cells:
            # Calculate relative position within the cell
            rel_x = (point.x - (cell.x * grid_config.x_step + grid_config.x_offset)) / grid_config.x_step
            rel_y = (point.y - (cell.y * grid_config.y_step + grid_config.y_offset)) / grid_config.y_step
            
            # Convert to image coordinates (flip y-axis)
            px = int((cell.x - min_cell_x) * cell_width + rel_x * cell_width)
            py = int(img_height - ((cell.y - min_cell_y) * cell_height + rel_y * cell_height))
            
            if 0 <= px < img_width and 0 <= py < img_height:
                image.putpixel((px, py), (0, 0, 255))

    # Draw cameras
    for camera in cameras:
        cell = Cell.fromPoint(camera.x, camera.y, grid_config.x_step, grid_config.y_step,
                            grid_config.x_offset, grid_config.y_offset)
        if cell in significant_cells:
            # Calculate relative position within the cell
            rel_x = (camera.x - (cell.x * grid_config.x_step + grid_config.x_offset)) / grid_config.x_step
            rel_y = (camera.y - (cell.y * grid_config.y_step + grid_config.y_offset)) / grid_config.y_step
            
            # Convert to image coordinates (flip y-axis)
            px = int((cell.x - min_cell_x) * cell_width + rel_x * cell_width)
            py = int(img_height - ((cell.y - min_cell_y) * cell_height + rel_y * cell_height))
            
            if 0 <= px < img_width and 0 <= py < img_height:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if 0 <= px + dx < img_width and 0 <= py + dy < img_height:
                            image.putpixel((px + dx, py + dy), (75, 0, 130))

    return image


def save_stats(grid_stats: Dict[Cell, Dict[str, any]], z_stats: Dict[str, float], output_path: str):
    """Save grid statistics to a YAML file, sorted by point count"""
    sorted_stats = sorted(
        [(cell.encode(), stats) for cell, stats in grid_stats.items()],
        key=lambda x: x[1]["point_count"],
        reverse=True
    )

    # Format the data for YAML output
    EPS = 1e-9
    yaml_data = {
        "z_camera_range": [float(z_stats['lowest_camera_z']), float(z_stats['highest_camera_z'])],
        "z_crop_range": [int(z_stats['min_z'] + EPS), int(z_stats['max_z'] + EPS)],
        "z_points_range": [float(z_stats['min_point_z']), float(z_stats['max_point_z'])],
        "z_median_point": float(z_stats['median_point_z']),
        "cells": []
    }

    # Add cells with cell_id as the first field
    for cell_id, stats in sorted_stats:
        cell_data = {
            "cell_id": cell_id,
            "points": stats["point_count"],
            "cameras": stats["camera_count"]
        }
        yaml_data["cells"].append(cell_data)

    # Write to YAML file with block style
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"🟢 Statistics saved to {output_path}")


def plan(config):
    """
    Process COLMAP reconstruction to create visualization and statistics.
    
    Args:
        config (addict.Dict): Configuration object containing:
            - plan.inp_colmap_dir: Input COLMAP reconstruction directory
            - plan.out_stats_file: Output path for statistics YAML file
            - plan.out_plot_file: Output path for visualization image
            - plan.grid.x_step: X grid step size
            - plan.grid.y_step: Y grid step size
            - plan.grid.x_offset: X grid offset
            - plan.grid.y_offset: Y grid offset
            - plan.min_points_per_cell: Minimum number of points per cell
            - plan.min_cameras_per_cell: Minimum number of cameras per cell
    """
    # Validate required config parameters
    required_params = [
        'inp_colmap_dir',
        'out_stats_file',
        'out_plot_file',
        'grid',
        'min_points_per_cell',
        'min_cameras_per_cell'
    ]
    
    for param in required_params:
        assert hasattr(config.plan, param), f"Missing required parameter in config: plan.{param}"
    
    # Validate grid parameters
    grid_params = ['x_step', 'y_step', 'x_offset', 'y_offset']
    for param in grid_params:
        assert hasattr(config.plan.grid, param), f"Missing grid parameter in config: plan.grid.{param}"
    
    # Read point cloud and camera positions
    colmap_dir = config.plan.inp_colmap_dir
    if not os.path.isdir(colmap_dir):
        raise FileNotFoundError(f"COLMAP directory not found: {colmap_dir}")

    points = read_point_cloud(colmap_dir)
    cameras = read_camera_positions(colmap_dir)
    
    # Get grid parameters
    grid_config = config.plan.grid
    
    # Compute grid statistics
    grid_stats, bounds, z_stats = compute_grid_stats(
        points, cameras, 
        x_step=grid_config.x_step,
        y_step=grid_config.y_step,
        x_offset=grid_config.x_offset,
        y_offset=grid_config.y_offset
    )

    # Filter cells based on minimum requirements
    filtered_stats = {
        cell: stats for cell, stats in grid_stats.items()
        if stats["point_count"] >= config.plan.min_points_per_cell and 
           stats["camera_count"] >= config.plan.min_cameras_per_cell
    }

    if not filtered_stats:
        raise ValueError(
            f"No cells found with at least {config.plan.min_points_per_cell} points "
            f"and {config.plan.min_cameras_per_cell} cameras"
        )

    # Create and save visualization
    image = create_visualization(
        filtered_stats, 
        points, 
        cameras, 
        bounds, 
        grid_config=grid_config,
        min_points_per_cell=config.plan.min_points_per_cell
    )
    os.makedirs(os.path.dirname(config.plan.out_plot_file), exist_ok=True)
    image.save(config.plan.out_plot_file)
    
    # Print summary statistics
    print(f"\n📊 Found {len(grid_stats)} total cells, {len(filtered_stats)} significant")
    print("    (min points: {}, min cameras: {})".format(
        config.plan.min_points_per_cell,
        config.plan.min_cameras_per_cell
    ))
    print("\nZ-coordinate ranges:")
    print(f"  • Cameras: [{z_stats['lowest_camera_z']:.2f}, {z_stats['highest_camera_z']:.2f}]")
    print(f"  • Crop: [{z_stats['min_z']:.2f}, {z_stats['max_z']:.2f}]")
    print(f"  • Points: [{z_stats['min_point_z']:.2f}, {z_stats['max_point_z']:.2f}]")
    print(f"  • Median point Z: {z_stats['median_point_z']:.2f}")
    
    print(f"\n🟢 Visualization saved to {config.plan.out_plot_file}")

    # Save statistics
    save_stats(filtered_stats, z_stats, config.plan.out_stats_file)