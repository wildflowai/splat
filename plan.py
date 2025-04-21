"""
This module processes COLMAP reconstructions to create a visualization of point clouds and camera positions.
It divides the space into a grid and highlights cells with significant point density.
"""

import pycolmap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from typing import List, Dict, Tuple, NamedTuple
from core.cell import Cell
import yaml
import os
import math
import json
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
                      grid_size: float) -> Tuple[Dict[Cell, Dict[str, any]], Tuple[float, float, float, float], Dict[str, float]]:
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
        cell = Cell.fromPoint(point.x, point.y, grid_size)
        grid_cells[cell]["point_count"] += 1
        grid_cells[cell]["points_z"].append(point.z)
        if grid_cells[cell]["bbox"] is None:
            grid_cells[cell]["bbox"] = cell.bbox()

    # Collect camera statistics
    for camera in cameras:
        cell = Cell.fromPoint(camera.x, camera.y, grid_size)
        grid_cells[cell]["camera_count"] += 1
        grid_cells[cell]["cameras_z"].append(camera.z)
        if grid_cells[cell]["bbox"] is None:
            grid_cells[cell]["bbox"] = cell.bbox()

    # Calculate site-wide z-coordinate statistics
    camera_z = [c.z for c in cameras]
    point_z = [p.z for p in points]
    
    z_stats = {
        "highest_camera_z": max(camera_z),
        "lowest_camera_z": min(camera_z),
        "min_z": math.floor(min(camera_z) - _DEPTH_BELOW_LOWEST),
        "max_z": math.ceil(max(camera_z) + _HEIGHT_ABOVE_HIGHEST),
        "median_point_z": float(np.median(point_z)),
        "min_point_z": min(point_z),
        "max_point_z": max(point_z)
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
                        grid_size: float,
                        cell_px: int = 300,
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

    img_width = (max_cell_x - min_cell_x + 1) * cell_px
    img_height = (max_cell_y - min_cell_y + 1) * cell_px
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

    for cell, stats in significant_cells.items():
        rel_x = (cell.x - min_cell_x) * cell_px
        rel_y = (cell.y - min_cell_y) * cell_px

        draw.rectangle([rel_x, rel_y, rel_x + cell_px,
                       rel_y + cell_px], outline='red')

        cell_text = f"{cell.x} {cell.y}\n{stats['point_count']}\n{stats['camera_count']}"
        text_bbox = draw.textbbox((0, 0), cell_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = rel_x + (cell_px - text_width) // 2
        text_y = rel_y + (cell_px - text_height) // 2

        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((text_x + dx, text_y + dy), cell_text, font=font, fill='black')
        draw.text((text_x, text_y), cell_text, font=font, fill='green')

    for point in points:
        px, py = world_to_image_coords(
            point.x, point.y, grid_size, min_cell_x, min_cell_y, cell_px)
        if 0 <= px < img_width and 0 <= py < img_height:
            image.putpixel((px, py), (0, 0, 255))

    for camera in cameras:
        px, py = world_to_image_coords(
            camera.x, camera.y, grid_size, min_cell_x, min_cell_y, cell_px)
        if 0 <= px < img_width and 0 <= py < img_height:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if 0 <= px + dx < img_width and 0 <= py + dy < img_height:
                        image.putpixel((px + dx, py + dy), (75, 0, 130))

    return image


def save_stats(grid_stats: Dict[Cell, Dict[str, any]], z_stats: Dict[str, float], output_path: str):
    """Save grid statistics to a JSON file, sorted by point count"""
    sorted_stats = sorted(
        [(cell.encode(), stats) for cell, stats in grid_stats.items()],
        key=lambda x: x[1]["point_count"],
        reverse=True
    )

    # Format the data for JSON output
    json_data = {
        "site_stats": {
            "z_coordinates": {
                "camera_range": [round(z_stats['lowest_camera_z'], 2), round(z_stats['highest_camera_z'], 2)],
                "crop_range": [round(z_stats['min_z'], 2), round(z_stats['max_z'], 2)],
                "points_range": [round(z_stats['min_point_z'], 2), round(z_stats['max_point_z'], 2)],
                "median_point_z": round(z_stats['median_point_z'], 2)
            }
        },
        "cells": []
    }

    # Add cells with consistent field ordering
    for cell_id, stats in sorted_stats:
        min_x, min_y, max_x, max_y = stats["bbox"]
        cell_data = {
            "cell_id": cell_id,
            "bbox": [min_x, max_x, min_y, max_y],
            "points": stats["point_count"],
            "cameras": stats["camera_count"]
        }
        json_data["cells"].append(cell_data)

    # Write to JSON file with nice formatting and arrays in single lines
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        # First convert to JSON string with standard indentation
        json_str = json.dumps(json_data, indent=2)
        
        # Function to format arrays in a single line
        def format_array(match):
            content = match.group(1)
            # Remove newlines and extra spaces between array elements
            elements = [e.strip() for e in content.split(',')]
            return '[' + ', '.join(elements) + ']'
        
        # Apply the formatting to arrays
        json_str = re.sub(r'\[\s*\n\s*([^[\]{}\n]+(,\s*\n\s*[^[\]{}\n]+)*)\s*\n\s*\]', format_array, json_str)
        f.write(json_str)
    
    print(f"🟢 Statistics saved to {output_path}")

    # Print a summary to console
    print("\n📊 Site Statistics:")
    print("\nZ-coordinate ranges:")
    z_coords = json_data['site_stats']['z_coordinates']
    print(f"  • Cameras: [{z_coords['camera_range'][0]:.2f}, {z_coords['camera_range'][1]:.2f}]")
    print(f"  • Crop: [{z_coords['crop_range'][0]:.2f}, {z_coords['crop_range'][1]:.2f}]")
    print(f"  • Points: [{z_coords['points_range'][0]:.2f}, {z_coords['points_range'][1]:.2f}]")
    print(f"  • Median point Z: {z_coords['median_point_z']:.2f}")


def plan(config):
    """
    Process COLMAP reconstruction to create visualization and statistics.
    
    Args:
        config (addict.Dict): Configuration object containing:
            - plan.inp_colmap_dir: Input COLMAP reconstruction directory
            - plan.out_stats_file: Output path for statistics JSON file
            - plan.out_plot_file: Output path for visualization image
            - plan.grid_step: Grid size in meters
            - plan.min_points_per_cell: Minimum number of points per cell
            - plan.min_cameras_per_cell: Minimum number of cameras per cell
    """
    # Validate required config parameters
    required_params = [
        'inp_colmap_dir',
        'out_stats_file',
        'out_plot_file',
        'grid_step',
        'min_points_per_cell',
        'min_cameras_per_cell'
    ]
    
    for param in required_params:
        assert hasattr(config.plan, param), f"Missing required parameter in config: plan.{param}"
    
    # Read point cloud and camera positions
    colmap_dir = config.plan.inp_colmap_dir
    if not os.path.isdir(colmap_dir):
        raise FileNotFoundError(f"COLMAP directory not found: {colmap_dir}")

    points = read_point_cloud(colmap_dir)
    cameras = read_camera_positions(colmap_dir)
    
    # Compute grid statistics
    grid_stats, bounds, z_stats = compute_grid_stats(points, cameras, config.plan.grid_step)

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
        config.plan.grid_step,
        min_points_per_cell=config.plan.min_points_per_cell
    )
    os.makedirs(os.path.dirname(config.plan.out_plot_file), exist_ok=True)
    image.save(config.plan.out_plot_file)

    # Save statistics
    save_stats(filtered_stats, z_stats, config.plan.out_stats_file)
    
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
    print(f"🟢 Statistics saved to {config.plan.out_stats_file}")
