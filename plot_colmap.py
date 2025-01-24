"""
This tool processes COLMAP reconstructions to create a visualization of point clouds and camera positions.
It divides the space into a grid and highlights cells with significant point density.

Example usage:
python plot_colmap.py \
    --input_path /Users/nsv/Downloads/m9_bin/sparse/0 \
    --output_image m9_pc.png \
    --output_stats m9_stats.csv \
    --grid_size 5 \
    --cell_px 300 \
    --min_points 100

Arguments:
    input_path: Path to COLMAP sparse reconstruction
    output_image: Path to save visualization PNG
    output_stats: Path to save cell statistics CSV
    grid_size: Size of each grid cell in meters (default: 10)
    cell_px: Size of each cell in output image pixels (default: 300)
    min_points: Minimum points required to display a cell (default: 100)
"""

import pycolmap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, NamedTuple
from core.cell import Cell


class Point2D(NamedTuple):
  x: float
  y: float
  name: str  # point_id or image_name


class GridStats(NamedTuple):
  cell_x: int
  cell_y: int
  point_count: int
  camera_count: int


def read_point_cloud(model_path: str) -> List[Point2D]:
  """Extract 2D points from COLMAP reconstruction"""
  model = pycolmap.Reconstruction(model_path)
  return [Point2D(point.xyz[0], point.xyz[1], str(point_id))
          for point_id, point in model.points3D.items()]


def read_camera_positions(model_path: str) -> List[Point2D]:
  """Extract camera positions from COLMAP reconstruction"""
  model = pycolmap.Reconstruction(model_path)
  return [Point2D(img.projection_center()[0], img.projection_center()[1], img_name)
          for img_name, img in model.images.items()]


def compute_grid_stats(points: List[Point2D], cameras: List[Point2D],
                       grid_size: float) -> Tuple[Dict[Cell, Dict[str, int]], Tuple[float, float, float, float]]:
  """Compute statistics for each grid cell"""
  # Calculate bounds
  all_points = points + cameras
  x_coords = [p.x for p in all_points]
  y_coords = [p.y for p in all_points]
  bounds = (min(x_coords), max(x_coords), min(y_coords), max(y_coords))

  # Count points and cameras per cell
  grid_cells = defaultdict(lambda: {"point_count": 0, "camera_count": 0})

  for point in points:
    cell = Cell.fromPoint(
        int(point.x - bounds[0]), int(point.y - bounds[2]), grid_size)
    grid_cells[cell]["point_count"] += 1

  for camera in cameras:
    cell = Cell.fromPoint(
        int(camera.x - bounds[0]), int(camera.y - bounds[2]), grid_size)
    grid_cells[cell]["camera_count"] += 1

  return grid_cells, bounds


def create_visualization(grid_stats: Dict[Cell, Dict[str, int]],
                         points: List[Point2D],
                         cameras: List[Point2D],
                         bounds: Tuple[float, float, float, float],
                         grid_size: float,
                         cell_px: int,
                         min_points_per_cell: int) -> Image.Image:
  """Create visualization of the point cloud and grid"""
  x_min, x_max, y_min, y_max = bounds

  # Filter significant cells
  significant_cells = {cell: stats for cell, stats in grid_stats.items()
                       if stats["point_count"] > min_points_per_cell}

  if not significant_cells:
    raise ValueError("No cells with sufficient points found")

  # Get grid dimensions
  cell_coords = [(cell.x, cell.y) for cell in significant_cells.keys()]
  min_cell_x = min(c[0] for c in cell_coords)
  max_cell_x = max(c[0] for c in cell_coords)
  min_cell_y = min(c[1] for c in cell_coords)
  max_cell_y = max(c[1] for c in cell_coords)

  # Create image
  img_width = int((max_cell_x - min_cell_x + 1) * cell_px)
  img_height = int((max_cell_y - min_cell_y + 1) * cell_px)
  image = Image.new('RGB', (img_width, img_height), color='white')
  draw = ImageDraw.Draw(image)

  # Try multiple common font paths with a larger size
  font_paths = [
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
      "/System/Library/Fonts/Helvetica.ttc",              # macOS
      "C:\\Windows\\Fonts\\arial.ttf",                    # Windows
      "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Ubuntu
      "arial.ttf",                                        # Generic
      "/usr/share/fonts/TTF/DejaVuSans.ttf",             # Arch Linux
  ]

  font = None
  for font_path in font_paths:
    try:
      font = ImageFont.truetype(font_path, 40)  # Increased size to 120
      break
    except:
      continue

  if font is None:
    font = ImageFont.load_default()
    print("Warning: Could not load TrueType font, using default font")

  # Draw grid cells and cell information
  for cell, stats in significant_cells.items():
    rel_x = (cell.x - min_cell_x) * cell_px
    rel_y = (cell.y - min_cell_y) * cell_px

    # Draw cell rectangle
    draw.rectangle([rel_x, rel_y, rel_x + cell_px,
                   rel_y + cell_px], outline='red')

    # Prepare cell text
    cell_text = f"{cell.x} {cell.y}\n{stats['point_count']}\n{stats['camera_count']}"

    # Calculate text position (center of cell)
    text_bbox = draw.textbbox((0, 0), cell_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = rel_x + (cell_px - text_width) // 2
    text_y = rel_y + (cell_px - text_height) // 2

    # Draw text stroke (black border)
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
      draw.text((text_x + dx, text_y + dy), cell_text, font=font, fill='black')

    # Draw text (green)
    draw.text((text_x, text_y), cell_text, font=font, fill='green')

  # Draw points
  for point in points:
    px = int((point.x - x_min) / grid_size * cell_px) - min_cell_x * cell_px
    py = int((point.y - y_min) / grid_size * cell_px) - min_cell_y * cell_px
    if 0 <= px < img_width and 0 <= py < img_height:
      image.putpixel((int(px), int(py)), (0, 0, 255))  # Blue

  # Draw cameras (now in indigo color: RGB(75, 0, 130))
  for camera in cameras:
    px = int((camera.x - x_min) / grid_size * cell_px) - min_cell_x * cell_px
    py = int((camera.y - y_min) / grid_size * cell_px) - min_cell_y * cell_px
    if 0 <= px < img_width and 0 <= py < img_height:
      for dx in range(-2, 3):
        for dy in range(-2, 3):
          if 0 <= px+dx < img_width and 0 <= py+dy < img_height:
            image.putpixel((int(px+dx), int(py+dy)), (75, 0, 130))  # Indigo

  return image


def save_stats(grid_stats: Dict[Cell, Dict[str, int]], output_path: str):
  """Save grid statistics to a text file, sorted by point count"""
  # Convert to list and sort by point count (descending)
  sorted_stats = sorted(
      [(cell.encode(), stats) for cell, stats in grid_stats.items()],
      key=lambda x: x[1]["point_count"],
      reverse=True
  )

  with open(output_path, 'w') as f:
    f.write("cell_id,point_count,camera_count\n")
    for cell_id, stats in sorted_stats:
      f.write(f"{cell_id},{stats['point_count']},{stats['camera_count']}\n")


def main():
  parser = argparse.ArgumentParser(
      description='Process COLMAP point cloud and create visualization')
  parser.add_argument('--input_path', required=True,
                      help='Path to COLMAP reconstruction')
  parser.add_argument(
      '--output_image', default='point_cloud_grid.png', help='Output image path')
  parser.add_argument('--output_stats', default='grid_stats.csv',
                      help='Output statistics file path')
  parser.add_argument('--grid_size', type=float,
                      default=10.0, help='Grid size in meters')
  parser.add_argument('--cell_px', type=int, default=300,
                      help='Cell size in pixels')
  parser.add_argument('--min_points', type=int, default=100,
                      help='Minimum points per cell')
  args = parser.parse_args()

  # Process data
  points = read_point_cloud(args.input_path)
  cameras = read_camera_positions(args.input_path)
  grid_stats, bounds = compute_grid_stats(points, cameras, args.grid_size)

  # Create and save visualization
  image = create_visualization(grid_stats, points, cameras, bounds,
                               args.grid_size, args.cell_px, args.min_points)
  image.save(args.output_image)

  # Save statistics
  save_stats(grid_stats, args.output_stats)

  print(f"Processing complete. Image saved to {args.output_image}")
  print(f"Statistics saved to {args.output_stats}")


if __name__ == "__main__":
  main()
