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
    --min_points 100 \
    --highlight_rect -20 -10 -10 0 \
    --highlight_cell "-2x-1y10s" 3
"""

import pycolmap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, NamedTuple, Optional
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
  all_points = points + cameras
  x_coords = [p.x for p in all_points]
  y_coords = [p.y for p in all_points]
  bounds = (min(x_coords), max(x_coords), min(y_coords), max(y_coords))

  grid_cells = defaultdict(lambda: {"point_count": 0, "camera_count": 0})

  for point in points:
    cell = Cell.fromPoint(point.x, point.y, grid_size)
    grid_cells[cell]["point_count"] += 1

  for camera in cameras:
    cell = Cell.fromPoint(camera.x, camera.y, grid_size)
    grid_cells[cell]["camera_count"] += 1

  return grid_cells, bounds


def world_to_image_coords(x: float, y: float, grid_size: float, min_cell_x: int, min_cell_y: int, cell_px: int) -> Tuple[int, int]:
  """Convert world coordinates to image pixel coordinates."""
  cell_x = int(x // grid_size) - min_cell_x
  cell_y = int(y // grid_size) - min_cell_y
  offset_x = (x % grid_size) / grid_size
  offset_y = (y % grid_size) / grid_size
  px = int(cell_x * cell_px + offset_x * cell_px)
  py = int(cell_y * cell_px + offset_y * cell_px)
  return px, py


def create_visualization(grid_stats: Dict[Cell, Dict[str, int]],
                         points: List[Point2D],
                         cameras: List[Point2D],
                         bounds: Tuple[float, float, float, float],
                         grid_size: float,
                         cell_px: int,
                         min_points_per_cell: int,
                         highlight_rect: Tuple[float,
                                               float, float, float] = None,
                         highlight_cell: Optional[str] = None,
                         highlight_cell_margin: float = 0) -> Image.Image:
  """Create visualization of the point cloud and grid"""
  x_min, x_max, y_min, y_max = bounds

  significant_cells = {cell: stats for cell, stats in grid_stats.items()
                       if stats["point_count"] > min_points_per_cell}

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

  # Draw highlight rectangle if provided
  if highlight_rect:
    rect_min_x, rect_min_y, rect_max_x, rect_max_y = highlight_rect
    px_min_x, px_min_y = world_to_image_coords(
        rect_min_x, rect_min_y, grid_size, min_cell_x, min_cell_y, cell_px)
    px_max_x, px_max_y = world_to_image_coords(
        rect_max_x, rect_max_y, grid_size, min_cell_x, min_cell_y, cell_px)
    draw.rectangle([px_min_x, px_min_y, px_max_x, px_max_y],
                   outline='red', width=2)

  # Draw highlight cell if provided
  if highlight_cell:
    cell = Cell(highlight_cell)
    min_x, min_y, max_x, max_y = cell.bbox(margin=highlight_cell_margin)
    px_min_x, px_min_y = world_to_image_coords(
        min_x, min_y, grid_size, min_cell_x, min_cell_y, cell_px)
    px_max_x, px_max_y = world_to_image_coords(
        max_x, max_y, grid_size, min_cell_x, min_cell_y, cell_px)
    draw.rectangle([px_min_x, px_min_y, px_max_x, px_max_y],
                   outline='blue', width=2)

  return image


def save_stats(grid_stats: Dict[Cell, Dict[str, int]], output_path: str):
  """Save grid statistics to a text file, sorted by point count"""
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
  parser.add_argument('--highlight_rect', type=float, nargs=4,
                      help='Rectangle coordinates to highlight (min_x min_y max_x max_y)')
  parser.add_argument('--highlight_cell', type=str, nargs='+',
                      help='Cell ID to highlight (e.g., "-2x-1y10s") followed by optional margin (default 0)')
  args = parser.parse_args()

  # Process highlight_cell arguments
  highlight_cell = None
  highlight_cell_margin = 0
  if args.highlight_cell:
    highlight_cell = args.highlight_cell[0]
    if len(args.highlight_cell) > 1:
      highlight_cell_margin = float(args.highlight_cell[1])

  points = read_point_cloud(args.input_path)
  cameras = read_camera_positions(args.input_path)
  grid_stats, bounds = compute_grid_stats(points, cameras, args.grid_size)

  image = create_visualization(grid_stats, points, cameras, bounds,
                               args.grid_size, args.cell_px, args.min_points,
                               args.highlight_rect, highlight_cell, highlight_cell_margin)
  image.save(args.output_image)

  save_stats(grid_stats, args.output_stats)

  print(f"Processing complete. Image saved to {args.output_image}")
  print(f"Statistics saved to {args.output_stats}")


if __name__ == "__main__":
  main()
