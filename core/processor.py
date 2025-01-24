from typing import List, Tuple, Dict
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from .ply_reader import PLYStreamer
from .grid import GridAnalyzer
from .visualizer import GridVisualizer
from .types import Point2D
from .cell import Cell


def process_ply_files(input_files: List[str],
                      grid_size: int,
                      cell_px: int,
                      min_points: int,
                      output_image: str = None,
                      output_stats: str = None,
                      show_progress: bool = True) -> Tuple[Image.Image, Dict[Cell, Dict[str, int]], List[Point2D]]:
  """
  Process PLY files and return visualization, statistics and points.

  Args:
      input_files: List of PLY file paths
      grid_size: Size of each grid cell
      cell_px: Pixels per cell in visualization
      min_points: Minimum points per cell
      output_image: Path to save intermediate visualizations
      output_stats: Path to save intermediate statistics
      show_progress: Whether to show progress bars and info

  Returns:
      Tuple of (visualization image, grid statistics, all points)
  """
  grid_analyzer = GridAnalyzer(grid_size)
  visualizer = GridVisualizer(cell_px)
  all_points = []

  # Create progress iterator
  file_iter = tqdm(PLYStreamer.stream_points(input_files),
                   total=len(input_files),
                   desc="Processing PLY files") if show_progress else PLYStreamer.stream_points(input_files)

  for i, (points, filename) in enumerate(file_iter):
    grid_analyzer.add_points(points)
    all_points.extend(points)

    grid_stats, bounds = grid_analyzer.get_stats()

    # Create current visualization
    image = visualizer.create_visualization(
        grid_stats, bounds, grid_size, min_points, all_points)

    # Save intermediate results if paths provided
    if output_image:
      image.save(output_image)
    if output_stats:
      save_grid_stats(grid_stats, output_stats)

    if show_progress:
      total_points = sum(stats['point_count'] for stats in grid_stats.values())
      total_cells = len(grid_stats)
      print(f"\nProcessed {i+1}/{len(input_files)} files: {filename}")
      print(f"Total points: {total_points:,}")
      print(f"Total cells: {total_cells}")

  return image, grid_stats, all_points


def save_grid_stats(grid_stats: Dict[Cell, Dict[str, int]], output_path: str):
  """Save grid statistics to CSV file."""
  sorted_stats = sorted(
      [(cell, stats) for cell, stats in grid_stats.items()],
      key=lambda x: x[1]['point_count'],
      reverse=True
  )

  with open(output_path, 'w') as f:
    f.write("cell_id,point_count\n")
    for cell, stats in sorted_stats:
      f.write(f"{cell},{stats['point_count']}\n")
