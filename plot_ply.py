"""
Process multiple PLY shards and create a single visualization.

Example usage:
python plot_ply.py \
    --input_dir /Users/nsv/corals_m9_grid_5 \
    --output_image plot_ply_viz.png \
    --output_stats plot_ply_stats.csv \
    --grid_size 5 \
    --cell_px 300

Alternative usage from Python:

  from pathlib import Path
  from core.processor import process_ply_files, save_grid_stats

  # Get input files
  ply_files = sorted(str(p) for p in Path("input_dir").glob("*.ply"))

  # Process files
  image, stats, points = process_ply_files(
      ply_files,
      grid_size=5,
      cell_px=300,
      min_points=100
  )

  # Use results as needed
  image.save("output.png")
  save_grid_stats(stats, "stats.csv")
"""

import argparse
from pathlib import Path
from core.processor import process_ply_files, save_grid_stats


def main():
  parser = argparse.ArgumentParser(
      description='Process PLY shards and create visualization')
  parser.add_argument('--input_dir', required=True,
                      help='Directory containing PLY shards')
  parser.add_argument('--output_image', required=True,
                      help='Output visualization path')
  parser.add_argument('--output_stats', required=True,
                      help='Output statistics path')
  parser.add_argument('--grid_size', type=int,
                      default=5, help='Grid size in meters')
  parser.add_argument('--cell_px', type=int, default=300,
                      help='Cell size in pixels')
  parser.add_argument('--min_points', type=int, default=100,
                      help='Minimum points per cell')
  args = parser.parse_args()

  # Get all PLY files
  ply_files = sorted(str(p) for p in Path(args.input_dir).glob("*.ply"))
  if not ply_files:
    raise ValueError(f"No PLY files found in {args.input_dir}")

  # Process files with output paths
  image, grid_stats, _ = process_ply_files(
      ply_files,
      args.grid_size,
      args.cell_px,
      args.min_points,
      output_image=args.output_image,
      output_stats=args.output_stats
  )


if __name__ == "__main__":
  main()
