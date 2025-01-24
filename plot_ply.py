"""
Process multiple PLY shards and create a single visualization.

Example usage:
python plot_ply.py \
    --input_dir /Users/nsv/corals_m9_grid_5 \
    --output_image plot_ply_viz.png \
    --output_stats plot_ply_stats.csv \
    --grid_size 5 \
    --cell_px 300
"""

import argparse
from pathlib import Path
from tqdm import tqdm
from core.ply_reader import PLYStreamer
from core.grid import GridAnalyzer
from core.visualizer import GridVisualizer


def save_stats(grid_stats, output_path):
  """Save current statistics to CSV file"""
  sorted_stats = sorted(
      [(cell, stats) for cell, stats in grid_stats.items()],
      key=lambda x: x[1]['point_count'],
      reverse=True
  )

  with open(output_path, 'w') as f:
    f.write("cell_id,point_count\n")
    for cell, stats in sorted_stats:
      f.write(f"{cell},{stats['point_count']}\n")


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

  print(f"Found {len(ply_files)} PLY files to process")

  # Initialize analyzers
  grid_analyzer = GridAnalyzer(args.grid_size)
  visualizer = GridVisualizer(args.cell_px)

  # Keep track of all points
  all_points = []

  # Process one file at a time with progress bar
  for i, (points, filename) in enumerate(tqdm(PLYStreamer.stream_points(ply_files),
                                              total=len(ply_files),
                                              desc="Processing PLY files")):
    # Add points to the grid and to our collection
    grid_analyzer.add_points(points)
    all_points.extend(points)

    # Get current statistics and create visualization
    grid_stats, bounds = grid_analyzer.get_stats()

    try:
      # Create and save current visualization
      image = visualizer.create_visualization(
          grid_stats, bounds, args.grid_size, args.min_points, all_points)
      image.save(args.output_image)

      # Save current statistics
      save_stats(grid_stats, args.output_stats)

      # Print progress
      total_points = sum(stats['point_count'] for stats in grid_stats.values())
      total_cells = len(grid_stats)
      print(f"\nProcessed {i+1}/{len(ply_files)} files: {filename}")
      print(f"Total points: {total_points:,}")
      print(f"Total cells: {total_cells}")

    except ValueError as e:
      print(f"\nSkipping visualization for {filename}: {e}")
      continue

  print("\nProcessing complete!")
  print(f"Final visualization saved to: {args.output_image}")
  print(f"Final statistics saved to: {args.output_stats}")


if __name__ == "__main__":
  main()
