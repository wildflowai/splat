#!/usr/bin/env python3
"""
Splits a COLMAP reconstruction into smaller cells with sufficient points.

Usage:
    python export_cells.py --input_colmap <path> --input_tiles <csv> --output_folder <path> [--min_points <int>] [--margin <float>] [--min_max_z <min_z> <max_z>]

Example:
    python export_cells.py \
        --input_colmap "/Users/nsv/Downloads/corals/hope/sparse/0" \
        --input_tiles "hope_stats.csv" \
        --output_folder "/Users/nsv/Downloads/corals/hope_cells/" \
        --min_points 10000 \
        --margin 3 \
        --min_max_z -1000 1000
"""

import argparse
import csv
import os
from tqdm import tqdm
from core.cell import Cell
from core.crop_colmap import crop_colmap_model


def main():
  parser = argparse.ArgumentParser(
      description="Crop COLMAP models into cells with sufficient points.")
  parser.add_argument("--input_colmap", required=True,
                      help="Input COLMAP model path (e.g., 'sparse/0')")
  parser.add_argument("--input_tiles", required=True,
                      help="CSV file with tile statistics")
  parser.add_argument("--output_folder", required=True,
                      help="Base output folder for cropped models")
  parser.add_argument("--min_points", type=int, default=10000,
                      help="Minimum point count to process a cell")
  parser.add_argument("--margin", type=float, default=3.0,
                      help="Margin to add to the cell bounding box")
  parser.add_argument("--min_max_z", type=float, nargs=2, default=[-100000, 100000],
                      help="Min and max Z values for the bounding box (default: -100000 100000)")
  args = parser.parse_args()

  # Parse CSV file into a list of dictionaries
  with open(args.input_tiles, 'r') as f:
    reader = csv.DictReader(f)
    all_cells = list(reader)

  # Filter cells with sufficient points
  cells_to_process = [cell for cell in all_cells if int(
      cell['point_count']) >= args.min_points]

  # Print summary
  total_cells = len(all_cells)
  to_process = len(cells_to_process)
  print(f"Total cells found: {total_cells}")
  print(f"Cells with sufficient points (>= {args.min_points}): {to_process}")

  # Process filtered cells
  for cell_data in tqdm(cells_to_process, desc="Processing cells", unit="cell"):
    cell_id = cell_data['cell_id']
    cell = Cell(cell_id)
    min_x, min_y, max_x, max_y = cell.bbox(margin=args.margin)
    min_z, max_z = args.min_max_z
    bounding_box = (min_x, min_y, min_z, max_x, max_y, max_z)

    output_path = os.path.join(args.output_folder, cell_id)
    crop_colmap_model(args.input_colmap, output_path, bounding_box)


if __name__ == "__main__":
  main()
