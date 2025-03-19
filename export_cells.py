"""
Splits massive COLMAP reconstruction into smaller cells with sufficient points.

Example usage:
python export_cells.py \
    --input_colmap /Users/nsv/Downloads/corals/hope/sparse/0 \
    --input_tiles hope_stats.csv \
    --output_folder /Users/nsv/Downloads/corals/hope_cells/ \
    --min_points 10000 \
    --margin 3
"""

import argparse
import csv
import os
import subprocess
from tqdm import tqdm
from core.cell import Cell  # Assumes an existing Cell class is available


def main():
  # Parse command-line arguments
  parser = argparse.ArgumentParser(
      description="Crop COLMAP models for cells with sufficient points")
  parser.add_argument("--input_colmap", required=True,
                      help="Input COLMAP model path")
  parser.add_argument("--input_tiles", required=True,
                      help="Input CSV file with tile statistics")
  parser.add_argument("--output_folder", required=True,
                      help="Base output folder for cropped models")
  parser.add_argument("--min_points", type=int, default=10000,
                      help="Minimum point count to process a cell")
  parser.add_argument("--margin", type=float, default=3,
                      help="Margin to add to the cell bounding box")
  args = parser.parse_args()

  # Step 1: Parse the CSV file into a list of dictionaries
  with open(args.input_tiles, 'r') as f:
    reader = csv.DictReader(f)
    all_cells = list(reader)

  # Step 2: Filter cells with sufficient points
  cells_to_process = [cell for cell in all_cells if int(
      cell['point_count']) >= args.min_points]

  # Step 3: Print summary
  total_cells = len(all_cells)
  to_process = len(cells_to_process)
  print(f"Total cells found: {total_cells}")
  print(f"Cells with sufficient points (>= {args.min_points}): {to_process}")

  # Step 4: Process filtered cells with a progress bar
  for cell_data in tqdm(cells_to_process, desc="Processing cells", unit="cell"):
    cell_id = cell_data['cell_id']
    point_count = cell_data['point_count']

    # Get bounding box using the Cell class
    cell = Cell(cell_id)
    min_x, min_y, max_x, max_y = cell.bbox(margin=args.margin)

    # Define boundary with fixed z-range (-1000 to 1000)
    boundary = f"{min_x},{min_y},-1000,{max_x},{max_y},1000"

    # Set up output path
    output_path = os.path.join(args.output_folder, cell_id, "sparse", "0")
    os.makedirs(output_path, exist_ok=True)

    # Build and execute COLMAP model_cropper command
    cmd = [
        "colmap",
        "model_cropper",
        "--input_path", args.input_colmap,
        "--output_path", output_path,
        "--boundary", boundary
    ]
    subprocess.run(cmd, check=True)  # Raises an error if the command fails


if __name__ == "__main__":
  main()
