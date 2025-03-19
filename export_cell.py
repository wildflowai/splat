#!/usr/bin/env python3
"""
Crop a COLMAP sparse reconstruction using a 3D bounding box.

Usage:
    python export_cell.py --input_path <path> --output_path <path> --bounding_box <min_x min_y min_z max_x max_y max_z>

Example:
    python export_cell.py \
        --input_path "sparse/0" \
        --output_path "output/cropped" \
        --bounding_box -10 -10 -10 10 10 10
"""

import argparse
from core.crop_colmap import crop_colmap_model


def main():
  parser = argparse.ArgumentParser(
      description="Crop a COLMAP sparse reconstruction.")
  parser.add_argument("--input_path", required=True,
                      help="Input sparse reconstruction directory (e.g., 'sparse/0')")
  parser.add_argument("--output_path", required=True,
                      help="Output directory for the cropped model")
  parser.add_argument("--bounding_box", type=float, nargs=6, required=True,
                      help="Bounding box as six floats: min_x min_y min_z max_x max_y max_z")
  args = parser.parse_args()

  crop_colmap_model(args.input_path, args.output_path, args.bounding_box)


if __name__ == "__main__":
  main()
