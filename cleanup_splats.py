"""
Usage example:
python cleanup_splats.py \
    --colmap_folder="/Users/nsv/Downloads/corals/m9/m9_bin/sparse/0" \
    --input_ply="/Users/nsv/Downloads/exported_march-19_-2x-1y10s-mcmc.ply" \
    --output_ply="/Users/nsv/Downloads/clean_splats.ply" \
    --deleted_splats_ply="/Users/nsv/Downloads/deleted_splats.ply" \
    --cell="-2x-1y10s" \
    --radius=0.5 \
    --min_points=10
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree
import pycolmap
from tqdm import tqdm
import os
from core.cell import Cell  # For getting cell bounding box


def read_point_cloud(colmap_folder: str) -> np.ndarray:
  """
  Read 3D points from a COLMAP reconstruction folder.

  Args:
      colmap_folder (str): Path to the COLMAP folder containing the reconstruction.

  Returns:
      np.ndarray: Array of shape (N, 3) with x, y, z coordinates of the point cloud.
  """
  model = pycolmap.Reconstruction(colmap_folder)
  points = [point.xyz for point in model.points3D.values()]
  return np.array(points, dtype=np.float32)


def read_ply_file(ply_path: str) -> tuple[np.ndarray, np.ndarray]:
  """
  Read vertex data from a .ply file, extracting positions and full vertex data.

  Args:
      ply_path (str): Path to the input .ply file.

  Returns:
      tuple: (positions, vertex_data)
          - positions: np.ndarray of shape (M, 3) with x, y, z coordinates.
          - vertex_data: np.ndarray with all vertex attributes.
  """
  ply_data = PlyData.read(ply_path)
  vertex_data = ply_data['vertex'].data
  positions = np.vstack(
      (vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
  return positions.astype(np.float32), vertex_data


def filter_splats(splat_positions: np.ndarray, point_cloud_kdtree: KDTree, indices_to_check: np.ndarray, radius: float, min_points: int) -> np.ndarray:
  """
  Filter splats based on proximity to point cloud points.

  Args:
      splat_positions (np.ndarray): Array of shape (M, 3) with splat x, y, z coordinates.
      point_cloud_kdtree (KDTree): KDTree built from the COLMAP point cloud.
      indices_to_check (np.ndarray): Indices of splats to check.
      radius (float): Radius within which to count points.
      min_points (int): Minimum number of points required within the radius.

  Returns:
      np.ndarray: Indices of splats to keep (those with >= min_points within radius).
  """
  keep_indices = []
  for i in tqdm(indices_to_check, desc="Filtering splats"):
    pos = splat_positions[i]
    indices = point_cloud_kdtree.query_ball_point(pos, r=radius)
    count = len(indices)
    if count >= min_points:
      keep_indices.append(i)
  return np.array(keep_indices, dtype=np.int64)


def save_ply_file(output_path: str, vertex_data: np.ndarray, indices: np.ndarray):
  """
  Save filtered vertex data to a new .ply file.

  Args:
      output_path (str): Path to save the output .ply file.
      vertex_data (np.ndarray): Original vertex data from the input .ply file.
      indices (np.ndarray): Indices of vertices to include in the output file.
  """
  filtered_data = vertex_data[indices]
  vertex_element = PlyElement.describe(filtered_data, 'vertex')
  PlyData([vertex_element], text=False).write(output_path)


def main():
  parser = argparse.ArgumentParser(
      description="Filter a 3D Gaussian splatting .ply file based on COLMAP point cloud proximity and optional cell bounding box."
  )
  parser.add_argument("--colmap_folder", required=True,
                      help="Path to the COLMAP reconstruction folder")
  parser.add_argument("--input_ply", required=True,
                      help="Path to the input 3D Gaussian splatting .ply file")
  parser.add_argument("--output_ply", required=True,
                      help="Path to save the filtered .ply file")
  parser.add_argument("--deleted_splats_ply",
                      help="Path to save the deleted splats .ply file (optional)")
  parser.add_argument(
      "--cell", help="Cell ID to filter splats within its bounding box (e.g., '-2x-1y10s')")
  parser.add_argument("--radius", type=float, default=0.5,
                      help="Radius in meters to check for point cloud points (default: 0.5)")
  parser.add_argument("--min_points", type=int, default=10,
                      help="Minimum number of point cloud points within radius (default: 10)")

  args = parser.parse_args()

  # Validate input files exist
  if not os.path.isdir(args.colmap_folder):
    raise FileNotFoundError(f"COLMAP folder not found: {args.colmap_folder}")
  if not os.path.isfile(args.input_ply):
    raise FileNotFoundError(f"Input .ply file not found: {args.input_ply}")

  print("Reading COLMAP point cloud...")
  point_cloud = read_point_cloud(args.colmap_folder)
  print(f"Loaded {len(point_cloud)} points from COLMAP point cloud.")

  print("Building KD-tree for point cloud...")
  kdtree = KDTree(point_cloud)

  print("Reading input .ply file...")
  splat_positions, vertex_data = read_ply_file(args.input_ply)
  print(f"Loaded {len(splat_positions)} splats from input .ply file.")

  # Determine which splats to check based on cell bounding box (if provided)
  if args.cell:
    print(f"Filtering splats within cell {args.cell} bounding box...")
    cell_obj = Cell(args.cell)
    min_x, min_y, max_x, max_y = cell_obj.bbox()
    # Filter splats within the bbox (ignoring z-coordinate for now)
    bbox_mask = (
        (splat_positions[:, 0] >= min_x) &
        (splat_positions[:, 0] <= max_x) &
        (splat_positions[:, 1] >= min_y) &
        (splat_positions[:, 1] <= max_y)
    )
    indices_to_check = np.where(bbox_mask)[0]
  else:
    # If no cell is provided, check all splats
    indices_to_check = np.arange(len(splat_positions))

  print(f"Checking {len(indices_to_check)} splats for proximity condition...")
  keep_indices = filter_splats(
      splat_positions, kdtree, indices_to_check, args.radius, args.min_points)
  print(f"Keeping {len(keep_indices)} splats out of {len(splat_positions)}.")

  print(f"Saving filtered splats to {args.output_ply}...")
  save_ply_file(args.output_ply, vertex_data, keep_indices)

  # Save deleted splats if requested
  if args.deleted_splats_ply:
    delete_indices = np.setdiff1d(np.arange(len(vertex_data)), keep_indices)
    if len(delete_indices) > 0:
      print(
          f"Saving {len(delete_indices)} deleted splats to {args.deleted_splats_ply}...")
      save_ply_file(args.deleted_splats_ply, vertex_data, delete_indices)
    else:
      print("No splats were deleted; skipping deleted splats file.")

  print("Done!")


if __name__ == "__main__":
  main()
