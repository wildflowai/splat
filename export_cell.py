"""
Usage:
python export_cell.py \
    --input_path /path/to/sparse/0 \
    --output_path /path/to/output/cropped \
    --bounding_box min_x min_y min_z max_x max_y max_z

Example:
python export_cell.py \
    --input_path "/Users/nsv/Downloads/corals/m9/m9_bin/sparse/0" \
    --output_path "/Users/nsv/Downloads/corals/m9_test/cropped" \
    --bounding_box -20 -10 -1000 0 10 1000
"""

import pycolmap
import numpy as np
import argparse
import os
from tqdm import tqdm


def main():
  # Set up command-line argument parsing
  parser = argparse.ArgumentParser(
      description='Crop a COLMAP sparse reconstruction based on a 3D bounding box.')
  parser.add_argument('--input_path', required=True,
                      help='Path to the COLMAP sparse reconstruction directory (e.g., sparse/0)')
  parser.add_argument('--output_path', required=True,
                      help='Path to the output directory for the cropped COLMAP model')
  parser.add_argument('--bounding_box', type=float, nargs=6, required=True,
                      help='3D bounding box as six values: min_x min_y min_z max_x max_y max_z')
  args = parser.parse_args()

  # Load the COLMAP reconstruction from the input path
  model = pycolmap.Reconstruction(args.input_path)

  # Extract bounding box coordinates
  min_x, min_y, min_z, max_x, max_y, max_z = args.bounding_box

  # Filter images based on camera positions within the bounding box
  kept_images = {}
  for image_id, image in model.images.items():
    pos = image.projection_center()
    if (min_x <= pos[0] <= max_x and
        min_y <= pos[1] <= max_y and
            min_z <= pos[2] <= max_z):
      kept_images[image_id] = image

  # Filter 3D points based on their positions within the bounding box
  kept_points = {}
  for point3D_id, point3D in model.points3D.items():
    xyz = point3D.xyz
    if (min_x <= xyz[0] <= max_x and
        min_y <= xyz[1] <= max_y and
            min_z <= xyz[2] <= max_z):
      kept_points[point3D_id] = point3D

  # Create a new reconstruction object for the cropped model
  new_model = pycolmap.Reconstruction()

  # Add cameras used by the kept images
  kept_camera_ids = set(image.camera_id for image in kept_images.values())
  for camera_id in kept_camera_ids:
    camera = model.cameras[camera_id]
    new_model.add_camera(camera)

  # Add kept images with filtered 2D-3D correspondences
  image_original_idx_to_new_idx = {}
  for image_id, image in tqdm(kept_images.items(), desc="Processing images"):
    original_points2D = image.points2D

    # Use the invalid point3D_id value from COLMAP
    invalid_point3D_id = 18446744073709551615  # Max uint64 value for "invalid"

    # Create new Point2D objects with the correct invalid point3D_id
    kept_points2D = [
        pycolmap.Point2D(pt.xy.reshape(2, 1), invalid_point3D_id)
        for pt in original_points2D if pt.point3D_id in kept_points
    ]

    # Map original 2D point indices to new indices after filtering
    original_indices = [
        i for i, pt in enumerate(original_points2D) if pt.point3D_id in kept_points
    ]
    image_original_idx_to_new_idx[image_id] = {
        original_i: new_i for new_i, original_i in enumerate(original_indices)
    }

    # Extract the quaternion and translation from the camera pose
    quaternion = image.cam_from_world.rotation.quat
    translation = image.cam_from_world.translation

    # Create a new Image object
    new_image = pycolmap.Image(
        name=image.name,
        camera_id=image.camera_id,
        image_id=image_id
    )

    # Set the pose using Rigid3d
    new_image.cam_from_world = pycolmap.Rigid3d(
        rotation=pycolmap.Rotation3d(quaternion),
        translation=translation
    )

    # Set the filtered 2D points
    new_image.points2D = kept_points2D

    # Add the image to the new model
    new_model.add_image(new_image)

  # Add kept 3D points with filtered tracks
  for point3D_id, point3D in tqdm(kept_points.items(), desc="Processing 3D points"):
    adjusted_track = pycolmap.Track()
    for track_el in point3D.track.elements:
      if track_el.image_id in kept_images:
        new_idx = image_original_idx_to_new_idx[track_el.image_id].get(
            track_el.point2D_idx)
        if new_idx is not None:
          adjusted_track.add_element(track_el.image_id, new_idx)

    if len(adjusted_track.elements) == 0:
      continue

    new_model.add_point3D(
        point3D.xyz,
        adjusted_track,
        color=point3D.color
    )

  # Create the output directory structure
  output_dir = os.path.join(args.output_path, 'sparse', '0')
  os.makedirs(output_dir, exist_ok=True)

  # Write the cropped model to the output directory
  new_model.write(output_dir)

  # Create an empty cropped.txt file in the output_path
  with open(os.path.join(args.output_path, 'cropped.txt'), 'w') as f:
    pass  # This creates an empty file

  print(f"Cropped COLMAP model saved to {output_dir}")


if __name__ == "__main__":
  main()
