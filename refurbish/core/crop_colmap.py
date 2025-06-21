import os
import pycolmap
from tqdm import tqdm


def crop_colmap_model(input_path, output_path, bounding_box):
  """
  Crop a COLMAP sparse reconstruction based on a 3D bounding box.

  Args:
      input_path (str): Path to the input sparse reconstruction directory (e.g., 'sparse/0').
      output_path (str): Path to the output directory for the cropped model.
      bounding_box (tuple): Six floats defining the bounding box (min_x, min_y, min_z, max_x, max_y, max_z).

  Raises:
      FileNotFoundError: If the input path does not exist.
      ValueError: If the output model directory exists and is not empty.
  """
  # Validate input path
  if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

  # Prepare output directory
  model_output_dir = os.path.join(output_path, "sparse", "0")
  if os.path.exists(model_output_dir) and os.listdir(model_output_dir):
    raise ValueError(f"Output directory '{model_output_dir}' is not empty.")
  os.makedirs(model_output_dir, exist_ok=True)

  # Load reconstruction
  model = pycolmap.Reconstruction(input_path)

  # Unpack bounding box
  min_x, min_y, min_z, max_x, max_y, max_z = bounding_box

  # Filter images by camera position
  kept_images = {
      img_id: img for img_id, img in model.images.items()
      if (min_x <= (pos := img.projection_center())[0] <= max_x and
          min_y <= pos[1] <= max_y and
          min_z <= pos[2] <= max_z)
  }

  # Filter 3D points by position
  kept_points = {
      pt_id: pt for pt_id, pt in model.points3D.items()
      if (min_x <= (xyz := pt.xyz)[0] <= max_x and
          min_y <= xyz[1] <= max_y and
          min_z <= xyz[2] <= max_z)
  }

  # Initialize new reconstruction
  new_model = pycolmap.Reconstruction()

  # Add relevant cameras
  for camera_id in {img.camera_id for img in kept_images.values()}:
    new_model.add_camera(model.cameras[camera_id])

  # Process images with filtered 2D-3D correspondences
  idx_map = {}
  for img_id, img in tqdm(kept_images.items(), desc="Processing images"):
    points2D = img.points2D
    invalid_id = 18446744073709551615  # COLMAP's invalid point3D_id
    kept_points2D = [
        pycolmap.Point2D(pt.xy.reshape(2, 1), invalid_id)
        for pt in points2D if pt.point3D_id in kept_points
    ]
    original_indices = [i for i, pt in enumerate(
        points2D) if pt.point3D_id in kept_points]
    idx_map[img_id] = {orig_i: new_i for new_i,
                       orig_i in enumerate(original_indices)}

    new_img = pycolmap.Image(
        name=img.name, camera_id=img.camera_id, image_id=img_id)
    new_img.cam_from_world = pycolmap.Rigid3d(
        rotation=pycolmap.Rotation3d(img.cam_from_world.rotation.quat),
        translation=img.cam_from_world.translation
    )
    new_img.points2D = kept_points2D
    new_model.add_image(new_img)

  # Process 3D points with updated tracks
  for pt_id, pt in tqdm(kept_points.items(), desc="Processing 3D points"):
    track = pycolmap.Track()
    for el in pt.track.elements:
      if el.image_id in kept_images and (new_idx := idx_map[el.image_id].get(el.point2D_idx)) is not None:
        track.add_element(el.image_id, new_idx)
    if track.elements:
      new_model.add_point3D(pt.xyz, track, color=pt.color)

  # Write output
  new_model.write(model_output_dir)
