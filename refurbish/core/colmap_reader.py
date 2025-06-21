"""Functions for reading COLMAP reconstructions."""

import pycolmap
from typing import List
from .types import Point2D


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
