"""Functions for streaming PLY files."""

from typing import Iterator, List, Tuple
import numpy as np
from pathlib import Path
from plyfile import PlyData
from .types import Point2D


class PLYStreamer:
  @staticmethod
  def stream_points(ply_paths: List[str]) -> Iterator[Tuple[List[Point2D], str]]:
    """
    Stream points from multiple PLY files one file at a time.
    Returns a tuple of (points, filename) for each file.
    """
    for ply_path in ply_paths:
      try:
        # Read PLY file
        ply_data = PlyData.read(ply_path)
        vertex_data = ply_data['vertex'].data
        filename = Path(ply_path).stem

        # Convert vertex data to points
        # Keep x,y as floats from PLY file
        points = [Point2D(
            x=float(vertex['x']),
            y=float(vertex['y']),
            name=f"{filename}_{i}"
        ) for i, vertex in enumerate(vertex_data)]

        yield points, filename

        # Clear memory
        del ply_data
        del vertex_data
        del points

      except Exception as e:
        print(f"Error reading {ply_path}: {e}")
        continue
