"""Grid-based point cloud analysis."""

from typing import Dict, Tuple, List
from collections import defaultdict
from .types import Point2D
from .cell import Cell


class GridAnalyzer:
  def __init__(self, grid_size: int):
    self.grid_size = grid_size
    self.grid_cells = defaultdict(lambda: {"point_count": 0})
    self.bounds = None

  def update_bounds(self, points: List[Point2D]):
    """Update grid bounds with new points"""
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]

    if self.bounds is None:
      self.bounds = (min(x_coords), max(x_coords),
                     min(y_coords), max(y_coords))
    else:
      self.bounds = (
          min(self.bounds[0], min(x_coords)),
          max(self.bounds[1], max(x_coords)),
          min(self.bounds[2], min(y_coords)),
          max(self.bounds[3], max(y_coords))
      )

  def add_points(self, points: List[Point2D]):
    """Add points to grid cells"""
    self.update_bounds(points)

    for point in points:
      cell = Cell.fromPoint(
          point.x,
          point.y,
          self.grid_size
      )
      self.grid_cells[cell]["point_count"] += 1

  def get_stats(self) -> Tuple[Dict[Cell, Dict[str, int]], Tuple[int, int, int, int]]:
    """Get current grid statistics and bounds"""
    return self.grid_cells, self.bounds
