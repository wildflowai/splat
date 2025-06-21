"""Common data types used across modules."""

from typing import NamedTuple


class Point2D(NamedTuple):
  x: float
  y: float
  name: str
