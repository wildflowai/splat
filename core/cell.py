from dataclasses import dataclass
import re

_ENCODING = re.compile(r'(-?\d+)x(-?\d+)y(\d+)s')


@dataclass(frozen=True)
class Cell:
  x: int
  y: int
  s: int = 5

  def __init__(self, encoded: str):
    m = _ENCODING.fullmatch(encoded)
    if not m:
      raise ValueError(f"Invalid format: {encoded}")
    x, y, s = map(int, m.groups())
    object.__setattr__(self, 'x', x)
    object.__setattr__(self, 'y', y)
    object.__setattr__(self, 's', s)

  @classmethod
  def fromPoint(cls, x: float, y: float, s: int = 5) -> 'Cell':
    """Creates a Cell instance from x and y coordinates.

    Args:
        cls: The class to instantiate.
        x (int): The x-coordinate of the cell.
        y (int): The y-coordinate of the cell.
        s (int, optional): The size of the cell (default is 5).

    Returns:
        Cell: A new Cell instance.
    """
    res = cls.__new__(cls)
    object.__setattr__(res, 'x', int(x // s))
    object.__setattr__(res, 'y', int(y // s))
    object.__setattr__(res, 's', int(s))
    return res

  def bbox(self, margin: int = 0) -> tuple:
    """Returns the bounding box of the cell.

    The bounding box is defined as (min_x, min_y, max_x, max_y) and can be expanded
    by an optional margin.

    Args:
        margin (int, optional): The margin to add to the bounding box (default is 0).

    Returns:
        tuple: A tuple containing (min_x, min_y, max_x, max_y) of the bounding box.
    """
    min_x = self.x * self.s - margin
    min_y = self.y * self.s - margin
    max_x = min_x + self.s + 2 * margin
    max_y = min_y + self.s + 2 * margin
    return (min_x, min_y, max_x, max_y)

  def __str__(self) -> str:
    """Convert cell to string representation."""
    return f"{self.x}x{self.y}y{self.s}s"

  def encode(self) -> str:
    return str(self)
