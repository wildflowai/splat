from dataclasses import dataclass
import re

_ENCODING = re.compile(r'(\d+)x(\d+)#(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)#(-?\d+)x(-?\d+)y')


@dataclass(frozen=True)
class Cell:
  x: int  # grid x coordinate
  y: int  # grid y coordinate
  x_step: int = 5  # x grid step size
  y_step: int = 5  # y grid step size
  x_offset: int = 0  # x grid offset
  y_offset: int = 0  # y grid offset

  def __init__(self, encoded: str, x_step: int = 5, y_step: int = 5, x_offset: int = 0, y_offset: int = 0):
    """Initialize a cell from its encoded string representation.
    
    Args:
        encoded: String in format "XSxYS#minX_minY_maxX_maxY#Xx-Yy"
        x_step: X grid step size (default: 5)
        y_step: Y grid step size (default: 5)
        x_offset: X grid offset (default: 0)
        y_offset: Y grid offset (default: 0)
    """
    m = _ENCODING.fullmatch(encoded)
    if not m:
      raise ValueError(f"Invalid format: {encoded}")
    enc_x_step, enc_y_step, min_x, min_y, max_x, max_y, x, y = map(int, m.groups())
    
    # Verify that the step sizes match
    if enc_x_step != x_step or enc_y_step != y_step:
      raise ValueError(f"Step sizes in encoding ({enc_x_step}x{enc_y_step}) don't match provided values ({x_step}x{y_step})")
    
    # Verify that the bounding box matches the grid coordinates
    expected_min_x = x * x_step + x_offset
    expected_min_y = y * y_step + y_offset
    expected_max_x = expected_min_x + x_step
    expected_max_y = expected_min_y + y_step
    
    if (min_x != expected_min_x or min_y != expected_min_y or 
        max_x != expected_max_x or max_y != expected_max_y):
      raise ValueError(f"Bounding box {min_x}_{min_y}_{max_x}_{max_y} doesn't match grid coordinates {x}x{y}y")
    
    object.__setattr__(self, 'x', x)
    object.__setattr__(self, 'y', y)
    object.__setattr__(self, 'x_step', x_step)
    object.__setattr__(self, 'y_step', y_step)
    object.__setattr__(self, 'x_offset', x_offset)
    object.__setattr__(self, 'y_offset', y_offset)

  @classmethod
  def fromPoint(cls, x: float, y: float, x_step: int = 5, y_step: int = 5, 
               x_offset: int = 0, y_offset: int = 0) -> 'Cell':
    """Creates a Cell instance from x and y coordinates.

    Args:
        x: The x-coordinate of the point
        y: The y-coordinate of the point
        x_step: X grid step size (default: 5)
        y_step: Y grid step size (default: 5)
        x_offset: X grid offset (default: 0)
        y_offset: Y grid offset (default: 0)

    Returns:
        Cell: A new Cell instance containing the point
    """
    res = cls.__new__(cls)
    # Calculate grid coordinates
    grid_x = int((x - x_offset) // x_step)
    grid_y = int((y - y_offset) // y_step)
    
    object.__setattr__(res, 'x', grid_x)
    object.__setattr__(res, 'y', grid_y)
    object.__setattr__(res, 'x_step', x_step)
    object.__setattr__(res, 'y_step', y_step)
    object.__setattr__(res, 'x_offset', x_offset)
    object.__setattr__(res, 'y_offset', y_offset)
    return res

  def bbox(self, margin: int = 0) -> tuple:
    """Returns the bounding box of the cell.

    The bounding box is defined as (min_x, min_y, max_x, max_y) and can be expanded
    by an optional margin.

    Args:
        margin: The margin to add to the bounding box (default: 0)

    Returns:
        tuple: A tuple containing (min_x, min_y, max_x, max_y) of the bounding box
    """
    min_x = self.x * self.x_step + self.x_offset - margin
    min_y = self.y * self.y_step + self.y_offset - margin
    max_x = min_x + self.x_step + 2 * margin
    max_y = min_y + self.y_step + 2 * margin
    return (min_x, min_y, max_x, max_y)

  def __str__(self) -> str:
    """Convert cell to string representation."""
    min_x, min_y, max_x, max_y = self.bbox()
    return f"{self.x_step}x{self.y_step}#{min_x}_{min_y}_{max_x}_{max_y}#{self.x}x{self.y}y"

  def encode(self) -> str:
    return str(self)

  def get_neighbors(self) -> list['Cell']:
    """Get the 8 neighboring cells."""
    neighbors = []
    for dx in [-1, 0, 1]:
      for dy in [-1, 0, 1]:
        if dx == 0 and dy == 0:
          continue
        neighbor = Cell.__new__(Cell)
        object.__setattr__(neighbor, 'x', self.x + dx)
        object.__setattr__(neighbor, 'y', self.y + dy)
        object.__setattr__(neighbor, 'x_step', self.x_step)
        object.__setattr__(neighbor, 'y_step', self.y_step)
        object.__setattr__(neighbor, 'x_offset', self.x_offset)
        object.__setattr__(neighbor, 'y_offset', self.y_offset)
        neighbors.append(neighbor)
    return neighbors
