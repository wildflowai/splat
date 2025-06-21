"""Visualization functions for point cloud grid."""

from typing import Dict, Tuple, List
from PIL import Image, ImageDraw, ImageFont
from .types import Point2D
from .cell import Cell


class GridVisualizer:
  def __init__(self, cell_px: int = 300):
    self.cell_px = cell_px
    self.font = self._init_font()

  def _init_font(self) -> ImageFont.FreeTypeFont:
    """Initialize font for text rendering"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "arial.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]

    for font_path in font_paths:
      try:
        return ImageFont.truetype(font_path, 40)
      except:
        continue

    print("Warning: Could not load TrueType font, using default font")
    return ImageFont.load_default()

  def create_visualization(self,
                           grid_stats: Dict[Cell, Dict[str, int]],
                           bounds: Tuple[int, int, int, int],
                           grid_size: int,
                           min_points_per_cell: int,
                           points: List[Point2D]) -> Image.Image:
    """Create visualization of the point cloud grid"""
    x_min, x_max, y_min, y_max = bounds

    # Get grid dimensions for ALL cells
    cell_coords = [(cell.x, cell.y) for cell in grid_stats.keys()]
    min_cell_x = min(c[0] for c in cell_coords)
    max_cell_x = max(c[0] for c in cell_coords)
    min_cell_y = min(c[1] for c in cell_coords)
    max_cell_y = max(c[1] for c in cell_coords)

    # Create image
    img_width = (max_cell_x - min_cell_x + 1) * self.cell_px
    img_height = (max_cell_y - min_cell_y + 1) * self.cell_px
    image = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(image)

    # Draw points first (as a background)
    for point in points:
        # Convert point coordinates to image coordinates
      # Get the cell coordinate for this point
      cell_x = int(point.x // grid_size)
      cell_y = int(point.y // grid_size)

      # Calculate relative position within the cell (0 to 1)
      rel_x = (point.x % grid_size) / grid_size
      rel_y = (point.y % grid_size) / grid_size

      # Convert to pixel coordinates
      px = ((cell_x - min_cell_x) + rel_x) * self.cell_px
      py = ((cell_y - min_cell_y) + rel_y) * self.cell_px

      # Only draw if point is within image bounds
      if 0 <= px < img_width and 0 <= py < img_height:
        # Draw point as a small blue dot
        image.putpixel((int(px), int(py)), (0, 0, 255))  # Blue

    # Draw ALL grid cells and cell information
    for cell, stats in grid_stats.items():
      rel_x = (cell.x - min_cell_x) * self.cell_px
      rel_y = (cell.y - min_cell_y) * self.cell_px

      # Draw cell rectangle
      draw.rectangle([rel_x, rel_y, rel_x + self.cell_px,
                      rel_y + self.cell_px], outline='red')

      # Prepare cell text - display integer coordinates
      cell_text = f"{int(cell.x)} {int(cell.y)}\n{stats['point_count']}"

      # Calculate text position (center of cell)
      text_bbox = draw.textbbox((0, 0), cell_text, font=self.font)
      text_width = text_bbox[2] - text_bbox[0]
      text_height = text_bbox[3] - text_bbox[1]
      text_x = rel_x + (self.cell_px - text_width) // 2
      text_y = rel_y + (self.cell_px - text_height) // 2

      # Draw text stroke (black border)
      for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((text_x + dx, text_y + dy),
                  cell_text, font=self.font, fill='black')

      # Draw text (green)
      draw.text((text_x, text_y), cell_text,
                font=self.font, fill='green')

    return image
