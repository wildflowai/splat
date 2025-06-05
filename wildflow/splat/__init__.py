"""
wildflow.splat - creating 3D Gaussian Splatting models of coral reefs.
"""

from ._core import Config, Patch
from .split import split_point_cloud

__version__ = "0.1.0"
__all__ = ["Config", "Patch", "split_point_cloud"]
