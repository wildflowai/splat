"""
wildflow.splat - creating 3D Gaussian Splatting models of coral reefs.

This module provides tools for optimal camera pose partitioning and 
point cloud processing for 3D Gaussian splatting workflows.
"""

# Core partitioning functionality
from .patches import Point2D, BoundingBox, PartitionOptions, patches

# PLY processing functionality
from ._core import Config, Patch
from .split import split_point_cloud

__version__ = "0.1.0"

# Public API
__all__ = [
    "Point2D",
    "BoundingBox", 
    "PartitionOptions",
    "patches",
    "Config",
    "Patch", 
    "split_point_cloud",
]
