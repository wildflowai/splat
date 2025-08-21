"""
wildflow.splat - creating 3D Gaussian Splatting models of coral reefs.

This module provides tools for optimal camera pose partitioning and 
point cloud processing for 3D Gaussian splatting workflows.
"""

# Core partitioning functionality
from .patches import patches, BoundingBox

# PLY processing functionality
from ._core import Config, Patch
from .split import split_point_cloud

# COLMAP camera splitting functionality
from ._core import CameraConfig, CameraPatch
from .split import split_cameras

__version__ = "0.1.4"

# Public API
__all__ = [
    "patches",
    "BoundingBox", 
    "Config",
    "Patch", 
    "split_point_cloud",
    "CameraConfig",
    "CameraPatch",
    "split_cameras",
]
