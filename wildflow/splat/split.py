"""
PLY point cloud processing for 3D Gaussian splatting workflows.

Fast spatial partitioning with multi-threaded processing and COLMAP-compatible output.
"""

from ._core import Config, Patch, split_ply as _split_ply
from typing import Dict, Any

__all__ = ["split_point_cloud"]


def split_point_cloud(config: Config) -> Dict[str, Any]:
    """
    Split PLY point cloud into spatial patches for 3D processing.

    Processes PLY files with configurable spatial bounds and sampling.
    Produces COLMAP-compatible binary outputs for 3D reconstruction workflows.

    Args:
        config: Configuration with input PLY file and spatial patch definitions

    Returns:
        Dictionary with processing results:
        - 'points_loaded': Number of points processed from PLY file
        - 'total_points_written': Points written across all patches  
        - 'patches_written': Number of spatial patches created

    Example:
        >>> config = Config("model.ply")
        >>> patch = Patch("section_1.bin")
        >>> config.add_patch(patch)
        >>> results = split_point_cloud(config)
        >>> print(f"Processed {results['total_points_written']} points")
    """
    return _split_ply(config)
