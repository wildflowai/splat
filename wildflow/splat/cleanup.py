"""
Gaussian Splatting Cleanup Utility (Python Bindings)

This utility provides Python bindings for a high-performance Rust-based Gaussian splat cleanup tool.
It allows you to filter Gaussian splat point clouds based on various criteria, including bounding box,
surface area, and proximity to a COLMAP point cloud.
"""

from ._core import CleanConfig, cleanup_ply as _cleanup_ply

__all__ = ["cleanup_point_cloud", "CleanConfig"]

def cleanup_point_cloud(config: CleanConfig):
    """
    Cleans a PLY point cloud based on the provided configuration.

    This function reads an input PLY file containing Gaussian splats, applies the specified filters,
    and writes the kept and (optionally) discarded splats to separate output PLY files.

    Args:
        config (CleanConfig): An instance of the CleanConfig class, which encapsulates all the
                              configuration parameters for the cleanup operation.

    Returns:
        None
    """
    return _cleanup_ply(config)
