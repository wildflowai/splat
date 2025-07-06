"""
PLY cleanup functionality.
"""

from ._core import CleanConfig, cleanup_ply as _cleanup_ply

__all__ = ["cleanup_point_cloud"]

def cleanup_point_cloud(config: CleanConfig):
    """
    Cleans a PLY point cloud based on the provided configuration.

    Args:
        config: The configuration for the cleanup process.
    """
    return _cleanup_ply(config)
