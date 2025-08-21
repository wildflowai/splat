"""
PLY point cloud processing for 3D Gaussian splatting workflows.

Fast spatial partitioning with multi-threaded processing and COLMAP-compatible output.
"""

from ._core import Config, Patch, split_ply as _split_ply
from ._core import CameraConfig, CameraPatch, split_cameras as _split_cameras
from typing import Dict, Any

__all__ = ["split_point_cloud", "split_cameras"]


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


def split_cameras(config: CameraConfig) -> Dict[str, Any]:
    """
    Split COLMAP reconstruction into spatial patches by camera positions.

    Processes COLMAP binary files with configurable spatial bounds.
    Produces cropped COLMAP reconstructions for patch-based processing.

    Args:
        config: Configuration with input COLMAP path and spatial patch definitions

    Returns:
        Dictionary with processing results:
        - 'cameras_loaded': Number of cameras in source reconstruction
        - 'images_loaded': Number of images in source reconstruction
        - 'points3d_loaded': Number of 3D points in source reconstruction
        - 'patches_written': Number of spatial patches created
        - 'total_cameras_written': Cameras written across all patches
        - 'total_images_written': Images written across all patches
        - 'total_points3d_written': 3D points written across all patches

    Example:
        >>> config = CameraConfig("/path/to/colmap/sparse/0")
        >>> patch = CameraPatch("/output/patch1")
        >>> patch.set_bounds(-1.0, -1.0, 1.0, 1.0)  # min_x, min_y, max_x, max_y
        >>> config.add_patch(patch)
        >>> results = split_cameras(config)
        >>> print(f"Created {results['patches_written']} patches")
    """
    return _split_cameras(config)
