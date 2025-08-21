"""
PLY point cloud processing for 3D Gaussian splatting workflows.

Fast spatial partitioning with multi-threaded processing and COLMAP-compatible output.
"""

from ._core import Config, Patch, split_ply as _split_ply
from ._core import CameraConfig, CameraPatch, split_cameras as _split_cameras, split_cameras_json as _split_cameras_json
from typing import Dict, Any, Union
import json

__all__ = ["split_point_cloud", "split_cameras", "split_cameras_from_patches"]


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


def split_cameras(config: Union[CameraConfig, Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Split COLMAP reconstruction into spatial patches by camera positions.

    Processes COLMAP binary files with configurable spatial bounds.
    Produces cropped COLMAP reconstructions for patch-based processing.

    Args:
        config: Configuration with input COLMAP path and spatial patch definitions.
               Can be:
               - CameraConfig object (legacy)
               - Dictionary with config data (recommended)
               - JSON string with config data

    Returns:
        Dictionary with processing results:
        - 'cameras_loaded': Number of cameras in source reconstruction
        - 'images_loaded': Number of images in source reconstruction
        - 'points3d_loaded': Number of 3D points in source reconstruction
        - 'patches_written': Number of spatial patches created
        - 'total_cameras_written': Cameras written across all patches
        - 'total_images_written': Images written across all patches
        - 'total_points3d_written': 3D points written across all patches

    Examples:
        >>> # Dict interface (recommended)
        >>> result = split_cameras({
        ...     "input_path": "/path/to/colmap/sparse/0", 
        ...     "min_z": -2.0, "max_z": 10.0,
        ...     "patches": [
        ...         {"output_path": "/output/p0", "min_x": -1, "min_y": -1, "max_x": 1, "max_y": 1}
        ...     ]
        ... })
        
        >>> # JSON interface 
        >>> import json
        >>> config_dict = {...}  # same as above
        >>> result = split_cameras(json.dumps(config_dict))
        
        >>> # Legacy object interface
        >>> config = CameraConfig("/path/to/colmap/sparse/0")
        >>> patch = CameraPatch("/output/patch1") 
        >>> patch.set_bounds(-1.0, -1.0, 1.0, 1.0)
        >>> config.add_patch(patch)
        >>> result = split_cameras(config)
    """
    # Handle different input types
    if isinstance(config, str):
        # JSON string
        return _split_cameras_json(config)
    elif isinstance(config, dict):
        # Dict - convert to JSON
        return _split_cameras_json(json.dumps(config))
    else:
        # Legacy CameraConfig object
        return _split_cameras(config)


def split_cameras_from_patches(input_path: str, patches: list, min_z: float = float('-inf'), max_z: float = float('inf')) -> Dict[str, Any]:
    """
    Simplified interface for your specific workflow.
    
    Args:
        input_path: Path to COLMAP sparse reconstruction  
        patches: List of patch dictionaries with output_path and bounds
        min_z, max_z: Z bounds for filtering
    
    Returns:
        Dictionary with processing results
        
    Example:
        >>> patches = [
        ...     {"output_path": "/output/p0", "min_x": -1, "min_y": -1, "max_x": 1, "max_y": 1},
        ...     {"output_path": "/output/p1", "min_x": 0, "min_y": 0, "max_x": 2, "max_y": 2}
        ... ]
        >>> result = split_cameras_from_patches("/path/to/colmap", patches, -2.0, 10.0)
    """
    config = {
        "input_path": input_path,
        "min_z": min_z,
        "max_z": max_z, 
        "patches": patches
    }
    return split_cameras(config)
