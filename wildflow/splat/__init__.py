"""
wildflow.splat - 3D Gaussian Splatting workflows made simple.

Unified dict/JSON API for all spatial processing operations.
"""

from ._core import Config, Patch, split_ply as _split_ply
from ._core import split_cameras_json as _split_cameras_json
from ._core import BoundingBox, patches as _patches
from ._core import cleanup_splats as _cleanup_splats
from ._core import merge_ply_files_py as _merge_ply_files
from typing import Dict, Any, Union, List, Tuple
import json

__version__ = "0.1.4"

__all__ = [
    "split_point_cloud",
    "split_cameras",
    "patches",
    "cleanup_splats",
    "merge_ply_files",
]


def split_point_cloud(config: Union[Config, Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Split PLY point cloud into spatial patches.
    
    Args:
        config: Point cloud configuration. Can be:
                - Config object (legacy)
                - Dictionary with config data (recommended)  
                - JSON string with config data

    Returns:
        Dict with 'points_loaded', 'total_points_written', 'patches_written'
        
    Example:
        >>> result = split_point_cloud({
        ...     "input_file": "model.ply",
        ...     "min_z": -2.0, "max_z": 10.0,
        ...     "sample_percentage": 5.0,
        ...     "patches": [
        ...         {"output_file": "/output/p0.bin", "min_x": -1, "min_y": -1, "max_x": 1, "max_y": 1}
        ...     ]
        ... })
    """
    if isinstance(config, (str, dict)):
        if isinstance(config, dict):
            config = json.dumps(config)
        data = json.loads(config) if isinstance(config, str) else config
        
        cfg = Config(data["input_file"])
        if "min_z" in data: cfg.min_z = data["min_z"]
        if "max_z" in data: cfg.max_z = data["max_z"]  
        if "sample_percentage" in data: cfg.sample_percentage = data["sample_percentage"]
        
        for patch_data in data.get("patches", []):
            patch = Patch(patch_data["output_file"])
            patch.min_x = patch_data["min_x"]
            patch.min_y = patch_data["min_y"] 
            patch.max_x = patch_data["max_x"]
            patch.max_y = patch_data["max_y"]
            cfg.add_patch(patch)
            
        return _split_ply(cfg)
    else:
        return _split_ply(config)


def split_cameras(config: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Split COLMAP reconstruction into spatial patches by camera positions.
    
    Args:
        config: Camera configuration. Can be:
                - Dictionary with config data (recommended)
                - JSON string with config data

    Returns:
        Dict with camera/image/point counts for loaded data and written patches
        
    Configuration format:
        {
            "input_path": str,        # Path to COLMAP sparse reconstruction (e.g., "/path/to/sparse/0")
            "min_z": float,           # Optional: minimum Z coordinate filter
            "max_z": float,           # Optional: maximum Z coordinate filter
            "save_points3d": bool,    # Optional: save points3D.bin for all patches (default: False)
            "patches": [              # List of spatial patches to create
                {
                    "output_path": str,      # Exact output directory path (no /sparse/0 added)
                    "min_x": float,          # Minimum X coordinate
                    "min_y": float,          # Minimum Y coordinate
                    "max_x": float,          # Maximum X coordinate
                    "max_y": float           # Maximum Y coordinate
                }
            ]
        }
        
    Example:
        >>> result = split_cameras({
        ...     "input_path": "/path/to/colmap/sparse/0", 
        ...     "min_z": -2.0, "max_z": 10.0,
        ...     "save_points3d": True,  # Global setting for all patches
        ...     "patches": [
        ...         {
        ...             "output_path": "/output/patch_0/sparse/0",  # Specify exact path
        ...             "min_x": -1, "min_y": -1, "max_x": 1, "max_y": 1
        ...         }
        ...     ]
        ... })
    """
    if isinstance(config, str):
        return _split_cameras_json(config)
    elif isinstance(config, dict):
        return _split_cameras_json(json.dumps(config))
    else:
        raise TypeError("config must be a dictionary or JSON string")


def patches(camera_positions: Union[List[Tuple[float, float]], Dict[str, Any]], 
           max_cameras: int = 700, 
           buffer_meters: float = 1.5,
           target_bins: int = 100) -> List[Dict[str, float]]:
    """
    Generate optimal spatial patches from camera positions.
    
    Args:
        camera_positions: Camera (x,y) positions OR dict with config
        max_cameras: Maximum cameras per patch
        buffer_meters: Buffer around patches  
        target_bins: Optimization granularity
        
    Returns:
        List of patch dictionaries with min_x, max_x, min_y, max_y
        
    Examples:
        >>> camera_positions = [(0, 0), (1, 1), (2, 2)]
        >>> patches_list = patches(camera_positions, max_cameras=500)
        
        >>> result = patches({
        ...     "camera_positions": [(0, 0), (1, 1), (2, 2)],
        ...     "max_cameras": 500,
        ...     "buffer_meters": 2.0
        ... })
    """
    if isinstance(camera_positions, dict):
        data = camera_positions
        positions = data["camera_positions"]
        max_cams = data.get("max_cameras", max_cameras)
        buffer = data.get("buffer_meters", buffer_meters) 
        bins = data.get("target_bins", target_bins)
    else:
        positions = camera_positions
        max_cams = max_cameras
        buffer = buffer_meters
        bins = target_bins
    
    bbox_objects = _patches(positions, max_cams, buffer, bins)
    return [
        {
            "min_x": bbox.min_x,
            "max_x": bbox.max_x, 
            "min_y": bbox.min_y,
            "max_y": bbox.max_y,
            "width": bbox.width,
            "height": bbox.height
        }
        for bbox in bbox_objects
    ]


def cleanup_splats(config: Union[Dict[str, Any], str]) -> None:
    """
    Clean 3D Gaussian splatting model by filtering splats.
    
    Args:
        config: Cleanup configuration. Can be:
                - Dictionary with config data (recommended)
                - JSON string with config data

    Returns:
        None (writes cleaned PLY files)
        
    Example:
        >>> cleanup_splats({
        ...     "input_file": "splats.ply",
        ...     "output_file": "cleaned_splats.ply", 
        ...     "max_area": 10.0,
        ...     "min_x": -5, "max_x": 5,
        ...     "colmap_points_file": "points3D.bin",
        ...     "radius": 0.5,
        ...     "min_neighbors": 5
        ... })
    """
    if isinstance(config, dict):
        config_json = json.dumps(config)
    else:
        config_json = config
        
    return _cleanup_splats(config_json)


def merge_ply_files(config: Union[Dict[str, Any], str]) -> None:
    """
    Merge multiple PLY files into one by concatenating vertex data.
    
    Args:
        config: Merge configuration. Can be:
                - Dictionary with config data (recommended)
                - JSON string with config data

    Returns:
        None (writes merged PLY file)
        
    Configuration format:
        {
            "input_files": [str],     # List of input PLY file paths
            "output_file": str        # Output merged PLY file path
        }
        
    Example:
        >>> merge_ply_files({
        ...     "input_files": ["clean-p0.ply", "clean-p1.ply", "clean-p2.ply"],
        ...     "output_file": "full-model.ply"
        ... })
        
    Note:
        - All input PLY files must have identical structure (same properties)
        - Only vertex data is merged (faces are not supported)
        - Header is copied from first file with updated vertex count
    """
    if isinstance(config, dict):
        config_json = json.dumps(config)
    else:
        config_json = config
        
    return _merge_ply_files(config_json)