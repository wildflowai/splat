"""
Split COLMAP camera poses into patches for GPU training.

When you have a large 3D scene with many camera poses (like from COLMAP), 
you need to split it into smaller patches that can fit on your GPU for 
3D Gaussian Splatting training. This finds the optimal way to partition 
your cameras into rectangular patches.

Basic usage:
    >>> from wildflow.splat import patches
    >>> # Your camera positions as (x, y) tuples
    >>> cameras = [(0, 0), (1, 1), (2, 2), (10, 10), (11, 11)]
    >>> # Split into patches with max 3 cameras each
    >>> result = patches(cameras, max_cameras=3)
    >>> print(f"Split into {len(result)} patches")
    
Advanced usage:
    >>> # More control over the splitting
    >>> result = patches(
    ...     cameras, 
    ...     max_cameras=100,    # Max cameras per patch
    ...     buffer_meters=1.5,  # Safety buffer around patches
    ...     target_bins=50      # Granularity of splitting
    ... )
    
Loading from COLMAP:
    >>> import pycolmap
    >>> model = pycolmap.Reconstruction("path/to/colmap")
    >>> cameras = [(img.projection_center()[0], img.projection_center()[1]) 
    ...            for img in model.images.values()]
    >>> result = patches(cameras)
"""

from typing import List, Tuple, Sequence
from ._core import patches as _patches, BoundingBox

# Simple data structures  
Point = Tuple[float, float]

def patches(camera_positions: Sequence[Point], 
           max_cameras: int = 700,
           buffer_meters: float = 1.5, 
           target_bins: int = 100) -> List[BoundingBox]:
    """
    Split camera positions into optimal rectangular patches for GPU training.
    
    Args:
        camera_positions: List of (x, y) camera positions
        max_cameras: Maximum cameras allowed per patch
        buffer_meters: Safety buffer around each patch in meters
        target_bins: Granularity of the splitting algorithm (higher = more precise)
        
    Returns:
        List of BoundingBox patches that cover all cameras optimally
        
    Example:
        >>> cameras = [(0, 0), (1, 1), (2, 2)]
        >>> result = patches(cameras, max_cameras=10)
        >>> print(f"Created {len(result)} patches")
    """
    # Convert to list of tuples for Rust compatibility
    camera_list = [(float(pos[0]), float(pos[1])) for pos in camera_positions]
    
    # Call the high-performance Rust implementation
    return _patches(camera_list, max_cameras, buffer_meters, target_bins) 