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

from typing import List, Tuple, Union, NamedTuple
from dataclasses import dataclass
import math

# Simple data structures
Point = Union[Tuple[float, float], NamedTuple]

@dataclass(frozen=True)
class BoundingBox:
    """A rectangular patch covering some cameras."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    
    @property
    def width(self) -> float:
        return self.max_x - self.min_x
    
    @property  
    def height(self) -> float:
        return self.max_y - self.min_y


def patches(camera_positions: List[Point], 
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
    if not camera_positions:
        raise ValueError("Camera positions cannot be empty")
    
    if len(camera_positions) <= max_cameras:
        # All cameras fit in one patch
        xs = [p[0] for p in camera_positions]
        ys = [p[1] for p in camera_positions]
        return [BoundingBox(
            min_x=min(xs) - buffer_meters,
            max_x=max(xs) + buffer_meters,
            min_y=min(ys) - buffer_meters,
            max_y=max(ys) + buffer_meters
        )]
    
    # Run the complex partitioning algorithm
    return _partition_with_dp(camera_positions, max_cameras, buffer_meters, target_bins)


# =============================================================================
# Implementation details below - you probably don't need to read this
# =============================================================================

class _Point2D(NamedTuple):
    x: float
    y: float

class _DpState(NamedTuple):
    patch_count: int
    aspect_cost: float
    patches: List[BoundingBox]
    
    @classmethod
    def infinity(cls):
        return cls(999999, float('inf'), [])
    
    @classmethod
    def empty(cls):
        return cls(0, 1.0, [])
    
    def __add__(self, other):
        return _DpState(
            self.patch_count + other.patch_count,
            self.aspect_cost * other.aspect_cost,
            self.patches + other.patches
        )
    
    def __lt__(self, other):
        if self.patch_count != other.patch_count:
            return self.patch_count < other.patch_count
        return self.aspect_cost < other.aspect_cost

class _Binner:
    def __init__(self, min_val: float, max_val: float, target_bins: int):
        if max_val <= min_val:
            raise ValueError(f"Invalid range: [{min_val}, {max_val}]")
        
        # Create nice bin boundaries
        rough = (max_val - min_val) / target_bins
        power = 10 ** math.floor(math.log10(rough))
        nice_numbers = [1, 2, 5, 10]
        nice = next(x for x in nice_numbers if x * power >= rough) * power
        
        self.width = nice
        self.start = math.floor(min_val / nice) * nice
        self.count = math.ceil((max_val - self.start) / nice)
    
    def bin_id(self, x: float) -> int:
        return max(0, min(self.count - 1, int((x - self.start) / self.width)))
    
    def bounds(self, i: int) -> Tuple[float, float]:
        return (self.start + i * self.width, self.start + (i + 1) * self.width)

class _FastCounter:
    """Fast range counting using two-pointers technique."""
    def __init__(self, cameras_sorted_by_y: List[_Point2D]):
        self.cameras = cameras_sorted_by_y
        self.left = 0
        self.right = 0
        self.last_y_start = float('-inf')
    
    def count_in_y_range(self, y_start: float, y_end: float) -> int:
        """Count cameras in Y range [y_start, y_end] - amortized O(1)"""
        # Reset if going backwards (shouldn't happen in our algorithm)
        if y_start < self.last_y_start:
            self.left = 0
            self.right = 0
        
        # Move left pointer to first camera >= y_start
        while self.left < len(self.cameras) and self.cameras[self.left].y < y_start:
            self.left += 1
            
        # Move right pointer to first camera > y_end  
        while self.right < len(self.cameras) and self.cameras[self.right].y <= y_end:
            self.right += 1
            
        self.last_y_start = y_start
        return self.right - self.left

def _pack_y_direction(cameras_in_x_range: List[_Point2D], 
                     all_cameras: List[_Point2D],
                     x_left: float, x_right: float,
                     y_binner: _Binner,
                     buffer_m: float, 
                     max_cameras: int) -> _DpState:
    """Pack cameras in Y direction for a given X range."""
    if not cameras_in_x_range:
        return _DpState.empty()
    
    # Find Y range needed
    cameras_by_y = sorted(cameras_in_x_range, key=lambda p: p.y)
    min_y = cameras_by_y[0].y
    max_y = cameras_by_y[-1].y
    
    min_bin = y_binner.bin_id(min_y)
    max_bin = y_binner.bin_id(max_y)
    
    # Pre-filter cameras in buffered X range and sort by Y for fast counting
    buffered_cameras = [cam for cam in all_cameras 
                       if x_left - buffer_m <= cam.x <= x_right + buffer_m]
    buffered_cameras_by_y = sorted(buffered_cameras, key=lambda p: p.y)
    
    # Initialize fast counter for Y-range queries
    fast_counter = _FastCounter(buffered_cameras_by_y) if buffered_cameras_by_y else None
    
    patches = []
    current_bin = min_bin
    
    # Greedily create patches
    while current_bin <= max_bin:
        best_end_bin = None
        
        # Find largest Y range that fits camera constraint
        for end_bin in range(current_bin, max_bin + 1):
            y_start = y_binner.bounds(current_bin)[0]
            y_end = y_binner.bounds(end_bin)[1]
            
            # Count cameras in buffered Y range using fast counter
            if fast_counter:
                camera_count = fast_counter.count_in_y_range(
                    y_start - buffer_m, y_end + buffer_m
                )
            else:
                camera_count = 0
            
            if camera_count <= max_cameras:
                best_end_bin = end_bin
            else:
                break
        
        if best_end_bin is None:
            # Can't fit even one bin - problem impossible
            return _DpState.infinity()
        
        # Create the patch
        y_start = y_binner.bounds(current_bin)[0]  
        y_end = y_binner.bounds(best_end_bin)[1]
        
        patch = BoundingBox(
            min_x=x_left - buffer_m,
            max_x=x_right + buffer_m,
            min_y=y_start - buffer_m,
            max_y=y_end + buffer_m
        )
        
        patches.append(patch)
        current_bin = best_end_bin + 1
        
    # Calculate aspect ratio cost
    total_cost = 1.0
    for patch in patches:
        aspect_ratio = max(patch.width, patch.height) / min(patch.width, patch.height)
        total_cost *= aspect_ratio
    
    return _DpState(len(patches), total_cost, patches)

def _partition_with_dp(camera_positions: List[Point],
                      max_cameras: int,
                      buffer_meters: float, 
                      target_bins: int) -> List[BoundingBox]:
    """Run the dynamic programming algorithm to find optimal partitioning."""
    
    # Convert to internal format
    cameras = [_Point2D(p[0], p[1]) for p in camera_positions]
    cameras_by_x = sorted(cameras, key=lambda p: p.x)
    
    # Calculate bounds
    min_x = min(cam.x for cam in cameras)
    max_x = max(cam.x for cam in cameras)
    min_y = min(cam.y for cam in cameras)
    max_y = max(cam.y for cam in cameras)
    
    # Create binners for X and Y axes
    x_binner = _Binner(min_x, max_x, target_bins)
    y_binner = _Binner(min_y, max_y, target_bins)
    
    # Dynamic programming on X axis
    dp = [_DpState.infinity()] * x_binner.count
    
    for cur_bin in range(x_binner.count):
        for prev_bin in range(cur_bin + 1):
            # X range for this partition
            x_left = x_binner.bounds(prev_bin)[0]
            x_right = x_binner.bounds(cur_bin)[1]
            
            # Get cameras in this X range
            if cur_bin == x_binner.count - 1:  # Last bin includes right boundary
                cameras_in_range = [c for c in cameras_by_x if x_left <= c.x <= x_right]
            else:
                cameras_in_range = [c for c in cameras_by_x if x_left <= c.x < x_right]
            
            # Pack Y direction for this X range
            new_state = _pack_y_direction(
                cameras_in_range, cameras, x_left, x_right,
                y_binner, buffer_meters, max_cameras
            )
            
            # Combine with previous solution
            if prev_bin > 0:
                new_state = new_state + dp[prev_bin - 1]
            
            # Keep if better
            if new_state < dp[cur_bin]:
                dp[cur_bin] = new_state
    
    final_solution = dp[x_binner.count - 1]
    
    if final_solution.patch_count >= 999999:
        raise ValueError("No solution found - try increasing max_cameras or reducing buffer")
    
    print(f"✅ Created {final_solution.patch_count} patches (aspect cost: {final_solution.aspect_cost:.2f})")
    
    return final_solution.patches 