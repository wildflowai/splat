"""
Camera pose partitioning for 3D Gaussian splatting workflows.

Optimal spatial partitioning of camera poses using dynamic programming
with configurable constraints and two-pointers optimization.
"""

from typing import List, NamedTuple, Optional
import math

# Configuration constants
DEFAULT_TARGET_BINS = 100
DEFAULT_BUFFER_M = 1.5
DEFAULT_MAX_CAMERAS = 700

# Magic numbers for nice bin boundaries
NICE_NUMBERS = [1, 2, 5, 10]
INFINITY_VALUE = 999999999

__all__ = ["Point2D", "BoundingBox", "PartitionOptions", "patches"]


class Point2D(NamedTuple):
    """Immutable 2D point with x, y coordinates."""
    x: float
    y: float


class BoundingBox(NamedTuple):
    """Immutable axis-aligned bounding box."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def width(self) -> float:
        """Get width of bounding box."""
        return self.max_x - self.min_x

    def height(self) -> float:
        """Get height of bounding box."""
        return self.max_y - self.min_y

    def contains_point(self, p: Point2D) -> bool:
        """Check if point is inside this bounding box (inclusive left, exclusive right)."""
        return (self.min_x <= p.x < self.max_x and 
                self.min_y <= p.y < self.max_y)

    def area(self) -> float:
        """Get area of bounding box."""
        return self.width() * self.height()

    def aspect_ratio(self) -> float:
        """Get aspect ratio (max dimension / min dimension)."""
        width, height = self.width(), self.height()
        assert width > 0 and height > 0, f"Invalid dimensions: {width}x{height}"
        return max(width, height) / min(width, height)


class PartitionOptions(NamedTuple):
    """Configuration options for camera pose partitioning."""
    target_bins: int = DEFAULT_TARGET_BINS
    buffer_m: float = DEFAULT_BUFFER_M
    max_cameras: int = DEFAULT_MAX_CAMERAS
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.target_bins > 0, f"target_bins must be positive, got {self.target_bins}"
        assert self.buffer_m >= 0, f"buffer_m must be non-negative, got {self.buffer_m}"
        assert self.max_cameras > 0, f"max_cameras must be positive, got {self.max_cameras}"


class _Binner:
    """Private class for creating nice bin boundaries for a given range."""
    
    def __init__(self, min_val: float, max_val: float, target_bins: int = DEFAULT_TARGET_BINS):
        assert max_val > min_val, f"max_val ({max_val}) must be greater than min_val ({min_val})"
        assert target_bins > 0, f"target_bins must be positive, got {target_bins}"
        
        rough = (max_val - min_val) / target_bins
        power = 10 ** math.floor(math.log10(rough))
        nice = next(x for x in NICE_NUMBERS if x * power >= rough) * power
        
        self.width = nice
        self.start = math.floor(min_val / nice) * nice
        self.count = math.ceil((max_val - self.start) / nice)
        
        assert self.count > 0, f"Invalid bin count: {self.count}"

    def bin_id(self, x: float) -> int:
        """Get bin ID for a given value."""
        return max(0, min(self.count - 1, int((x - self.start) / self.width)))

    def bounds(self, i: int) -> tuple[float, float]:
        """Get (start, end) bounds for bin i."""
        assert 0 <= i < self.count, f"Bin index {i} out of range [0, {self.count})"
        return (self.start + i * self.width, self.start + (i + 1) * self.width)


class _DpState(NamedTuple):
    """Private class for dynamic programming state."""
    min_patches: int
    extra_cost: float
    patches: List[BoundingBox]

    @classmethod
    def infinity(cls) -> '_DpState':
        """Create an infinity state (worst possible solution)."""
        return cls(min_patches=INFINITY_VALUE, extra_cost=float('inf'), patches=[])
    
    @classmethod 
    def empty(cls) -> '_DpState':
        """Create an empty state (neutral for combining)."""
        return cls(min_patches=0, extra_cost=1.0, patches=[])

    def __add__(self, other: '_DpState') -> '_DpState':
        """Combine two DP states."""
        return _DpState(
            min_patches=self.min_patches + other.min_patches,
            extra_cost=self.extra_cost * other.extra_cost,
            patches=self.patches + other.patches
        )

    def __lt__(self, other: '_DpState') -> bool:
        """Compare DP states: first by number of patches, then by extra cost."""
        if self.min_patches != other.min_patches:
            return self.min_patches < other.min_patches
        return self.extra_cost < other.extra_cost


class _TwoPointersCounter:
    """
    Private class for efficient range counting using two-pointers technique.
    Amortized O(1) per query when queries have increasing ranges.
    """
    
    def __init__(self, points_by_y: List[Point2D]):
        assert points_by_y, "Points list cannot be empty"
        # Verify points are sorted by y-coordinate
        for i in range(1, len(points_by_y)):
            assert points_by_y[i-1].y <= points_by_y[i].y, "Points must be sorted by y-coordinate"
        
        self.points = points_by_y
        self.left_ptr = 0
        self.right_ptr = 0
        self.last_start = float('-inf')
    
    def count_in_range(self, y_start: float, y_end: float) -> int:
        """Count points in [y_start, y_end] range."""
        assert y_start <= y_end, f"Invalid range: [{y_start}, {y_end}]"
        
        # Reset pointers if query goes backwards (shouldn't happen in our greedy algorithm)
        if y_start < self.last_start:
            self.left_ptr = 0
            self.right_ptr = 0
        
        # Move left pointer to first point >= y_start
        while self.left_ptr < len(self.points) and self.points[self.left_ptr].y < y_start:
            self.left_ptr += 1
        
        # Move right pointer to first point > y_end
        while self.right_ptr < len(self.points) and self.points[self.right_ptr].y <= y_end:
            self.right_ptr += 1
        
        self.last_start = y_start
        count = self.right_ptr - self.left_ptr
        assert count >= 0, f"Invalid count: {count}"
        return count


def _pack_y_axis(x_filtered_points: List[Point2D], all_points_by_x: List[Point2D], 
                 x_left: float, x_right: float, by: _Binner, buffer_m: float, max_cameras: int) -> _DpState:
    """
    Private function to pack points in the given x-range into y-axis aligned bounding boxes.
    Uses two-pointers technique for O(1) amortized point counting.
    """
    assert x_left < x_right, f"Invalid x-range: [{x_left}, {x_right}]"
    assert buffer_m >= 0, f"Buffer must be non-negative, got {buffer_m}"
    assert max_cameras > 0, f"max_cameras must be positive, got {max_cameras}"
    assert all_points_by_x, "all_points_by_x cannot be empty"
    
    if not x_filtered_points:
        return _DpState.empty()
    
    # Sort the x-filtered points by y for efficient processing
    points_by_y = sorted(x_filtered_points, key=lambda p: p.y)
    assert points_by_y, "No points after filtering"
    
    # Find y-range to cover
    min_y = points_by_y[0].y
    max_y = points_by_y[-1].y
    assert min_y <= max_y, f"Invalid y-range: [{min_y}, {max_y}]"
    
    min_bin = by.bin_id(min_y)
    max_bin = by.bin_id(max_y)
    assert min_bin <= max_bin, f"Invalid bin range: [{min_bin}, {max_bin}]"
    
    # Pre-compute points in buffered x-range for fast counting
    buffered_x_left = x_left - buffer_m
    buffered_x_right = x_right + buffer_m
    buffered_points_by_y = sorted([p for p in all_points_by_x 
                                  if buffered_x_left <= p.x <= buffered_x_right], 
                                 key=lambda p: p.y)
    
    patches = []
    extra_cost = 1.0
    current_bin = min_bin
    
    # Initialize two-pointers for efficient range counting
    if buffered_points_by_y:  # Only create counter if there are points
        point_counter = _TwoPointersCounter(buffered_points_by_y)
    
    # Greedily pack bins
    while current_bin <= max_bin:
        best_end_bin = None
        
        # Find the largest range starting from current_bin that fits constraint
        for end_bin in range(current_bin, max_bin + 1):
            y_start, y_end = by.bounds(current_bin)[0], by.bounds(end_bin)[1]
            assert y_start <= y_end, f"Invalid y bounds: [{y_start}, {y_end}]"
            
            # Count points in buffered area using two-pointers
            buffered_y_start = y_start - buffer_m
            buffered_y_end = y_end + buffer_m
            
            if buffered_points_by_y:
                point_count = point_counter.count_in_range(buffered_y_start, buffered_y_end)
            else:
                point_count = 0
            
            assert point_count >= 0, f"Invalid point count: {point_count}"
            
            if point_count <= max_cameras:
                best_end_bin = end_bin
            else:
                break
        
        # If we can't fit even one bin, the problem is infeasible
        if best_end_bin is None:
            return _DpState.infinity()
        
        # Create the bounding box including buffer
        y_start, y_end = by.bounds(current_bin)[0], by.bounds(best_end_bin)[1]
        bbox = BoundingBox(
            min_x=x_left - buffer_m,
            max_x=x_right + buffer_m, 
            min_y=y_start - buffer_m,
            max_y=y_end + buffer_m
        )
        
        # Validate bounding box
        assert bbox.width() > 0 and bbox.height() > 0, f"Invalid bbox dimensions: {bbox.width()}x{bbox.height()}"
        
        # Calculate aspect ratio cost based on core area (without buffer)
        core_width = x_right - x_left
        core_height = y_end - y_start
        if core_width > 0 and core_height > 0:
            aspect_ratio = max(core_width, core_height) / min(core_width, core_height)
            extra_cost *= aspect_ratio
        
        patches.append(bbox)
        current_bin = best_end_bin + 1
    
    assert patches, "No patches created"
    assert extra_cost > 0, f"Invalid extra_cost: {extra_cost}"
    return _DpState(min_patches=len(patches), extra_cost=extra_cost, patches=patches)


def patches(camera_poses_2d: List[Point2D], 
           options: Optional[PartitionOptions] = None) -> List[BoundingBox]:
    """
    Partition 2D camera poses into optimal bounding boxes using dynamic programming.
    
    This function solves the camera pose partitioning problem by finding the minimum number
    of axis-aligned bounding boxes that cover all camera poses while respecting constraints
    on maximum cameras per patch and buffer zones.
    
    Args:
        camera_poses_2d: List of 2D camera positions to partition
        options: Configuration options (uses defaults if None)
        
    Returns:
        List of bounding boxes that partition the points optimally
        
    Raises:
        AssertionError: If inputs are invalid or no solution exists
        
    Example:
        >>> from wildflow.splat import patches, Point2D, PartitionOptions
        >>> poses = [Point2D(0, 0), Point2D(1, 1), Point2D(2, 2)]
        >>> options = PartitionOptions(max_cameras=10, buffer_m=0.1)
        >>> result = patches(poses, options)
        >>> print(f"Created {len(result)} patches")
    """
    # Use default options if none provided
    if options is None:
        options = PartitionOptions()
    
    # Validate inputs
    assert camera_poses_2d, "Camera poses list cannot be empty"
    options.validate()
    
    # Sort points by x-coordinate for fast filtering
    points_by_x = sorted(camera_poses_2d, key=lambda p: p.x)
    
    # Calculate bounds
    min_x = min(p.x for p in camera_poses_2d)
    max_x = max(p.x for p in camera_poses_2d)
    min_y = min(p.y for p in camera_poses_2d)
    max_y = max(p.y for p in camera_poses_2d)
    
    assert min_x < max_x, f"All points have same x-coordinate: {min_x}"
    assert min_y < max_y, f"All points have same y-coordinate: {min_y}"
    
    # Create binners
    bx = _Binner(min_x, max_x, options.target_bins)
    by = _Binner(min_y, max_y, options.target_bins)
    
    # Pre-filter points by x-ranges and cache results
    x_filtered_points = {}
    
    # Initialize DP table
    dp = [_DpState.infinity()] * bx.count
    
    # Fill DP table
    for cur_bin in range(bx.count):
        for prev_bin in range(cur_bin + 1):
            # Define x-range for this partition
            x_left, x_right = bx.bounds(prev_bin)[0], bx.bounds(cur_bin)[1]
            assert x_left < x_right, f"Invalid x-range: [{x_left}, {x_right}]"
            
            # For the rightmost partition, include the right boundary
            is_rightmost = (cur_bin == bx.count - 1)
            
            # Get cached or compute filtered points for this x-range
            cache_key = (prev_bin, cur_bin, is_rightmost)
            if cache_key not in x_filtered_points:
                if is_rightmost:
                    filtered = [p for p in points_by_x if x_left <= p.x <= x_right]
                else:
                    filtered = [p for p in points_by_x if x_left <= p.x < x_right]
                x_filtered_points[cache_key] = filtered
            
            # Get the optimal y-axis packing for this x-range
            new_state = _pack_y_axis(x_filtered_points[cache_key], points_by_x, x_left, x_right, 
                                   by, options.buffer_m, options.max_cameras)
            
            # Combine with previous state if exists
            if prev_bin > 0:
                new_state = new_state + dp[prev_bin - 1]
            
            # Update if better
            if new_state < dp[cur_bin]:
                dp[cur_bin] = new_state
    
    final_state = dp[bx.count - 1]
    assert final_state.min_patches != INFINITY_VALUE, "No solution found - try increasing max_cameras or reducing buffer_m"
    assert final_state.patches, "No patches in final solution"
    
    print(f"Final solution: {final_state.min_patches} patches, aspect ratio cost: {final_state.extra_cost:.4f}")
    
    # Validate final solution
    total_points_covered = 0
    for bbox in final_state.patches:
        points_in_box = [p for p in camera_poses_2d if bbox.contains_point(p)]
        total_points_covered += len(points_in_box)
        assert len(points_in_box) <= options.max_cameras, f"Box contains {len(points_in_box)} points, exceeds max_cameras={options.max_cameras}"
    
    # Note: total_points_covered might be less than len(camera_poses_2d) due to exclusive right boundaries
    print(f"Solution validation: {total_points_covered}/{len(camera_poses_2d)} points covered by {len(final_state.patches)} patches")
    
    return final_state.patches 