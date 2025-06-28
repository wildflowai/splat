"""
Test cases for wildflow.splat.patches function.

Following Google style guide with clear naming and 3-block structure.
"""

import pytest
from wildflow.splat import patches, BoundingBox


def test_patches_empty_list_raises_error():
    """Test patches with empty camera list raises ValueError."""
    camera_positions = []
    
    with pytest.raises(ValueError, match="Camera positions cannot be empty"):
        patches(camera_positions)


def test_patches_single_camera_returns_one_patch():
    """Test patches with single camera returns exactly one patch."""
    camera_positions = [(5.0, 10.0)]
    
    result = patches(camera_positions, max_cameras=100, buffer_meters=1.0)
    
    assert len(result) == 1
    assert isinstance(result[0], BoundingBox)
    assert result[0].min_x == 4.0  # 5.0 - 1.0 buffer
    assert result[0].max_x == 6.0  # 5.0 + 1.0 buffer
    assert result[0].min_y == 9.0  # 10.0 - 1.0 buffer
    assert result[0].max_y == 11.0 # 10.0 + 1.0 buffer


def test_patches_few_cameras_returns_one_patch():
    """Test patches with few cameras that fit in one patch returns one patch."""
    camera_positions = [(0, 0), (1, 1), (2, 2)]
    
    result = patches(camera_positions, max_cameras=10, buffer_meters=0.5)
    
    assert len(result) == 1
    assert result[0].min_x == -0.5  # 0 - 0.5 buffer
    assert result[0].max_x == 2.5   # 2 + 0.5 buffer
    assert result[0].min_y == -0.5  # 0 - 0.5 buffer
    assert result[0].max_y == 2.5   # 2 + 0.5 buffer


def test_patches_exactly_max_cameras_returns_one_patch():
    """Test patches with exactly max_cameras count returns one patch."""
    camera_positions = [(i, i) for i in range(5)]  # Exactly 5 cameras
    
    result = patches(camera_positions, max_cameras=5, buffer_meters=0.0)
    
    assert len(result) == 1
    assert result[0].min_x == 0.0
    assert result[0].max_x == 4.0
    assert result[0].min_y == 0.0
    assert result[0].max_y == 4.0


def test_patches_exceeds_max_cameras_returns_multiple_patches():
    """Test patches with more than max_cameras returns multiple patches."""
    camera_positions = [(i, 0) for i in range(10)]  # 10 cameras in a line
    
    result = patches(camera_positions, max_cameras=3, buffer_meters=0.0)
    
    assert len(result) >= 2  # Should need at least 2 patches
    assert all(isinstance(patch, BoundingBox) for patch in result)
    assert all(patch.width > 0 and patch.height > 0 for patch in result)


def test_patches_clustered_cameras_returns_multiple_patches():
    """Test patches with clustered cameras returns appropriate number of patches."""
    cluster1 = [(i, j) for i in range(3) for j in range(3)]     # 9 cameras at origin
    cluster2 = [(i+10, j+10) for i in range(3) for j in range(3)] # 9 cameras at (10,10)
    camera_positions = cluster1 + cluster2
    
    result = patches(camera_positions, max_cameras=10, buffer_meters=1.0)
    
    assert len(result) >= 2  # Should create at least 2 patches for 2 clusters
    assert len(result) <= 4  # But not too many
    assert all(patch.width > 0 and patch.height > 0 for patch in result)


def test_patches_large_buffer_creates_larger_patches():
    """Test patches with large buffer creates patches with larger dimensions."""
    camera_positions = [(0, 0), (1, 1)]
    
    result_small_buffer = patches(camera_positions, buffer_meters=0.1)
    result_large_buffer = patches(camera_positions, buffer_meters=5.0)
    
    assert len(result_small_buffer) == len(result_large_buffer) == 1
    assert result_large_buffer[0].width > result_small_buffer[0].width
    assert result_large_buffer[0].height > result_small_buffer[0].height


def test_patches_different_target_bins_returns_valid_results():
    """Test patches with different target_bins returns valid partitioning."""
    camera_positions = [(i, j) for i in range(5) for j in range(5)]  # 5x5 grid
    
    result_coarse = patches(camera_positions, target_bins=10, max_cameras=15, buffer_meters=0.5)
    result_fine = patches(camera_positions, target_bins=50, max_cameras=15, buffer_meters=0.5)
    
    assert len(result_coarse) >= 1
    assert len(result_fine) >= 1
    assert all(isinstance(patch, BoundingBox) for patch in result_coarse)
    assert all(isinstance(patch, BoundingBox) for patch in result_fine)


def test_patches_negative_coordinates_handles_correctly():
    """Test patches with negative coordinates handles correctly."""
    camera_positions = [(-5, -3), (-2, -1), (1, 2), (4, 5)]
    
    result = patches(camera_positions, max_cameras=10, buffer_meters=1.0)
    
    assert len(result) >= 1
    assert result[0].min_x <= -6.0  # Should include -5 - 1 buffer
    assert result[0].max_x >= 5.0   # Should include 4 + 1 buffer
    assert result[0].min_y <= -4.0  # Should include -3 - 1 buffer
    assert result[0].max_y >= 6.0   # Should include 5 + 1 buffer


def test_patches_zero_buffer_creates_tight_patches():
    """Test patches with zero buffer creates patches without extra margin."""
    camera_positions = [(0, 0), (10, 10)]
    
    result = patches(camera_positions, buffer_meters=0.0, max_cameras=10)
    
    assert len(result) == 1
    assert result[0].min_x == 0.0
    assert result[0].max_x == 10.0
    assert result[0].min_y == 0.0
    assert result[0].max_y == 10.0

# ------------------ BASIC FUNCTIONALITY ------------------

def test_single_patch_simple_case():
    """Single patch with a few cameras, all within limits."""
    cameras = [(0, 0), (1, 1), (2, 2)]
    result = patches(cameras, max_cameras=10, buffer_meters=1.0)
    assert len(result) == 1
    bbox = result[0]
    assert bbox.min_x <= -1 and bbox.max_x >= 3
    assert bbox.min_y <= -1 and bbox.max_y >= 3

def test_multiple_patches_simple_line():
    """Simple line of cameras that exceeds max_cameras."""
    cameras = [(i, 0) for i in range(6)]
    result = patches(cameras, max_cameras=3, buffer_meters=0.5)
    assert len(result) >= 2
    total_covered = sum([(bbox.max_x - bbox.min_x) * (bbox.max_y - bbox.min_y) for bbox in result])
    assert all([(bbox.max_x - bbox.min_x) > 0 for bbox in result])

def test_clustered_cameras_two_groups():
    """Two clusters of 50 cameras each, should produce 2 patches."""
    cameras = [(0 + i * 0.1, 0 + j * 0.1) for i in range(5) for j in range(10)] + \
              [(100 + i * 0.1, 100 + j * 0.1) for i in range(5) for j in range(10)]
    result = patches(cameras, max_cameras=60, buffer_meters=2.0)
    assert len(result) == 2
    assert all([len([pt for pt in cameras if bbox.min_x <= pt[0] <= bbox.max_x and bbox.min_y <= pt[1] <= bbox.max_y]) <= 60 for bbox in result])

# ------------------ GEOMETRIC EDGE CASES ------------------

def test_single_camera():
    """One camera should still produce one valid patch."""
    cameras = [(10, 10)]
    result = patches(cameras, max_cameras=5, buffer_meters=1.0)
    assert len(result) == 1
    bbox = result[0]
    assert bbox.min_x <= 9 and bbox.max_x >= 11


def test_all_same_position():
    """All cameras in the exact same spot."""
    cameras = [(5, 5)] * 10
    result = patches(cameras, max_cameras=10, buffer_meters=0.5)
    assert len(result) == 1
    bbox = result[0]
    assert bbox.min_x <= 4.5 and bbox.max_x >= 5.5


def test_horizontal_line():
    cameras = [(i, 0) for i in range(10)]
    result = patches(cameras, max_cameras=5, buffer_meters=1.0)
    assert all([(bbox.max_y - bbox.min_y) > 0 for bbox in result])


def test_vertical_line():
    cameras = [(0, i) for i in range(10)]
    result = patches(cameras, max_cameras=5, buffer_meters=1.0)
    assert all([(bbox.max_x - bbox.min_x) > 0 for bbox in result])


def test_negative_coordinates():
    cameras = [(-10, -10), (-5, -5), (0, 0)]
    result = patches(cameras, max_cameras=10, buffer_meters=1.0)
    assert len(result) == 1
    bbox = result[0]
    assert bbox.min_x <= -11 and bbox.max_y >= 1

# ------------------ CONSTRAINT EDGE CASES ------------------

def test_exact_max():
    cameras = [(i, 0) for i in range(10)]
    result = patches(cameras, max_cameras=10, buffer_meters=1.0)
    assert len(result) == 1

def test_one_over_max():
    cameras = [(i, 0) for i in range(11)]
    result = patches(cameras, max_cameras=10, buffer_meters=1.0)
    assert len(result) >= 2


def test_zero_buffer():
    cameras = [(0, 0), (1, 1), (2, 2)]
    result = patches(cameras, max_cameras=10, buffer_meters=0.0)
    bbox = result[0]
    assert bbox.min_x <= 0 and bbox.max_x >= 2


def test_large_buffer_small_area():
    cameras = [(1, 1), (1.2, 1.2), (1.1, 1.3)]
    result = patches(cameras, max_cameras=10, buffer_meters=5.0)
    assert len(result) == 1
    bbox = result[0]
    assert (bbox.max_x - bbox.min_x) > 9

# ------------------ STRESS TESTS ------------------

def test_many_clusters():
    cameras = []
    for i in range(10):
        cameras.extend([(i * 20 + x * 0.1, i * 20 + y * 0.1) for x in range(3) for y in range(3)])
    result = patches(cameras, max_cameras=9, buffer_meters=1.0)
    assert len(result) == 10


def test_dense_vs_sparse():
    dense = [(i * 0.1, 0) for i in range(100)]
    sparse = [(i * 10, 10) for i in range(100)]
    result = patches(dense + sparse, max_cameras=50, buffer_meters=1.0)
    assert len(result) >= 3


def test_l_shaped():
    cameras = [(i, 0) for i in range(10)] + [(9, j) for j in range(1, 10)]
    result = patches(cameras, max_cameras=5, buffer_meters=1.0)
    assert len(result) >= 2


def test_random_distribution():
    import random
    random.seed(42)
    cameras = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(200)]
    result = patches(cameras, max_cameras=50, buffer_meters=2.0)
    assert all([len([pt for pt in cameras if bbox.min_x <= pt[0] <= bbox.max_x and bbox.min_y <= pt[1] <= bbox.max_y]) <= 50 for bbox in result])

# ------------------ REAL-WORLD-LIKE ------------------

def test_realistic_grid():
    cameras = [(i, j) for i in range(10) for j in range(10)]  # 100 cameras
    result = patches(cameras, max_cameras=20, buffer_meters=1.0)
    assert len(result) >= 5
    covered = set()
    for bbox in result:
        for pt in cameras:
            if bbox.min_x <= pt[0] <= bbox.max_x and bbox.min_y <= pt[1] <= bbox.max_y:
                covered.add(pt)
    assert len(covered) == len(cameras)


def test_large_scale_scene():
    cameras = [(i * 0.5, j * 0.5) for i in range(200) for j in range(5)]
    result = patches(cameras, max_cameras=100, buffer_meters=2.0)
    assert all([len([pt for pt in cameras if bbox.min_x <= pt[0] <= bbox.max_x and bbox.min_y <= pt[1] <= bbox.max_y]) <= 100 for bbox in result])


if __name__ == "__main__":
    print("Running all tests...")
    
    # Get all test functions from current module using globals()
    import sys
    current_module = sys.modules[__name__]
    test_functions = [getattr(current_module, name) for name in dir(current_module) 
                     if name.startswith('test_') and callable(getattr(current_module, name))]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__} passed")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
    
    print(f"\n🎉 Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("💡 Tip: Run with 'python -m pytest tests/test_patches.py -v' for better output")
        exit(1) 