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


if __name__ == "__main__":
    print("Running basic tests...")
    
    try:
        test_patches_single_camera_returns_one_patch()
        print("✅ Single camera test passed")
        
        test_patches_few_cameras_returns_one_patch()
        print("✅ Few cameras test passed")
        
        test_patches_exactly_max_cameras_returns_one_patch()
        print("✅ Exact max cameras test passed")
        
        test_patches_zero_buffer_creates_tight_patches()
        print("✅ Zero buffer test passed")
        
        print("\n🎉 All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise 