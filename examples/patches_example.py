"""
Example usage of wildflow.splat.patches library for COLMAP camera partitioning.
Interactive sliders allow you to explore how different parameters affect the partitioning.
"""

import pycolmap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import math
from wildflow.splat import patches as splat_patches, BoundingBox

# Constants
COLMAP_FOLDER = "/Users/nsv/corals/soneva/0/"
MAX_CAMERAS = 50
BUFFER_METERS = 0.0
TARGET_BINS = 100

def rotate30(cameras: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Rotate camera positions by 30 degrees counter-clockwise."""
    angle_rad = math.radians(30)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    
    rotated = []
    for x, y in cameras:
        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle
        rotated.append((x_rot, y_rot))
    
    return rotated

def create_interactive_plot(cameras: list[tuple[float, float]]) -> None:
    """Create interactive plot with sliders for parameter adjustment."""
    # Create figure with space for sliders at bottom (smaller to fit on screen)
    fig = plt.figure(figsize=(12, 9))
    
    # Main plot takes up most of the space
    ax_main = plt.axes((0.1, 0.3, 0.8, 0.6))
    
    # Create slider axes with proper spacing
    ax_max_cameras = plt.axes((0.2, 0.2, 0.6, 0.03))
    ax_buffer = plt.axes((0.2, 0.15, 0.6, 0.03))
    ax_target_bins = plt.axes((0.2, 0.1, 0.6, 0.03))
    
    # Create sliders with nice snapping values
    slider_max_cameras = Slider(ax_max_cameras, 'Max Cameras', 1, 2000, valinit=MAX_CAMERAS, valfmt='%d')
    # Buffer slider with 100 beautiful steps (0.02 increments)
    slider_buffer = Slider(ax_buffer, 'Buffer (m)', 0.0, 2.0, valinit=BUFFER_METERS, valfmt='%.2f', 
                          valstep=0.02)
    slider_target_bins = Slider(ax_target_bins, 'Target Bins', 10, 300, valinit=TARGET_BINS, valfmt='%d')
    
    def update_plot(val=None):
        """Update the plot when sliders change."""
        # Clear the main plot
        ax_main.clear()
        
        # Get current slider values
        max_cameras = int(slider_max_cameras.val)
        buffer_meters = slider_buffer.val
        target_bins = int(slider_target_bins.val)
        
        # Run partitioning with current parameters
        try:
            bboxes = splat_patches(
                cameras, 
                max_cameras=max_cameras,
                buffer_meters=buffer_meters,
                target_bins=target_bins
            )
            
            # Plot camera positions
            x_coords = [pos[0] for pos in cameras]
            y_coords = [pos[1] for pos in cameras]
            ax_main.scatter(x_coords, y_coords, c='red', s=20, alpha=0.7,
                           edgecolors='black', linewidth=0.3, zorder=2)
            
            # Draw bounding boxes
            for i, bbox in enumerate(bboxes):
                rect = patches.Rectangle(
                    (bbox.min_x, bbox.min_y), bbox.width, bbox.height,
                    linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.3, zorder=1
                )
                ax_main.add_patch(rect)
                
                # Add box number
                center_x = bbox.min_x + bbox.width / 2
                center_y = bbox.min_y + bbox.height / 2
                ax_main.text(center_x, center_y, str(i+1),
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=12, color='blue', fontweight='bold', zorder=3,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Update title with current parameters
            ax_main.set_title(f'Camera Partitioning: {len(bboxes)} patches '
                             f'(max_cameras={max_cameras}, buffer={buffer_meters:.2f}m, bins={target_bins})')
            
        except Exception as e:
            # Show error message if partitioning fails
            ax_main.text(0.5, 0.5, f'Error: {str(e)}', 
                        transform=ax_main.transAxes, ha='center', va='center',
                        fontsize=14, color='red', bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.8))
            ax_main.set_title('Partitioning Failed - Try Different Parameters')
        
        ax_main.set_xlabel('X Position')
        ax_main.set_ylabel('Y Position')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_aspect('equal')
        
        # Redraw
        fig.canvas.draw()
    
    # Connect sliders to update function
    slider_max_cameras.on_changed(update_plot)
    slider_buffer.on_changed(update_plot)
    slider_target_bins.on_changed(update_plot)
    
    # Initial plot
    update_plot()
    
    plt.show()

def main():
    # Load camera positions using the exact pattern from docstring
    print("Loading camera positions...")
    model = pycolmap.Reconstruction(COLMAP_FOLDER)
    cameras = [(img.projection_center()[0], img.projection_center()[1]) 
               for img in model.images.values()]
    print(f"Loaded {len(cameras)} camera positions")
    
    # Rotate cameras (can be commented out easily)
    cameras = rotate30(cameras)
    
    # Create interactive visualization
    print("Creating interactive visualization...")
    print("Use the sliders to adjust parameters and see how they affect the partitioning!")
    create_interactive_plot(cameras)

if __name__ == "__main__":
    main() 