"""
wildflow - AI tools for 3D computer vision and photogrammetry processing
"""

__version__ = "0.1.0"

# Import available tools
try:
    from wildflow import splat
    __all__ = ["splat"]
except ImportError:
    # splat module not available
    __all__ = []

# Future modules:
# from wildflow import sfm          # Structure-from-Motion processing
# from wildflow import gaussian     # 3D Gaussian splatting utilities
# from wildflow import analysis     # Computer vision analysis tools
