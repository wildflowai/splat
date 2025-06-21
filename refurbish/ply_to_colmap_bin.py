"""
PLY to COLMAP Binary Converter
------------------------------
This module converts PLY point cloud files to COLMAP points3D.bin format.

The COLMAP binary format for points3D.bin:
- Number of points (uint64)
- For each point:
  - ID (uint64)
  - XYZ (3 * double)
  - RGB (3 * uint8)
  - Error/Uncertainty (double)
  - Track length (uint64) - usually 0 for imported points
"""

import os
import struct
import numpy as np
import time
import mmap
from collections import defaultdict

def read_ply_binary_optimized(file_path):
    """Optimized PLY binary reader using memory mapping and numpy operations"""
    with open(file_path, 'rb') as f:
        # Parse header
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        
        header_str = header.decode('utf-8', errors='ignore')
        header_lines = header_str.strip().split('\n')
        
        # Extract header information
        vertex_count = 0
        properties = []
        is_binary_le = True
        
        for line in header_lines:
            line = line.strip()
            if line == "ply":
                continue
            elif line.startswith("format "):
                format_type = line.split()[1]
                is_binary_le = "little_endian" in format_type
            elif line.startswith("element vertex "):
                vertex_count = int(line.split()[2])
            elif line.startswith("property "):
                properties.append(line.split())
        
        # Map property names to their positions and types
        prop_types = {}
        prop_sizes = {"float": 4, "double": 8, "uchar": 1, "int": 4}
        
        row_size = 0
        for prop in properties:
            prop_type = prop[1]
            prop_name = prop[2]
            prop_types[prop_name] = (prop_type, row_size)
            row_size += prop_sizes.get(prop_type, 4)
        
        # Memory map the file for faster reading
        header_size = len(header)
        data_size = row_size * vertex_count
        
        # Use memory mapping for large files
        mm = mmap.mmap(f.fileno(), header_size + data_size, access=mmap.ACCESS_READ)
        mm.seek(header_size)
        
        # Prepare arrays for points and colors
        points = np.zeros((vertex_count, 3), dtype=np.float32)
        colors = np.ones((vertex_count, 3), dtype=np.uint8) * 255
        
        # Check if color properties exist
        has_color = False
        if all(name in prop_types for name in ['red', 'green', 'blue']):
            color_props = ['red', 'green', 'blue']
            has_color = True
        elif all(name in prop_types for name in ['r', 'g', 'b']):
            color_props = ['r', 'g', 'b']
            has_color = True
        
        # Create format string for struct unpack
        format_dict = {'float': 'f', 'double': 'd', 'uchar': 'B', 'int': 'i'}
        endian = '<' if is_binary_le else '>'
        
        # Read all vertex data at once
        for i in range(vertex_count):
            vertex_offset = header_size + i * row_size
            
            # Extract XYZ coordinates
            for j, coord in enumerate(['x', 'y', 'z']):
                prop_type, offset = prop_types[coord]
                size = prop_sizes[prop_type]
                format_char = format_dict[prop_type]
                
                value_bytes = mm[vertex_offset + offset:vertex_offset + offset + size]
                value = struct.unpack(f"{endian}{format_char}", value_bytes)[0]
                points[i, j] = value
            
            # Extract RGB colors if present
            if has_color:
                for j, color_prop in enumerate(color_props):
                    prop_type, offset = prop_types[color_prop]
                    size = prop_sizes[prop_type]
                    format_char = format_dict[prop_type]
                    
                    value_bytes = mm[vertex_offset + offset:vertex_offset + offset + size]
                    value = struct.unpack(f"{endian}{format_char}", value_bytes)[0]
                    colors[i, j] = value
        
        mm.close()
        return points, colors

def write_points3d_bin(output_path, points, colors):
    """Write points to COLMAP binary format using numpy operations and buffer writing"""
    point_count = len(points)
    
    # Pre-allocate buffer for better performance
    # Each point requires: 8 (id) + 24 (xyz) + 3 (rgb) + 8 (error) + 8 (track_length) = 51 bytes
    buffer_size = 8 + (point_count * 51)  # 8 bytes for the point count
    buffer = bytearray(buffer_size)
    
    # Pack number of points
    struct.pack_into('<Q', buffer, 0, point_count)
    offset = 8
    
    # Convert points to double precision for COLMAP format
    points_double = points.astype(np.float64)
    
    # Ensure colors are uint8
    colors_uint8 = np.array(colors, dtype=np.uint8)
    
    # Pre-allocate error and track length values
    error = -1.0
    track_length = 0
    
    # Write all points
    for i in range(point_count):
        # Point ID (1-indexed)
        struct.pack_into('<Q', buffer, offset, i + 1)
        offset += 8
        
        # XYZ coordinates
        struct.pack_into('<ddd', buffer, offset, 
                         points_double[i, 0], 
                         points_double[i, 1], 
                         points_double[i, 2])
        offset += 24
        
        # RGB color
        struct.pack_into('<BBB', buffer, offset,
                         colors_uint8[i, 0],
                         colors_uint8[i, 1],
                         colors_uint8[i, 2])
        offset += 3
        
        # Error and track length
        struct.pack_into('<dQ', buffer, offset, error, track_length)
        offset += 16
    
    # Write buffer to file
    with open(output_path, 'wb') as f:
        f.write(buffer)

def convert_ply_to_bin(input_ply_path, output_bin_path):
    """
    Convert PLY point cloud to COLMAP points3D.bin format
    
    Args:
        input_ply_path: Path to input PLY file
        output_bin_path: Path to output points3D.bin file
    """
    input_ply_path = os.path.normpath(input_ply_path)
    output_bin_path = os.path.normpath(output_bin_path)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_bin_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    print(f"Reading PLY file: {input_ply_path}")
    points, colors = read_ply_binary_optimized(input_ply_path)
    print(f"Read {len(points)} points in {time.time() - start_time:.2f}s")
    
    start_time = time.time()
    print(f"Writing points3D.bin to {output_bin_path}")
    write_points3d_bin(output_bin_path, points, colors)
    print(f"Wrote {len(points)} points in {time.time() - start_time:.2f}s")
    
    return len(points)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PLY file to COLMAP points3D.bin")
    parser.add_argument("input_ply", type=str, help="Path to input PLY file")
    parser.add_argument("output_bin", type=str, help="Path to output points3D.bin file")
    
    args = parser.parse_args()
    
    convert_ply_to_bin(args.input_ply, args.output_bin) 