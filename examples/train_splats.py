# Example workflow for training 3d gaussian splatting models using Postshot CLI.

#!/usr/bin/env python3

import os
import subprocess
import re
import pycolmap
from pathlib import Path
from wildflow import splat
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from cache import cached
from functional import seq
from operator import itemgetter
from typing import Dict, Any

class Config:
    # Input paths
    sparse_path = r"C:\Users\...\colmap\sparse\0"
    images_path = r"C:\Users\...\colmap\corrected"
    pointcloud_path = r"C:\Users\...\pc.ply"
    
    # Output paths
    output_path = r"C:\Users\...\experiments\train-splats\output"
    
    # Training settings
    max_cameras = 1200
    buffer_meters = 0.8
    sample_percentage = 5.0
    train_steps = 30
    postshot_exe = r"C:\Users\...\Jawset Postshot\bin\postshot-cli.exe"
    
    # Cleanup settings
    cleanup_params = {
        "max_area": 0.004,
        "min_neighbors": 20,
        "radius": 0.2
    }



@cached("Creating patches", "patches", ["patches", "min_z", "max_z"])
def step1_create_patches():
    model = pycolmap.Reconstruction(Config.sparse_path)
    camera_poses = [img.projection_center() for img in model.images.values()]
    camera_z_values = [pos[2] for pos in camera_poses]
    cameras_2d = [(pos[0], pos[1]) for pos in camera_poses]
    
    print(f"Total cameras: {len(cameras_2d)}")
    
    min_z = min(camera_z_values) - 2.0
    max_z = max(camera_z_values) + 0.5
    
    max_cameras = Config.max_cameras
    buffer_meters = Config.buffer_meters
    
    patches_list = splat.patches(cameras_2d, max_cameras=max_cameras, buffer_meters=buffer_meters)

    print(f"✓ Created {len(patches_list)} patches with {max_cameras} max cameras and {buffer_meters}m buffer")
    print(f"  Z range: [{min_z:.1f}, {max_z:.1f}]")
    return patches_list, min_z, max_z

@cached("Splitting cameras", "cameras")
def step2_split_cameras(patches_list, min_z, max_z):
    result = splat.split_cameras({
        "input_path": Config.sparse_path,
        "min_z": min_z,
        "max_z": max_z,
        "save_points3d": False,
        "patches": [
            {**patch, "output_path": f"{Config.output_path}/p{i}/sparse/0"}
            for i, patch in enumerate(patches_list)
        ]
    })
    print(f"✓ Split cameras: {result['total_cameras_written']} cameras, {result['total_images_written']} images")
    return {"result": result, "patches_count": len(patches_list)}

@cached("Splitting point cloud", "pointcloud")
def step3_split_pointcloud(patches_list, min_z, max_z):
    coords = lambda p: {k: p[k] for k in ('min_x', 'max_x', 'min_y', 'max_y')}
    
    result = splat.split_point_cloud({
        "input_file": Config.pointcloud_path,
        "min_z": min_z,
        "max_z": max_z,
        "sample_percentage": Config.sample_percentage,
        "patches": [
            {**coords(patch), "output_file": f"{Config.output_path}/p{i}/sparse/0/points3D.bin"}
            for i, patch in enumerate(patches_list)
        ]
    })
    
    print(f"✓ Split point cloud: {result['points_loaded']} → {result['total_points_written']} points")
    return {"result": result}

def train_patch(patch_idx, gpu_id):
    paths = {
        "sparse": Path(Config.output_path) / f"p{patch_idx}" / "sparse" / "0",
        "output": Path(Config.output_path) / f"p{patch_idx}" / f"raw-p{patch_idx}.ply",
        "log": Path(Config.output_path) / f"p{patch_idx}" / f"train_p{patch_idx}_gpu{gpu_id}.log"
    }
    
    cmd = [
        Config.postshot_exe, "train",
        "-i", str(paths["sparse"] / "cameras.bin"),
        "-i", str(paths["sparse"] / "images.bin"), 
        "-i", str(paths["sparse"] / "points3D.bin"),
        "-i", Config.images_path,
        "-p", "Splat3",
        # "-p", "Splat ADC",
        "--gpu", str(gpu_id),
        "--train-steps-limit", str(Config.train_steps),
        "--show-train-error",
        "--export-splat-ply", str(paths["output"])
    ]
    
    # Log the PowerShell command with proper line continuations
    cmd_str = f'& "{cmd[0]}" {cmd[1]} `\n' + ' `\n'.join([f'  {arg}' for arg in cmd[2:]])
    
    print(f"  Starting p{patch_idx} on GPU {gpu_id}")
    
    try:
        with open(paths["log"], 'w') as f:
            f.write(f"# Command executed:\n{cmd_str}\n\n")
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                          creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0, check=True)
        
        # Parse training metrics from log
        metrics = parse_training_log(paths["log"])
        
        print(f"  ✓ p{patch_idx} complete - SSIM: {metrics['ssim']}, Splats: {metrics['splats']}, Time: {metrics['time']}")
        return {
            "patch": patch_idx, 
            "gpu": gpu_id, 
            "success": True, 
            "metrics": metrics
        }
    
    except subprocess.CalledProcessError as e:
        print(f"  ✗ p{patch_idx} failed on GPU {gpu_id}: {e}")
        return {"patch": patch_idx, "gpu": gpu_id, "success": False, "error": str(e)}


def parse_training_log(log_file: Path) -> Dict[str, Any]:
    """Parse training log to extract SSIM, splat count, and training time."""
    try:
        lines = log_file.read_text().split('\n')
        
        # Find last line with SSIM score
        ssim_line = None
        for line in reversed(lines):
            if 'SSIM' in line and 'Training Radiance Field' in line:
                ssim_line = line
                break
        
        if not ssim_line:
            return {"ssim": "N/A", "splats": "N/A", "time": "N/A"}
        
        # Extract SSIM score (e.g., "SSIM 0.789")
        ssim_match = re.search(r'SSIM\s+([\d.]+)', ssim_line)
        ssim = float(ssim_match.group(1)) if ssim_match else "N/A"
        
        # Extract splat count (e.g., "6.95 MSplats")
        splats_match = re.search(r'([\d.]+)\s+MSplats', ssim_line)
        splats = float(splats_match.group(1)) if splats_match else "N/A"
        
        # Extract training time (e.g., "Elapsed: 47 m 00 s")
        time_match = re.search(r'Elapsed:\s+(\d+)\s+m\s+(\d+)\s+s', ssim_line)
        if time_match:
            minutes, seconds = int(time_match.group(1)), int(time_match.group(2))
            time = "{}m {}s".format(minutes, seconds)
        else:
            time = "N/A"
        
        return {"ssim": ssim, "splats": splats, "time": time}
        
    except Exception as e:
        print(f"  Warning: Could not parse log {log_file}: {e}")
        return {"ssim": "N/A", "splats": "N/A", "time": "N/A"}


@cached("Training splats", "training")
def step4_train_splats(patches_list):
    to_train = [
        i for i in range(len(patches_list))
        if not (Path(Config.output_path) / f"p{i}" / f"raw-p{i}.ply").exists()
    ]
    
    if not to_train:
        print("✓ All patches already trained")
        return {"completed": True, "results": []}
    
    print(f"Training {len(to_train)} patches on 2 GPUs...")
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            lambda args: train_patch(*args),
            [(patch_idx, patch_idx % 2) for patch_idx in to_train]
        ))
    
    print(f"✓ Training completed for {len(results)} patches")
    return {"completed": True, "results": results}

@cached("Cleaning up splats", "cleanup")
def step5_cleanup_splats(patches_list, min_z, max_z):
    # Remove buffer from patch boundaries to get non-overlapping regions
    def get_core_boundaries(patch):
        buffer = Config.buffer_meters
        return {
            "min_x": patch["min_x"] + buffer,
            "max_x": patch["max_x"] - buffer,
            "min_y": patch["min_y"] + buffer,
            "max_y": patch["max_y"]  # Diver's mistake - keep original Y boundaries
        }
    
    # Create disposed splats folder for examination
    disposed_folder = Path(Config.output_path) / "disposed-splats"
    disposed_folder.mkdir(exist_ok=True)

    # Create clean folder for output files
    clean_folder = Path(Config.output_path) / "clean"
    clean_folder.mkdir(exist_ok=True)
    
    results = [
        splat.cleanup_splats({
            "input_file": f"{Config.output_path}/p{i}/raw-p{i}.ply",
            "output_file": str(clean_folder / f"clean-p{i}.ply"),
            "disposed_file": str(disposed_folder / f"disposed-p{i}.ply"),  # Save disposed splats for examination
            **get_core_boundaries(patch),
            "min_z": min_z,
            "max_z": max_z,
            **Config.cleanup_params
        }) or print(f"  ✓ Cleaned p{i} (disposed splats saved to disposed-splats/)")
        for i, patch in enumerate(patches_list)
        if (Path(Config.output_path) / f"p{i}" / f"raw-p{i}.ply").exists()
        and not (clean_folder / f"clean-p{i}.ply").exists()
    ]
    
    print(f"✓ Cleaned {len(results)} splat files")
    print(f"  Disposed splats saved to: {disposed_folder}")
    return {"completed": True, "cleaned_count": len(results)}


@cached("Merging cleaned PLY files", "merge")
def merge_clean_ply_files(clean_folder: Path) -> None:
    """Merge all cleaned PLY files into one full-model.ply file using Rust backend."""
    print("🔵 Merging cleaned PLY files...")
    
    # Find all clean-p*.ply files
    clean_files = list(clean_folder.glob("clean-p*.ply"))
    
    if not clean_files:
        print("⚠️  No cleaned PLY files found to merge")
        return
    
    print(f"Found {len(clean_files)} cleaned PLY files to merge")
    
    output_file = clean_folder / "full-model.ply"
    
    try:
        # Prepare config for Rust merge function
        config = {
            "input_files": [str(f) for f in clean_files],
            "output_file": str(output_file)
        }
        
        # Call Rust merge function
        splat.merge_ply_files(config)
        
        print(f"✓ Successfully merged PLY files into {output_file}")
        
    except Exception as e:
        print(f"✗ Error merging files: {e}")
        print("  Note: Using Rust backend for robust PLY handling")


def auto_cleanup_missing_checkpoints():
    """Automatically clean up output files when checkpoints are missing."""
    print("🧹 Checking for missing checkpoints...")
    
    cache_dir = Path(Config.output_path) / "cache"
    if not cache_dir.exists():
        print("  No cache directory found - starting fresh")
        return
    
    # Check which checkpoints exist
    existing_checkpoints = set()
    for checkpoint_file in cache_dir.glob("*.json"):
        existing_checkpoints.add(checkpoint_file.stem)
    
    print(f"  Found checkpoints: {list(existing_checkpoints)}")
    
    # Clean up based on missing checkpoints
    if "patches" not in existing_checkpoints:
        print("  ❌ Patches checkpoint missing - cleaning patch folders")
        for patch_dir in Path(Config.output_path).glob("p*"):
            if patch_dir.is_dir():
                import shutil
                shutil.rmtree(patch_dir)
                print(f"    Deleted {patch_dir}")
    
    if "cameras" not in existing_checkpoints:
        print("  ❌ Cameras checkpoint missing - cleaning camera data")
        # Camera data is in patch folders, already handled above
    
    if "pointcloud" not in existing_checkpoints:
        print("  ❌ Pointcloud checkpoint missing - cleaning point cloud data")
        # Point cloud data is in patch folders, already handled above
    
    if "training" not in existing_checkpoints:
        print("  ❌ Training checkpoint missing - cleaning training outputs")
        # Training outputs are in patch folders, already handled above
    
    if "cleanup" not in existing_checkpoints:
        print("  ❌ Cleanup checkpoint missing - cleaning cleaned files")
        clean_dir = Path(Config.output_path) / "clean"
        if clean_dir.exists():
            import shutil
            shutil.rmtree(clean_dir)
            print(f"    Deleted {clean_dir}")
        
        disposed_dir = Path(Config.output_path) / "disposed-splats"
        if disposed_dir.exists():
            import shutil
            shutil.rmtree(disposed_dir)
            print(f"    Deleted {disposed_dir}")
    
    if "merge" not in existing_checkpoints:
        print("  ❌ Merge checkpoint missing - cleaning merged files")
        full_model = Path(Config.output_path) / "full-model.ply"
        if full_model.exists():
            import shutil
            full_model.unlink()
            print(f"    Deleted {full_model}")
    
    print("  ✅ Cleanup complete")


def main():
    print("=== Coral Reef 3D Gaussian Splatting Workflow ===")
    
    # Ensure output directory exists
    Path(Config.output_path).mkdir(exist_ok=True)
    
    # Auto-cleanup missing checkpoints
    auto_cleanup_missing_checkpoints()
    
    # Run workflow steps
    patches_list, min_z, max_z = step1_create_patches()
    step2_split_cameras(patches_list, min_z, max_z)
    step3_split_pointcloud(patches_list, min_z, max_z)
    
    training_result = step4_train_splats(patches_list)
    
    # Only proceed to cleanup if at least some training succeeded
    successful_patches = [r for r in training_result.get("results", []) if r.get("success")]
    if successful_patches:
        step5_cleanup_splats(patches_list, min_z, max_z)
        
        # Merge all cleaned PLY files into one full model
        clean_folder = Path(Config.output_path) / "clean"
        if clean_folder.exists():
            merge_clean_ply_files(clean_folder)
    else:
        print("⚠️  Skipping cleanup - no successful training results")

    # Display training summary
    if successful_patches:
        print("\n📊 Training Summary:")
        for result in successful_patches:
            patch_idx = result["patch"]
            metrics = result.get("metrics", {})
            print(f"  p{patch_idx}: SSIM={metrics.get('ssim', 'N/A')}, "
                  f"Splats={metrics.get('splats', 'N/A')}, Time={metrics.get('time', 'N/A')}")
    
    print("\n🎉 Workflow completed successfully!")
    print(f"Results saved to: {Config.output_path}")

if __name__ == "__main__":
    main()
