from wildflow.splat.cleanup import cleanup_point_cloud, CleanConfig

def main():
    """
    Example usage of the cleanup_point_cloud function.
    """
    config = CleanConfig(
        input_file="/Users/nsv/Downloads/test/splats.ply",
        output_file="/Users/nsv/Downloads/test/splats_cleaned_area.ply",
        max_area=0.05,
    )
    config.colmap_points_file = "/Users/nsv/Downloads/test/points3D.bin"
    config.min_neighbors = 10
    config.radius = 0.1
    config.discarded_file = "/Users/nsv/Downloads/test/splats_discarded_area.ply"

    print("Running cleanup with only surface area filter...")
    try:
        cleanup_point_cloud(config)
        print("Cleanup with surface area filter successful!")
    except Exception as e:
        print(f"An error occurred during area filter cleanup: {e}")

if __name__ == "__main__":
    main()
