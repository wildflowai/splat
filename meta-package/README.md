# wildflow

Modeling natural ecosystems.

## Installation

Install all components:

```bash
pip install wildflow
```

or install components separately:

```bash
pip install wildflow-splat  # Just the PLY processing tools
```

## Quick Start

```python
from wildflow import splat

# Process PLY point cloud
config = splat.Config("model.ply")
patch = splat.Patch("output.bin")
config.add_patch(patch)

results = splat.split_point_cloud(config)
print(f"Processed {results['total_points_written']} points")
```

## Requirements

- Python 3.8+

## License

MIT License
