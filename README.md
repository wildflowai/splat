Super hacky library to work with coral reef splats.

You can swim with a few GoPros around a reef (e.g. [wildflow.ai/protocol](https://wildflow.ai/protocol)) and then turn the footage into 3D models (e.g. [wildflow.ai/demo](https://wildflow.ai/demo)) to track changes over time, run different analysis on top of it, and ultimately see which conservation/restoration methods work best.

This is a bunch of primitives to process the data.

# Usage
Install with
```
pip install wildflow
```
So you can play with it from python:
```py
from wildflow import splat
splat.split(...)
```
# Workflow

## SfM workflow
Turns images from cameras 3D point cloud and 

![](/images/wildflow-3dgs-wf.svg)

# Local Development

This library uses Rust extensions built with Maturin. To set up locally:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Install dependencies and build
pip install maturin
pip install -r requirements.txt
maturin develop
```

After making changes to Rust code, rebuild with `maturin develop`.
