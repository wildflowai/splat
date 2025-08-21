Super hacky library to work with coral reef splats.

You can swim with a few GoPros around a reef (e.g. [wildflow.ai/protocol](https://wildflow.ai/protocol)) and then turn the footage into 3D models (e.g. [wildflow.ai/demo](https://wildflow.ai/demo)) to track changes over time, run different analysis on top of it, and ultimately see which conservation/restoration methods work best.


# Installation

### From PyPI
```bash
pip install wildflow
```

### From GitHub (latest development version)
```bash
pip install git+https://github.com/wildflowai/splat.git
```

**Note**: Installing from GitHub requires Rust to be installed on your system.

## Using the library
```py
from wildflow import splat
splat.split(...)
```

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

