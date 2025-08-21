Super hacky library to work with coral reef splats.

You can swim with a few GoPros around a reef (e.g. [wildflow.ai/protocol](https://wildflow.ai/protocol)) and then turn the footage into 3D models (e.g. [wildflow.ai/demo](https://wildflow.ai/demo)) to track changes over time, run different analysis on top of it, and ultimately see which conservation/restoration methods work best.

This is a bunch of primitives to process the data.

# Usage

## Installation

### From PyPI
```bash
pip install wildflow
```

### From GitHub Releases (pre-compiled wheels)

**Note**: Pre-compiled wheels are only available for releases created after the workflow was updated. If no wheels are available, use the PyPI or source installation methods above.

#### Check if wheels are available
Visit [GitHub Releases](https://github.com/wildflowai/splat/releases) and look for `.whl` files in the assets section.

#### If wheels are available:
```bash
# Get the latest release for your platform
curl -s https://api.github.com/repos/wildflowai/splat/releases/latest | \
  grep "browser_download_url.*whl" | \
  grep "win_amd64" | \
  cut -d '"' -f 4 | \
  xargs pip install

# Or for macOS ARM64
curl -s https://api.github.com/repos/wildflowai/splat/releases/latest | \
  grep "browser_download_url.*whl" | \
  grep "macosx.*arm64" | \
  cut -d '"' -f 4 | \
  xargs pip install

# Or for Linux x86_64
curl -s https://api.github.com/repos/wildflowai/splat/releases/latest | \
  grep "browser_download_url.*whl" | \
  grep "linux_x86_64" | \
  cut -d '"' -f 4 | \
  xargs pip install
```

#### PowerShell (Windows)
```powershell
$latest = Invoke-RestMethod -Uri "https://api.github.com/repos/wildflowai/splat/releases/latest"
$wheel = $latest.assets | Where-Object { $_.name -like "*win_amd64.whl" } | Select-Object -First 1
if ($wheel) {
    pip install $wheel.browser_download_url
} else {
    Write-Host "No wheels available for this release. Use PyPI installation instead: pip install wildflow"
}
```

### From GitHub Source (requires Rust)

**Prerequisites**: This package requires Rust to compile native extensions.

#### Windows
1. Install Rust from [rustup.rs](https://rustup.rs/):
   - Download and run `rustup-init.exe`
   - Follow the installation prompts
   - Restart your terminal/PowerShell
2. Install the package:
   ```bash
   pip install git+https://github.com/wildflowai/splat.git
   ```

#### macOS/Linux
```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Install the package
pip install git+https://github.com/wildflowai/splat.git
```

## Using the library
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

