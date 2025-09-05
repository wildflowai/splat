# PLY to Orthographic PNG Renderer

Converts PLY files to orthographic PNG images using SuperSplat.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

## Usage

```python
from splats_ortho import render_ortho

render_ortho("model.ply", "output.png", width=4096, height=4096)
```

Or command line:
```bash
python splats_ortho.py model.ply output.png 4096 4096
```

That's it!
