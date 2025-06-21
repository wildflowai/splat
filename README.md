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
