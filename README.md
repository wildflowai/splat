Super hacky library to work with splats.

Usage:

```sh
git clone https://github.com/wildflowai/splat.git
cd splat
pip install -r requirements.txt
```

# Train Reef Splats Workflow

Someone swim with a few GoPros around a reef and we create 3D model so that you can see corals in the browser and how they change over time (e.g. [wildflow.ai/demo](https://wildflow.ai/demo))

This is a workflow below takes Metashape project as input, trains 3DGS model and deploys it for users to see it.

![](/images/wildflow-3dgs-wf.svg)

# 0. Metashape | 100 years

Metashape project that we recieved for a given coral reef site. Camera positions reconstructed. Everything is scaled correctly. Time-series 3D models are co-registered (aligned in space).

```
out:
  metashape/site/time
                /project_file

retention: R0
```

# 1. Prep

## 1.a. Colmap | H100 linux | 2 hours

Export camera positions, point cloud, and images (warped) in Colmap format. Keep image size under 2-3MB (sometimes they blow up to 10MB after this).

```
dep: 0
out:
  3dgs/site/time/1_prep/all/images
                           /sparse/0
retention: R1 - because metashape and takes time
```

## 1.b. Colour correction | H100 linux or Windows with GPU | 5 hours

Lightroom or AI colour correction. Keep original warped images.

```
dep: 1.a
out:
  3dgs/site/time/1_prep/all/images
                           /original_images
retention: R1 - because it takes time to correct
```

## 1.c. Train split | H100 linux | 15 min per patch

Split large model into smaller ones for training so that it fits into VRAM. Usually 15x15 meters approx. Use Metashape API to save in COLMAP format these patches.

```
dep: 0
out:
  3dgs/site/time/1_prep/patch/sparse/0

retention: R1 - because of metashape api
```

## 1.d. Copy images | Any computer would do | 5 min

Copy colour corrected images from the folder with all picturs here, so we have a colmap project for a given patch. Alternatively move colmap artifacts to the folder with all images.

```
dep: 1.b, 1.c
out:
  3dgs/site/time/1_prep/patch/images
retention: R4 - disposable
```

# 2. Train | WINDOWS

Train 3DGS for each splat using Postshot. Only works on Windows. Keep both high-resolution and low-resolution models.

```
dep: 1.d

inp:
  3dgs/site/time/1_prep/patch

out:
  3dgs/site/time/2_train/high/site-time-patch.ply
  3dgs/site/time/2_train/low/site-time-patch.ply

retention: R2 - training takes time and windows
```

# 3. Cleanup

## 3.a Merge and split | Any computer would do

Split large patches into smaller ones 5x5 cells. Each cell should have 50cm margin (needed for cleaning algo to work well).

```
dep: 2

out:
  3dgs/site/time/3_clean/high-raw/site-time-cell.ply
  3dgs/site/time/3_clean/low-raw/site-time-cell.ply

retention: R4
```

## 3.b Cleanup | H100 linux

For each cell run cleanup process to remove wacky splats.
High-res keep as 5x5 cells.
Low-res merge together into one model.

```
dep: 3.a

out:
  3dgs/site/time/3_clean/high/site-time-cell.ply
  3dgs/site/time/3_clean/low/site-time-cell.ply

retention: R4 - because we deploy this data straightaway if it's good
```

# 4. Deploy | Any computer would do

Convert clean models into Octree format. Deploy everything to GCS bucket (production) along with credits doc. Now users could fly over coral reefs.

```
dep: 3.b
inp: credits.md
out: gs://wildflow/site-time-hash/...

retention: R1 - because it's user-facing
```
