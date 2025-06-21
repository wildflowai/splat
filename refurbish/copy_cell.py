
"""
mkdir -p /home/nozdrenkov/corals/colmap/m9_crop_-3x0y5s/sparse/0

colmap model_cropper \
  --input_path /home/nozdrenkov/corals/colmap/m9_txt/sparse/0/  \
  --output_path /home/nozdrenkov/corals/colmap/m9_crop_-3x0y5s/sparse/0  \
  --boundary -17,-2,-1000,-8,7,1000

python copy_cell

"""
import os
import shutil
import pycolmap
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

all_images_dir = "/home/nozdrenkov/corals/colmap/m9_txt/images"
model_path = "/home/nozdrenkov/corals/colmap/m9_crop_-3x0y5s/sparse/0"
output_images_dir = "/home/nozdrenkov/corals/colmap/m9_crop_-3x0y5s/images"

os.makedirs(output_images_dir, exist_ok=True)
model = pycolmap.Reconstruction(model_path)


def copy_image(img_name):
  src = os.path.join(all_images_dir, img_name)
  dst = os.path.join(output_images_dir, img_name)
  if os.path.exists(src):
    shutil.copy2(src, dst)
    return True
  else:
    return False


images = [img.name for _, img in model.images.items()]
copied_count = 0

with ThreadPoolExecutor() as executor:
  results = list(tqdm(executor.map(copy_image, images),
                 total=len(images), desc="Copying images"))
  copied_count = sum(results)

print(f"Done. {copied_count} images copied.")
