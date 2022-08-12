import json
import shutil
from os import makedirs
from os.path import join, expanduser

from tqdm import tqdm

from gpv2 import file_paths
from gpv2.data.webqa_dataset import WebQaDataset
from gpv2.utils import image_utils

OUTPUT_DIR = expanduser("~/webqa-valtest")
makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_IMAGES = join(OUTPUT_DIR, "images")
makedirs(OUTPUT_IMAGES, exist_ok=True)


def main():
  image_ids = []
  for f in ["test_image_info.json", "val_image_info.json"]:
    src = join(file_paths.WEBQA_DIR, f)

    shutil.copy(src, join(OUTPUT_DIR, f))
    with open(src, "r") as f:
      data = json.load(f)
    for ex in data:
      image_ids.append(ex["image"]["image_id"])

  for image_id in tqdm(image_ids, desc="copy"):
    src = join(file_paths.WEB_IMAGES_DIR, image_id)
    shutil.copy(src, OUTPUT_IMAGES)



if __name__ == '__main__':
  main()
