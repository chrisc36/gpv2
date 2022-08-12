from collections import Counter
from os.path import join, exists

from typing import List

from gpv2 import file_paths
from gpv2.data.dataset import Dataset, VqaExample

import json


def get_image_id(image_id):
  for subset in ["train2014", "val2014"]:
    filename = f"{subset}/COCO_{subset}_{str(image_id).zfill(12)}.jpg"
    if exists(join(file_paths.COCO_IMAGES, filename)):
      return f'coco/' + filename
  raise ValueError(f"Unable to find image file for image id {image_id}")


@Dataset.register("okvqa2")
class OkVqa(Dataset):
  def __init__(self, split, mask_rationale_answers=False):
    self.split = split
    self.mask_rationale_answers = mask_rationale_answers

  def get_source_name(self) -> str:
    return "okvqa"

  def get_name(self) -> str:
    name = f"okvqa-{self.split}"
    return name

  def load(self) -> List:
    if self.mask_rationale_answers:
      filename = f"okvqa2_{self.split}_v1p0_w_rationale_newans.json"
    else:
      filename = f"okvqa2_{self.split}_v1p0_w_rationale.json"
    out = []
    with open(join(file_paths.OKVQA2_HOME, filename)) as f:
      data = json.load(f)
      for ex in data:
        if self.split == "test":
          image_id = f'okvqa-test/{str(ex["image_id"]).zfill(12)}.jpg'
        else:
          image_id = get_image_id(ex["image_id"])
        out.append(VqaExample(
          f"okvqa--{ex['question_id']}",
          image_id=image_id,
          question=ex["question"],
          answers=Counter(ex["direct_answers"]),
          meta=dict(distractors=[v for k, v in ex.items() if k.startswith("mc_distractor")],
                    rationale=ex["rationale"],
                    mc_correct_answer=ex["mc_correct_answer"]
                    )
        ))
    return out


def main():
  data = OkVQa("val").load()
  print(len(data))


if __name__ == '__main__':
  main()