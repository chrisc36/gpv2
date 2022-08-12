from typing import List

from dataclasses import dataclass

from gpv2.data.dataset import Task, VqaExample
from gpv2.data.gpv_datasets import GpvDataset
from gpv2.data.grit import GritDataset
from gpv2.experimental.visualize import HtmlVisualizer
from gpv2.train.runner import load_gpv_predictions


@dataclass
class ConsistencyTest:
  image_id: str
  questions: List[str]


targets = [
  ConsistencyTest(
    "coco-cap275201",
    ["How are the people in this image related?"],
  )
]

def test():
  ConsistencyTest("")


def main():
  pass


def show_coco_cap():
  data = GpvDataset(Task.CAPTIONING, "val", sample=200).load()
  viz = HtmlVisualizer(True)
  pred = load_gpv_predictions("/home/chrisc/gpv2/models/gpv2/r0/tmp/gpv-cap-val--basic")

  html = []
  for ex in data:
    html += viz.get_image_html(ex.image_id, None)
    cap = pred[ex.get_gpv_id()].text[0]
    html += [f"<div>{ex.image_id}</div>"]
    html += [f"<div>{cap}</div>"]

  with open("out.html", "w") as f:
    f.write("\n".join(html))


if __name__ == '__main__':
  show_coco_cap()