import json
import pickle
from os.path import join, exists
from typing import List

import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from gpv2 import file_paths
from gpv2.data.dataset import Task, LocalizationExample
from gpv2.image_featurizer.vinvl_featurizer import VinvlCollate
from gpv2.model.gpv_example import GPVExample
from gpv2.utils import image_utils
from gpv2.utils.pytorch_utils import to_device
from gpv2.vinvl.get_vinvl import get_vinvl
import torch
from torch import nn
from gpv2.data.dce_dataset import DceDataset
from gpv2.vinvl.structures.bounding_box import BoxList


def main():
  device = torch.device("cuda")
  label_map = join(file_paths.DATA_DIR, "vinvl", "VG-SGG-dicts-vgoi6-clipped.json")
  with open(label_map) as f:
    label_map = json.load(f)
  id_to_label = {int(k): v for k, v in label_map["idx_to_label"].items()}
  dataset = DceDataset(Task.LOCALIZATION, "test")
  examples = dataset.load()

  vinvl_model, eval_transform = get_vinvl()
  vinvl_model.to(device)

  img_collate_fn = VinvlCollate(eval_transform, False)

  def collate(_batch: List[LocalizationExample]):
    gpv_exs = []
    for ex in _batch:
      gpv_exs.append(GPVExample(ex.gpv_id, ex.image_id, None, None))
    features = img_collate_fn.collate(gpv_exs)[0]
    del features["query_boxes"]
    return _batch, features

  loader = DataLoader(examples, batch_size=4, shuffle=False, collate_fn=collate)

  predictions = {}
  for examples, batch in tqdm(loader, ncols=100):
    batch = to_device(batch, device)
    with torch.no_grad():
      out, backbone_features = vinvl_model(batch["images"], None, True)
    for box, ex in zip(out, examples):
      box: BoxList = box
      boxes = box.bbox

      labels = box.get_field("labels").cpu().numpy()
      labels = [id_to_label[i] for i in labels]

      scores = box.get_field("scores")
      w, h = box.size
      scale = torch.as_tensor([w, h, w, h], dtype=boxes.dtype, device=device).unsqueeze(0)
      boxes = boxes/scale
      boxes = torchvision.ops.box_convert(boxes, box.mode, "cxcywh")
      boxes = boxes.cpu().numpy()
      scores = scores.cpu().numpy()
      predictions[ex.gpv_id] =   dict(labels=labels, boxes=boxes, scores=scores)

  with open(f"out-{dataset.part}.pkl", "wb") as f:
    pickle.dump(predictions, f)


if __name__ == '__main__':
  main()