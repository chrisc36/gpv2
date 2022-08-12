import argparse
from os import mkdir
from os.path import join, exists
from typing import Dict

import imagesize
import torch
import torchvision.ops

from gpv2.data.grit import GritDataset
from gpv2.model.load_model import load_model
from gpv2.model.model import GPVExampleOutput
from gpv2.train.runner import load_gpv_predictions
from gpv2.utils import py_utils, image_utils
import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  parser.add_argument("output_dir")
  parser.add_argument("--split", default="ablation")
  parser.add_argument("--loc_th", type=float, default=0.5)
  args = parser.parse_args()

  if not exists(args.output_dir):
    mkdir(args.output_dir)
  model = py_utils.select_run_dir(args.model)
  eval_dir = join(model, "eval")

  for task in ["categorization", "vqa", "localization", "refexp"]:
    include_bboxes = task in {"localization", "refexp"}
    eval_name = GritDataset(task, args.split).get_name() + "--default"
    eval_name = join(eval_dir, eval_name)
    if not exists(eval_name):
      print(f"{eval_name} does not exist")
      continue
    print(f"Loading {eval_name}")
    pred: Dict[str, GPVExampleOutput] = load_gpv_predictions(eval_name, include_bboxes)
    out = []
    examples = {x.get_gpv_id(): x for x in GritDataset(task, args.split).load()}
    for example_id, ex in pred.items():
      if include_bboxes:
        if task == "localization":
          keep = ex.relevance > args.loc_th
        else:
          ixs = np.argsort(-ex.relevance)
          ex.boxes = ex.boxes[ixs]
          ex.relevance = ex.relevance[ixs]
          keep = slice(0, 1)

        bboxes = torchvision.ops.box_convert(torch.as_tensor(ex.boxes[keep]), "cxcywh", "xyxy")
        w, h = image_utils.get_image_size(examples[example_id].image_id)
        bboxes = bboxes.numpy() * np.array([w, h, w, h]).reshape((1, 4))
        bboxes = np.round(bboxes).astype(np.int64).tolist()
        if len(bboxes) > 0 and task == "localization":
          conf = float(ex.relevance[keep].min())
        elif len(bboxes) > 0 and task == "refexp":
          conf = float(ex.relevance[0])
        else:
          conf = 0.5
      else:
        conf = float(np.exp(ex.text_logprobs[0]))
        bboxes = None
      assert 0 <= conf <= 1
      ex_dict = dict(
        example_id=example_id.split("/", 1)[1],
        bboxes=bboxes,
        confidence=conf
      )
      if task not in {"localization", "refexp"}:
        ex_dict["words"] = ex.text[0]
      out.append(ex_dict)

    output_f = join(args.output_dir, f"{task}.json")
    print(f"Saving to {output_f}")
    py_utils.dump_json_object(out, join(args.output_dir, f"{task}.json"))

  param_f = join(args.output_dir, "params.json")
  py_utils.dump_json_object(dict(params_in_millions=370), param_f)


if __name__ == '__main__':
  main()