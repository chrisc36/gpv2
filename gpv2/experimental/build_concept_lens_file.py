import argparse
import json
import logging
from collections import Counter, defaultdict
from os.path import join, exists, basename

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from gpv2.train.runner import load_gpv_predictions
from gpv2.utils import image_utils, py_utils
from gpv2.data.dataset import Task, ClsExample, VqaExample, CaptioningExample, LocalizationExample
from gpv2.data.webqa_dataset import WebQaExample
from gpv2.experimental.covid_dataset import WebCovidCap, WebCovidLoc, WebCovid
from gpv2.train.evaluator import *


def get_answer_type(answer, webqa_words):
  try:
    float(answer)
    return "num"
  except ValueError:
    pass

  if answer in {"one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten"}:
    return "num"

  if answer in {
    "red", "tan", "blue", "white", "black", "brown", "yellow",
    "orange", "green", "pink", "grey", "purple",
  }:
    return "color"

  if any(x in set(COCO_CATEGORIES) for x in answer.split()):
    return "in-coco"

  if any(x in answer for x in webqa_words):
    return "in-webqa"

  return "other"


def build(name, coco=False):
  from gpv2.data.gpv_datasets import Task

  if name == "cap":
    dataset = WebCovidCap()
  elif name == "loc":
    dataset = WebCovidLoc()
  elif name == "cls":
    dataset = WebCovid("val")
  else:
    raise RuntimeError()

  task = dataset.get_task()

  instances = dataset.load()
  np.random.shuffle(instances)

  models = {
    "gpv2": ("models/webqa-covid-ep3", "basic"),
  }

  predictions = {}
  for k, (src, eval_name) in models.items():
    predictions[k] = load_gpv_predictions(
      join(src, "r0", "eval", f"{dataset.get_name()}--{eval_name}"),
      load_boxes=task == Task.LOCALIZATION
    )

  # loc_eval = LocalizationEvaluator()
  all_coco = set(py_utils.flatten_list(SYNONYMS[x] for x in COCO_CATEGORIES))

  data = []
  for instance in tqdm(instances, ncols=100):
    row = {}
    row["id"] = instance.get_gpv_id()
    row["image_id"] = str(instance.image_id).split("/")[1]

    image_path = str(instance.image_id).split("/")[1]

    # if isinstance(instance, LocalizationExample):
    #   boxes = []
    #   for box in instance.bboxes:
    #     box_data = {}
    #     x1, y1, w, h = box
    #     for name, val in zip(["x1", "y1", "x2", "y2"], [x1, y1, x1+w, y1+h]):
    #       box_data[name] = float(val)
    #     box_data["color"] = "green"
    #     boxes.append(box_data)
    #   row["image"] = dict(image=image_path, boxes=boxes)
    #
    # else:
    row["image"] = image_path

    if isinstance(instance, ClsExample):
      row["category"] = instance.category
    elif isinstance(instance, WebQaExample):
      row["category"] = instance.answer
    elif isinstance(instance, LocalizationExample):
      row["category"] = instance.category
    elif isinstance(instance, (CaptioningExample, LocalizationExample)):
      pass
    else:
      raise ValueError()

    if task not in {Task.LOCALIZATION, Task.CAPTIONING} and len(predictions) > 1:
      outs = [pred[instance.get_gpv_id()].text[0] for pred in predictions.values()]
      if all(outs[0] == x for x in outs):
        row["agree"] = "True"
      else:
        row["agree"] = "False"

    for model_name, pred in predictions.items():
      output = pred[instance.get_gpv_id()]
      answers = output.text

      if task == Task.CAPTIONING:
        row[f"{model_name}:answer"] = answers[0]

      if task != Task.LOCALIZATION:
        row[model_name] = answers[0]
        row[f"{model_name}:conf"] = output.text_logprobs[0]
        if len(answers) > 0:
          table = []
          for ans, lp in zip(answers[:5], output.text_logprobs[:5]):
            table.append(dict(ans=ans, prob=np.exp(lp)*100))
          row[f"{model_name}:answers"] = table

      else:
        # if coco:
        #   w, h = image_utils.get_image_size(instance.image_id)
        #   missing /= torch.tensor([w, h, w, h]).unsqueeze(0)
        #   assert torch.all(missing <= 1.0)

        pred_boxes = torchvision.ops.box_convert(torch.as_tensor(output.boxes), "cxcywh", "xyxy")

        boxes = []
        for ix in np.argsort(-output.relevance)[:3]:
          box_data = {}
          x1, y1, x2, y2 = pred_boxes[ix].numpy()

          for name, val in zip(["x1", "y1", "x2", "y2"], [x1, y1, x2, y2]):
            box_data[name] = float(val)
          box_data["rel"] = float(output.relevance[ix])
          boxes.append(box_data)

          box_data["color"] = "green"

        row[model_name] = dict(image=image_path, boxes=boxes)

      if isinstance(instance, ClsExample):
        syns = OPENSCE_SYNONYMS.get(instance.category, [instance.category])
        row[f"{model_name}:accuracy"] = float(answers[0] in syns)
      elif isinstance(instance, WebQaExample):
        row[f"{model_name}:accuracy"] = float(answers[0] == instance.answer)
      elif isinstance(instance, VqaExample):
        res = vqa_eval.evaluate_examples([instance], {instance.get_gpv_id(): output})[0]
        row[f"{model_name}:acc"] = float(res["acc"])
        if coco_finder.find(output.text[0]):
          row[f"{model_name}:coco-answer"] = "True"
        else:
          row[f"{model_name}:coco-answer"] = "False"
    data.append(row)

  if coco:
    out = f"/home/chrisc/concept-lens/coco-{task.value}.json"
  else:
    out = f"/home/chrisc/concept-lens/{task.value}.json"
  logging.info(f"Saving to {out}")
  with open(out, "w") as f:
    json.dump(data, f, indent=2)


def main():
  parser = argparse.ArgumentParser()
  args = parser.parse_args()

  py_utils.add_stdout_logger()
  for t in ["loc"]:
    build(t)


if __name__ == '__main__':
  main()

