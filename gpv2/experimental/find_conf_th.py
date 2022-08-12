from dataclasses import replace

from gpv2.data.dataset import Task
from gpv2.data.dce_dataset import DceDataset
from gpv2.data.grit import GritDataset
from gpv2.model.model import GPVExampleOutput
from gpv2.train.evaluator import LocalizationEvaluator
from gpv2.train.runner import load_gpv_predictions
import numpy as np


def prune(ex: GPVExampleOutput, th: float):
  keep = ex.relevance > th
  return replace(ex, relevance=np.ones_like(ex.relevance[keep]), boxes=ex.boxes[keep])


def main():
  dataset = DceDataset(Task.LOCALIZATION, "val")
  examples = dataset.load()
  eval_dir = f"models/gpv2/r0/eval/opensce-val-det-v2--basic//"
  predictions = load_gpv_predictions(eval_dir, True)
  predictions = {k.replace("opensce", "dce"): v for k, v in predictions.items()}

  evaluator = LocalizationEvaluator()

  for th in [0.1, 0.2, 0.3]:
    pred = {k: prune(v, th) for k, v in predictions.items()}
    print(th)
    print(evaluator.evaluate(examples, pred))


if __name__ == '__main__':
  main()