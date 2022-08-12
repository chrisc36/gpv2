import json
import os
import shutil
from collections import defaultdict
from os.path import expanduser, join, exists

from gpv2.experimental.okvqa import OkVqa
from gpv2.utils.py_utils import load_json_object

MODELS = {
 "base": "models/okvqa2/base",
 "rationale-input": "models/okvqa2/input-rationales",
  "rationale-input-masked": "models/okvqa2/input-rationales-answer-masked",
}

EVALS = {
  "mc": "mc",
  "da": "default"
}


DATASETS = {
  "test": OkVqa("test"),
  "val": OkVqa("val")
}

OUTPUT_DIR = join(expanduser("~/"), "okvqa-out")
if not exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
  scores = defaultdict(lambda : defaultdict(dict))
  all_scores = {}
  for model_name, src in MODELS.items():
    for dataset_name, dataset in DATASETS.items():
      if "masked" in model_name:
        dataset = OkVqa(dataset.split, True)
      for eval_name, eval_dirname in EVALS.items():
        target = join(src, "r0", "eval", dataset.get_name() + "--" + eval_dirname)
        pred_file = join(target, "predictions.json")
        if not exists(target) or not exists(pred_file):
          print(f"Missing {target}")
          continue
        shutil.copy(pred_file, join(OUTPUT_DIR, f"{model_name}_{dataset_name}_{eval_name}.json"))
        score = load_json_object(join(target, "eval.json"))["stats"]["all/score"]
        date = load_json_object(join(target, "config.json"))["date"]
        scores[dataset_name][model_name][eval_name] = score
        name = f"{model_name}_{dataset_name}_{eval_name}"
        print(name, date, score)
        all_scores[name] = score

  print(json.dumps(all_scores, indent=2))



if __name__ == '__main__':
  main()