import logging
from argparse import ArgumentParser
from os.path import join, dirname
from typing import List

from gpv2.data.dataset import Task, Dataset
from gpv2.data.dce_dataset import DceDataset
from gpv2.data.gpv_datasets import GpvDataset
from gpv2.data.grit import GritDataset
from gpv2.experimental.okvqa import OkVqa
from gpv2.data.webqa_dataset import WebQaDataset
from gpv2.utils import pytorch_utils, py_utils
from gpv2.utils.py_utils import load_json_object


def add_dataset_args(parser: ArgumentParser, sample=True,
                     part_default="val", task_default="coco"):
  """Add arguments to `parser` so it can be used to specify datasets"""

  parser.add_argument("--part", default=part_default,
                      choices=["val", "test", "train"],
                      help="Subset of the dataset")
  parser.add_argument("--datasets", default=[task_default], required=task_default is None, nargs="+",
                      help="Dataset name")
  if sample:
    parser.add_argument("--sample", type=int)


def get_datasets_from_args(args, model_dir=None, sample=True) -> List[Dataset]:
  """Get datasets from the args from `add_dataset_args`"""

  if model_dir is not None:
    # Figure out what gpv_split the model was trained on by reading `trainer.json`
    # We do this we so can correctly run the sce_split or regular split GpvDataset
    # that matches how the model was trained.
    if py_utils.is_model_dir(model_dir):
      trainer = load_json_object(join(model_dir, "trainer.json"))
    else:
      trainer = load_json_object(join(dirname(model_dir), "trainer.json"))
    train_split = set()
    for ds in trainer["train_datasets"]:
      ds = ds["dataset"]
      if ds["type"] == "gpv":
        train_split.add(ds["gpv_split"])

    if len(train_split) == 0:
      trained_on_sce = None
    elif len(train_split) > 1:
      raise ValueError()
    else:
      trained_on_sce = list(train_split)[0]
  else:
    trained_on_sce = False

  part = args.part
  sample = None if not sample else getattr(args, "sample", None)

  to_show = []
  dce_tasks = set()
  coco_tasks = set()
  for dataset in args.datasets:
    if dataset.startswith("grit"):
      parts = dataset.split("-")[1:]
      split = "ablation" if len(parts) == 1 else parts[1]
      to_show.append(GritDataset(parts[0], split))

    elif dataset in "webqa":
      to_show.append(WebQaDataset(part, sample=sample))

    elif dataset in "webqa-cov":
      from gpv2.experimental.covid_dataset import WebCovid
      to_show.append(WebCovid(part, sample=sample))

    elif dataset in "webqa-cap":
      from gpv2.experimental.covid_dataset import WebCovidCap
      to_show.append(WebCovidCap())
    elif dataset in "webqa-loc":
      from gpv2.experimental.covid_dataset import WebCovidLoc
      to_show.append(WebCovidLoc())

    elif dataset in "okvqa":
      if py_utils.is_model_dir(model_dir):
        trainer = load_json_object(join(model_dir, "trainer.json"))
      else:
        trainer = load_json_object(join(dirname(model_dir), "trainer.json"))
      train_ds = trainer["train_datasets"]
      assert len(train_ds) == 1
      train_ds = train_ds[0]["dataset"]
      assert train_ds["type"] == "okvqa2"
      strip = train_ds.get("mask_rationale_answers", False)
      logging.info(f"mask answer rationales set to {strip} based on train data")
      to_show.append(OkVqa(part, strip))
    elif dataset == "coco":
      coco_tasks.update(Task)
    elif dataset == "dce":
      dce_tasks.update(Task)
    elif dataset in {x.value for x in Task}:
      coco_tasks.add(Task(dataset))
    elif dataset.startswith("coco-"):
      coco_tasks.add(Task(dataset.split("-")[1]))
    elif dataset.startswith("dce-"):
      dce_tasks.add(Task(dataset.split("-")[1]))
    else:
      raise NotImplementedError(dataset)

  for task in coco_tasks:
    to_show += [GpvDataset(task, part, trained_on_sce, sample=sample)]
  for task in dce_tasks:
    to_show += [DceDataset(task, part, sample=sample)]
  return to_show