import json
import logging
import os
from argparse import ArgumentParser
from copy import deepcopy

import torch.utils.data
from transformers import AutoConfig

from gpv2.data.dataset import Task, WebQaExample
from gpv2.data.webqa_dataset import WebQaDataset
from gpv2.data.webqa_templates import DefaultWebQueryGenerator
from gpv2.experiments.trainer_cli import add_train_args, get_trainer_from_args, \
  run_trainer_from_args
from gpv2.image_featurizer.precomputed_features import Hdf5FeatureExtractor
from gpv2.model.gpv2 import T5GpvPerBox
from gpv2.model.layers import Linear, BasicBoxEmbedder
from gpv2.model.loss import DetrLocalizationLoss, LocalizationLoss, BasicGPVLoss
from gpv2.model.model import BeamSearchSpec
from gpv2.model.preprocess_example import WebQaPreprocessor, LocalizationPreprocessor, \
  VqaPreprocessor, ClassificationPreprocessor, CaptioningPreprocessor
from gpv2.train.evaluator import WebQaEvaluator, ResultKey
from gpv2.train.optimizer import DelayedWarmupScheduleBuilder, AdamWBuilder, OptimizerBuilder, \
  ParameterGroup, AllParameters
from gpv2.train.trainer import Trainer, EvaluationSetup, TrainerDataset, RunArgs
from gpv2.utils import py_utils
from gpv2.utils.to_params import to_params
from gpv2.data.gpv_datasets import *
from gpv2.experimental.covid_dataset import *
from gpv2.experiments.trainer_cli import COCO_EVAL
from gpv2.train.runner import DataLoaderBuilder
from gpv2.utils.pytorch_utils import get_devices


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
  parser = ArgumentParser()
  parser.add_argument("--model", choices=["t5-small", "t5-base", "t5-large"], default=None)
  parser.add_argument("--init_from")
  parser.add_argument("--save_each_epoch", action="store_true")
  parser.add_argument("--no_find_unused", action="store_true")
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight_decay", type=float, default=1e-4)
  parser.add_argument("--webqa_sample", type=float, default=1.0)
  parser.add_argument("--webqa_subset",  default=None)
  parser.add_argument("--debug", choices=["tiny", "small", "med", "large"], default=None)

  parser.add_argument("--device", nargs="+", default=None,
                      help="List of integer GPU devices to train on")
  parser.add_argument("--output_dir")

  args = parser.parse_args()

  py_utils.add_stdout_logger()

  if args.model is None:
    if args.debug in ["tiny", "small"] and args.init_from is None:
      # Hack the default to be t5-small if not manually specified
      args.model = "t5-small"
    else:
      args.model = "t5-base"

  conf = AutoConfig.from_pretrained(args.model)
  t5_dim = conf.d_model

  localization_loss = DetrLocalizationLoss(1, 5, 2, 1, 0.5, 1, 5, 2, ['labels'])

  # Current best
  model = T5GpvPerBox(
    args.model,
    loss=BasicGPVLoss(localization_loss),
    image_feature_extractor=Hdf5FeatureExtractor("vinvl", BasicBoxEmbedder()),
    # image_feature_extractor=PretrainedDetrFeaturizer(),
    image_joiner=Linear(2048+5, t5_dim),
    all_lower_case=True,
    initialize_from=args.init_from,
    contrast_query="other",
    convert_to_relevance="raw",
    combine_with_objectness="multiply",
    embed_objectness_score=False,
    preprocessors=[
      WebQaPreprocessor(DefaultWebQueryGenerator()),
      CaptioningPreprocessor(),
      ClassificationPreprocessor(),
      VqaPreprocessor(),
      LocalizationPreprocessor()
    ]
  )

  groups = [ParameterGroup(
    AllParameters(),
    group_name="all",
    overrides=dict(delay=0.0, warmup=0.1, lr=args.lr),
    allow_overlap=True
  )]

  scheduler = DelayedWarmupScheduleBuilder()
  optimizer = AdamWBuilder(
    lr=args.lr,
    weight_decay=args.weight_decay,
    parameter_groups=groups
  )

  print("Optimizer:")
  print(json.dumps(to_params(optimizer, OptimizerBuilder), indent=2))

  train_ds = []
  eval_ds = []
  for task in Task:
    train_ds.append(TrainerDataset(
      GpvDataset(task, "train", sample=8000), logging_name="tr-"+str(task),
      train_sample=100 if args.debug else 2000,
      eval_sample=0, eval_setup=COCO_EVAL[task]
    ))
    eval_ds.append(TrainerDataset(
      GpvDataset(task, "val"), logging_name="val-"+str(task),
      eval_sample=100 if args.debug else 2000, train_sample=None,
      eval_setup=COCO_EVAL[task]
    ))

  trainer: Trainer = Trainer(
    train_datasets=train_ds,
    eval_datasets=eval_ds,
    train_loader=DataLoaderBuilder(
      64, 2, True,
      prefetch_factor=2, persist_workers=False),
    loss_logging_ema=0.995, monitor_ema=0.995,
    optimizer=optimizer, step_schedule=scheduler,
    epochs=3,
  )

  webqq_eval = EvaluationSetup(
    WebQaEvaluator(),
    dict(beam_search_spec=BeamSearchSpec(1, 5))
  )
  webqa_train = WebCovid("train", 100 if args.debug else None)
  webqa_val = WebCovid("val", 100 if args.debug else None)
  trainer.train_datasets.append(TrainerDataset(webqa_train, "web-tr", eval_setup=webqq_eval))
  trainer.eval_datasets.append(TrainerDataset(
    webqa_val, "web-eval", eval_setup=webqq_eval))
  trainer.best_model_key = [ResultKey("accuracy", dataset_name="web-eval")]
  trainer.save_each_epoch = args.save_each_epoch
  trainer.find_unused_parameters = False

  devices = RunArgs.build(get_devices(args.device), False, 1)
  trainer.train(model, args.output_dir, devices)


if __name__ == '__main__':
  main()
