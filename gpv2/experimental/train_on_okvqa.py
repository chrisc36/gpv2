import json
import logging
import os
from argparse import ArgumentParser
from copy import deepcopy

import torch.utils.data
from transformers import AutoConfig

from gpv2.data.dataset import Task
from gpv2.experimental.okvqa import OkVqa
from gpv2.experiments.trainer_cli import add_train_args, get_trainer_from_args, \
  run_trainer_from_args
from gpv2.image_featurizer.precomputed_features import Hdf5FeatureExtractor
from gpv2.model.gpv2 import T5GpvPerBox
from gpv2.model.layers import Linear, BasicBoxEmbedder
from gpv2.model.loss import DetrLocalizationLoss, LocalizationLoss, BasicGPVLoss
from gpv2.model.model import BeamSearchSpec
from gpv2.model.preprocess_example import *
from gpv2.train.evaluator import VqaEvaluator, ResultKey
from gpv2.train.optimizer import DelayedWarmupScheduleBuilder, AdamWBuilder, OptimizerBuilder, \
  ParameterGroup, AllParameters
from gpv2.train.runner import DataLoaderBuilder
from gpv2.train.trainer import Trainer, TrainerDataset, RunArgs, EvaluationSetup
from gpv2.utils import py_utils
from gpv2.utils.pytorch_utils import get_devices
from gpv2.utils.to_params import to_params

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
  parser = ArgumentParser()
  parser.add_argument("--model", choices=["t5-small", "t5-base", "t5-large"], default="t5-base")
  parser.add_argument("--init_from")
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--output_dir")
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=4)
  parser.add_argument("--override", action="store_true")
  parser.add_argument("--mask_rationale_answers", action="store_true")
  parser.add_argument("--rationales", choices=["none", "input", "target"])
  parser.add_argument("--device", nargs="+")
  parser.add_argument("--weight_decay", type=float, default=1e-4)
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  conf = AutoConfig.from_pretrained(args.model)
  t5_dim = conf.d_model

  localization_loss = DetrLocalizationLoss(1, 5, 2, 1, 0.5, 1, 5, 2, ['labels'])
  if args.rationales == "none":
    pre = VqaPreprocessor()
    evaluator = VqaEvaluator(False)
  elif args.rationales == "input":
    pre = VqaRationaleInputPreprocessor()
    evaluator = VqaEvaluator(False)
  elif args.rationales == "target":
    pre = VqaRationaleTargetPreprocessor()
    evaluator = VqaEvaluator(True)
  else:
    raise ValueError()

  # Current best
  model = T5GpvPerBox(
    args.model,
    loss=BasicGPVLoss(localization_loss),
    image_feature_extractor=Hdf5FeatureExtractor("vinvl", BasicBoxEmbedder()),
    image_joiner=Linear(2048+5, t5_dim),
    all_lower_case=True,
    initialize_from=args.init_from,
    contrast_query="other",
    convert_to_relevance="raw",
    combine_with_objectness="multiply",
    embed_objectness_score=False,
    preprocessors=[pre]
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

  eval_setup = EvaluationSetup(evaluator, dict(beam_search_spec=BeamSearchSpec(1, 10)))

  trainer: Trainer = Trainer(
    train_datasets=[TrainerDataset(
      OkVqa("train", args.mask_rationale_answers), "train", eval_setup=eval_setup, eval_sample=2000
    )],
    eval_datasets=[
      TrainerDataset(OkVqa("val", args.mask_rationale_answers), "val", eval_setup=eval_setup)
    ],
    loss_logging_ema=0.995,
    monitor_ema=0.995,
    optimizer=optimizer,
    step_schedule=scheduler,
    epochs=args.epochs,
    best_model_key=ResultKey("score", dataset_name="val"),
    train_loader=DataLoaderBuilder(args.batch_size, num_workers=4)
  )

  devices = RunArgs.build(get_devices(args.device), False, 1)
  trainer.train(model, args.output_dir, devices, override=args.override)


if __name__ == '__main__':
  main()
