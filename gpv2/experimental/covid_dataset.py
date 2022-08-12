import logging
import sys
from os.path import join, exists
from typing import Union, Optional, Dict, Any, List

from dataclasses import dataclass, replace

import numpy as np

from gpv2 import file_paths
from gpv2.data.dataset import Dataset, WebQaExample
from gpv2.model.model import PredictionArg
from gpv2.utils.py_utils import load_json_object, dump_json_object, int_to_str
from gpv2.data.webqa_dataset import load_webqa_file


@Dataset.register("webqa-covid")
class WebCovid(Dataset):

  QTYPES_NAME_TO_TYPES = {
    "1n": "1n",
    "1": ("1n", "1v", "1a"),
    "1and2": ("1n", "1v", "1a", "2a", "2v"),
    "1q": ("q", "1n", "1v", "1a"),
    "q": ("q", ),
    "basic": ("q", "1n", "1v", "1a", "2a", "2v")
  }
  QTYPES_TYPES_TO_NAMES = {
    frozenset(v): k for k, v in QTYPES_NAME_TO_TYPES.items()
  }

  def __init__(self, split: str, sample=None, qtypes="basic"):
    if split not in {"test", "val", "train"}:
      raise ValueError(split)
    if isinstance(qtypes, str):
      self.qtypes = self.QTYPES_NAME_TO_TYPES[qtypes]
    else:
      assert len(qtypes) == len(set(qtypes))
      self.qtypes = qtypes
    self.sample = sample
    self.split = split

  def get_task(self):
    from gpv2.data.gpv_datasets import Task
    return Task.CLS

  def get_source_name(self) -> str:
    return "covid-webqa"

  def get_qtypes_name(self):
    if len(self.qtypes) == 1:
      return self.qtypes[0]
    else:
      return self.QTYPES_TYPES_TO_NAMES[frozenset(self.qtypes)]

  def get_name(self) -> str:
    name = f"cwebqa"
    name += f"-{self.get_qtypes_name()}"
    name += f"-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def get_answer_options(self, synonyms=False):
    raise NotImplementedError()

  def load(self) -> List[WebQaExample]:
    file = join(file_paths.WEBQA_DIR, f"covid_{self.split}_image_info.json")
    instances = load_webqa_file(file, self.qtypes, f"cweb-{self.split}")
    if self.split == "val":
      instances += load_webqa_file(join(file_paths.WEBQA_DIR, f"covid_test_image_info.json"),
                                   self.qtypes, f"cweb-test")
    if self.sample:
      instances.sort(key=lambda x: x.gpv_id)
      np.random.RandomState(613423).shuffle(instances)
      return instances[:self.sample]
    else:
      return instances


from gpv2.data.gpv_datasets import CaptioningExample, Caption, LocalizationExample


@Dataset.register("web-covid-cap")
class WebCovidCap(Dataset):

  def get_name(self) -> str:
    return "web-covid-cap"

  def get_task(self):
    from gpv2.data.gpv_datasets import Task
    return Task.CAPTIONING

  def load(self) -> List:
    examples = WebCovid("val", qtypes="q").load()
    out = []
    for ex in examples:
      out.append(CaptioningExample(
        ex.gpv_id, ex.image_id, [Caption(ex.gpv_id, None)]))
    return out


@Dataset.register("web-covid-loc")
class WebCovidLoc(Dataset):

  def get_name(self) -> str:
    return "web-covid-loc"

  def get_task(self):
    from gpv2.data.gpv_datasets import Task
    return Task.LOCALIZATION

  def load(self) -> List:
    examples = WebCovid("val", qtypes="q").load()
    out = []
    for i, ex in enumerate(examples):
      if ex.noun is not None:
        out.append(LocalizationExample(
          f"web-covid-loc-{i}",
          ex.image_id, None, category=ex.noun
        ))
    return out



