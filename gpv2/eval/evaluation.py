import json
import logging
from datetime import datetime
from os.path import isdir, join
from typing import Tuple, Any, Callable, Dict

from gpv2 import file_paths
from gpv2.data.dataset import Dataset, Task, CaptioningExample
from gpv2.data.dce_dataset import DceDataset
from gpv2.data.gpv_datasets import GpvDataset, COCO_CATEGORIES
from gpv2.data.webqa_dataset import WebQaDataset
from gpv2.experimental.boosting_eval import BoostingDceClsEvaluator
from gpv2.experimental.okvqa import OkVqa
from gpv2.data.synonyms import SYNONYMS
from gpv2.train.evaluator import Evaluator, VqaEvaluator, CaptionEvaluator, LocalizationEvaluator, \
  ClsEvaluator, DceClsEvaluator, DceVqaEvaluator, ResultKey, WebQaEvaluator
from gpv2.utils import py_utils
from gpv2.utils.py_utils import dump_json_object, load_json_object
from gpv2.utils.to_params import to_params


def save_evaluation(prefix_or_dir: str, evaluator: Evaluator, stats: Dict[ResultKey, Any]):
  """Save the results in `prefix_or_dir`

  :param prefix_or_dir: Where to save the results
  :param evaluator: Evaluator used, save for book-keeping purposes
  :param stats: The states to save
  """
  if isdir(prefix_or_dir) and not prefix_or_dir.endswith("/"):
    prefix_or_dir += "/"
  cache_file = prefix_or_dir + "eval.json"
  to_save = {("all" if k.subset_name is None else k.subset_name) + "/" + k.metric_name: v
             for k, v in stats.items()}
  to_save = dict(
    stats=to_save,
    evaluator=to_params(evaluator, Evaluator),
    date=datetime.now().strftime("%m%d-%H%M%S"),
    version=6,
  )
  dump_json_object(to_save, cache_file)


def get_webqa_is_verified(part):
  assert part in {"test", "val"}
  src = join(file_paths.WEBQA_DIR, f"{part}_image_info_turk_verified.json")
  logging.info(f"Loading {src}")
  data = load_json_object(join(file_paths.WEBQA_DIR, f"{part}_image_info_turk_verified.json"))
  verified = set(("web/"+x["image"]["image_id"], x["noun"], x["verb"], x["adj"]) for x in data)
  def is_verified(ex):
    return (ex.image_id, ex.noun, ex.verb, ex.adj) in verified
  return is_verified


def get_evaluator(dataset: Dataset) -> Tuple[Evaluator, Callable]:
  """Gets the Evaluator and optionally a function to map examples to subset for `dataset`

  Returns None if the dataset requires a test-server evaluations.
  """

  if isinstance(dataset, OkVqa):
    return VqaEvaluator(strip_rationales=True), None

  if isinstance(dataset, WebQaDataset):
    is_verified = get_webqa_is_verified(dataset.split)

    def get_subsets(x):
      qtype = x.qtype
      subsets = [qtype, qtype[1:]]
      if qtype[1:] in "nva":
        subsets.append("nva")
      if not is_verified(x):
        subsets = [f"unverified-{x}" for x in qtype]
      return subsets

    return WebQaEvaluator(), get_subsets

  if isinstance(dataset, DceDataset):
    if dataset.task == Task.CAPTIONING:
      logging.warning("OpenSce caption eval not supported, use the nocaps server")
      return None, None

    if dataset.task in {Task.CLS, Task.CLS_IN_CONTEXT}:
      unseen = GpvDataset.UNSEEN_GROUPS[Task.CLS]
      seen = set(py_utils.flatten_list(SYNONYMS[x] for x in COCO_CATEGORIES if x not in unseen))
      unseen = set(py_utils.flatten_list(SYNONYMS[x] for x in unseen))

      def get_subsets(x):
        if x.category in seen:
          return ["seen"]
        elif x.category not in unseen:
          return ["unseen"]
        else:
          return []
    else:
      get_subsets = None

    cls_eval = DceClsEvaluator()
    cls_eval = BoostingDceClsEvaluator(list(range(0, 20)), False, 20, cls_eval)
    return {
             Task.CLS: cls_eval,
             Task.LOCALIZATION: LocalizationEvaluator(),
             Task.VQA: DceVqaEvaluator(),
             Task.CLS_IN_CONTEXT: cls_eval
           }[dataset.task], get_subsets

  per_caption = True
  if isinstance(dataset, GpvDataset):
    unseen_split = dataset.split == "test" and dataset.gpv_split
  else:
    unseen_split = False

  if unseen_split:
    def get_subsets(x):
      if isinstance(x, CaptioningExample):
        if per_caption:
          target_cap = [cap for cap in x.captions if x.gpv_id == cap.gpv_id]
          is_unseen = len(target_cap[0].meta["gpv1-unseen"]) > 0
        else:
          raise NotImplementedError()
      else:
        is_unseen = len(x.meta["gpv1-unseen"]) > 0
      return ["unseen"] if is_unseen else ["seen"]
  else:
    get_subsets = None

  if isinstance(dataset, GpvDataset) and dataset.gpv_split:
    # Use the gpv_split evaluation protocol from GPV1
    per_caption = True
  else:
    per_caption = False

  evaluator = {
    Task.VQA: VqaEvaluator(),
    Task.CAPTIONING: CaptionEvaluator(per_caption=per_caption, bleu=None),
    Task.LOCALIZATION: LocalizationEvaluator(),
    Task.CLS: ClsEvaluator(),
    Task.CLS_IN_CONTEXT: ClsEvaluator(),
  }[dataset.get_task()]
  return evaluator, get_subsets
