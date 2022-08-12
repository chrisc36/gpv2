from numbers import Number
from typing import List, Dict

from dataclasses import replace
import numpy as np

from gpv2.data.gpv_datasets import COCO_CATEGORIES
from gpv2.data.synonyms import SYNONYMS
from gpv2.model.model import GPVExampleOutput
from gpv2.train.evaluator import ResultKey, PerExampleEvaluator, DceClsEvaluator, Evaluator
from gpv2.utils import py_utils


@Evaluator.register("boosting-opensce-cls")
class BoostingDceClsEvaluator(PerExampleEvaluator):
  def __init__(self, search_range, coco_syn, top_n, sub_eval: DceClsEvaluator,
               report_boost_results=True):
    self.search_range = search_range
    self.sub_eval = sub_eval
    self.coco_syn = coco_syn
    self.top_n = top_n
    self.coco_answer = set(COCO_CATEGORIES)
    self.report_boost_results = report_boost_results
    if self.coco_syn:
      self.coco_answer = set(py_utils.flatten_list(SYNONYMS[x] for x in self.coco_answer))

  def evaluate(
      self,
      examples: List,
      predictions: Dict[str, GPVExampleOutput],
      allow_partial=False,
      mean=True,
      subset_mapping=None
  ) -> Dict[ResultKey, Number]:
    if mean is False:
      raise ValueError()

    if self.top_n is not None:
      for k, v in predictions.items():
        if len(v.text) < self.top_n:
          raise ValueError(f"Example has {len(v.text)} beams, but top n is {self.top_n}")
      predictions = {k: v.set_beams_to_keep(self.top_n) for k, v in predictions.items()}

    stats = {}

    all_scores = []
    for ex in examples:
      p = predictions[ex.get_gpv_id()]
      coco_ixs = []
      for i, a in enumerate(p.text):
        if a in self.coco_answer:
          coco_ixs.append(i)
      coco_ixs = np.array(coco_ixs)
      scores = []
      if len(coco_ixs) > 0:
        for th in self.search_range:
          boosted = np.array(p.text_logprobs)
          boosted[coco_ixs] -= th
          answer = p.text[np.argmax(boosted)]
          scores.append(answer == ex.category)
      else:
        scores = [p.text[0] == ex.category] * len(self.search_range)
      all_scores.append(scores)
    all_scores = np.array(all_scores).mean(0)

    if self.report_boost_results:
      for ix, th in enumerate(self.search_range):
        stats[ResultKey(f"boost{th}-accuracy")] = all_scores[ix]

    best_th_ix = np.argmax(all_scores)
    best_th = self.search_range[best_th_ix]

    stats[ResultKey(metric_name="boost")] = best_th

    revised = {}

    for k, p in predictions.items():
      boosted = np.array(p.text_logprobs)
      for i, a in enumerate(p.text):
        if a in self.coco_answer:
          boosted[i] -= best_th
      re_sort = np.argsort(-boosted)
      boosted = boosted[re_sort]
      revised[k] = replace(p, text_logprobs=boosted, text=[p.text[i] for i in re_sort])

    for k, v in self.sub_eval.evaluate(examples, revised, subset_mapping=subset_mapping).items():
      stats[replace(k, metric_name=f"boost-{k.metric_name}")] = v
    stats.update(self.sub_eval.evaluate(examples, predictions, subset_mapping=subset_mapping))
    return {k: v for k, v in stats.items()}
