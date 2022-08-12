import string
from os.path import join

import nltk.corpus
import torch
from dataclasses import replace
from tqdm import tqdm

from gpv2 import file_paths
from gpv2.data.dataset import Task, LocalizationExample, VqaExample
from gpv2.data.dce_dataset import DceDataset
from gpv2.data.gpv_datasets import GpvDataset
from gpv2.experimental.visualize import HtmlVisualizer, BoxesToVisualize, save_html
from gpv2.model.gpv2 import T5GpvPerBox
from gpv2.model.load_model import load_model
from gpv2.model.model import BeamSearchSpec
from gpv2.train.runner import load_gpv_predictions
from gpv2.utils import py_utils
from gpv2.utils.py_utils import select_run_dir
from gpv2.utils.pytorch_utils import to_device


import numpy as np


STOPWORDS = set(nltk.corpus.stopwords.words('english'))


def extract_keywords(query, caption):
  text = caption.split()
  if query is not None:
    text += query.split()
  parts = [x.strip(string.punctuation) for x in text]
  return list(set(x for x in parts if x not in STOPWORDS))


def main(model):
  py_utils.add_stdout_logger()

  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  print(f"Loading dataset...")
  # dataset = DceDataset(Task.CAPTIONING, "val")
  dataset = GpvDataset(Task.VQA, "val", False)
  examples = dataset.load()

  np.random.RandomState(232143).shuffle(examples)

  print("Loading predictions..")
  # pred = load_gpv_predictions(join(model, "r0/eval/opensce-val-cap--basic"))

  if isinstance(dataset, GpvDataset):
    pred = load_gpv_predictions(join(model, f"r0/eval/gpv-{dataset.task.value}-val-basic"))
  else:
    raise ValueError()

  updated_pred = {}
  for k, v in pred.items():
    updated_pred[k.replace("opensce", "dce")] = v
  pred = updated_pred

  print(f"Loading model...")
  run = select_run_dir(model)
  model: T5GpvPerBox = load_model(run, device=device)

  model.set_prediction_args(beam_search_spec=BeamSearchSpec(10, 30))

  html = []
  viz = HtmlVisualizer(embed_image=True)

  for ex in tqdm(examples[:50], ncols=100):
    batch = [ex]
    batch = [model.preprocess_example(x) for x in batch]
    if isinstance(batch[0].query, list):
      batch[0] = replace(batch[0], query=batch[0].query[:1])
    caption = pred[ex.get_gpv_id()].text[0]

    if isinstance(ex, VqaExample):
      query = ex.question
    else:
      query = None

    keywords = extract_keywords(query, caption)

    batch = []
    for keyword in keywords:
      batch.append(LocalizationExample("", ex.image_id, bboxes=None, category=keyword))
    batch = [model.preprocess_example(x) for x in batch]
    fe = model.get_collate()(batch)
    fe = to_device(fe, device)

    predict = model.predict(**fe, no_box_sort=True)

    html.append(f"<h2>{query} => {caption}</h2>")
    html.append(f"<div>{keywords}</div>")

    html += ['<div style="display: table; width: 100%">']
    for key, out in zip(keywords, predict):
      html += ['<div style="display: table-cell">']
      html += [f"<div>{key}</div>"]
      html += viz.get_image_html_boxes(ex.image_id, boxes=[
        BoxesToVisualize(out.boxes, out.relevance, "cxcywh", "red", True)
      ])
      html += ["</div>"]
    html += ["</div>"]

    html += [f"<h5>MAX</h5>"]

    rel = np.stack([x.relevance for x in predict], 0).max(0)
    html += viz.get_image_html_boxes(ex.image_id, boxes=[
      BoxesToVisualize(predict[0].boxes, rel, "cxcywh", "red", True)
    ], width=400, height=300)


    # keywords += ["other"]
    # batch[0] = replace(batch[0], relevance_query=keywords)
    # fe = model.get_collate()(batch)
    # fe = to_device(fe, device)
    #
    # with torch.no_grad():
    #   predict, other = model.predict_multi_relevance(**fe, constrast_with_last=True)
    #   predict = predict.cpu().numpy()

    # boxes = fe["image_inputs"]["regions"].boxes[0, :-1].cpu().numpy()
    # html.append(f"<h2>{batch[0].query} => {caption}</h2>")
    # html.append(f"<div>{keywords}</div>")
    #
    # html += ['<div style="display: table; width: 100%">']
    # for ix, key in enumerate(keywords[:-1]):
    #   html += ['<div style="display: table-cell">']
    #   html += [f"<div>{key}</div>"]
    #   html += viz.get_image_html_boxes(ex.image_id, boxes=[
    #     BoxesToVisualize(boxes, predict[:, ix], "cxcywh", "red", True)
    #   ])
    #   html += ["</div>"]
    # html += ["</div>"]
    #
    # html += [f"<h5>MAX</h5>"]
    # html += viz.get_image_html_boxes(ex.image_id, boxes=[
    #   BoxesToVisualize(boxes, predict.max(1)/predict.max(), "cxcywh", "red", True)
    # ], width=400, height=300)
    #
    # html += [f"<h5>OTHER</h5>"]
    # other = np.exp(other.cpu().numpy())
    # other = other.max() - other
    # html += viz.get_image_html_boxes(ex.image_id, boxes=[
    #   BoxesToVisualize(boxes, other/other.max(), "cxcywh", "red", True)
    # ], width=400, height=300)


  save_html(html, "/home/chrisc/visualization/out.html")


if __name__ == '__main__':
  main("models/gpv2")