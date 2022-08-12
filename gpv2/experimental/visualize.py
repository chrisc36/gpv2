import base64
import os
import string
from typing import List, Dict, Optional, Union

import logging
from os.path import join, isdir, relpath, dirname, exists

import PIL.Image
from cv2 import cv2
import h5py
import imagesize
import numpy as np
import torch
import torchvision.ops
from PIL import Image
from dataclasses import dataclass
from torchvision.ops import box_convert

from gpv2 import file_paths
from gpv2.utils import image_utils


def html_rect(x1, y1, x2, y2, rel=None, rank=None, color="black", border_width="medium", label=None):
  rect_style = {
    "position": "absolute",
    "top": y1,
    "left": x1,
    "height": y2-y1,
    "width": x2-x1,
    "border-style": "solid",
    "border-color": color,
    "border-width": border_width,
    "box-sizing": "border-box"
  }
  rect_style_str = "; ".join(f"{k}: {v}" for k, v in rect_style.items())

  text_style = {
    "position": "absolute",
    "top": y1-5,
    "left": x1+3,
    "color": color,
    "background-color": "black",
    "z-index": 9999,
    "padding-right": "5px",
    "padding-left": "5px",
  }
  text_style_str = "; ".join(f"{k}: {v}" for k, v in text_style.items())

  if rel is None and rank is None:
    container = ""
  else:
    container = f'class=box'
    if rel:
      container += f' data-rel="{rel}"'
    if rank:
      container += f' data-rank="{rank}"'

  if rel is None and label is None:
    text = ''
  elif label is not None:  # Label overrides rel
    text = f'  <div style="{text_style_str}">{label}</div>'
  else:
    text = f'  <div style="{text_style_str}">{rel:0.2f}</div>'

  html = [
    f'<div {container}>',
    f'  <div style="{rect_style_str}"></div>',
    text,
    "</div>"
  ]
  return html


def get_color_text(word, color, suffix=None):
  span = f'<span style="color: {color}">{word}</span>'
  if suffix is not None:
    span += suffix
  return span


def _html(tag, prod, style=""):
  return f'<{tag} style="{style}">{prod}</{tag}>'


@dataclass
class BoxesToVisualize:
  boxes: Union[np.ndarray, torch.Tensor]
  scores: Optional[np.ndarray]
  format: str
  color: str
  normalized: bool = False
  labels: List[str] = None


class HtmlVisualizer:

  def __init__(self, embed_image=False):
    self._file_map = None
    self.embed_image = embed_image

  def get_cropped_image(self, image_id, crop):
    image_file = image_utils.get_image_file(image_id)
    cropped_file = image_utils.get_cropped_img_key(image_id, crop)
    cropped_file = join("cropped", cropped_file + ".jpg")
    cropped_dir = join(file_paths.PRECOMPUTED_FEATURES_DIR, "copped-images")
    cropped_full_path = join(cropped_dir, cropped_file)

    if not exists(cropped_full_path):
      if not exists(cropped_dir):
        os.mkdir(cropped_dir)
      logging.info(f"Building cropped image {cropped_full_path}")
      img = PIL.Image.open(image_file)
      img = image_utils.crop_img(img, crop)
      img.save(cropped_full_path)
    return cropped_full_path

  def get_image_html_boxes(self, image_id, boxes: List[BoxesToVisualize],
                           width=None, height=None, crop=None, wrap="div"):
    html = []
    html += [f'<{wrap} style="display: inline-block; position: relative;">']

    if isinstance(image_id, tuple):
      src, (image_w, image_h) = image_id
    else:
      image_file = image_utils.get_image_file(image_id)
      if crop:
        cropped_file = image_utils.get_cropped_img_key(image_id, crop)
        cropped_file = join("cropped", cropped_file + ".jpg")
        cropped_full_path = join(file_paths.VISUALIZATION_DIR, cropped_file)
        if not exists(cropped_full_path):
          logging.info(f"Building cropped image {cropped_full_path}")
          img = PIL.Image.open(image_file)
          img = image_utils.crop_img(img, crop)
          img.save(cropped_full_path)
        src = cropped_file
        image_w, image_h = imagesize.get(cropped_full_path)
      else:
        # TODO This depends on the details of the filepath...
        src = image_utils.get_image_file(image_id)
        image_w, image_h = image_utils.get_image_size(image_id)

    if self.embed_image:
      if crop:
        src = cropped_full_path
      img = cv2.imread(src)
      encoded_image = base64.b64encode(cv2.imencode('.jpg', img)[1])
      src = 'data:image/jpg;base64,{}'.format(encoded_image.decode())
      image_attr = dict(src=src)
    else:
      image_attr = dict(src=src)
    if width:
      image_attr["width"] = width
    if height:
      image_attr["height"] = height
    attr_str = " ".join(f"{k}={v}" for k, v in image_attr.items())
    html += [f'<img {attr_str}>']

    for box_set in boxes:
      if box_set.normalized:
        if width is None:
          w_factor = image_w
          h_factor = image_h
        else:
          w_factor = width
          h_factor = height
      elif width is not None:
        w_factor = width / image_w
        h_factor = height / image_h
      else:
        w_factor = 1
        h_factor = 1

      task_rel = box_set.scores
      task_boxes = box_set.boxes
      if task_rel is not None:
        ixs = np.argsort(-task_rel)
      else:
        ixs = np.arange(len(task_boxes))
      task_boxes = box_convert(torch.as_tensor(task_boxes), box_set.format, "xyxy").numpy()
      for rank, ix in enumerate(ixs):
        box = task_boxes[ix]
        rel = None if task_rel is None else task_rel[ix]
        x1, y1, x2, y2 = box
        html += html_rect(
          x1*w_factor, y1*h_factor, x2*w_factor, y2*h_factor,
          rel=rel, rank=rank+1,
          color=box_set.color,
          label=None if box_set.labels is None else box_set.labels[ix]
        )

    html += [f'</{wrap}>']
    return html

  def get_image_html(self, image_id, boxes, task_boxes=None, task_rel=None,
                     color="rgb(200,0,200)", width=None, height=None, crop=None):
    to_show = []
    if boxes is not None:
      to_show.append(BoxesToVisualize(boxes, None, "xywh", "blue", False))
    if task_boxes is not None:
      to_show.append(BoxesToVisualize(task_boxes, task_rel, "cxcywh", color, True))
    return self.get_image_html_boxes(image_id, to_show, width, height, crop)


def get_table_html(rows):
  html = []
  style = """
table td {
border: thin solid; 
}
table th {
border: thin solid;
}
  """
  html.append("<style>")
  html.append(style)
  html.append("</style>")

  html += ["<div>"]
  html += ['<table style="font-size:20px; margin-left: auto; margin-right: auto; border-collapse: collapse;">']

  all_keys = rows[0]
  for row in rows[1:]:
    all_keys.update(row)
  cols = list(all_keys)

  html += ['\t<tr>']
  for col in cols:
    html += [_html("th", col, "text-align:center")]
  html += ["\t</tr>"]

  for row in rows:
    html += [f'\t<tr>']
    for k in all_keys:
      html += [f'<td style="text-align:center">']
      val = [""] if k not in row else row[k]
      if isinstance(val, list):
        html += val
      else:
        html.append(str(val))
      html += ["</td>"]
    html += ["\t</tr>"]
  html += ["</table>"]
  html += ["</div>"]

  return html


def save_html(html, out, sliders=True):
  logging.info(f"Writing to {out}")
  if sliders:
    with open("with_slider_template.html") as f:
      template = string.Template(f.read())
    html = template.substitute(html_contents="\n".join(html))
  else:
    html = "\n".join(html)
  with open(out, "w") as f:
    f.write(html)
