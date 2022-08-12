from typing import List, Tuple, Dict, Any

from gpv2 import file_paths
from gpv2.data.dataset import Task
from gpv2.detr.detr_misc import nested_tensor_from_tensor_list
from gpv2.detr.detr_roi_head import create_detr_roi_head
from gpv2.image_featurizer.image_featurizer import ImageFeatureExtractor, ImageCollater, \
  gather_qboxes_and_targets, BoxTargets, ImageRegionFeatures
import torch
import torchvision
from allennlp.common import Params
from dataclasses import replace

import torchvision.transforms as T
import numpy as np
import skimage.io as skio
from skimage.transform import resize

from gpv2.model.gpv_example import GPVExample
from gpv2.utils import image_utils

NORMALIZE_TRANSFORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_stocastic_transforms(task: Task):
  if task in {task.CLS}:
    transforms = [T.RandomApply([
      T.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8), T.RandomHorizontalFlip(), T.RandomGrayscale(p=0.2)]
  elif task in {Task.LOCALIZATION, Task.CLS_IN_CONTEXT}:
    transforms = [
      T.RandomApply([
        T.ColorJitter(0.4, 0.4, 0.4, 0.1)
      ], p=0.8),
      T.RandomGrayscale(p=0.2),
    ]
  elif task in {task.VQA, Task.CAPTIONING}:
    transforms = [
      T.RandomApply([
        T.ColorJitter(0.2, 0.2, 0.2, 0.0)
      ], p=0.8),
    ]
  else:
    raise NotImplementedError(task)
  return transforms


def get_train_transforms(task: Task):
  return T.Compose(
    [
      T.ToPILImage(mode='RGB')
    ] + get_stocastic_transforms(task) +
    [
      T.ToTensor(),
      NORMALIZE_TRANSFORM
    ]
  )


def get_eval_transform():
  return T.Compose([
    T.ToPILImage(mode='RGB'),
    T.ToTensor(),
    NORMALIZE_TRANSFORM
  ])


def load_image_data(example, image_size):
  img_file = image_utils.get_image_file(example.image_id)
  crop = example.crop
  try:
    img = skio.imread(img_file)
    if len(img.shape) == 2:
      img = np.tile(np.expand_dims(img, 2), (1, 1, 3))
    else:
      img = img[:, :, :3]
  except (OSError, ValueError) as e:
    raise ValueError(f"Error reading image {img_file}")

  if crop:
    img = image_utils.crop_img(img, crop)

  original_image_size = img.shape[:2]  # HxW

  if image_size:
    img = resize(img, image_size, anti_aliasing=True)
    img = (255 * img).astype(np.uint8)
  return img, original_image_size


def get_detr_model(num_queries=100, pretrained="coco_sce", lr_backbone=0,
                   load_object_classifier=False):
  cfg = dict(
    num_queries=num_queries,
    num_classes=91 if load_object_classifier else 1,
    hidden_dim=256,
    lr_backbone=lr_backbone,
    nheads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    backbone="resnet50",
    position_embedding="sine",
    masks=False,
    dilation=False,
    dropout=0.1,
    dim_feedforward=2048,
    pre_norm=False,
    aux_loss=False,
    frozenbatchnorm=True,
    last_layer_only=True,
  )
  from omegaconf import OmegaConf
  model = create_detr_roi_head(OmegaConf.create(cfg))

  if pretrained:
    state_dict = torch.load(
      file_paths.PRETRAINED_DETR_MODELS[pretrained], map_location="cpu")['model']
    if not load_object_classifier:
      del state_dict["class_embed.weight"]
      del state_dict["class_embed.bias"]
      model.load_state_dict(state_dict, strict=False)
    else:
      model.load_state_dict(state_dict)

  return model


class DetrCollator(ImageCollater):
  def __init__(self, is_train, train_preprocess=True):
    super().__init__()
    self.train_preprocess = train_preprocess
    self.is_train = is_train
    self.train_transforms = {task: get_train_transforms(task) for task in Task}
    self.eval_transform = get_eval_transform()
    self.image_size = (480, 640)

  def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], BoxTargets]:
    image_tensors = []
    for example in batch:
      if self.is_train and getattr(self, "train_preprocess", True):
        trans = self.train_transforms[example.task]
      else:
        trans = self.eval_transform
      img, size = load_image_data(example, self.image_size)
      image_tensors.append(trans(img))

    images = nested_tensor_from_tensor_list(image_tensors)
    qboxes, targets = gather_qboxes_and_targets(batch)
    return dict(images=images, query_boxes=qboxes), targets


@ImageFeatureExtractor.register("detr")
class PretrainedDetrFeaturizer(ImageFeatureExtractor):
  """Pretrained DETR model, used in GPV1"""

  def __init__(self, freeze_backbone=True, freeze_extractor=True, init_relevance=False,
               pretrained_model="coco_sce", clip_boxes=False, full_classifier=False, train_preprocess=True):
    super().__init__()
    self.train_preprocess = train_preprocess
    self.full_classifier = full_classifier
    self.init_relevance = init_relevance
    self.freeze_backbone = freeze_backbone
    self.freeze_extractor = freeze_extractor
    self.pretrained_model = pretrained_model
    self.clip_boxes = clip_boxes
    if init_relevance:
      raise NotImplementedError()
    self.detr = get_detr_model(
      pretrained=self.pretrained_model, load_object_classifier=True)
    self._freeze()

  def _load_from_state_dict(self, *args, **kwargs):
    super()._load_from_state_dict(*args, **kwargs)
    self._freeze()

  def set_freeze(self, freeze_backbone, freeze_extractor):
    self.freeze_backbone = freeze_backbone
    self.freeze_extractor = freeze_extractor
    self._freeze()

  def _freeze(self):
    for n, p in self.detr.named_parameters():
      if n.startswith("class_embed."):
        p.requires_grad = True
      if n.startswith("backbone."):
        p.requires_grad = not self.freeze_backbone
      else:
        p.requires_grad = not self.freeze_extractor

  def get_collate(self, is_train=False):
    return DetrCollator(is_train, train_preprocess=self.train_preprocess)

  def forward(self, images, query_boxes) -> ImageRegionFeatures:
    if any(x is not None for x in query_boxes):
      raise NotImplementedError("Query boxes not supported")
    out = self.detr(images)

    boxes = out["pred_boxes"]
    if self.clip_boxes:
      # Detr can give us out-of-bound boxes, it is built so cx, cy, w, h are
      # between 0 and 1, but that can still lead to invalid x1 y1 x2 y2 coordinates
      c = torchvision.ops.box_convert(boxes.view(-1, 4), "cxcywh", "xyxy")
      c = torch.clip(c, 0.0, 1.0)
      boxes = torchvision.ops.box_convert(c, "xyxy", "cxcywh").view(*boxes.size())

    return ImageRegionFeatures(
      boxes,
      out["detr_hs"].squeeze(0),
      out["pred_relevance_logits"]
    )

