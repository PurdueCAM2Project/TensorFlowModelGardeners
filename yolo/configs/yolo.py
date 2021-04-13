# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""YOLO configuration definition."""
import tensorflow as tf
from typing import ClassVar, Dict, List, Optional, Tuple, Union
import dataclasses
import os

from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import common

from yolo.configs import backbones
import numpy as np

COCO_INPUT_PATH_BASE = 'coco'
IMAGENET_TRAIN_EXAMPLES = 1281167
IMAGENET_VAL_EXAMPLES = 50000
IMAGENET_INPUT_PATH_BASE = 'imagenet-2012-tfrecord'


# default param classes
@dataclasses.dataclass
class ModelConfig(hyperparams.Config):

  @property
  def input_size(self):
    if self._input_size is None:
      return [None, None, 3]
    else:
      return self._input_size

  @input_size.setter
  def input_size(self, input_size):
    self._input_size = input_size

  def as_dict(self):  # get_build_model_dict(self):
    #model_cfg = getattr(self.model, self.model.type)
    #model_kwargs = model_cfg.as_dict()
    model_kwargs = super().as_dict()
    # print(model_kwargs)
    if self._boxes is not None:
      model_kwargs.update({'_boxes': [str(b) for b in self.boxes]})
    else:
      model_kwargs.update({'_boxes': None})

    return model_kwargs

  @property
  def backbone(self):
    if isinstance(self.base, str):
      # TODO: remove the automatic activation setter
      # self.norm_activation.activation = Yolo._DEFAULTS[self.base].activation
      return Yolo._DEFAULTS[self.base].backbone
    else:
      return self.base.backbone

  @backbone.setter
  def backbone(self, val):
    self.base.backbone = val

  @property
  def decoder(self):
    if isinstance(self.base, str):
      return Yolo._DEFAULTS[self.base].decoder
    else:
      return self.base.decoder

  @decoder.setter
  def decoder(self, val):
    self.base.decoder = val

  @property
  def darknet_weights_file(self):
    if isinstance(self.base, str):
      return Yolo._DEFAULTS[self.base].darknet_weights_file
    else:
      return self.base.darknet_weights_file

  @property
  def darknet_weights_cfg(self):
    if isinstance(self.base, str):
      return Yolo._DEFAULTS[self.base].darknet_weights_cfg
    else:
      return self.base.darknet_weights_cfg

  @property
  def boxes(self):
    if self._boxes is None:
      return None
    boxes = []
    for box in self._boxes:
      # print(box)
      if isinstance(box, list) or isinstance(box, tuple):
        boxes.append(box)
      elif isinstance(box, str):
        if box[0] == '(' or box[0] == '[':
          f = []
          for b in box[1:-1].split(','):
            f.append(float(b.strip()))
          boxes.append(f)
        else:
          f = []
          for b in box.split(','):
            f.append(float(b.strip()))
          boxes.append(f)
      elif isinstance(box, int):
        raise IOError('unsupported input type, only strings or tuples')
    # print(boxes)
    return boxes

  def set_boxes(self, box_list):
    setter = []
    for value in box_list:
      # print(value)
      value = str(list(value))
      setter.append(value[1:-1])
    self._boxes = setter

  @boxes.setter
  def boxes(self, box_list):
    setter = []
    for value in box_list:
      value = str(list(value))
      setter.append(value[1:-1])
    self._boxes = setter


# dataset parsers
@dataclasses.dataclass
class Mosaic(hyperparams.Config):
  output_size: List[int] = dataclasses.field(default_factory=lambda: [640, 640])
  mosaic_frequency: float = 0.667
  crop_area: List[int] = dataclasses.field(default_factory=lambda: [0.25, 1.0])
  crop_area_mosaic: List[int] = dataclasses.field(
      default_factory=lambda: [0.25, 0.95])
  random_crop: bool = True
  random_crop_mosaic: bool = True


@dataclasses.dataclass
class Parser(hyperparams.Config):
  image_w: int = 640
  image_h: int = 640
  fixed_size: bool = True
  max_num_instances: int = 200
  min_process_size: int = 640
  letter_box: bool = False
  random_flip: bool = True
  pct_rand: float = 0.0
  jitter_im: float = 0.6
  jitter_boxes: float = 0.2
  aug_rand_translate: float = 0.05
  aug_rand_saturation: float = 0.7  #1.5
  aug_rand_brightness: float = 0.6  #1.5
  aug_rand_hue: float = 0.1  #0.015
  aug_rand_zoom: float = 0.5
  aug_rand_angle: float = 7.0
  use_tie_breaker: bool = True
  use_scale_xy: bool = True
  anchor_thresh: float = 4.0
  mosaic: Mosaic = Mosaic()


# pylint: disable=missing-class-docstring
@dataclasses.dataclass
class TfExampleDecoder(hyperparams.Config):
  regenerate_source_id: bool = False


@dataclasses.dataclass
class TfExampleDecoderLabelMap(hyperparams.Config):
  regenerate_source_id: bool = False
  label_map: str = ''


@dataclasses.dataclass
class DataDecoder(hyperparams.OneOfConfig):
  type: Optional[str] = 'simple_decoder'
  simple_decoder: TfExampleDecoder = TfExampleDecoder()
  label_map_decoder: TfExampleDecoderLabelMap = TfExampleDecoderLabelMap()


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''  #'gs://tensorflow2/coco_records/train/2017*'
  tfds_name: str = None  #'coco'
  tfds_split: str = None  #'train'
  global_batch_size: int = 64
  is_training: bool = True
  dtype: str = 'float16'
  decoder: DataDecoder = DataDecoder()
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  tfds_download: bool = True
  cache: bool = False


@dataclasses.dataclass
class YoloDecoder(hyperparams.Config):
  """if the name is specified, or version is specified we ignore 
  input parameters and use version and name defaults"""
  version: Optional[str] = None
  type: Optional[str] = None
  embed_fpn: bool = False
  fpn_path_len: int = 4
  path_process_len: int = 6
  max_level_process_len: Optional[int] = None
  embed_spp: bool = False
  xy_exponential: bool = False


def _build_dict(min_level, max_level, value):
  return lambda: {str(key): value for key in range(min_level, max_level + 1)}


def _build_path_scales(min_level, max_level):
  return lambda: {str(key): 2**key for key in range(min_level, max_level + 1)}


@dataclasses.dataclass
class YoloLossLayer(hyperparams.Config):
  min_level: int = 3
  max_level: int = 5
  ignore_thresh: Dict = dataclasses.field(
      default_factory=_build_dict(min_level, max_level, 0.7))
  truth_thresh: Dict = dataclasses.field(
      default_factory=_build_dict(min_level, max_level, 1.0))
  loss_type: Dict = dataclasses.field(
      default_factory=_build_dict(min_level, max_level, 'ciou'))
  iou_normalizer: Dict = dataclasses.field(
      default_factory=_build_dict(min_level, max_level, 0.75))
  cls_normalizer: Dict = dataclasses.field(
      default_factory=_build_dict(min_level, max_level, 1.0))
  obj_normalizer: Dict = dataclasses.field(
      default_factory=_build_dict(min_level, max_level, 1.0))
  max_delta: Dict = dataclasses.field(
      default_factory=_build_dict(min_level, max_level, np.inf))
  new_cords: Dict = dataclasses.field(
      default_factory=_build_dict(min_level, max_level, True))
  scale_xy: Dict = dataclasses.field(
      default_factory=_build_dict(min_level, max_level, 2.0))
  path_scales: Dict = dataclasses.field(
      default_factory=_build_path_scales(min_level, max_level))
  objectness_smooth: Dict = dataclasses.field(
      default_factory=_build_dict(min_level, max_level, 0.0))
  use_nms: bool = False
  iou_thresh: float = 0.001
  nms_thresh: float = 0.6
  max_boxes: int = 200
  pre_nms_points: int = 500
  anchor_generation_scale: int = 512
  use_reduction_sum: bool = True


@dataclasses.dataclass
class YoloBase(hyperparams.OneOfConfig):
  backbone: backbones.Backbone = backbones.Backbone(
      type='darknet', darknet=backbones.DarkNet(model_id='cspdarknet53'))
  decoder: YoloDecoder = YoloDecoder(version='v3', type='regular')
  darknet_weights_file: str = 'yolov3.weights'
  darknet_weights_cfg: str = 'yolov3.cfg'


@dataclasses.dataclass
class Yolo(ModelConfig):
  num_classes: int = 80
  _input_size: Optional[List[int]] = None
  min_level: int = 3
  max_level: int = 5
  boxes_per_scale: int = 3
  base: Union[str, YoloBase] = YoloBase()
  subdivisions: int = 1
  use_sam: bool = False
  filter: YoloLossLayer = YoloLossLayer(
      min_level=min_level, max_level=max_level)
  norm_activation: common.NormActivation = common.NormActivation(
      activation='leaky',
      use_sync_bn=True,
      norm_momentum=0.99,
      norm_epsilon=0.001)
  decoder_activation: str = 'leaky'
  # _boxes: Optional[List[str]] = dataclasses.field(default_factory=lambda: [
  #     '(12, 16)', '(19, 36)', '(40, 28)', '(36, 75)', '(76, 55)', '(72, 146)',
  #     '(142, 110)', '(192, 243)', '(459, 401)'
  # ])

  _boxes: Optional[List[str]] = None


# model task
@dataclasses.dataclass
class YoloTask(cfg.TaskConfig):
  model: Yolo = Yolo(base='v4')
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  weight_decay: float = 5e-4
  annotation_file: Optional[str] = None
  gradient_clip_norm: float = 0.0
  per_category_metrics: bool = False

  load_darknet_weights: bool = False
  darknet_load_decoder: bool = False
  init_checkpoint_modules: str = None  #'backbone'


@dataclasses.dataclass
class YoloSubDivTask(YoloTask):
  subdivisions: int = 4


COCO_INPUT_PATH_BASE = 'coco'
COCO_TRIAN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000


@exp_factory.register_config_factory('yolo_custom')
def yolo_custom() -> cfg.ExperimentConfig:
  """COCO object detection with YOLO."""
  train_batch_size = 1
  eval_batch_size = 1
  base_default = 1200000
  num_batches = 1200000 * 64 / train_batch_size

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(
          #            mixed_precision_dtype='float16',
          #            loss_scale='dynamic',
          num_gpus=2),
      task=YoloTask(
          model=Yolo(),
          train_data=DataConfig(  # input_path=os.path.join(
              # COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(),
              shuffle_buffer_size=2),
          validation_data=DataConfig(
              # input_path=os.path.join(COCO_INPUT_PATH_BASE,
              #                        'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              shuffle_buffer_size=2)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=2000,
          summary_interval=8000,
          checkpoint_interval=10000,
          train_steps=num_batches,
          validation_steps=1000,
          validation_interval=10,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [
                          int(400000 / base_default * num_batches),
                          int(450000 / base_default * num_batches)
                      ],
                      'values': [
                          0.00261 * train_batch_size / 64,
                          0.000261 * train_batch_size / 64,
                          0.0000261 * train_batch_size / 64
                      ]
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 1000 * 64 // num_batches,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


@exp_factory.register_config_factory('yolo_tpu')
def yolo_tpu() -> cfg.ExperimentConfig:
  """COCO object detection with YOLO."""
  train_batch_size = 1
  eval_batch_size = 1
  base_default = 1200000
  num_batches = 1200000 * 64 / train_batch_size

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=YoloTask(
          model=Yolo(),
          train_data=DataConfig(  # input_path=os.path.join(
              # COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(),
              shuffle_buffer_size=10000),
          validation_data=DataConfig(
              # input_path=os.path.join(COCO_INPUT_PATH_BASE,
              #                        'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              shuffle_buffer_size=10000)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=2000,
          summary_interval=8000,
          checkpoint_interval=10000,
          train_steps=num_batches,
          validation_steps=1000,
          validation_interval=10,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [
                          int(400000 / base_default * num_batches),
                          int(450000 / base_default * num_batches)
                      ],
                      'values': [
                          0.00261 * train_batch_size / 64,
                          0.000261 * train_batch_size / 64,
                          0.0000261 * train_batch_size / 64
                      ]
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 1000 * 64 // num_batches,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config
