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
class Parser(hyperparams.Config):
  image_w: int = 416
  image_h: int = 416
  fixed_size: bool = True
  jitter_im: float = 0.3
  jitter_boxes: float = 0.005
  min_process_size: int = 320
  max_process_size: int = 608
  max_num_instances: int = 200
  random_flip: bool = True
  pct_rand: float = 0.5
  letter_box: bool = False
  cutmix: bool = False
  mosaic: bool = True #too slow
  aug_rand_saturation: bool = True
  aug_rand_brightness: bool = True
  aug_rand_zoom: bool = False
  aug_rand_hue: bool = True
  seed: int = 10
  use_tie_breaker: bool = True


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  tfds_name: str = 'voc'
  tfds_split: str = 'train'
  global_batch_size: int = 32
  is_training: bool = True
  dtype: str = 'float16'
  decoder = None
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  tfds_download: bool = True


@dataclasses.dataclass
class YoloDecoder(hyperparams.Config):
  """if the name is specified, or version is specified we ignore input parameters and use version and name defaults"""
  version: Optional[str] = None
  type: Optional[str] = None
  embed_fpn: bool = False
  fpn_path_len: int = 4
  path_process_len: int = 6
  max_level_process_len: Optional[int] = None
  embed_spp: bool = False
  xy_exponential: bool = False


@dataclasses.dataclass
class YoloLossLayer(hyperparams.Config):
  iou_thresh: float = 0.2
  nms_thresh: float = 0.6
  ignore_thresh: float = 0.5
  loss_type: str = 'ciou'
  max_boxes: int = 200
  anchor_generation_scale: int = 416
  use_nms: bool = False
  iou_normalizer: float = 0.75
  cls_normalizer: float = 1.0
  obj_normalizer: float = 1.0


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
  dilate: bool = False
  filter: YoloLossLayer = YoloLossLayer()
  norm_activation: common.NormActivation = common.NormActivation(
      activation='leaky',
      use_sync_bn=False,
      norm_momentum=0.99,
      norm_epsilon=0.001)
  decoder_activation: str = 'leaky'
  _boxes: Optional[List[str]] = dataclasses.field(default_factory=lambda: [
      '(12, 16)', '(19, 36)', '(40, 28)', '(36, 75)', '(76, 55)', '(72, 146)',
      '(142, 110)', '(192, 243)', '(459, 401)'
  ])


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

  load_darknet_weights: bool = True
  darknet_load_decoder: bool = True


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


@exp_factory.register_config_factory('yolo_subdiv_custom')
def yolo_subdiv_custom() -> cfg.ExperimentConfig:
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
      task=YoloSubDivTask(
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
