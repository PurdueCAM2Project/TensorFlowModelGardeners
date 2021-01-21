import os
from typing import List, Optional, Tuple
import dataclasses
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.beta.configs import common

from yolo.configs import backbones


@dataclasses.dataclass
class Parser(hyperparams.Config):
  aug_rand: bool = True
  aug_rand_saturation: bool = False
  aug_rand_brightness: bool = False
  aug_rand_zoom: bool = False
  aug_rand_rotate: bool = False
  aug_rand_hue: bool = False
  aug_rand_aspect: bool = False
  scale: Tuple[int, int] = (128, 448)
  seed: int = 10


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  tfds_name: str = ''
  tfds_split: str = 'train'
  global_batch_size: int = 10
  is_training: bool = True
  dtype: str = 'float16'
  decoder = None
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  tfds_download: bool = True


@dataclasses.dataclass
class ImageClassificationModel(hyperparams.Config):
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = backbones.Backbone(
      type='darknet', resnet=backbones.DarkNet())
  dropout_rate: float = 0.0
  norm_activation: common.NormActivation = common.NormActivation()
  # Adds a BatchNormalization layer pre-GlobalAveragePooling in classification
  add_head_batch_norm: bool = False


@dataclasses.dataclass
class Losses(hyperparams.Config):
  one_hot: bool = True
  label_smoothing: float = 0.0
  l2_weight_decay: float = 0.0


@dataclasses.dataclass
class ImageClassificationTask(cfg.TaskConfig):
  """The model config."""
  model: ImageClassificationModel = ImageClassificationModel()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  losses: Losses = Losses()
  gradient_clip_norm: float = 0.0
  logging_dir: str = None


@exp_factory.register_config_factory('darknet_classification')
def image_classification() -> cfg.ExperimentConfig:
  """Image classification general."""
  return cfg.ExperimentConfig(
      task=ImageClassificationTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
