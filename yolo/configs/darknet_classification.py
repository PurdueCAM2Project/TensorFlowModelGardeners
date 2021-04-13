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
  tfds_name: str = 'imagenet2012'  # 'cats_vs_dogs' #
  tfds_split: str = 'train'
  tfds_data_dir: str = '/media/vbanna/DATA_SHARE/tfds'
  global_batch_size: int = 1
  is_training: bool = True
  dtype: str = 'float16'
  decoder = None
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  tfds_download: bool = True


@dataclasses.dataclass
class ImageClassificationModel(hyperparams.Config):
  num_classes: int = 1000
  input_size: List[int] = dataclasses.field(
      default_factory=lambda: [256, 256, 3])
  backbone: backbones.Backbone = backbones.Backbone(
      type='darknet', resnet=backbones.DarkNet(model_id='cspdarknet'))
  dropout_rate: float = 0.0
  norm_activation: common.NormActivation = common.NormActivation(
      activation='leaky', use_sync_bn=False)
  # Adds a BatchNormalization layer pre-GlobalAveragePooling in classification
  add_head_batch_norm: bool = False
  min_level: Optional[int] = None
  max_level: int = 5
  subdivisions: int = 8
  darknet_weights_file: str = 'cache://csdarknet53.weights'
  darknet_weights_cfg: str = 'cache://csdarknet53.cfg'


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
  load_darknet_weights: bool = False
  init_checkpoint_modules: str = 'backbone'


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
