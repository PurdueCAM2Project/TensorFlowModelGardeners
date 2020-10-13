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
from typing import Dict, List, Optional, Tuple, Union
import dataclasses
import os

from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import backbones
from official.vision.beta.configs import common

@dataclasses.dataclass
class YoloCFG(hyperparams.Config):
    @property
    def boxes(self):
        boxes = []
        for box in self._boxes:
            f = []
            for b in box.split(","):
                f.append(int(b.strip()))
            boxes.append(f)
        return boxes
    
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
    fixed_size: bool = False
    jitter_im: float = 0.1
    jitter_boxes: float = 0.005
    net_down_scale: int = 32
    min_process_size: int = 320
    max_process_size: int = 608
    max_num_instances: int = 200
    random_flip: bool = True
    pct_rand: float = 0.5
    seed: int = 10
    shuffle_buffer_size: int = 10000

@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
    """Input config for training."""
    input_path: str = ''
    global_batch_size: int = 10
    is_training: bool = True
    dtype: str = 'float16'
    decoder = None
    parser: Parser = Parser()
    shuffle_buffer_size: int = 10000

# Loss and Filter definitions
@dataclasses.dataclass
class Gridpoints(hyperparams.Config):
    low_memory: bool = False
    reset: bool = True

@dataclasses.dataclass
class YoloLoss(hyperparams.Config):
    ignore_thresh: float = 0.7
    truth_thresh: float = 1.0
    use_tie_breaker: bool = None

@dataclasses.dataclass
class YoloLayer(hyperparams.Config):
    generator_params: Gridpoints = Gridpoints()
    iou_thresh: float = 0.45
    class_thresh: float = 0.45
    max_boxes: int = 200
    anchor_generation_scale: int = 416
    use_nms: bool = True
    
# model definition
@dataclasses.dataclass
class Yolov4regular(YoloCFG):
    model: str = "regular"
    backbone: Optional[Dict] = None #"regular"
    neck: Optional[Dict] = None #"regular"
    head: Optional[Dict] = None #"regular"
    head_filter: YoloLayer = YoloLayer()
    _boxes: List[str] = dataclasses.field(default_factory=lambda:["12, 16", "19, 36", "40, 28", "36, 75", "76, 55", "72, 146", "142, 110" ,"192, 243", "459, 401"])
    masks: Dict = dataclasses.field(default_factory=lambda: {"1024": [6, 7, 8], "512": [3, 4, 5], "256": [0, 1, 2]})
    path_scales: Dict = dataclasses.field(default_factory=lambda: {"1024": 32, "512": 16, "256": 8})
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {"1024": 1.05, "512": 1.1, "256": 1.2})
    use_tie_breaker: bool = None

@dataclasses.dataclass
class Yolov3regular(YoloCFG):
    model: str = "regular"
    backbone: Optional[Dict] = None #"regular"
    head: Optional[Dict] = None #"regular"
    head_filter: YoloLayer = YoloLayer()
    _boxes: List[str] = dataclasses.field(default_factory=lambda:["10, 13", "16, 30", "33, 23", "30, 61", "62, 45", "59, 119", "116, 90" ,"156, 198", "373, 326"])
    masks: Dict = dataclasses.field(default_factory=lambda: {"1024": [6, 7, 8], "512": [3, 4, 5], "256": [0, 1, 2]})
    path_scales: Dict = dataclasses.field(default_factory=lambda: {"1024": 32, "512": 16, "256": 8})
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {"1024": 1.0, "512": 1.0, "256": 1.0})
    use_tie_breaker: bool = None

@dataclasses.dataclass
class Yolov3spp(YoloCFG):
    model: str = "spp"
    backbone: Optional[Dict] = None #"regular"
    head: Optional[Dict] = None #"spp"
    head_filter: YoloLayer = YoloLayer()
    _boxes: List[str] = dataclasses.field(default_factory=lambda:["10, 13", "16, 30", "33, 23", "30, 61", "62, 45", "59, 119", "116, 90" ,"156, 198", "373, 326"])
    masks: Dict = dataclasses.field(default_factory=lambda: {"1024": [6, 7, 8], "512": [3, 4, 5], "256": [0, 1, 2]})
    path_scales: Dict = dataclasses.field(default_factory=lambda: {"1024": 32, "512": 16, "256": 8})
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {"1024": 1.0, "512": 1.0, "256": 1.0})
    use_tie_breaker: bool = None

@dataclasses.dataclass
class Yolov3tiny(YoloCFG):
    model: str = "tiny"
    backbone: Optional[Dict] = None #"tiny"
    head: Optional[Dict] = None #"tiny"
    head_filter: YoloLayer = YoloLayer()
    _boxes: List[str] = dataclasses.field(default_factory=lambda:["10, 14", "23, 27", "37, 58","81, 82", "135, 169", "344, 319"])
    masks: Dict = dataclasses.field(default_factory=lambda: {"1024": [3, 4, 5], "256": [0, 1, 2]})
    path_scales: Dict = dataclasses.field(default_factory=lambda: {"1024": 32, "256": 8})
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {"1024": 1.0, "256": 1.0})
    use_tie_breaker: bool = None

@dataclasses.dataclass
class Yolo(hyperparams.OneOfConfig):
    type: str = "v4"
    v3: Yolov3regular = Yolov3regular()
    v3_spp: Yolov3spp = Yolov3spp()
    v3_tiny: Yolov3tiny = Yolov3tiny()
    v4: Yolov4regular = Yolov4regular()

# model task
@dataclasses.dataclass
class YoloTask(cfg.TaskConfig):
    _input_size: Optional[List[int]] = None
    model:Yolo = Yolo()
    loss:YoloLoss = YoloLoss()
    train_data: DataConfig = DataConfig(is_training=True)
    validation_data: DataConfig = DataConfig(is_training=False)
    num_classes: int = 80
    min_level: int = 3
    max_level: int = 5
    weight_decay: float = 5e-4
    gradient_clip_norm: float = 0.0

    init_checkpoint_modules: str = 'all'  # all or backbone
    init_checkpoint: Optional[str] = None
    annotation_file: Optional[str] = None
    per_category_metrics = False

    load_original_weights: bool = True
    backbone_from_darknet: bool = True
    head_from_darknet: bool = False

    @property
    def input_size(self):
        if self._input_size == None:
            return [None, None, 3]
        else:
            return self._input_size
    
    @input_size.setter 
    def input_size(self, input_size):
        self._input_size = input_size



    

