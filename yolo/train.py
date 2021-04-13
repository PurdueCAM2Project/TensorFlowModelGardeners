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
"""TensorFlow Model Garden Vision training driver."""
from yolo.utils.run_utils import prep_gpu
try:
  prep_gpu()
except BaseException:
  print('GPUs ready')

from absl import app
from absl import flags
import gin
import sys

from official.core import train_utils
# pylint: disable=unused-import
from yolo.common import registry_imports
# pylint: enable=unused-import
from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import task_factory
from official.core import train_lib
from official.modeling import performance
import pprint

FLAGS = flags.FLAGS
"""
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
"""
"""
get the cache file:
scp -i ./jaeyounkim-purdue-1 cache.zip  purdue@34.105.118.198:~/

train darknet:
python3 -m yolo.train --mode=train_and_eval --experiment=darknet_classification --model_dir=../checkpoints/darknet53 --config_file=yolo/configs/experiments/darknet53.yaml
python3.8 -m yolo.train --mode=train_and_eval --experiment=darknet_classification --model_dir=../checkpoints/dilated_darknet53 --config_file=yolo/configs/experiments/dilated_darknet53.yaml

finetune darknet:
nohup python3 -m yolo.train --mode=train_and_eval --experiment=darknet_classification --model_dir=../checkpoints/darknet53_remap_fn --config_file=yolo/configs/experiments/darknet53_leaky_fn_tune.yaml >> darknet53.log & tail -f darknet53.log

train yolo-v4:
nohup python3 -m yolo.train --mode=train_and_eval --experiment=yolo_custom --model_dir=../checkpoints/yolov4- --config_file=yolo/configs/experiments/yolov4.yaml  >> yolov4.log & tail -f yolov4.log

nohup python3.8 -m yolo.train --mode=train_and_eval --experiment=yolo_subdiv_custom --model_dir=../checkpoints/yolov4_subdiv64 --config_file=yolo/configs/experiments/yolov4_subdiv.yaml  >> yolov4.log & tail -f yolov4.log
nohup python3 -m yolo.train --mode=train_and_eval --experiment=yolo_custom --model_dir=../checkpoints/yolov4- --config_file=yolo/configs/experiments/yolov4-1gpu.yaml  >> yolov4-1gpu.log & tail -f yolov4-1gpu.log

nohup python3.8 -m yolo.train --mode=train_and_eval --experiment=yolo_custom --model_dir=../checkpoints/yolov3-1gpu_mosaic --config_file=yolo/configs/experiments/yolov3-1gpu.yaml  >> yolov3-1gpu.log & tail -f yolov3-1gpu.log

evalaute Yolo:
nohup python3.8 -m yolo.train --mode=train_and_eval --experiment=yolo_custom --model_dir=../checkpoints/yolov4- --config_file=yolo/configs/experiments/yolov4-eval.yaml  >> yolov4-eval.log & tail -f yolov4-eval.log
"""


def subdivison_adjustment(params):

  if hasattr(params.task.model,
             'subdivisions') and params.task.model.subdivisions > 1:
    print('adjustment is needed')
    subdivisons = params.task.model.subdivisions
    params.task.train_data.global_batch_size //= subdivisons
    # params.task.validation_data.global_batch_size //= subdivisons
    params.trainer.train_steps *= subdivisons
    # params.trainer.validation_steps *= subdivisons
    params.trainer.validation_interval = (params.trainer.validation_interval //
                                          subdivisons) * subdivisons
    params.trainer.checkpoint_interval = (params.trainer.checkpoint_interval //
                                          subdivisons) * subdivisons
    params.trainer.steps_per_loop = (params.trainer.steps_per_loop //
                                     subdivisons) * subdivisons
    params.trainer.summary_interval = (params.trainer.summary_interval //
                                       subdivisons) * subdivisons

    if params.trainer.optimizer_config.learning_rate.type == 'stepwise':
      bounds = params.trainer.optimizer_config.learning_rate.stepwise.boundaries
      params.trainer.optimizer_config.learning_rate.stepwise.boundaries = [
          subdivisons * bound for bound in bounds
      ]

    if params.trainer.optimizer_config.learning_rate.type == 'polynomial':
      params.trainer.optimizer_config.learning_rate.polynomial.decay_steps *= subdivisons

    if params.trainer.optimizer_config.optimizer.type == 'sgd':
      print(params.trainer.optimizer_config.optimizer.type)
      params.trainer.optimizer_config.optimizer.type = 'sgd_accum'
      params.trainer.optimizer_config.optimizer.sgd_accum.accumulation_steps = subdivisons
      params.trainer.optimizer_config.optimizer.sgd_accum.momentum = params.trainer.optimizer_config.optimizer.sgd.momentum
      params.trainer.optimizer_config.optimizer.sgd_accum.decay = params.trainer.optimizer_config.optimizer.sgd.decay

    if params.trainer.optimizer_config.warmup.type == 'linear':
      params.trainer.optimizer_config.warmup.linear.warmup_steps *= subdivisons

  print(params.as_dict())
  # sys.exit()
  return params


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  print(FLAGS.experiment)
  params = train_utils.parse_configuration(FLAGS)

  params = subdivison_adjustment(params)
  model_dir = FLAGS.model_dir
  if 'train' in FLAGS.mode and model_dir != None:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
  # dtype is float16
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype,
                                           params.runtime.loss_scale)
  if params.runtime.worker_hosts != '' and params.runtime.worker_hosts is not None:
    num_workers = distribute_utils.configure_cluster(
        worker_hosts=params.runtime.worker_hosts,
        task_index=params.runtime.task_index)
    print(num_workers)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)

  with distribution_strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)

  train_lib.run_experiment(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      model_dir=model_dir)


if __name__ == '__main__':
  import datetime

  a = datetime.datetime.now()
  tfm_flags.define_flags()
  app.run(main)
  b = datetime.datetime.now()

  print('\n\n\n\n\n\n\n {b - a}')
