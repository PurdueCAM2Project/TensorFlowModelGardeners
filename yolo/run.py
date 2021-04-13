from yolo.utils.run_utils import prep_gpu
# try:
#
# except BaseException:
#   print("GPUs ready")

from absl import app
from absl import flags
import gin
import sys

from official.core import train_utils
# pylint: disable=unused-import
from yolo.common import registry_imports
# pylint: enable=unused-import
from official.common import flags as tfm_flags

from typing import Tuple, List
from official.core import train_utils
from official.modeling import performance
from official.core import task_factory
import os

from yolo.demos import video_detect_gpu as vgu
from yolo.demos import video_detect_cpu as vcu
import tensorflow as tf
"""
export GOOGLE_APPLICATION_CREDENTIALS=<key>.json 
python3.8 -m yolo.run --experiment=yolo_custom --out_resolution 416 --config_file=yolo/configs/experiments/yolov4-eval.yaml --video ../videos/nyc.mp4  --max_batch 5 
"""
"""
python3.8 -m yolo.run --experiment=yolo_custom --out_resolution 416 --config_file=yolo/configs/experiments/yolov3-eval.yaml --video ../videos/nyc.mp4  --max_batch 9
"""
"""
python3.8 -m yolo.run --experiment=yolo_custom --out_resolution 416 --config_file=yolo/configs/experiments/yolov4-tiny-eval.yaml --video ../videos/nyc.mp4  --max_batch 9
"""

FLAGS = flags.FLAGS


def define_flags():
  """Defines flags."""
  flags.DEFINE_bool('gpu', default=True, help='The experiment type registered.')

  flags.DEFINE_string(
      'experiment', default=None, help='The experiment type registered.')

  flags.DEFINE_string(
      'model_dir',
      default=None,
      help='The directory where the model and training/evaluation summaries'
      'are stored.')

  flags.DEFINE_multi_string(
      'config_file',
      default=None,
      help='YAML/JSON files which specifies overrides. The override order '
      'follows the order of args. Note that each file '
      'can be used as an override template to override the default parameters '
      'specified in Python. If the same parameter is specified in both '
      '`--config_file` and `--params_override`, `config_file` will be used '
      'first, followed by params_override.')

  flags.DEFINE_string(
      'params_override',
      default=None,
      help='a YAML/JSON string or a YAML file which specifies additional '
      'overrides over the default parameters and those specified in '
      '`--config_file`. Note that this is supposed to be used only to override '
      'the model parameters, but not the parameters like TPU specific flags. '
      'One canonical use case of `--config_file` and `--params_override` is '
      'users first define a template config file using `--config_file`, then '
      'use `--params_override` to adjust the minimal set of tuning parameters, '
      'for example setting up different `train_batch_size`. The final override '
      'order of parameters: default_model_params --> params from config_file '
      '--> params in params_override. See also the help message of '
      '`--config_file`.')

  flags.DEFINE_string(
      'tpu',
      default=None,
      help='The Cloud TPU to use for training. This should be either the name '
      'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
      'url.')

  flags.DEFINE_string(
      'tf_data_service', default=None, help='The tf.data service address')

  flags.DEFINE_string('video', default=None, help='path to video to run on')

  flags.DEFINE_bool(
      'preprocess_gpu', default=False, help='preprocess on the gpu')

  flags.DEFINE_bool('print_conf', default=True, help='preprocess on the gpu')

  flags.DEFINE_integer(
      'process_size', default=416, help='preprocess on the gpu')

  flags.DEFINE_integer('max_batch', default=None, help='preprocess on the gpu')

  flags.DEFINE_float('wait_time', default=None, help='preprocess on the gpu')

  flags.DEFINE_integer(
      'out_resolution', default=416, help='preprocess on the gpu')

  flags.DEFINE_integer('scale_que', default=1, help='preprocess on the gpu')


def load_model(experiment='yolo_custom', config_path=[], model_dir=''):
  CFG = train_utils.ParseConfigOptions(
      experiment=experiment, config_file=config_path)
  params = train_utils.parse_configuration(CFG)

  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype,
                                           params.runtime.loss_scale)

  task = task_factory.get_task(params.task, logging_dir=model_dir)
  model = task.build_model()

  if model_dir is not None and model_dir != '':
    optimizer = task.create_optimizer(params.trainer.optimizer_config,
                                      params.runtime)
    # optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.SGD(), dynamic = True)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    status = ckpt.restore(tf.train.latest_checkpoint(model_dir))

    # status.expect_partial().assert_existing_objects_matched()
    print(dir(status), status)
  else:
    task.initialize(model)

  return task, model, params


def load_flags(CFG):
  params = train_utils.parse_configuration(CFG)
  model_dir = CFG.model_dir

  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype,
                                           params.runtime.loss_scale)

  task = task_factory.get_task(params.task, logging_dir=model_dir)
  model = task.build_model()

  if model_dir is not None and model_dir != '':
    optimizer = task.create_optimizer(params.trainer.optimizer_config,
                                      params.runtime)
    # optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.SGD(), dynamic = True)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    status = ckpt.restore(tf.train.latest_checkpoint(model_dir))

    # try:
    #   status.expect_partial().assert_existing_objects_matched()
    # except:
    #   print("this checkpoint could not assert all components consumed, componnets may not match")
    print(dir(status), status)
  else:
    task.initialize(model)

  return task, model, params


def main(_):
  prep_gpu()
  task, model, params = load_flags(FLAGS)

  if FLAGS.gpu:
    cap = vgu.FastVideo(
        FLAGS.video,
        model=model,
        process_width=FLAGS.process_size,
        process_height=FLAGS.process_size,
        preprocess_with_gpu=FLAGS.preprocess_gpu,
        classes=params.task.model.num_classes,
        print_conf=FLAGS.print_conf,
        max_batch=FLAGS.max_batch,
        disp_h=FLAGS.out_resolution,
        scale_que=FLAGS.scale_que,
        wait_time=FLAGS.wait_time)
    cap.run()
  else:
    vcu.runner(model, FLAGS.video, FLAGS.process_size, FLAGS.out_resolution)


if __name__ == '__main__':
  import datetime

  a = datetime.datetime.now()
  define_flags()
  app.run(main)
  b = datetime.datetime.now()

  print('\n\n\n\n\n\n\n {b - a}')
