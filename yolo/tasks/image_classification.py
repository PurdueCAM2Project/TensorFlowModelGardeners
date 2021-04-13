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
"""Image classification task definition."""
import tensorflow as tf
from official.core import input_reader
from official.core import task_factory
from yolo.configs import darknet_classification as exp_cfg
from yolo.dataloaders.decoders import classification_tfds_decoder as cli
from yolo.dataloaders import classification_input
from yolo.dataloaders import classification_vision
from official.vision.beta.tasks import image_classification
from yolo.losses import cross_entropy_loss
from official.modeling import tf_utils
import logging


@task_factory.register_task_cls(exp_cfg.ImageClassificationTask)
class ImageClassificationTask(image_classification.ImageClassificationTask):
  """A task for image classification."""

  def initialize(self, model: tf.keras.Model):

    if self.task_config.load_darknet_weights:
      from yolo.utils import DarkNetConverter
      from yolo.utils._darknet2tf.load_weights import split_converter
      from yolo.utils._darknet2tf.load_weights2 import load_weights_backbone
      from yolo.utils._darknet2tf.load_weights2 import load_weights_prediction_layers
      from yolo.utils.downloads.file_manager import download

      weights_file = self.task_config.model.darknet_weights_file
      config_file = self.task_config.model.darknet_weights_cfg

      if ('cache://' not in weights_file and 'cache://' not in config_file):
        list_encdec = DarkNetConverter.read(config_file, weights_file)
      else:
        import os
        path = os.path.abspath('cache')
        if (not os.path.isdir(path)):
          os.mkdir(path)

        cfg = f"{path}/cfg/{config_file.split('/')[-1]}"
        if not os.path.isfile(cfg):
          download(config_file.split('/')[-1])

        wgt = f"{path}/weights/{weights_file.split('/')[-1]}"
        if not os.path.isfile(wgt):
          download(weights_file.split('/')[-1])

        list_encdec = DarkNetConverter.read(cfg, wgt)

      splits = model.backbone._splits
      if 'neck_split' in splits.keys():
        encoder, decoder, _ = split_converter(list_encdec,
                                              splits['backbone_split'],
                                              splits['neck_split'])
      else:
        encoder, decoder = split_converter(list_encdec,
                                           splits['backbone_split'])
        neck = None

      load_weights_backbone(model.backbone, encoder)
    else:
      """Loading pretrained checkpoint."""
      if not self.task_config.init_checkpoint:
        return

      ckpt_dir_or_file = self.task_config.init_checkpoint
      if tf.io.gfile.isdir(ckpt_dir_or_file):
        ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

      # Restoring checkpoint.
      if self.task_config.init_checkpoint_modules == 'all':
        ckpt = tf.train.Checkpoint(**model.checkpoint_items)
        status = ckpt.restore(ckpt_dir_or_file)
        #status.assert_consumed()
        status.expect_partial().assert_existing_objects_matched()
      elif self.task_config.init_checkpoint_modules == 'backbone':
        ckpt = tf.train.Checkpoint(backbone=model.backbone)
        status = ckpt.restore(ckpt_dir_or_file)
        status.expect_partial().assert_existing_objects_matched()
      else:
        assert "Only 'all' or 'backbone' can be used to initialize the model."

      logging.info('Finished loading pretrained checkpoint from %s',
                   ckpt_dir_or_file)
      # model.backbone.trainable = False
      # model.head.trainable = False

  def build_losses(self, labels, model_outputs, aux_losses=None):
    """Sparse categorical cross entropy loss.

    Args:
      labels: labels.
      model_outputs: Output logits of the classifier.
      aux_losses: auxiliarly loss tensors, i.e. `losses` in keras.Model.

    Returns:
      The total loss tensor.
    """
    losses_config = self.task_config.losses
    if losses_config.one_hot:
      total_loss = tf.keras.losses.categorical_crossentropy(
          labels,
          model_outputs,
          from_logits=True,
          label_smoothing=losses_config.label_smoothing)
      # total_loss = cross_entropy_loss.ce_loss(labels, model_outputs,
      #                                         losses_config.label_smoothing)
      #total_loss = tf.math.reduce_sum(total_loss)
    else:
      total_loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, model_outputs, from_logits=False)
    total_loss = tf_utils.safe_mean(total_loss)
    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    return total_loss

  def build_metrics(self, training=True):
    """Gets streaming metrics for training/validation."""
    k = 5  #self.task_config.evaluation.top_k
    if self.task_config.losses.one_hot:
      metrics = [
          tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
          tf.keras.metrics.TopKCategoricalAccuracy(
              k=k, name='top_{}_accuracy'.format(k))
      ]
    else:
      metrics = [
          tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
          tf.keras.metrics.SparseTopKCategoricalAccuracy(
              k=k, name='top_{}_accuracy'.format(k))
      ]
    return metrics

  def build_inputs(self, params, input_context=None):
    """Builds classification input."""

    num_classes = self.task_config.model.num_classes
    input_size = self.task_config.model.input_size

    if params.tfds_name is not None:
      decoder = cli.Decoder()
    else:
      decoder = classification_input.Decoder()

    parser = classification_input.Parser(
        output_size=input_size[:2],
        num_classes=num_classes,
        aug_rand_saturation=params.parser.aug_rand or
        params.parser.aug_rand_saturation,
        aug_rand_brightness=params.parser.aug_rand or
        params.parser.aug_rand_brightness,
        aug_rand_zoom=params.parser.aug_rand or params.parser.aug_rand_zoom,
        aug_rand_rotate=params.parser.aug_rand or params.parser.aug_rand_rotate,
        aug_rand_hue=params.parser.aug_rand or params.parser.aug_rand_hue,
        aug_rand_aspect=params.parser.aug_rand or params.parser.aug_rand_aspect,
        scale=params.parser.scale,
        seed=params.parser.seed,
        dtype=params.dtype)

    # parser = classification_vision.Parser(
    #   output_size = input_size[:2],
    #   aug_policy = 'randaug',
    #   dtype=params.dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)
    return dataset

  def train_step(self, inputs, model, optimizer, metrics=None):
    """Does forward and backward.
    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.
    Returns:
      A dictionary of logs.
    """
    features, labels = inputs
    if self.task_config.losses.one_hot:
      labels = tf.one_hot(labels, self.task_config.model.num_classes)

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      #tf.print(tf.argmax(outputs, axis = -1), tf.argmax(labels, axis = -1))
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      loss = self.build_losses(
          model_outputs=outputs, labels=labels, aux_losses=model.losses)
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer,
                    tf.keras.mixed_precision.experimental.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(optimizer,
                  tf.keras.mixed_precision.experimental.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)

    # Apply gradient clipping.
    if self.task_config.gradient_clip_norm > 0:
      grads, _ = tf.clip_by_global_norm(grads,
                                        self.task_config.gradient_clip_norm)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})

    # tf.print(logs, end='\r')

    # ret = '\033[F' * (len(logs.keys()))
    # tf.print(ret, end='\n')
    return logs
