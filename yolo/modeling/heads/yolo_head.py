import tensorflow as tf
from yolo.modeling.layers import nn_blocks


class YoloHead(tf.keras.layers.Layer):

  def __init__(self,
               min_level,
               max_level,
               classes=80,
               boxes_per_level=3,
               output_extras=0,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer="glorot_uniform",
               subdivisions=8,
               kernel_regularizer=None,
               bias_regularizer=None,
               activation=None,
               **kwargs):
    super().__init__(**kwargs)
    self._min_level = min_level
    self._max_level = max_level

    self._key_list = [
        str(key) for key in range(self._min_level, self._max_level + 1)
    ]

    self._classes = classes
    self._boxes_per_level = boxes_per_level
    self._output_extras = output_extras

    self._output_conv = (classes + output_extras + 5) * boxes_per_level

    self._base_config = dict(
        filters=self._output_conv,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bn=False,
        activation=activation,
        subdivisions=subdivisions,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)

  def build(self, input_shape):
    self._head = dict()
    for key in self._key_list:
      self._head[key] = nn_blocks.ConvBN(**self._base_config)
    # super().build(input_shape)

  def call(self, inputs):
    outputs = dict()
    for key in self._key_list:
      outputs[key] = self._head[key](inputs[key])
    return outputs

  @property
  def output_depth(self):
    return (self._classes + self._output_extras + 5) * self._boxes_per_level

  @property
  def num_boxes(self):
    if self._min_level is None or self._max_level is None:
      raise Exception(
          "model has to be built before number of boxes can be determined")
    return (self._max_level - self._min_level + 1) * self._boxes_per_level

  def get_config(self):
    config = dict(
        classes=self._classes,
        boxes_per_level=self._boxes_per_level,
        output_extras=self._output_extras,
        xy_exponential=self._xy_exponential,
        exp_base=self._exp_base,
        xy_scale_base=self._xy_scale_base,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
    )
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
