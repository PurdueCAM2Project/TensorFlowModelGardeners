# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Normalization layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import distribution_strategy_context as ds
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.distribute import reduce_util
from tensorflow.python.keras.layers import normalization

import tensorflow as tf


class SubDivBatchNormalization(normalization.BatchNormalizationBase):

  _USE_V2_BEHAVIOR = True

  def __init__(self,
               axis=-1,
               subdivisions=1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               fused=None,
               trainable=True,
               adjustment=None,
               name=None,
               **kwargs):

    super(SubDivBatchNormalization, self).__init__(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=None
        if not renorm else False,  #if subdivisions <= 1 else False, #alter this
        trainable=trainable,
        virtual_batch_size=None,
        name=name,
        **kwargs)

    self.subdivisions = subdivisions

  def build(self, input_shape):
    super().build(input_shape)
    input_shape = tensor_shape.TensorShape(input_shape)
    if not input_shape.ndims:
      raise ValueError('Input has undefined rank:', input_shape)
    ndims = len(input_shape)

    # Convert axis to list and resolve negatives
    if isinstance(self.axis, int):
      self.axis = [self.axis]

    for idx, x in enumerate(self.axis):
      if x < 0:
        self.axis[idx] = ndims + x

    # Validate axes
    for x in self.axis:
      if x < 0 or x >= ndims:
        raise ValueError('Invalid axis: %d' % x)
    if len(self.axis) != len(set(self.axis)):
      raise ValueError('Duplicate axis: %s' % self.axis)

    # TODO(yaozhang): if input is not 4D, reshape it to 4D and reshape the
    # output back to its original shape accordingly.
    # self.fused = None

    if self.fused:
      if self._USE_V2_BEHAVIOR:
        # TODO(b/173253101): Using fused in the 5D case is currently disabled
        # due to a regression on UNet, so it is only currently only supported in
        # the 4D case.
        if self.fused is None:
          self.fused = ndims == 4
        elif self.fused and ndims != 4:
          raise ValueError('Batch normalization layers with `fused=True` only '
                           'support 4D or 5D input tensors. '
                           'Received tensor with shape: %s' %
                           (tuple(input_shape),))
      else:
        assert self.fused is not None
        self.fused = (ndims == 4 and self._fused_can_be_used())
      # TODO(chrisying): fused batch norm is currently not supported for
      # multi-axis batch norm and by extension virtual batches. In some cases,
      # it might be possible to use fused batch norm but would require reshaping
      # the Tensor to 4D with the axis in 1 or 3 (preferred 1) which is
      # particularly tricky. A compromise might be to just support the most
      # common use case (turning 5D w/ virtual batch to NCHW)

      if self.axis == [1] and ndims == 4:
        self._data_format = 'NCHW'
      elif self.axis == [1] and ndims == 5:
        self._data_format = 'NCDHW'
      elif self.axis == [3] and ndims == 4:
        self._data_format = 'NHWC'
      elif self.axis == [4] and ndims == 5:
        self._data_format = 'NDHWC'
      elif ndims == 5:
        # 5D tensors that can be passed in but should not use fused batch norm
        # due to unsupported axis.
        self.fused = False
      else:
        raise ValueError('Unsupported axis, fused batch norm only supports '
                         'axis == [1] or axis == [3] for 4D input tensors or '
                         'axis == [1] or axis == [4] for 5D input tensors')

    axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
    for x in axis_to_dim:
      if axis_to_dim[x] is None:
        raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                         input_shape)
    self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

    # get the shape for my weights based on input shape
    if len(axis_to_dim) == 1 and self.virtual_batch_size is None:
      # Single axis batch norm (most common/default use-case)
      param_shape = (list(axis_to_dim.values())[0],)
    else:
      # Parameter shape is the original shape but with 1 in all non-axis dims
      param_shape = [
          axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)
      ]
      if self.virtual_batch_size is not None:
        # When using virtual batches, add an extra dim at index 1
        param_shape.insert(1, 1)
        for idx, x in enumerate(self.axis):
          self.axis[idx] = x + 1  # Account for added dimension

    try:
      # Disable variable partitioning when creating the moving mean and variance
      if hasattr(self, '_scope') and self._scope:
        partitioner = self._scope.partitioner
        self._scope.set_partitioner(None)
      else:
        partitioner = None

      if self.subdivisions > 1:
        self.aggregated_sum_batch = self.add_weight(
            name='agg_sum',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.moving_mean_initializer,
            synchronization=tf_variables.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.SUM,
            experimental_autocast=False)

        self.aggregated_square_sum_batch = self.add_weight(
            name='agg_square_sum',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.moving_variance_initializer,
            synchronization=tf_variables.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.SUM,
            experimental_autocast=False)

        self.local_count = self.add_weight(
            name='local_sum',
            shape=(),
            dtype=tf.int32,
            initializer=tf.zeros_initializer(),
            synchronization=tf_variables.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.SUM,
            experimental_autocast=False)

        self.aggregated_batch_size = self.add_weight(
            name='net_batches',
            shape=(),
            dtype=tf.int32,
            initializer=tf.zeros_initializer(),
            synchronization=tf_variables.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.SUM,
            experimental_autocast=False)

    finally:
      if partitioner:
        self._scope.set_partitioner(partitioner)
    self.built = True

  def _assign_subdiv_moving_average(self, variable, value, momentum,
                                    subdivsions, count):
    with K.name_scope('AssignSubDivMovingAvg') as scope:
      with ops.colocate_with(variable):
        decay = ops.convert_to_tensor_v2_with_dispatch(
            1.0 - momentum, name='decay')
        if decay.dtype != variable.dtype.base_dtype:
          decay = math_ops.cast(decay, variable.dtype.base_dtype)

        # get the aggregated update
        update_delta = (variable - math_ops.cast(value, variable.dtype)) * decay

        # update at the end of last step
        update_delta = array_ops.where((count + 1) % subdivisions == 0,
                                       update_delta, K.zeros_like(update_delta))
        return state_ops.assign_sub(variable, update_delta, name=scope)

  def _assign_subdiv_new_value(self, variable, value, subdivisions, count):
    with K.name_scope('AssignNewValue') as scope:
      with ops.colocate_with(variable):
        update_value = array_ops.where((count + 1) % subdivisions == 0, value,
                                       variable)
        return state_ops.assign(variable, update_value, name=scope)

  def _assign_subdiv_rotating_sum(self, variable, value, subdivisions, count,
                                  inputs_size):
    with K.name_scope('AssignSubDivRotatedSum') as scope:
      with ops.colocate_with(variable):
        # reduce it for the current
        update_delta = value  #/subdivisions

        # if the input size is 0
        if inputs_size is not None:
          update_delta = array_ops.where(inputs_size > 0, update_delta,
                                         K.zeros_like(update_delta))

        # if we are starting a new batch set the variable to 0 by removing it
        # from update delta then add the delta to the variable to get
        # rid of the value variable
        update_delta = array_ops.where(count % subdivisions == 0,
                                       update_delta - variable, update_delta)
        return state_ops.assign_add(variable, update_delta, name=scope)

  def _subdiv_calculate_mean_and_var(self, inputs, reduction_axes, keep_dims):
    # calculate the
    net_sum = math_ops.reduce_sum(
        inputs, axis=reduction_axes, keepdims=keep_dims)
    squared_mean = math_ops.reduce_sum(
        math_ops.square(inputs), axis=reduction_axes, keepdims=keep_dims)

    if self._support_zero_size_input():
      # Keras assumes that batch dimension is the first dimension for Batch
      # Normalization.
      input_batch_size = array_ops.shape(inputs)[0]
    else:
      input_batch_size = None

    # get the number of total params you are averaging including batchsize(local)
    axes_vals = [
        (array_ops.shape_v2(inputs))[i] for i in range(1, len(reduction_axes))
    ]
    multiplier = math_ops.cast(math_ops.reduce_prod(axes_vals), dtypes.float32)

    squared_mean = squared_mean / multiplier
    net_sum = net_sum / multiplier

    if input_batch_size is None:
      mean, variance = nn.moments(inputs, reduction_axes, keep_dims=keep_dims)
    else:
      batches_ = math_ops.cast(input_batch_size, self._param_dtype)
      mean = net_sum / batches_
      variance = squared_mean / batches_ - math_ops.square(
          array_ops.stop_gradient(mean))

    return mean, net_sum, variance, squared_mean, input_batch_size

  def subdiv_moments(self, inputs, reduction_axes, keep_dims):
    # mean and variance only for the current batch
    mean, net_sum, variance, squared_mean, input_batch_size = self._subdiv_calculate_mean_and_var(
        inputs, reduction_axes, keep_dims)

    if self._support_zero_size_input():
      input_batch_size = 0 if input_batch_size is None else input_batch_size
      mean = array_ops.where(input_batch_size > 0, mean, K.zeros_like(mean))
      net_sum = array_ops.where(input_batch_size > 0, net_sum,
                                K.zeros_like(net_sum))
      variance = array_ops.where(input_batch_size > 0, variance,
                                 K.zeros_like(variance))
      squared_mean = array_ops.where(input_batch_size > 0, squared_mean,
                                     K.zeros_like(squared_mean))
    return mean, net_sum, variance, squared_mean, input_batch_size

  def _subdiv_batch_norm(self, inputs, training=None):
    # tf.print('bn', self.local_count)
    training = self._get_training_value(training)

    inputs_dtype = inputs.dtype.base_dtype
    if inputs_dtype in (dtypes.float16, dtypes.bfloat16):
      # Do all math in float32 if given 16-bit inputs for numeric stability.
      # In particular, it's very easy for variance to overflow in float16 and
      # for safety we also choose to cast bfloat16 to float32.
      inputs = math_ops.cast(inputs, dtypes.float32)

    params_dtype = self._param_dtype

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]
    if self.virtual_batch_size is not None:
      del reduction_axes[1]  # Do not reduce along virtual batch dim

    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

    def _broadcast(v):
      if (v is not None and len(v.shape) != ndims and
          reduction_axes != list(range(ndims - 1))):
        return array_ops.reshape(v, broadcast_shape)
      return v

    scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

    # what does this do...
    def _compose_transforms(scale, offset, then_scale, then_offset):
      if then_scale is not None:
        scale *= then_scale
        offset *= then_scale
      if then_offset is not None:
        offset += then_offset
      return (scale, offset)

    # is training value true false or None
    training_value = control_flow_util.constant_value(training)
    update_value = (self.local_count + 1) % self.subdivisions == 0
    if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
      mean, variance = self.moving_mean, self.moving_variance
    else:
      # training_value could be True or None -> None means determine at runtime
      if self.adjustment:
        adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
        # Adjust only during training.
        adj_scale = control_flow_util.smart_cond(
            training, lambda: adj_scale, lambda: array_ops.ones_like(adj_scale))
        adj_bias = control_flow_util.smart_cond(
            training, lambda: adj_bias, lambda: array_ops.zeros_like(adj_bias))
        scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

      keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1

      # normalization stats for the current batch important = mean and squared_mean
      mean, net_sum, variance, squared_mean, input_batch_size = self.subdiv_moments(
          math_ops.cast(inputs, self._param_dtype),
          reduction_axes,
          keep_dims=keep_dims)

      # aggregate the things
      def _update_aggragate_sum():
        return self._assign_subdiv_rotating_sum(self.aggregated_sum_batch,
                                                net_sum, self.subdivisions,
                                                self.local_count,
                                                input_batch_size)

      def _update_aggragate_squared_sum():
        return self._assign_subdiv_rotating_sum(
            self.aggregated_square_sum_batch, squared_mean, self.subdivisions,
            self.local_count, input_batch_size)

      def _update_aggragate_batch_size():
        return self._assign_subdiv_rotating_sum(self.aggregated_batch_size,
                                                input_batch_size,
                                                self.subdivisions,
                                                self.local_count,
                                                input_batch_size)

      self.add_update(_update_aggragate_sum)
      self.add_update(_update_aggragate_squared_sum)
      self.add_update(_update_aggragate_batch_size)

      aggregated_mean = self.aggregated_sum_batch / math_ops.cast(
          self.aggregated_batch_size, params_dtype)
      aggregated_squared_mean = self.aggregated_square_sum_batch / math_ops.cast(
          self.aggregated_batch_size, params_dtype)
      aggregated_variance = aggregated_squared_mean - math_ops.square(
          aggregated_mean)

      moving_mean = self.moving_mean
      moving_variance = self.moving_variance

      # if we are training use the stats for this batch for normalizing this
      # value other wise use the moving average

      # should only happen when we update the moving values
      mean = control_flow_util.smart_cond(
          training,
          true_fn=lambda: mean,
          false_fn=lambda: ops.convert_to_tensor_v2_with_dispatch(moving_mean))
      variance = control_flow_util.smart_cond(
          training,
          true_fn=lambda: variance,
          false_fn=lambda: ops.convert_to_tensor_v2_with_dispatch(
              moving_variance))

      # circular update of the mean and variance
      new_mean = control_flow_util.smart_cond(
          update_value,
          true_fn=lambda: ops.convert_to_tensor_v2_with_dispatch(aggregated_mean
                                                                ),
          false_fn=lambda: moving_mean)

      new_variance = control_flow_util.smart_cond(
          update_value,
          true_fn=lambda: ops.convert_to_tensor_v2_with_dispatch(
              aggregated_variance),
          false_fn=lambda: moving_variance)

      # # should only be done when the moving mean is updated
      # tf.print(new_variance, self.local_count, update_value, self.aggregated_batch_size, self.aggregated_sum_batch)

      if self.renorm:
        r, d, new_mean, new_variance = self._renorm_correction_and_moments(
            new_mean, new_variance, training, input_batch_size)
        # When training, the normalized values (say, x) will be transformed as
        # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
        # = x * (r * gamma) + (d * gamma + beta) with renorm.
        r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
        d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
        scale, offset = _compose_transforms(r, d, scale, offset)

      def _do_update(var, value):
        """Compute the updates for mean and variance."""
        return self._assign_moving_average(var, value, self.momentum,
                                           self.aggregated_batch_size)

      def mean_update():
        true_branch = lambda: _do_update(self.moving_mean, new_mean)
        false_branch = lambda: self.moving_mean
        return control_flow_util.smart_cond(training, true_branch, false_branch)

      def variance_update():
        """Update the moving variance."""

        def true_branch_renorm():
          # We apply epsilon as part of the moving_stddev to mirror the training
          # code path.
          moving_stddev = _do_update(self.moving_stddev,
                                     math_ops.sqrt(new_variance + self.epsilon))
          return self._assign_new_value(
              self.moving_variance,
              # Apply relu in case floating point rounding causes it to go
              # negative.
              K.relu(moving_stddev * moving_stddev - self.epsilon))

        if self.renorm:
          true_branch = true_branch_renorm
        else:
          true_branch = lambda: _do_update(self.moving_variance, new_variance)

        false_branch = lambda: self.moving_variance
        return control_flow_util.smart_cond(training, true_branch, false_branch)

      def update_count():
        with K.name_scope('update_count') as scope:
          # update the local count
          return state_ops.assign_add(
              self.local_count, tf.cast(1, self.local_count.dtype), name=scope)

      self.add_update(mean_update)
      self.add_update(variance_update)
      self.add_update(update_count)

    mean = math_ops.cast(mean, inputs.dtype)
    variance = math_ops.cast(variance, inputs.dtype)
    if offset is not None:
      offset = math_ops.cast(offset, inputs.dtype)
    if scale is not None:
      scale = math_ops.cast(scale, inputs.dtype)
    outputs = nn.batch_normalization(inputs, _broadcast(mean),
                                     _broadcast(variance), offset, scale,
                                     self.epsilon)
    if inputs_dtype in (dtypes.float16, dtypes.bfloat16):
      outputs = math_ops.cast(outputs, inputs_dtype)

    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    if self.virtual_batch_size is not None:
      outputs = undo_virtual_batching(outputs)
    return outputs

  def call(self, inputs, training=None):
    training = self._get_training_value(training)
    if self.subdivisions <= 1 or self.subdivisions is None:
      return super().call(inputs, training=training)
    else:
      if self.renorm is False and training is False and self.fused:
        # outputs = self._fused_batch_norm(inputs, training=False)
        beta = self.beta if self.center else self._beta_const
        gamma = self.gamma if self.scale else self._gamma_const
        outputs, mean, variance = nn.fused_batch_norm(
            inputs,
            gamma,
            beta,
            mean=self.moving_mean,
            variance=self.moving_variance,
            epsilon=self.epsilon,
            is_training=False,
            data_format=self._data_format)
        return outputs
      return self._subdiv_batch_norm(inputs, training=training)


class SubDivSyncBatchNormalization(SubDivBatchNormalization):
  r"""Normalize and scale inputs or activations synchronously across replicas.
  Applies batch normalization to activations of the previous layer at each batch
  by synchronizing the global batch statistics across all devices that are
  training the model. For specific details about batch normalization please
  refer to the `tf.keras.layers.BatchNormalization` layer docs.
  If this layer is used when using tf.distribute strategy to train models
  across devices/workers, there will be an allreduce call to aggregate batch
  statistics across all replicas at every training step. Without tf.distribute
  strategy, this layer behaves as a regular `tf.keras.layers.BatchNormalization`
  layer.
  Example usage:
  ```
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16))
    model.add(tf.keras.layers.experimental.SyncBatchNormalization())
  ```
  Arguments:
    axis: Integer, the axis that should be normalized
      (typically the features axis).
      For instance, after a `Conv2D` layer with
      `data_format="channels_first"`,
      set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
      If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
      If False, `gamma` is not used.
      When the next layer is linear (also e.g. `nn.relu`),
      this can be disabled since the scaling
      will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    renorm: Whether to use [Batch Renormalization](
      https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
    trainable: Boolean, if `True` the variables will be marked as trainable.
  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode.
      - `training=True`: The layer will normalize its inputs using the
        mean and variance of the current batch of inputs.
      - `training=False`: The layer will normalize its inputs using the
        mean and variance of its moving statistics, learned during training.
  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
  Output shape:
    Same shape as input.
  """

  def __init__(self,
               axis=-1,
               subdivisions=1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               trainable=True,
               adjustment=None,
               name=None,
               **kwargs):

    # Currently we only support aggregating over the global batch size.
    super(SubDivSyncBatchNormalization, self).__init__(
        axis=axis,
        subdivisions=subdivisions,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=False,
        trainable=trainable,
        name=name,
        **kwargs)

  def _calculate_mean_and_var(self, x, axes, keep_dims):

    with K.name_scope('moments'):
      # The dynamic range of fp16 is too limited to support the collection of
      # sufficient statistics. As a workaround we simply perform the operations
      # on 32-bit floats before converting the mean and variance back to fp16
      y = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x
      replica_ctx = ds.get_replica_context()
      if replica_ctx:
        # local to me
        local_sum = math_ops.reduce_sum(y, axis=axes, keepdims=True)
        local_squared_sum = math_ops.reduce_sum(
            math_ops.square(y), axis=axes, keepdims=True)
        batch_size = math_ops.cast(array_ops.shape_v2(y)[0], dtypes.float32)
        # TODO(b/163099951): batch the all-reduces once we sort out the ordering
        # issue for NCCL. We don't have a mechanism to launch NCCL in the same
        # order in each replica nowadays, so we limit NCCL to batch all-reduces.

        # get the sum of all replicas (converge all devices)
        y_sum = replica_ctx.all_reduce(reduce_util.ReduceOp.SUM, local_sum)
        # get the sum from all replicas (converge all devices)
        y_squared_sum = replica_ctx.all_reduce(reduce_util.ReduceOp.SUM,
                                               local_squared_sum)
        # get the net batch size from all devices (converge all devices)
        global_batch_size = replica_ctx.all_reduce(reduce_util.ReduceOp.SUM,
                                                   batch_size)

        # get the number of total params you are averaging (local)
        axes_vals = [(array_ops.shape_v2(y))[i] for i in range(1, len(axes))]
        multiplier = math_ops.cast(
            math_ops.reduce_prod(axes_vals), dtypes.float32)
        multiplier = multiplier * global_batch_size

        # conver mean var (locally)
        mean = y_sum / multiplier
        y_squared_mean = y_squared_sum / multiplier
        # var = E(x^2) - E(x)^2
        variance = y_squared_mean - math_ops.square(mean)
      else:
        # if you only have one replica dont worry about it
        # Compute true mean while keeping the dims for proper broadcasting.
        mean = math_ops.reduce_mean(y, axes, keepdims=True, name='mean')
        # sample variance, not unbiased variance
        # Note: stop_gradient does not change the gradient that gets
        #       backpropagated to the mean from the variance calculation,
        #       because that gradient is zero
        variance = math_ops.reduce_mean(
            math_ops.squared_difference(y, mean),
            axes,
            keepdims=True,
            name='variance')
      if not keep_dims:
        mean = array_ops.squeeze(mean, axes)
        variance = array_ops.squeeze(variance, axes)
      if x.dtype == dtypes.float16:
        return (math_ops.cast(mean, dtypes.float16),
                math_ops.cast(variance, dtypes.float16))
      else:
        return (mean, variance)

  def _subdiv_calculate_mean_and_var(self, x, axes, keep_dims):

    with K.name_scope('moments'):
      # The dynamic range of fp16 is too limited to support the collection of
      # sufficient statistics. As a workaround we simply perform the operations
      # on 32-bit floats before converting the mean and variance back to fp16
      y = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x
      replica_ctx = ds.get_replica_context()

      if replica_ctx:
        # local to me

        local_sum = math_ops.reduce_sum(y, axis=axes, keepdims=True)
        local_squared_sum = math_ops.reduce_sum(
            math_ops.square(y), axis=axes, keepdims=True)
        batch_size = math_ops.cast(array_ops.shape_v2(y)[0], dtypes.float32)
        # TODO(b/163099951): batch the all-reduces once we sort out the ordering
        # issue for NCCL. We don't have a mechanism to launch NCCL in the same
        # order in each replica nowadays, so we limit NCCL to batch all-reduces.
        # get the sum of all replicas (converge all devices)
        y_sum = replica_ctx.all_reduce(reduce_util.ReduceOp.SUM, local_sum)
        # get the sum from all replicas (converge all devices)
        y_squared_sum = replica_ctx.all_reduce(reduce_util.ReduceOp.SUM,
                                               local_squared_sum)
        # get the net batch size from all devices (converge all devices)
        input_batch_size = replica_ctx.all_reduce(reduce_util.ReduceOp.SUM,
                                                  batch_size)

        #tf.print(replica_ctx.replica_id_in_sync_group, replica_ctx.num_replicas_in_sync, batch_size, self.aggregated_square_sum_batch, axes)
        # get the number of total params you are averaging (local)
        axes_vals = [(array_ops.shape_v2(y))[i] for i in range(1, len(axes))]
        multiplier_ = math_ops.cast(
            math_ops.reduce_prod(axes_vals), dtypes.float32)
        multiplier = multiplier_ * input_batch_size

        # conver mean var (locally)
        mean = y_sum / multiplier
        y_squared_mean = y_squared_sum / multiplier
        # var = E(x^2) - E(x)^2
        variance = y_squared_mean - math_ops.square(mean)
        net_sum = y_sum / multiplier_
        squared_mean = y_squared_sum / multiplier_

      else:
        # mean = math_ops.reduce_mean(y, axes, keepdims=True, name='mean')
        # # sample variance, not unbiased variance
        # # Note: stop_gradient does not change the gradient that gets
        # #       backpropagated to the mean from the variance calculation,
        # #       because that gradient is zero
        # variance = math_ops.reduce_mean(
        #     math_ops.squared_difference(y, array_ops.stop_gradient(mean)),
        #     axes,
        #     keepdims=True,
        #     name='variance')

        net_sum = math_ops.reduce_sum(y, axis=axes, keepdims=True)
        squared_mean = math_ops.reduce_sum(
            math_ops.square(y), axis=axes, keepdims=True)

        if self._support_zero_size_input():
          # Keras assumes that batch dimension is the first dimension for Batch
          # Normalization.
          input_batch_size = array_ops.shape(y)[0]
        else:
          input_batch_size = None

        # get the number of total params you are averaging including batchsize(local)
        axes_vals = [(array_ops.shape_v2(y))[i] for i in range(1, len(axes))]
        multiplier = math_ops.cast(
            math_ops.reduce_prod(axes_vals), dtypes.float32)

        squared_mean = squared_mean / multiplier
        net_sum = net_sum / multiplier

        if input_batch_size is None:
          mean, variance = nn.moments(y, axes, keep_dims=True)
          input_batch_size = 0
        else:
          batches_ = math_ops.cast(input_batch_size, self._param_dtype)
          # # if you only have one replica dont worry about it
          # # Compute true mean while keeping the dims for proper broadcasting.
          mean = net_sum / batches_
          variance = squared_mean / batches_ - math_ops.square(mean)

      input_batch_size = math_ops.cast(input_batch_size, dtypes.int32)
      if not keep_dims:
        mean = array_ops.squeeze(mean, axes)
        net_sum = array_ops.squeeze(net_sum, axes)
        variance = array_ops.squeeze(variance, axes)
        squared_mean = array_ops.squeeze(squared_mean, axes)
      if x.dtype == dtypes.float16:
        return (math_ops.cast(mean, dtypes.float16),
                math_ops.cast(net_sum, dtypes.float16),
                math_ops.cast(variance, dtypes.float16),
                math_ops.cast(squared_mean, dtypes.float16), input_batch_size)
      else:
        return (mean, net_sum, variance, squared_mean, input_batch_size)


class ShuffleBatchNormalization(normalization.BatchNormalizationBase):

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               trainable=True,
               adjustment=None,
               name=None,
               **kwargs):

    # Currently we only support aggregating over the global batch size.
    super(ShuffleBatchNormalization, self).__init__(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=False,
        trainable=trainable,
        virtual_batch_size=None,
        name=name,
        **kwargs)

  def _calculate_mean_and_var(self, x, axes, keep_dims):

    with K.name_scope('moments'):
      # The dynamic range of fp16 is too limited to support the collection of
      # sufficient statistics. As a workaround we simply perform the operations
      # on 32-bit floats before converting the mean and variance back to fp16
      y = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x
      # if you only have one replica dont worry about it
      # Compute true mean while keeping the dims for proper broadcasting.
      mean = math_ops.reduce_mean(y, axes, keepdims=True, name='mean')
      # sample variance, not unbiased variance
      # Note: stop_gradient does not change the gradient that gets
      #       backpropagated to the mean from the variance calculation,
      #       because that gradient is zero
      variance = math_ops.reduce_mean(
          math_ops.squared_difference(y, array_ops.stop_gradient(mean)),
          axes,
          keepdims=True,
          name='variance')

      replica_ctx = ds.get_replica_context()
      if replica_ctx:
        tf.print(replica_ctx.num_replicas_in_sync)
        tf.print(replica_ctx.replica_id_in_sync_group)

      if not keep_dims:
        mean = array_ops.squeeze(mean, axes)
        variance = array_ops.squeeze(variance, axes)
      if x.dtype == dtypes.float16:
        return (math_ops.cast(mean, dtypes.float16),
                math_ops.cast(variance, dtypes.float16))
      else:
        return (mean, variance)
