import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.ops import math_ops, state_ops, control_flow_ops, array_ops
# from tensorflow.python import ops
# from tensorflow.python.keras import backend_config

__all__ = ['SGDAccumulated']


# problem is that sub division cannot change between saves
class SGDAccumulated(OptimizerV2):
  """Optimizer that implements the Adam algorithm with gradient accumulation."""

  def __init__(self,
               accumulation_steps=1,
               accumulation_type='mean',
               learning_rate=0.01,
               momentum=0.0,
               nesterov=False,
               name='SGD',
               **kwargs):
    r"""Construct a new SGD optimizer.
    Args:
        accumulation_steps: An integer. Update gradient in every accumulation steps.
        learning_rate: A Tensor or a floating point value.    The learning rate.
        beta_1: A float value or a constant float tensor. The exponential decay
            rate for the 1st moment estimates.
        beta_2: A float value or a constant float tensor. The exponential decay
            rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
        amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
            the paper "On the Convergence of Adam and beyond".
        name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".    @compatibility(eager) When eager execution is
            enabled, `learning_rate`, `beta_1`, `beta_2`, and `epsilon` can each be
            a callable that takes no arguments and returns the actual value to use.
            This can be useful for changing these values across different
            invocations of optimizer functions. @end_compatibility
        **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for backward
            compatibility, recommended to use `learning_rate` instead.
    """

    super(SGDAccumulated, self).__init__(name, **kwargs)
    self._set_hyper('accumulation_steps', tf.cast(accumulation_steps, tf.int32))
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._momentum = False
    if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError('`momentum` must be between [0, 1].')
    self._set_hyper('momentum', momentum)
    self.nesterov = nesterov
    self._accumulation_type = accumulation_type

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, 'g')
    if self._momentum:
      for var in var_list:
        self.add_slot(var, 'momentum')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(SGDAccumulated, self)._prepare_local(var_device, var_dtype,
                                               apply_state)
    apply_state[(var_device, var_dtype)]['momentum'] = array_ops.identity(
        self._get_hyper('momentum', var_dtype))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    # tf.print('opt', self.iterations)
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    lr_t = coefficients['lr_t']
    accumulation_steps = self._get_hyper('accumulation_steps', 'int64')
    update_cond = tf.equal((self.iterations + 1) % accumulation_steps, 0)

    # steps = 500500 * accumulation steps
    sub_step = self.iterations % accumulation_steps + 1
    local_step = math_ops.cast(self.iterations // accumulation_steps + 1,
                               var_dtype)

    # used to control when updates happen (zero when substeps != accumulation steps)
    lr = tf.where(update_cond, lr_t, 0.0)

    #gradient accumulation sum
    if self._accumulation_type == 'sum':
      g = self.get_slot(var, 'g')  # accumulated gradient
      g_a = grad
      g_t = tf.where(tf.equal(sub_step, 1), g_a, g_a + g)
      g_t = state_ops.assign(g, g_t, use_locking=self._use_locking)
    elif self._accumulation_type == 'moving_avg':
      g = self.get_slot(var, 'g')  # accumulated gradient
      g_a = grad / math_ops.cast(accumulation_steps, var_dtype)
      g_t = tf.where(
          tf.equal(sub_step, 1), g_a,
          g + (g_a - g) / math_ops.cast(sub_step, var_dtype))
      g_t = state_ops.assign(g, g_t, use_locking=self._use_locking)
    else:
      g = self.get_slot(var, 'g')  # accumulated gradient
      g_a = grad / math_ops.cast(accumulation_steps, var_dtype)
      g_t = tf.where(tf.equal(sub_step, 1), g_a, g_a + g)
      g_t = state_ops.assign(g, g_t, use_locking=self._use_locking)

    # momentum update
    if self._momentum:
      momentum = coefficients['momentum']
      momentum_grad = self.get_slot(var, 'momentum')
      momentum_g = tf.where(update_cond,
                            momentum * momentum_grad + (1 - momentum) * g_t,
                            momentum_grad)
      var_update = state_ops.assign_sub(
          var, lr * momentum_g, use_locking=self._use_locking)
      with tf.control_dependencies([momentum_g]):
        momentum_g = state_ops.assign(
            momentum_grad, momentum_g, use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, momentum_g])
    # nestrov momentum
    # standard update
    else:
      var_update = state_ops.assign_sub(
          var, lr * g_t, use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update])

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # This method is only needed for momentum optimization.

    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    lr_t = coefficients['lr_t']
    accumulation_steps = self._get_hyper('accumulation_steps', 'int64')
    update_cond = tf.equal((self.iterations + 1) % accumulation_steps, 0)

    # steps = 500500 * accumulation steps
    sub_step = self.iterations % accumulation_steps + 1
    local_step = math_ops.cast(self.iterations // accumulation_steps + 1,
                               var_dtype)

    # used to control when updates happen (zero when substeps != accumulation steps)
    lr = tf.where(update_cond, lr_t, 0.0)

    #gradient accumulation
    g = self.get_slot(var, 'g')  # accumulated gradient
    g_a = grad / math_ops.cast(accumulation_steps, var_dtype)
    g_t = tf.where(
        tf.equal(sub_step, 1), g_a,
        g + (g_a - g) / math_ops.cast(sub_step, var_dtype))
    g_t = state_ops.assign(g, g_t, use_locking=self._use_locking)

    # momentum update
    momentum = coefficients['momentum']
    momuentum_grad = self.get_slot(var, 'momentum')
    momuentum_g = tf.where(update_cond,
                           momentum * momuentum_grad + (1 - momentum) * g_t,
                           momuentum_grad)
    var_update = state_ops.assign_sub(
        var, lr * momuentum_grad, use_locking=self._use_locking)
    momuentum_grad = state_ops.assign(
        momuentum_g, momuentum_grad, use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, momuentum_grad])
    # nestrov momentum

  def get_config(self):
    config = super(SGDAccumulated, self).get_config()
    config.update({
        'accumulation_steps':
            self._serialize_hyperparameter('accumulation_steps'),
        'learning_rate':
            self._serialize_hyperparameter('learning_rate'),
        'decay':
            self._serialize_hyperparameter('decay'),
        'momentum':
            self._serialize_hyperparameter('momentum'),
    })
    return config


# if __name__ == "__main__":
#   from yolo import run
#   import os
#   optimizer = SGDAccumulated(accumulation_steps = 8)

#   config = [os.path.abspath('yolo/configs/experiments/yolov4-eval.yaml')]
#   model_dir = "" #os.path.abspath("../checkpoints/yolo_dt8_norm_iou")

#   task, model, params = run.load_model(experiment='yolo_custom', config_path=config, model_dir=model_dir)

#   train_data = task.build_inputs(task.task_config.train_data)
#   validation_data = task.build_inputs(task.task_config.train_data)

#   model.compile(optimizer = optimizer)
