import tensorflow as tf
from tensorflow.python.ops import math_ops, state_ops, control_flow_ops, array_ops
from tensorflow.python import ops 
from tensorflow.python.keras.utils import control_flow_util
# from tensorflow.python.keras import backend_config


__all__ = ['SGDAccumulated']

# problem is that sub division cannot change between saves
class SGDAccumulated(tf.keras.optimizers.Optimizer):
  """Optimizer that implements the Adam algorithm with gradient accumulation."""

  def __init__(self,
                accumulation_steps = 1,
                accumulation_type = 'mean', 
                learning_rate=0.01,
                momentum=0.0,
                nesterov=False,
                name="SGD",
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
    self._optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    self._track_trackable(self._optimizer, 'core_optimizer')

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, 'g')

  def raw_sparse_update(self, grad, var, apply_state = None):
    def func():
      # tf.print("up")
      return self._optimizer._resource_apply_sparse(grad, var, apply_state = apply_state)
    return func

  def raw_dense_update(self, grad, var, apply_state = None):
    def func():
      # tf.print("up")
      return self._optimizer._resource_apply_dense(grad, var, apply_state = apply_state)
    return func

  def no_update(self, var):
    def func():
      # tf.print("no up")
      var_update = state_ops.assign(var, var, use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update])
    return func

  def _resource_apply_dense(self, grad, var, apply_state = None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    accumulation_steps = self._get_hyper('accumulation_steps', 'int64')
    update_cond = tf.equal((self.iterations + 1) % accumulation_steps, 0)
    sub_step = self.iterations % accumulation_steps

    g = self.get_slot(var, 'g') # accumulated gradient
    g_a = grad / math_ops.cast(accumulation_steps, var_dtype)
    g_t = tf.where(tf.equal(sub_step, 0), g_a, g_a + g)
    g_t = state_ops.assign(g, g_t, use_locking=self._use_locking)

    updates = control_flow_util.smart_cond(update_cond, 
                                true_fn = self.raw_dense_update(g_t, var, apply_state = apply_state), 
                                false_fn = self.no_update(var))
    return updates


  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    accumulation_steps = self._get_hyper('accumulation_steps', 'int64')
    update_cond = tf.equal((self.iterations + 1) % accumulation_steps, 0)
    sub_step = self.iterations % accumulation_steps

    g = self.get_slot(var, 'g') # accumulated gradient
    g_a = grad / math_ops.cast(accumulation_steps, var_dtype)
    g_t = tf.where(tf.equal(sub_step, 0), g_a, g_a + g)
    g_t = state_ops.assign(g, g_t, use_locking=self._use_locking)

    updates = control_flow_util.smart_cond(update_cond, 
                                true_fn = self.raw_dense_update(g_t, var, apply_state = apply_state), 
                                false_fn = self.no_update(var))
    return updates
  
  def _create_hypers(self):
    super()._create_hypers()
    self._optimizer._create_hypers()  # pylint: disable=protected-access
    

  def _prepare(self, var_list):
    states = super()._prepare(var_list=var_list)
    states.update(self._optimizer._prepare(var_list=var_list))
    return states  # pylint: disable=protected-access

  @property
  def iterations(self):
    return self._optimizer.iterations

  @iterations.setter
  def iterations(self, variable):
    self._optimizer.iterations = variable
    # self.iterations = variable

  @property
  def weights(self):
    return self._weights + self._optimizer.weights

  def variables(self):
    return self._weights + [self.iterations]

  @property
  def lr(self):
    return self._optimizer._get_hyper('learning_rate')

  @lr.setter
  def lr(self, lr):
    self._optimizer._set_hyper('learning_rate', lr)

  @property
  def learning_rate(self):
    return self._optimizer._get_hyper('learning_rate')

  @learning_rate.setter
  def learning_rate(self, learning_rate):  # pylint: disable=redefined-outer-name
    self._optimizer._set_hyper('learning_rate', learning_rate)

  # def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
  #   return self._optimizer._resource_apply_sparse_duplicate_indices(
  #       grad, var, indices)

  def get_config(self):
    config = super(SGDAccumulated, self).get_config()
    config.update({
        'accumulation_steps': self._serialize_hyperparameter('accumulation_steps'),
        'optimizer': tf.keras.optimizers.serialize(self._optimizer),
    })
    return config


  @classmethod
  def from_config(cls, config, custom_objects=None):
    optimizer = tf.keras.optimizers.deserialize(
        config.pop('optimizer'),
        custom_objects=custom_objects,
    )
    return cls(optimizer, **config)

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