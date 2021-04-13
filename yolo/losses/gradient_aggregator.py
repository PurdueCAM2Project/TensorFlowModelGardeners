import tensorflow as tf

# TODO: run sub dividied training. take the gradients and added them together in the update
# once you, when they ask for the result jsut return it, and reset the state
# i think you need to zip and then add them together.


class GradientAggregator(tf.Module):
  """Encapsulates metrics that perform a reduce operation on the values.
  Args:
    reduction: a `tf.keras.metrics.Reduction` enum value.
    name: string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """

  def __init__(self,
               train_vars,
               batch_size,
               subdivisions,
               loss_aggregation='MEAN',
               name='gradientAgg'):
    super().__init__(name=name)
    self._grad = [
        self.add_grad(name=f"{var.name}$grad", variable=var)
        for var in train_vars
    ]
    self._zeros = [
        self.add_grad(name=f"{var.name}$zeros", variable=var)
        for var in train_vars
    ]
    self._count = tf.Variable(name='count', initial_value=0.0, trainable=False)

    self._batch_size = batch_size
    self._subdivisions = subdivisions
    self._loss_aggregation = loss_aggregation

  def add_grad(self, name, variable, synchronization=None, aggregation=None):
    if aggregation is None:
      aggregation = tf.VariableAggregation.SUM

    # safe side for usage with TPU
    if synchronization is None:
      synchronization = tf.VariableSynchronization.ON_WRITE

    value = tf.zeros_like(variable)

    return tf.Variable(
        name=name,
        initial_value=value,
        trainable=False,
        synchronization=synchronization,
        aggregation=aggregation)

  def update_state(self, values, sample_weight=None):
    mod = 1.0
    if (self._count) % self._subdivisions == 0:
      mod = 0.0

    if self._loss_aggregation == 'SUM':
      for i in range(len(self._grad)):
        self._grad[i].assign(self._grad[i] * mod + (
            values[i] /
            tf.cast(self._subdivisions * self._batch_size, values[i].dtype)))
    elif self._loss_aggregation == 'MEAN':
      for i in range(len(self._grad)):
        self._grad[i].assign(self._grad[i] * mod +
                             (values[i] /
                              tf.cast(self._subdivisions, values[i].dtype)))
    else:
      for i in range(len(self._grad)):
        self._grad[i].assign(self._grad[i] * mod + values[i])
    self._count.assign_add(1)
    return

  def result(self):
    if (self._count) % self._subdivisions == 0:
      return self._grad, self._count
    else:
      return self._zeros, self._count


from yolo import run
if __name__ == '__main__':

  task, model, params = run.load_model(
      config_path=[
          '/media/vbanna/DATA_SHARE/Research/TensorFlowModelGardeners/yolo/configs/experiments/yolov4-tiny-eval.yaml'
      ],
      model_dir='')
  inputs = tf.ones((1, 416, 416, 3))
  optimizer = tf.keras.optimizers.RMSprop()

  task._task_config.validation_data.global_batch_size = 8
  subdivisions = 2
  task._task_config.validation_data.global_batch_size = task._task_config.validation_data.global_batch_size // subdivisions
  print(task._task_config.validation_data.global_batch_size)
  train_data = task.build_inputs(task._task_config.validation_data)
  sample = train_data.take(subdivisions * 1)
  subdiv_ag = GradientAggregator(
      model.trainable_variables,
      task._task_config.validation_data.global_batch_size,
      subdivisions,
      loss_aggregation='MEAN')

  i = 0
  loss = 0
  with tf.GradientTape() as tape:

    for data, label in sample:
      a = model(data, training=True)

    loss_, metric = task.build_losses(a['raw_output'], label)
    loss += loss_
  grad = tape.gradient(loss, model.trainable_variables)

  # subdiv_ag.update_state(grad)
  # if i % subdivisions == 0:
  #   grad_ag, count = subdiv_ag.result()
  # i += 1
  # print(grad_ag[0], i)

  # optimizer.apply_gradients(zip(grad_ag, model.trainable_variables))

  # grad_ag, count = subdiv_ag.result()

  print(grad[-1])
  # for variable in grad_ag:
  #   print(variable)
  # print(subdiv_ag.count())
  # subdiv_ag.reset_states()
