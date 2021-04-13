from yolo.tasks import yolo
import orbit
from official.core import exp_factory
from official.modeling import optimization

import tensorflow as tf
from absl.testing import parameterized
from yolo.utils.run_utils import prep_gpu
# try:
#   prep_gpu()
# except BaseException:
#   print("GPUs ready")


class YoloTaskTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(("yolo_v4_coco",))
  def test_task(self, config_name):
    config = exp_factory.get_exp_config(config_name)
    config.task.train_data.global_batch_size = 2

    task = yolo.YoloTask(config.task)
    model = task.build_model()
    metrics = task.build_metrics(training=False)
    strategy = tf.distribute.get_strategy()

    dataset = orbit.utils.make_distributed_dataset(strategy, task.build_inputs,
                                                   config.task.train_data)

    iterator = iter(dataset)
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    logs = task.train_step(next(iterator), model, optimizer, metrics=metrics)
    self.assertIn("loss", logs)
    logs = task.validation_step(next(iterator), model, metrics=metrics)
    self.assertIn("loss", logs)


if __name__ == "__main__":
  tf.test.main()
