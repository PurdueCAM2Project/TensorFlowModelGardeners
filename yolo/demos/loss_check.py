import os

from yolo.tasks import yolo
import orbit
from official.core import exp_factory
from official.modeling import optimization

import tensorflow as tf
from yolo.utils.run_utils import prep_gpu
from yolo import run


if __name__ == "__main__":
    try:
        prep_gpu()
    except:
        print("GPUs ready")

    config = exp_factory.get_exp_config('yolo_custom')
    config.task.train_data.global_batch_size = 1

    task = yolo.YoloTask(config.task)
    model = task.build_model()
    task.initialize(model)
    metrics = task.build_metrics(training=False)
    """
    config = [os.path.abspath('../configs/experiments/yolov4-eval.yaml')]
    model_dir = ''
    task, model = run.load_model(
        experiment='yolo_custom',
        config_path=config,
        model_dir=model_dir
    )
    """

    # prepare dataset
    strategy = tf.distribute.get_strategy()
    dataset = orbit.utils.make_distributed_dataset(strategy, task.build_inputs, config.task.validation_data)
    iterator = iter(dataset)
    one_data = next(iterator)

    # print(one_data)
    # tf.keras.preprocessing.image.save_img("sample2.png", one_data[0].numpy()[0], "channels_last")

    logs = task.validation_step(one_data, model)
