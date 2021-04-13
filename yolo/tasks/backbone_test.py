from yolo.configs import darknet_classification as dcfg
from yolo.tasks import image_classification as imc
from yolo.modeling.backbones import darknet
import tensorflow as tf

from yolo.utils.run_utils import prep_gpu
import matplotlib.pyplot as plt


def test_classification_input():
  with tf.device('/CPU:0'):
    config = dcfg.ImageClassificationTask(
        model=dcfg.ImageClassificationModel(
            backbone=dcfg.backbones.Backbone(
                type='darknet',
                darknet=dcfg.backbones.DarkNet(model_id='cspdarknet53')),
            darknet_weights_file='cache://csdarknet53.weights',
            darknet_weights_cfg='cache://csdarknet53.cfg'))
    task = imc.ImageClassificationTask(config)

    model = task.build_model()
    task.initialize(model)
    # config.train_data.dtype = "float32"
    # config.validation_data.dtype = "float32"
    train_data = task.build_inputs(config.train_data)
    test_data = task.build_inputs(config.validation_data)
  return train_data, test_data, model


def test_classification():
  dataset, dsp, model = test_classification_input()
  for l, (i, j) in enumerate(dsp):
    o = model(i)
    o = tf.keras.layers.Softmax()(o)
    val, ind = tf.math.top_k(o, k=5)
    print(ind, j)
    plt.imshow(i[0].numpy())
    plt.show()
    if l > 30:
      break
  return


# prep_gpu()
# test_classification()
