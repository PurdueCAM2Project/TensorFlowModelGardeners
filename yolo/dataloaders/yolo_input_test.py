from yolo.tasks import image_classification as imc
from yolo.utils.demos import utils, coco
from yolo.tasks import yolo
from yolo.configs import yolo as yolocfg
from yolo.configs import darknet_classification as dcfg
from official.core import input_reader
from yolo.dataloaders import yolo_input as YOLO_Detection_Input
from yolo.dataloaders import classification_input
from yolo.dataloaders.decoders import tfds_coco_decoder
from yolo.ops import box_ops
import matplotlib.pyplot as plt
import dataclasses
from official.modeling import hyperparams
from official.core import config_definitions as cfg
import tensorflow as tf

from yolo.modeling.layers import detection_generator
from official.vision.beta.ops import box_ops as bops
from official.vision.beta.tasks import retinanet
from official.vision.beta.configs import retinanet as retcfg


def test_yolo_input_task():
  with tf.device('/CPU:0'):
    config = yolocfg.YoloTask(
        model=yolocfg.Yolo(
            base='v4',
            min_level=3,
            norm_activation=yolocfg.common.NormActivation(activation='mish'),
            #norm_activation = yolocfg.common.NormActivation(activation="leaky"),
            #_boxes = ['(10, 14)', '(23, 27)', '(37, 58)', '(81, 82)', '(135, 169)', '(344, 319)'],
            #_boxes = ["(10, 13)", "(16, 30)", "(33, 23)","(30, 61)", "(62, 45)", "(59, 119)","(116, 90)", "(156, 198)", "(373, 326)"],
            # _boxes =  ['(10, 14)', '(23, 27)', '(37, 58)', '(81, 82)'], #, '(135, 169)'])
            _boxes=[
                '(12, 16)', '(19, 36)', '(40, 28)', '(36, 75)', '(76, 55)',
                '(72, 146)', '(142, 110)', '(192, 243)', '(459, 401)'
            ],
            # _boxes = ['[15.0, 23.0]', '[38.0, 57.0]', '[119.0, 67.0]', '[57.0, 141.0]', '[164.0, 156.0]', '[97.0, 277.0]', '[371.0, 184.0]', '[211.0, 352.0]', '[428.0, 419.0]'],
            # _boxes = None,
            filter=yolocfg.YoloLossLayer(use_nms=False)))
    task = yolo.YoloTask(config)

    # loading both causes issues, but oen at a time is not issue, why?
    config.train_data.dtype = 'float32'
    config.validation_data.dtype = 'float32'
    config.train_data.tfds_name = 'coco'
    config.validation_data.tfds_name = 'coco'
    config.train_data.tfds_split = 'train'
    config.validation_data.tfds_split = 'validation'
    train_data = task.build_inputs(config.train_data)
    test_data = task.build_inputs(config.validation_data)
  return train_data, test_data


def test_retinanet_input_task():
  with tf.device('/CPU:0'):
    config = retcfg.RetinaNetTask()
    task = retinanet.RetinaNetTask(config)

    # loading both causes issues, but oen at a time is not issue, why?
    config.train_data.dtype = 'float32'
    config.validation_data.dtype = 'float32'
    config.train_data.tfds_name = 'coco'
    config.validation_data.tfds_name = 'coco'
    config.train_data.tfds_split = 'train'
    config.validation_data.tfds_split = 'validation'
    train_data = task.build_inputs(config.train_data)
    test_data = task.build_inputs(config.validation_data)
  return train_data, test_data


def test_classification_input():
  with tf.device('/CPU:0'):
    config = dcfg.ImageClassificationTask()

    task = imc.ImageClassificationTask(config)
    # config.train_data.dtype = "float32"
    # config.validation_data.dtype = "float32"
    train_data = task.build_inputs(config.train_data)
    test_data = task.build_inputs(config.validation_data)
  return train_data, test_data


def test_classification_pipeline():
  dataset, dsp = test_classification_input()
  for l, (i, j) in enumerate(dataset):
    plt.imshow(i[0].numpy())
    plt.show()
    if l > 30:
      break
  return


import time


def test_yolo_pipeline(is_training=True):
  dataset, dsp = test_yolo_input_task()
  print(dataset, dsp)
  # shind = 3
  dip = 0
  drawer = utils.DrawBoxes(labels=coco.get_coco_names(), thickness=2)
  dfilter = detection_generator.YoloFilter()
  ltime = time.time()

  data = dataset if is_training else dsp
  data = data.take(15)
  for l, (i, j) in enumerate(data):
    ftime = time.time()
    i_ = tf.image.draw_bounding_boxes(i, j['bbox'], [[1.0, 0.0, 1.0]])

    gt = j['true_conf']
    inds = j['inds']

    obj3 = gt['3'][..., 0]
    obj4 = gt['4'][..., 0]
    obj5 = gt['5'][..., 0]

    for shind in range(1):
      fig, axe = plt.subplots(1, 5)

      image = i[shind]
      boxes = j["bbox"][shind]
      classes = j["classes"][shind]
      confidence = j["classes"][shind]

      draw_dict = {
          'bbox': boxes,
          'classes': classes,
          'confidence': confidence,
      }
      # print(tf.cast(bops.denormalize_boxes(boxes, image.shape[:2]), tf.int32))
      image = drawer(image, draw_dict)

      (true_box, ind_mask, true_class, best_iou_match, num_reps) = tf.split(
          j['upds']['5'], [4, 1, 1, 1, 1], axis=-1)

      true_xy = true_box[shind][..., 0:2] * 20
      ind_xy = tf.cast(j['inds']['5'][shind][..., 0:2], true_xy.dtype)
      x, y = tf.split(ind_xy, 2, axis=-1)
      ind_xy = tf.concat([y, x], axis=-1)
      tf.print(true_xy - ind_xy, summarize=-1)
      axe[0].imshow(i_[shind])
      axe[1].imshow(image)
      axe[2].imshow(obj3[shind].numpy())
      axe[3].imshow(obj4[shind].numpy())
      axe[4].imshow(obj5[shind].numpy())

      fig.set_size_inches(16.5, 5.5, forward=True)
      plt.tight_layout()
      plt.show()

    ltime = time.time()

    if l >= 100:
      break


def time_pipeline():
  dataset, dsp = test_yolo_input_task()
  print(dataset)
  times = []
  ltime = time.time()
  for l, (i, j) in enumerate(dataset):
    ftime = time.time()
    # print(tf.reduce_min(i))
    # print(l , ftime - ltime, end = ", ")
    times.append(ftime - ltime)
    ltime = time.time()
    if l >= 100:
      break

  plt.plot(times)
  plt.show()

  print(f"total time {sum(times)}")


def reduce(obj):
  return tf.math.ceil(
      tf.clip_by_value(
          tf.reduce_max(
              tf.reduce_sum(
                  tf.reshape(
                      obj, [obj.shape[0], obj.shape[1], obj.shape[2], 3, 3, 4]),
                  axis=-1),
              axis=-1), 0.0, 1.0))


def test_ret_pipeline():
  dataset, dsp = test_retinanet_input_task()
  print(dataset, dsp)

  shind = 0
  dip = 0
  drawer = utils.DrawBoxes(labels=coco.get_coco_names(), thickness=1)
  dfilter = detection_generator.YoloFilter()
  ltime = time.time()

  data = dsp
  data = data.take(10)

  lim1 = 0
  lim2 = 3
  for l, (i, j) in enumerate(data):
    ftime = time.time()

    i2 = drawer(i, j)  #
    i = tf.image.draw_bounding_boxes(i, j['bbox'], [[1.0, 0.0, 1.0]])

    gt = j['box_targets']

    obj3 = gt['3']
    obj4 = gt['4']
    obj5 = gt['5']
    obj6 = gt['6']
    obj7 = gt['7']

    obj3 = reduce(obj3)
    obj4 = reduce(obj4)
    obj5 = reduce(obj5)
    obj6 = reduce(obj6)
    obj7 = reduce(obj7)

    fig, axe = plt.subplots(1, 7)

    axe[0].imshow(i2[shind])
    axe[1].imshow(i[shind])
    axe[2].imshow(obj3[shind].numpy())
    axe[3].imshow(obj4[shind].numpy())
    axe[4].imshow(obj5[shind].numpy())
    axe[5].imshow(obj6[shind].numpy())
    axe[6].imshow(obj7[shind].numpy())

    fig.set_size_inches(10.5, 8.5, forward=True)
    plt.tight_layout()
    plt.show()

    ltime = time.time()

    if l >= 10:
      break


if __name__ == '__main__':
  # test_ret_pipeline()
  test_yolo_pipeline(is_training=True)
  # test_yolo_pipeline(is_training=False)
  # time_pipeline()
  # test_classification_pipeline()
  # from yolo.ops import preprocessing_ops as po
  # dataset, dsp = test_yolo_input_task()

  # dataset = dataset.unbatch()
  # dataset = dataset.batch(4)
  # drawer = utils.DrawBoxes(labels=coco.get_coco_names(), thickness=1)

  # for l, (image, sample) in enumerate(dataset):

  #   image, boxes, classes, num_instances = po.randomized_cutmix_split(image, sample['bbox'], sample['classes'])

  #   # print(num_instances, tf.shape(boxes))
  #   sample = {
  #     'bbox': boxes,
  #     'classes': classes
  #   }

  #   image = drawer(image, sample)
  #   fig, axe = plt.subplots(1, 4)

  #   axe[0].imshow(image[0])
  #   axe[1].imshow(image[1])
  #   axe[2].imshow(image[2])
  #   axe[3].imshow(image[3])

  #   fig.set_size_inches(18.5, 6.5, forward=True)
  #   plt.tight_layout()
  #   plt.show()
  #   if l > 5:
  #     break
