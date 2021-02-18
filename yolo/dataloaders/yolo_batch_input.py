import tensorflow as tf

from yolo.ops import preprocessing_ops
from yolo.ops import box_ops as box_utils
from official.vision.beta.ops import box_ops, preprocess_ops
from official.vision.beta.dataloaders import parser


def pad_max_instances(value, instances, pad_value=0, pad_axis=0):
  shape = tf.shape(value)
  if pad_axis < 0:
    pad_axis = tf.shape(shape)[0] + pad_axis
  dim1 = shape[pad_axis]
  take = tf.math.reduce_min([instances, dim1])
  value, _ = tf.split(value, [take, -1],
                      axis=pad_axis)  # value[:instances, ...]
  pad = tf.convert_to_tensor([tf.math.reduce_max([instances - dim1, 0])])
  nshape = tf.concat([shape[:pad_axis], pad, shape[(pad_axis + 1):]], axis=0)
  pad_tensor = tf.fill(nshape, tf.cast(pad_value, dtype=value.dtype))
  value = tf.concat([value, pad_tensor], axis=pad_axis)
  return value


class Parser(parser.Parser):
  def __init__(self,
               image_w=416,
               image_h=416,
               num_classes=80,
               fixed_size=False,
               use_tie_breaker=True,
               min_level=3,
               max_level=5,
               masks=None,
               cutmix=False,
               mosaic=True,
               max_num_instances=200,
               pct_rand=0.5,
               anchors=None,
               seed=10,
               dtype='float32'):
    self._net_down_scale = 2**max_level

    self._num_classes = num_classes

    self._image_w = (image_w // self._net_down_scale) * self._net_down_scale
    self._image_h = self._image_w if image_h is None else (
        image_h // self._net_down_scale) * self._net_down_scale

    self._anchors = anchors
    self._masks = {
        key: tf.convert_to_tensor(value)
        for key, value in masks.items()
    }
    self._use_tie_breaker = use_tie_breaker

    self._pct_rand = pct_rand
    self._max_num_instances = max_num_instances

    self._seed = seed
    self._cutmix = cutmix
    self._fixed_size = fixed_size
    self._mosaic = mosaic

    if dtype == 'float16':
      self._dtype = tf.float16
    elif dtype == 'bfloat16':
      self._dtype = tf.bfloat16
    elif dtype == 'float32':
      self._dtype = tf.float32
    else:
      raise Exception(
          'Unsupported datatype used in parser only {float16, bfloat16, or float32}'
      )

  def _build_grid(self, raw_true, width, batch=False, use_tie_breaker=False):
    mask = self._masks
    for key in self._masks.keys():
      if not batch:
        mask[key] = preprocessing_ops.build_grided_gt(
            raw_true, self._masks[key], width // 2**int(key),
            self._num_classes, raw_true['bbox'].dtype, use_tie_breaker)
      else:
        mask[key] = preprocessing_ops.build_batch_grided_gt(
            raw_true, self._masks[key], width // 2**int(key),
            self._num_classes, raw_true['bbox'].dtype, use_tie_breaker)

      mask[key] = tf.cast(mask[key], self._dtype)
    return mask

  def parse_train_data(self, image, label):
    if self._cutmix:
      batch_size = tf.shape(image)[0]
      if batch_size >= 1:
        image, boxes, classes, num_detections = preprocessing_ops.randomized_cutmix_batch(
            image, label['bbox'], label['classes'])
        label['bbox'] = pad_max_instances(boxes,
                                          self._max_num_instances,
                                          pad_axis=-2,
                                          pad_value=0)
        label['classes'] = pad_max_instances(classes,
                                             self._max_num_instances,
                                             pad_axis=-1,
                                             pad_value=-1)

    if self._mosaic:
      image, boxes, classes = preprocessing_ops.mosaic(image,
                                                       label['bbox'],
                                                       label['classes'],
                                                       self._image_w,
                                                       crop_delta=0.5)
      label['bbox'] = pad_max_instances(boxes,
                                        self._max_num_instances,
                                        pad_axis=-2,
                                        pad_value=0)
      label['classes'] = pad_max_instances(classes,
                                           self._max_num_instances,
                                           pad_axis=-1,
                                           pad_value=-1)
    randscale = self._image_w // self._net_down_scale
    if not self._fixed_size:
      do_scale = tf.greater(
          tf.random.uniform([], minval=0, maxval=1, seed=self._seed),
          1 - self._pct_rand)
      if do_scale:
        randscale = tf.random.uniform([],
                                      minval=10,
                                      maxval=21,
                                      seed=self._seed,
                                      dtype=tf.int32)
    width = randscale * self._net_down_scale
    image = tf.image.resize(image, (width, width))

    label['bbox'] = box_utils.yxyx_to_xcycwh(label['bbox'])
    best_anchors = preprocessing_ops.get_best_anchor_batch(
        label['bbox'],
        self._anchors,
        width=self._image_w,
        height=self._image_h)
    label['best_anchors'] = pad_max_instances(best_anchors,
                                              self._max_num_instances,
                                              pad_axis=-2,
                                              pad_value=0)

    grid = self._build_grid(label,
                            width,
                            batch=True,
                            use_tie_breaker=self._use_tie_breaker)
    label.update({'grid_form': grid})
    label['bbox'] = box_utils.xcycwh_to_yxyx(label['bbox'])
    return image, label

  def parse_eval_data(self, data, label):
    return data, label
