""" Detection Data parser and processing for YOLO.
Parse image and ground truths in a dataset to training targets and package them
into (image, labels) tuple for RetinaNet.
"""

# Import libraries
import tensorflow as tf
import tensorflow_addons as tfa

from yolo.ops import preprocessing_ops
from yolo.ops import box_ops as box_utils
from official.vision.beta.ops import box_ops, preprocess_ops
from official.vision.beta.dataloaders import parser
from yolo.ops import loss_utils as loss_ops


def pad_max_instances(value, instances, pad_value=0, pad_axis=0):
  shape = tf.shape(value)
  if pad_axis < 0:
    pad_axis = tf.shape(shape)[0] + pad_axis
  dim1 = shape[pad_axis]
  take = tf.math.reduce_min([instances, dim1])
  value, _ = tf.split(
      value, [take, -1], axis=pad_axis)  # value[:instances, ...]
  pad = tf.convert_to_tensor([tf.math.reduce_max([instances - dim1, 0])])
  nshape = tf.concat([shape[:pad_axis], pad, shape[(pad_axis + 1):]], axis=0)
  pad_tensor = tf.fill(nshape, tf.cast(pad_value, dtype=value.dtype))
  value = tf.concat([value, pad_tensor], axis=pad_axis)
  return value


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               image_w=416,
               image_h=416,
               num_classes=80,
               fixed_size=False,
               jitter_im=0.2,
               jitter_boxes=0.005,
               letter_box=False,
               use_tie_breaker=True,
               min_level=3,
               max_level=5,
               masks=None,
               cutmix = False,
               mosaic = True, 
               max_process_size=608,
               min_process_size=320,
               max_num_instances=200,
               random_flip=True,
               pct_rand=0.5,
               aug_rand_saturation=True,
               aug_rand_brightness=True,
               aug_rand_zoom=True,
               aug_rand_hue=True,
               anchors=None,
               seed=10,
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.
    Args:
      image_w: a `Tensor` or `int` for width of input image.
      image_h: a `Tensor` or `int` for height of input image.
      num_classes: a `Tensor` or `int` for the number of classes.
      fixed_size: a `bool` if True all output images have the same size.
      jitter_im: a `float` that is the maximum jitter applied to the image for
        data augmentation during training.
      jitter_boxes: a `float` that is the maximum jitter applied to the bounding
        box for data augmentation during training.
      net_down_scale: an `int` that down scales the image width and height to the
        closest multiple of net_down_scale.
      max_process_size: an `int` for maximum image width and height.
      min_process_size: an `int` for minimum image width and height ,
      max_num_instances: an `int` number of maximum number of instances in an image.
      random_flip: a `bool` if True, augment training with random horizontal flip.
      pct_rand: an `int` that prevents do_scale from becoming larger than 1-pct_rand.
      masks: a `Tensor`, `List` or `numpy.ndarrray` for anchor masks.
      aug_rand_saturation: `bool`, if True, augment training with random
        saturation.
      aug_rand_brightness: `bool`, if True, augment training with random
        brightness.
      aug_rand_zoom: `bool`, if True, augment training with random
        zoom.
      aug_rand_hue: `bool`, if True, augment training with random
        hue.
      anchors: a `Tensor`, `List` or `numpy.ndarrray` for bounding box priors.
      seed: an `int` for the seed used by tf.random
    """
    self._net_down_scale = 2**max_level

    self._num_classes = num_classes
    self._image_w = (image_w // self._net_down_scale) * self._net_down_scale
    self._image_h = self._image_w if image_h is None else (
        image_h // self._net_down_scale) * self._net_down_scale

    self._max_process_size = max_process_size
    self._min_process_size = min_process_size

    self._anchors = anchors
    self._masks = {
        key: tf.convert_to_tensor(value) for key, value in masks.items()
    }
    self._use_tie_breaker = use_tie_breaker

    self._jitter_im = 0.0 if jitter_im is None else jitter_im
    self._jitter_boxes = 0.0 if jitter_boxes is None else jitter_boxes
    self._pct_rand = pct_rand
    self._max_num_instances = max_num_instances
    self._random_flip = random_flip
    self._letter_box = letter_box

    self._aug_rand_saturation = aug_rand_saturation
    self._aug_rand_brightness = aug_rand_brightness
    self._aug_rand_zoom = aug_rand_zoom
    self._aug_rand_hue = aug_rand_hue

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
            raw_true, self._masks[key], width // 2**int(key), self._num_classes,
            raw_true['bbox'].dtype, use_tie_breaker)
      else:
        mask[key] = preprocessing_ops.build_batch_grided_gt(
            raw_true, self._masks[key], width // 2**int(key), self._num_classes,
            raw_true['bbox'].dtype, use_tie_breaker)

      mask[key] = tf.cast(mask[key], self._dtype)
    return mask

  def _parse_train_data(self, data):
    """Generates images and labels that are usable for model training.
        Args:
          data: a dict of Tensors produced by the decoder.
        Returns:
          images: the image tensor.
          labels: a dict of Tensors that contains labels.
        """

    image = data['image'] / 255

    # / 255
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']

    do_blur = tf.random.uniform([],
                                minval=0,
                                maxval=1,
                                seed=self._seed,
                                dtype=tf.float32)
    if do_blur > 0.9:
      image = tfa.image.gaussian_filter2d(image, filter_shape=7, sigma=15)
    elif do_blur > 0.7:
      image = tfa.image.gaussian_filter2d(image, filter_shape=5, sigma=6)
    elif do_blur > 0.4:
      image = tfa.image.gaussian_filter2d(image, filter_shape=5, sigma=3)

    image = tf.image.rgb_to_hsv(image)
    i_h, i_s, i_v = tf.split(image, 3, axis=-1)
    if self._aug_rand_hue:
      delta = preprocessing_ops.rand_uniform_strong(
          -0.1, 0.1
      )  # tf.random.uniform([], minval= -0.1,maxval=0.1, seed=self._seed, dtype=tf.float32)
      i_h = i_h + delta  # Hue
      i_h = tf.clip_by_value(i_h, 0.0, 1.0)
    if self._aug_rand_saturation:
      delta = preprocessing_ops.rand_scale(
          0.75
      )  # tf.random.uniform([], minval= 0.5,maxval=1.1, seed=self._seed, dtype=tf.float32)
      i_s = i_s * delta
    if self._aug_rand_brightness:
      delta = preprocessing_ops.rand_scale(
          0.75
      )  # tf.random.uniform([], minval= -0.15,maxval=0.15, seed=self._seed, dtype=tf.float32)
      i_v = i_v * delta
    image = tf.concat([i_h, i_s, i_v], axis=-1)
    image = tf.image.hsv_to_rgb(image)

    stddev = tf.random.uniform([],
                               minval=0,
                               maxval=40 / 255,
                               seed=self._seed,
                               dtype=tf.float32)
    noise = tf.random.normal(
        shape=tf.shape(image), mean=0.0, stddev=stddev, seed=self._seed)
    noise = tf.math.minimum(noise, 0.5)
    noise = tf.math.maximum(noise, 0)
    image += noise
    image = tf.clip_by_value(image, 0.0, 1.0)

    image_shape = tf.shape(image)[:2]

    if self._random_flip:
      image, boxes, _ = preprocess_ops.random_horizontal_flip(
          image, boxes, seed=self._seed)

    if self._jitter_boxes != 0.0:
      boxes = box_ops.denormalize_boxes(boxes, image_shape)
      boxes = box_ops.jitter_boxes(boxes, 0.025)
      boxes = box_ops.normalize_boxes(boxes, image_shape)

    if self._jitter_im != 0.0:
      image, boxes, classes = preprocessing_ops.random_jitter(
          image, boxes, classes, self._jitter_im, seed=self._seed)

    if self._aug_rand_zoom and not self._mosaic:
      image, boxes, classes = preprocessing_ops.random_zoom_crop(
          image, boxes, classes, self._jitter_im)

    shape = tf.shape(image)
    width = shape[1]
    height = shape[0]
    randscale = self._image_w // self._net_down_scale

    if self._fixed_size and not self._mosaic:
      do_scale = tf.greater(
          tf.random.uniform([], minval=0, maxval=1, seed=self._seed),
          1 - self._pct_rand)
      if do_scale:
        randscale = tf.random.uniform([],
                                      minval=10,
                                      maxval=15,
                                      seed=self._seed,
                                      dtype=tf.int32)

    if self._letter_box and not self._mosaic:
      image, boxes = preprocessing_ops.fit_preserve_aspect_ratio(
          image,
          boxes,
          width=width,
          height=height,
          target_dim=randscale * self._net_down_scale)
      width = randscale * self._net_down_scale
      height = randscale * self._net_down_scale
    else:
      minscale = tf.math.minimum(width, height)
      image, boxes, classes = preprocessing_ops.resize_crop_filter(
        image,
        boxes,
        classes,
        default_width=width,  # randscale * self._net_down_scale,
        default_height=height,  # randscale * self._net_down_scale,
        target_width=minscale,
        target_height=minscale,
        randomize=True)

    shape = tf.shape(image)
    width = shape[1]
    height = shape[0]
    image = tf.image.resize(image, (self._image_w, self._image_h))

    
    image = tf.clip_by_value(image, 0.0, 1.0)
    num_dets = tf.shape(classes)[0]
    # padding
    classes = preprocess_ops.clip_or_pad_to_fixed_size(classes,
                                                       self._max_num_instances,
                                                       -1)

    if self._fixed_size and not self._cutmix and not self._mosaic:
      boxes = box_utils.yxyx_to_xcycwh(boxes)
      best_anchors = preprocessing_ops.get_best_anchor(
          boxes, self._anchors, width=self._image_w, height=self._image_h)
      best_anchors = preprocess_ops.clip_or_pad_to_fixed_size(
          best_anchors, self._max_num_instances, 0)
      boxes = preprocess_ops.clip_or_pad_to_fixed_size(boxes,
                                                       self._max_num_instances,
                                                       0)
      labels = {
          'source_id': data['source_id'],
          'bbox': tf.cast(boxes, self._dtype),
          'classes': tf.cast(classes, self._dtype),
          'best_anchors': tf.cast(best_anchors, self._dtype),
          'width': width,
          'height': height,
          'num_detections': num_dets
      }
      grid = self._build_grid(
          labels, self._image_w, use_tie_breaker=self._use_tie_breaker)
      labels.update({'grid_form': grid})
      labels['bbox'] = box_utils.xcycwh_to_yxyx(labels['bbox'])
    else:
      boxes = preprocess_ops.clip_or_pad_to_fixed_size(boxes,
                                                       self._max_num_instances,
                                                       0)
                                  
      labels = {
          'source_id': data['source_id'],
          'bbox': tf.cast(boxes, self._dtype),
          'classes': tf.cast(classes, self._dtype),
          'width': width,
          'height': height,
          'num_detections': num_dets
      }
    return image, labels

  # broken for some reason in task, i think dictionary to coco evaluator has
  # issues
  def _parse_eval_data(self, data):
    """Generates images and labels that are usable for model training.
        Args:
          data: a dict of Tensors produced by the decoder.
        Returns:
          images: the image tensor.
          labels: a dict of Tensors that contains labels.
        """

    shape = tf.shape(data['image'])
    image = data['image'] / 255
    boxes = data['groundtruth_boxes']
    width = shape[1]
    height = shape[0]

    image, boxes = preprocessing_ops.fit_preserve_aspect_ratio(
        image, boxes, width=width, height=height, target_dim=self._image_w)
    boxes = box_utils.yxyx_to_xcycwh(boxes)

    best_anchors = preprocessing_ops.get_best_anchor(
        boxes, self._anchors, width=self._image_w, height=self._image_h)
    boxes = pad_max_instances(boxes, self._max_num_instances, 0)
    classes = pad_max_instances(data['groundtruth_classes'],
                                self._max_num_instances, -1)
    best_anchors = pad_max_instances(best_anchors, self._max_num_instances, 0)
    area = pad_max_instances(data['groundtruth_area'], self._max_num_instances,
                             0)
    is_crowd = pad_max_instances(
        tf.cast(data['groundtruth_is_crowd'], tf.int32),
        self._max_num_instances, 0)

    labels = {
        'source_id': data['source_id'],
        'bbox': tf.cast(boxes, self._dtype),
        'classes': tf.cast(classes, self._dtype),
        'area': tf.cast(area, self._dtype),
        'is_crowd': is_crowd,
        'best_anchors': tf.cast(best_anchors, self._dtype),
        'width': width,
        'height': height,
        'num_detections': tf.shape(data['groundtruth_classes'])[0]
    }

    # if self._fixed_size:
    grid = self._build_grid(
        labels,
        self._image_w,
        batch=False,
        use_tie_breaker=self._use_tie_breaker)
    labels.update({'grid_form': grid})
    labels['bbox'] = box_utils.xcycwh_to_yxyx(labels['bbox'])
    return image, labels

  def _postprocess_fn(self, image, label):
    '''
    if self._cutmix:
      batch_size = tf.shape(image)[0]
      if batch_size >= 1:
        image, boxes, classes, num_detections = preprocessing_ops.randomized_cutmix_batch(
            image, label['bbox'],label['classes'])
        label['bbox'] = pad_max_instances(
            boxes, self._max_num_instances, pad_axis=-2, pad_value=0)
        label['classes'] = pad_max_instances(
            classes, self._max_num_instances, pad_axis=-1, pad_value=-1)
    
    if self._mosaic:
      image, boxes, classes = preprocessing_ops.mosaic(
          image, label['bbox'],label['classes'], self._image_w, crop_delta=0.5)
      label['bbox'] = pad_max_instances(boxes,
                                        self._max_num_instances,
                                        pad_axis=-2,
                                        pad_value=0)
      label['classes'] = pad_max_instances(classes,
                                           self._max_num_instances,
                                           pad_axis=-1,
                                           pad_value=-1)
    '''
  
    return image, label

  def postprocess_fn(self, is_training):
    if is_training:
      return self._postprocess_fn if not self._fixed_size or self._cutmix or self._mosaic else None
    else:
      return None

  # def parse_fn(self, is_training):
  #   """Returns a parse fn that reads and parses raw tensors from the decoder.

  #   Args:
  #     is_training: a `bool` to indicate whether it is in training mode.

  #   Returns:
  #     parse: a `callable` that takes the serialized examle and generate the
  #       images, labels tuple where labels is a dict of Tensors that contains
  #       labels.
  #   """
  #   def parse(decoded_tensors):
  #     """Parses the serialized example data."""
  #     if is_training:
  #       return self._parse_train_data(decoded_tensors)
  #     else:
  #       return self._parse_train_data(decoded_tensors)

  #   return parse
