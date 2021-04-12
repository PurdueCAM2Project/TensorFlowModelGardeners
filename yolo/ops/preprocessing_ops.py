import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from yolo.ops import box_ops
from official.vision.beta.ops import preprocess_ops

def load_data_region(im, w, h, jitter, hue, saturation, exposure):
  oh, ow = tf.shape(im)[:2]

  dw = tf.cast(tf.cast(ow, dtype=tf.float32) * jitter, dtype=tf.int32)
  dh = tf.cast(tf.cast(oh, dtype=tf.float32) * jitter, dtype=tf.int32)

  pleft, pright = tf.random.uniform(shape=[2], minval = -dw, maxval=dw, dtype=tf.int32)
  ptop, pbot = tf.random.uniform(shape=[2], minval = -dh, maxval=dh, dtype=tf.int32)

  swidth = ow - pleft - pright
  sheight = oh - ptop - pbot

  cropped = darknet_crop(im, pleft, ptop, swidth, sheight)
  
  sized = tf.image.resize(im, [w,h])

  flip = tf.random.uniform([], minval=0, maxval=1, dtype=tf.int32)
  if flip:
    sized = tf.image.flip_left_right(sized)

  return random_distort_image(sized, hue, saturation, exposure)


''' WIP CANT CHANGE TENSOR OBJECTS
def darknet_crop(im, dx, dy, w, h):
   
  im_h, im_w, im_c = tf.shape(im) 
  for k in range(im_c):
    for j in range(1, w):
        r = constrain_int(j + dy, 0, im_h-1)
        c = constrain_int(i + dx, 0, im_w-1)
        cropped[i][j][k] = im[r][c][k]
  return cropped

def constrain_int(a, min, max):
  if a < min:
    return min
  if a > max:
    return max
  return a  
'''

def distort_image(im, hue, sat, val):
  orig_type = im.dtype
  im = tf.cast(im, dtype=tf.float32)
  h, w, c = tf.shape(im)

  im = tf.image.rgb_to_hsv(im)
  
  x = tf.squeeze(tf.slice(im, [0, 0, 0], [h, w, 1]))
  y = tf.squeeze(tf.slice(im, [0, 0, 1], [h, w, 1])) * sat
  z = tf.squeeze(tf.slice(im, [0, 0, 2], [h, w, 1])) * val
  im = tf.stack([x, y, z], axis=2)

  im = tf.image.hsv_to_rgb(im)
  im = tf.image.adjust_hue(im, hue)

  return tf.cast(im, dtype=orig_type)


def random_distort_image(im, hue, saturation, exposure):
  dhue = rand_uniform_strong(-hue, hue)
  dsat = rand_scale(saturation)
  dexp = rand_scale(exposure)
  return distort_image(im, dhue, dsat, dexp)


def rand_uniform_strong(minval, maxval, dtype=tf.float32):
  if minval > maxval:
    minval, maxval = maxval, minval
  return tf.random.uniform([], minval=minval, maxval=maxval, dtype=dtype)


def rand_scale(val, dtype=tf.float32):
  scale = rand_uniform_strong(1, val, dtype=dtype)
  do_ret = tf.random.uniform([], minval=0, maxval=1, dtype=tf.int32)
  if (do_ret == 1):
    return scale
  return 1.0 / scale


def shift_zeros(data, mask, axis=-2, fill=0):
  zeros = tf.zeros_like(data) + fill

  data_flat = tf.boolean_mask(data, mask)
  nonzero_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=-2)
  nonzero_mask = tf.sequence_mask(nonzero_lens, maxlen=tf.shape(mask)[-2])
  perm1 = tf.range(0, tf.shape(tf.shape(data))[0] - 2)
  perm2 = tf.roll(
      tf.range(tf.shape(tf.shape(data))[0] - 2,
               tf.shape(tf.shape(data))[0]),
      1,
      axis=-1)

  perm = tf.concat([perm1, perm2], axis=-1)
  nonzero_mask = tf.transpose(nonzero_mask, perm=perm)
  inds = tf.cast(tf.where(nonzero_mask), dtype=tf.int32)
  nonzero_data = tf.tensor_scatter_nd_update(
      zeros, tf.cast(tf.where(nonzero_mask), dtype=tf.int32), data_flat)

  return nonzero_data


def scale_image(image, resize=False, w=None, h=None):
  """Image Normalization.
    Args:
        image(tensorflow.python.framework.ops.Tensor): The image.
    Returns:
        A Normalized Function.
    """
  with tf.name_scope('scale_image'):
    image = tf.convert_to_tensor(image)
    if resize:
      image = tf.image.resize(image, size=(w, h))
    image = image / 255
  return image


def random_jitter(image, box, classes, t, seed=10):
  jx = 1 + tf.random.uniform(minval=-t, maxval=t, shape=(), dtype=tf.float32)
  jy = jx  # 1 + tf.random.uniform(minval=-t, maxval=t, shape=(), dtype=tf.float32)
  shape = tf.shape(image)

  if tf.shape(shape)[0] == 4:
    width = tf.cast(shape[2], jx.dtype) * jx
    height = tf.cast(shape[1], jy.dtype) * jy
  else:
    width = tf.cast(shape[1], jx.dtype) * jx
    height = tf.cast(shape[0], jy.dtype) * jy

  image = tf.image.resize(image, (height, width))
  return image, box, classes


def random_translate(image, box, classes, t, seed=10):
  t_x = tf.random.uniform(minval=-t, maxval=t, shape=(), dtype=tf.float32)
  t_y = tf.random.uniform(minval=-t, maxval=t, shape=(), dtype=tf.float32)
  box, classes = translate_boxes(box, classes, t_x, t_y)
  image = translate_image(image, t_x, t_y)
  return image, box, classes


def random_zoom_crop(image, boxes, classes, zoom_factor):
  jx = 1 + tf.random.uniform(
      minval=-zoom_factor, maxval=zoom_factor, shape=(), dtype=tf.float32)
  jy = 1 + tf.random.uniform(
      minval=-zoom_factor, maxval=zoom_factor, shape=(), dtype=tf.float32)
  shape = tf.shape(image)
  if tf.shape(shape)[0] == 4:
    width = shape[2]
    height = shape[1]
  else:
    width = shape[1]
    height = shape[0]
  image, boxes, classes = resize_crop_filter(
      image,
      boxes,
      classes,
      default_width=width,
      default_height=height,
      target_width=tf.cast(tf.cast(width, jx.dtype) * jx, tf.int32),
      target_height=tf.cast(tf.cast(height, jy.dtype) * jy, tf.int32),
      randomize=True)
  return image, boxes, classes


def translate_boxes(box, classes, translate_x, translate_y):
  with tf.name_scope('translate_boxs'):
    box = box_ops.yxyx_to_xcycwh(box)
    x, y, w, h = tf.split(box, 4, axis=-1)
    x = x + translate_x
    y = y + translate_y

    x_mask_lower = x >= 0
    y_mask_lower = y >= 0
    x_mask_upper = x < 1
    y_mask_upper = y < 1

    x_mask = tf.math.logical_and(x_mask_lower, x_mask_upper)
    y_mask = tf.math.logical_and(y_mask_lower, y_mask_upper)
    mask = tf.math.logical_and(x_mask, y_mask)

    x = shift_zeros(x, mask)  # tf.boolean_mask(x, mask)
    y = shift_zeros(y, mask)  # tf.boolean_mask(y, mask)
    w = shift_zeros(w, mask)  # tf.boolean_mask(w, mask)
    h = shift_zeros(h, mask)  # tf.boolean_mask(h, mask)
    classes = shift_zeros(tf.expand_dims(classes, axis=-1), mask)
    classes = tf.squeeze(classes, axis=-1)

    box = tf.concat([x, y, w, h], axis=-1)
    box = box_ops.xcycwh_to_yxyx(box)
  return box, classes


def translate_image(image, translate_x, translate_y):
  with tf.name_scope('translate_image'):
    if (translate_x != 0 and translate_y != 0):
      image_jitter = tf.convert_to_tensor([translate_x, translate_y])
      image_jitter.set_shape([2])
      image = tfa.image.translate(
          image, image_jitter * tf.cast(tf.shape(image)[1], tf.float32))
  return image


def resize_crop_filter(image,
                       boxes,
                       classes,
                       default_width,
                       default_height,
                       target_width,
                       target_height,
                       randomize=False):
  with tf.name_scope('resize_crop_filter'):
    dx = (tf.math.maximum(default_width, target_width) -
          tf.math.minimum(default_width, target_width)) // 2
    dy = (tf.math.maximum(default_height, target_height) -
          tf.math.minimum(default_height, target_height)) // 2

    if randomize:
      dx = tf.random.uniform([], minval=0, maxval=dx *
                             2, dtype=tf.int32) if dx != 0 else 0
      dy = tf.random.uniform([], minval=0, maxval=dy *
                             2, dtype=tf.int32) if dy != 0 else 0

    if target_width > default_width:
      image, boxes, classes = pad_filter_to_bbox(image, boxes, classes,
                                                 target_width, default_height,
                                                 dx, 0)
    elif target_width < default_width:
      image, boxes, classes = crop_filter_to_bbox(
          image, boxes, classes, target_width, default_height, dx, 0, fix=False)

    if target_height > default_height:
      image, boxes, classes = pad_filter_to_bbox(image, boxes, classes,
                                                 target_width, target_height, 0,
                                                 dy)
    elif target_height < default_height:
      image, boxes, classes = crop_filter_to_bbox(
          image, boxes, classes, target_width, target_height, 0, dy, fix=False)

  return image, boxes, classes


def crop_filter_to_bbox(image,
                        boxes,
                        classes,
                        target_width,
                        target_height,
                        offset_width,
                        offset_height,
                        fix=False):
  with tf.name_scope('resize_crop_filter'):
    shape = tf.shape(image)

    if tf.shape(shape)[0] == 4:
      width = shape[2]
      height = shape[1]
    else:  # tf.shape(shape)[0] == 3:
      width = shape[1]
      height = shape[0]

    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                          target_height, target_width)
    if fix:
      image = tf.image.pad_to_bounding_box(image, offset_height, offset_width,
                                           height, width)

    x_lower_bound = offset_width / width
    y_lower_bound = offset_height / height

    x_upper_bound = (offset_width + target_width) / width
    y_upper_bound = (offset_height + target_height) / height

    boxes = box_ops.yxyx_to_xcycwh(boxes)
    x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)

    x_mask_lower = x > x_lower_bound
    y_mask_lower = y > y_lower_bound
    x_mask_upper = x < x_upper_bound
    y_mask_upper = y < y_upper_bound

    x_mask = tf.math.logical_and(x_mask_lower, x_mask_upper)
    y_mask = tf.math.logical_and(y_mask_lower, y_mask_upper)

    mask = tf.math.logical_and(x_mask, y_mask)

    x = shift_zeros(x, mask)  # tf.boolean_mask(x, mask)
    y = shift_zeros(y, mask)  # tf.boolean_mask(y, mask)
    w = shift_zeros(w, mask)  # tf.boolean_mask(w, mask)
    h = shift_zeros(h, mask)  # tf.boolean_mask(h, mask)
    classes = shift_zeros(tf.expand_dims(classes, axis=-1), mask, fill=-1)
    classes = tf.squeeze(classes, axis=-1)

    if not fix:
      x = (x - x_lower_bound) * tf.cast(width / target_width, x.dtype)
      y = (y - y_lower_bound) * tf.cast(height / target_height, y.dtype)
      w = w * tf.cast(width / target_width, w.dtype)
      h = h * tf.cast(height / target_height, h.dtype)

    boxes = tf.cast(tf.concat([x, y, w, h], axis=-1), boxes.dtype)
    boxes = box_ops.xcycwh_to_yxyx(boxes)
  return image, boxes, classes


def pad_filter_to_bbox(image, boxes, classes, target_width, target_height,
                       offset_width, offset_height):
  with tf.name_scope('resize_crop_filter'):
    shape = tf.shape(image)

    if tf.shape(shape)[0] == 4:
      width = shape[2]
      height = shape[1]
    else:  # tf.shape(shape)[0] == 3:
      width = shape[1]
      height = shape[0]

    image = tf.image.pad_to_bounding_box(image, offset_height, offset_width,
                                         target_height, target_width)

    x_lower_bound = tf.cast(offset_width / width, tf.float32)
    y_lower_bound = tf.cast(offset_height / height, tf.float32)

    boxes = box_ops.yxyx_to_xcycwh(boxes)
    x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)

    x = (x + x_lower_bound) * tf.cast(width / target_width, x.dtype)
    y = (y + y_lower_bound) * tf.cast(height / target_height, y.dtype)
    w = w * tf.cast(width / target_width, w.dtype)
    h = h * tf.cast(height / target_height, h.dtype)

    boxes = tf.cast(tf.concat([x, y, w, h], axis=-1), boxes.dtype)
    boxes = box_ops.xcycwh_to_yxyx(boxes)
  return image, boxes, classes


def cut_out(image_full, boxes, classes, target_width, target_height,
            offset_width, offset_height):
  shape = tf.shape(image_full)

  if tf.shape(shape)[0] == 4:
    width = shape[2]
    height = shape[1]
  else:  # tf.shape(shape)[0] == 3:
    width = shape[1]
    height = shape[0]

  image_crop = tf.image.crop_to_bounding_box(
      image_full, offset_height, offset_width, target_height, target_width) + 1
  image_crop = tf.ones_like(image_crop)
  image_crop = tf.image.pad_to_bounding_box(image_crop, offset_height,
                                            offset_width, height, width)
  image_crop = 1 - image_crop

  x_lower_bound = offset_width / width
  y_lower_bound = offset_height / height

  x_upper_bound = (offset_width + target_width) / width
  y_upper_bound = (offset_height + target_height) / height

  boxes = box_ops.yxyx_to_xcycwh(boxes)

  x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)

  x_mask_lower = x > x_lower_bound
  y_mask_lower = y > y_lower_bound
  x_mask_upper = x < x_upper_bound
  y_mask_upper = y < y_upper_bound

  x_mask = tf.math.logical_and(x_mask_lower, x_mask_upper)
  y_mask = tf.math.logical_and(y_mask_lower, y_mask_upper)
  mask = tf.math.logical_not(tf.math.logical_and(x_mask, y_mask))

  x = shift_zeros(x, mask)  # tf.boolean_mask(x, mask)
  y = shift_zeros(y, mask)  # tf.boolean_mask(y, mask)
  w = shift_zeros(w, mask)  # tf.boolean_mask(w, mask)
  h = shift_zeros(h, mask)  # tf.boolean_mask(h, mask)
  classes = shift_zeros(tf.expand_dims(classes, axis=-1), mask, fill=-1)
  classes = tf.squeeze(classes, axis=-1)

  boxes = tf.cast(tf.concat([x, y, w, h], axis=-1), boxes.dtype)
  boxes = box_ops.xcycwh_to_yxyx(boxes)

  image_full *= image_crop
  return image_full, boxes, classes


def cutmix_1(image_to_crop, boxes1, classes1, image_mask, boxes2, classes2,
             target_width, target_height, offset_width, offset_height):
  with tf.name_scope('cutmix'):
    image, boxes, classes = cut_out(image_mask, boxes2, classes2, target_width,
                                    target_height, offset_width, offset_height)
    image_, boxes_, classes_ = crop_filter_to_bbox(
        image_to_crop,
        boxes1,
        classes1,
        target_width,
        target_height,
        offset_width,
        offset_height,
        fix=True)
    image += image_
    boxes = tf.concat([boxes, boxes_], axis=-2)
    classes = tf.concat([classes, classes_], axis=-1)

    boxes = box_ops.yxyx_to_xcycwh(boxes)
    x, y, w, h = tf.split(boxes, 4, axis=-1)

    mask = x > 0
    x = shift_zeros(x, mask)  # tf.boolean_mask(x, mask)
    y = shift_zeros(y, mask)  # tf.boolean_mask(y, mask)
    w = shift_zeros(w, mask)  # tf.boolean_mask(w, mask)
    h = shift_zeros(h, mask)  # tf.boolean_mask(h, mask)
    classes = shift_zeros(tf.expand_dims(classes, axis=-1), mask, fill=-1)
    classes = tf.squeeze(classes, axis=-1)

    boxes = tf.cast(tf.concat([x, y, w, h], axis=-1), boxes.dtype)
    boxes = box_ops.xcycwh_to_yxyx(boxes)

  return image, boxes, classes


def cutmix_batch(image, boxes, classes, target_width, target_height,
                 offset_width, offset_height):
  with tf.name_scope('cutmix_batch'):

    image_, boxes_, classes_ = cut_out(image, boxes, classes, target_width,
                                       target_height, offset_width,
                                       offset_height)
    image__, boxes__, classes__ = crop_filter_to_bbox(
        image,
        boxes,
        classes,
        target_width,
        target_height,
        offset_width,
        offset_height,
        fix=True)

    mix = tf.random.uniform([], minval=0, maxval=1)
    if mix > 0.5:
      i_split1, i_split2 = tf.split(image__, 2, axis=0)
      b_split1, b_split2 = tf.split(boxes__, 2, axis=0)
      c_split1, c_split2 = tf.split(classes__, 2, axis=0)

      image__ = tf.concat([i_split2, i_split1], axis=0)
      boxes__ = tf.concat([b_split2, b_split1], axis=0)
      classes__ = tf.concat([c_split2, c_split1], axis=0)

    image = image_ + image__
    boxes = tf.concat([boxes_, boxes__], axis=-2)
    classes = tf.concat([classes_, classes__], axis=-1)

    boxes = box_ops.yxyx_to_xcycwh(boxes)
    x, y, w, h = tf.split(boxes, 4, axis=-1)

    mask = x > 0
    x = shift_zeros(x, mask)  # tf.boolean_mask(x, mask)
    y = shift_zeros(y, mask)  # tf.boolean_mask(y, mask)
    w = shift_zeros(w, mask)  # tf.boolean_mask(w, mask)
    h = shift_zeros(h, mask)  # tf.boolean_mask(h, mask)
    classes = shift_zeros(tf.expand_dims(classes, axis=-1), mask, fill=-1)
    classes = tf.squeeze(classes, axis=-1)

    boxes = tf.cast(tf.concat([x, y, w, h], axis=-1), boxes.dtype)
    boxes = box_ops.xcycwh_to_yxyx(boxes)

    x = tf.squeeze(x, axis=-1)

    num_detections = tf.reduce_sum(tf.cast(x > 0, tf.int32), axis=-1)

  return image, boxes, classes, num_detections


def randomized_cutmix_batch(image, boxes, classes):
  shape = tf.shape(image)

  width = shape[2]
  height = shape[1]

  w_limit = 3 * width // 4
  h_limit = 3 * height // 4

  twidth = tf.random.uniform([],
                             minval=width // 4,
                             maxval=w_limit,
                             dtype=tf.int32)
  theight = tf.random.uniform([],
                              minval=height // 4,
                              maxval=h_limit,
                              dtype=tf.int32)

  owidth = tf.random.uniform([],
                             minval=0,
                             maxval=width - twidth,
                             dtype=tf.int32)
  oheight = tf.random.uniform([],
                              minval=0,
                              maxval=height - theight,
                              dtype=tf.int32)

  image, boxes, classes, num_detections = cutmix_batch(image, boxes, classes,
                                                       twidth, theight, owidth,
                                                       oheight)
  return image, boxes, classes, num_detections


def randomized_cutmix_split(image, boxes, classes):
  # this is not how it is really done
  mix = tf.random.uniform([], maxval=1, dtype=tf.int32)
  if mix == 1:
    i1, i2, i3, i4 = tf.split(image, 4, axis=0)
    b1, b2, b3, b4 = tf.split(boxes, 2, axis=0)
    c1, c2, c3, c4 = tf.split(classes, 2, axis=0)

    image = tf.concat([i1, i3, i2, i4], axis=0)
    boxes = tf.concat([b1, b3, b2, b4], axis=0)
    classes = tf.concat([b1, b3, b2, b4], axis=0)

  i_split1, i_split2 = tf.split(image, 2, axis=0)
  b_split1, b_split2 = tf.split(boxes, 2, axis=0)
  c_split1, c_split2 = tf.split(classes, 2, axis=0)

  i_split1, b_split1, c_split1, num_dets1 = randomized_cutmix_batch(
      i_split1, b_split1, c_split1)
  i_split2, b_split2, c_split2, num_dets2 = randomized_cutmix_batch(
      i_split2, b_split2, c_split2)
  image = tf.concat([i_split2, i_split1], axis=0)
  boxes = tf.concat([b_split2, b_split1], axis=0)
  classes = tf.concat([c_split2, c_split1], axis=0)
  num_detections = tf.concat([num_dets2, num_dets1], axis=0)
  #image, boxes, classes, num_detections = randomized_cutmix_batch(image, boxes, classes)

  return image, boxes, classes, num_detections


def fit_preserve_aspect_ratio(image,
                              boxes,
                              width=None,
                              height=None,
                              target_dim=None):
  if width is None or height is None:
    shape = tf.shape(data['image'])
    if tf.shape(shape)[0] == 4:
      width = shape[2]
      height = shape[1]
    else:
      width = shape[1]
      height = shape[0]

  clipper = tf.math.maximum(width, height)
  if target_dim is None:
    target_dim = clipper

  pad_width = clipper - width
  pad_height = clipper - height
  image = tf.image.pad_to_bounding_box(image, pad_height // 2, pad_width // 2,
                                       clipper, clipper)

  boxes = box_ops.yxyx_to_xcycwh(boxes)
  x, y, w, h = tf.split(boxes, 4, axis=-1)

  y *= tf.cast(height / clipper, tf.float32)
  x *= tf.cast(width / clipper, tf.float32)

  y += tf.cast((pad_height / clipper) / 2, tf.float32)
  x += tf.cast((pad_width / clipper) / 2, tf.float32)

  h *= tf.cast(height / clipper, tf.float32)
  w *= tf.cast(width / clipper, tf.float32)

  boxes = tf.concat([x, y, w, h], axis=-1)

  boxes = box_ops.xcycwh_to_yxyx(boxes)
  image = tf.image.resize(image, (target_dim, target_dim))
  return image, boxes


def get_best_anchor(y_true, anchors, width=1, height=1):
  """
    get the correct anchor that is assoiciated with each box using IOU
    Args:
      y_true: tf.Tensor[] for the list of bounding boxes in the yolo format
      anchors: list or tensor for the anchor boxes to be used in prediction
        found via Kmeans
      width: int for the image width
      height: int for the image height

    Return:
      tf.Tensor: y_true with the anchor associated with each ground truth
      box known
    """
  with tf.name_scope('get_anchor'):
    width = tf.cast(width, dtype=y_true.dtype)
    height = tf.cast(height, dtype=y_true.dtype)
    # split the boxes into center and width height
    anchor_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]

    # scale thhe boxes
    anchors = tf.convert_to_tensor(anchors, dtype=y_true.dtype)
    anchors_x = anchors[..., 0] / width
    anchors_y = anchors[..., 1] / height
    anchors = tf.stack([anchors_x, anchors_y], axis=-1)
    k = tf.shape(anchors)[0]
    # build a matrix of anchor boxes of shape [num_anchors, num_boxes, 4]
    anchors = tf.transpose(anchors, perm=[1, 0])

    anchor_xy = tf.tile(
        tf.expand_dims(anchor_xy, axis=-1), [1, 1, tf.shape(anchors)[-1]])
    anchors = tf.tile(
        tf.expand_dims(anchors, axis=0), [tf.shape(anchor_xy)[0], 1, 1])

    # stack the xy so, each anchor is asscoaited once with each center from
    # the ground truth input
    anchors = K.concatenate([anchor_xy, anchors], axis=1)
    anchors = tf.transpose(anchors, perm=[2, 0, 1])

    # copy the gt n times so that each anchor from above can be compared to
    # input ground truth to shape: [num_anchors, num_boxes, 4]
    truth_comp = tf.tile(
        tf.expand_dims(y_true[..., 0:4], axis=-1),
        [1, 1, tf.shape(anchors)[0]])
    truth_comp = tf.transpose(truth_comp, perm=[2, 0, 1])

    # compute intersection over union of the boxes, and take the argmax of
    # comuted iou for each box. thus each box is associated with the
    # largest interection over union
    iou_raw = box_ops.compute_iou(truth_comp, anchors)
    values, indexes = tf.math.top_k(
        tf.transpose(iou_raw, perm=[1, 0]),
        k=tf.cast(k, dtype=tf.int32),
        sorted=True)
    ind_mask = tf.cast(values > 0.213, dtype=indexes.dtype)

    # pad the indexs such that all values less than the thresh are -1
    # add one, multiply the mask to zeros all the bad locations
    # subtract 1 makeing all the bad locations 0.
    iou_index = tf.concat([
        K.expand_dims(indexes[..., 0], axis=-1),
        ((indexes[..., 1:] + 1) * ind_mask[..., 1:]) - 1
    ],
                          axis=-1)
    iou_index = iou_index[..., :6]

  return tf.cast(iou_index, dtype=tf.float32)


def get_best_anchor_batch(y_true, anchors, width=1, height=1):
  """
    get the correct anchor that is assoiciated with each box using IOU
    Args:
      y_true: tf.Tensor[] for the list of bounding boxes in the yolo format
      anchors: list or tensor for the anchor boxes to be used in prediction
        found via Kmeans
      width: int for the image width
      height: int for the image height

    Return:
      tf.Tensor: y_true with the anchor associated with each ground truth
      box known
    """
  with tf.name_scope('get_anchor'):
    width = tf.cast(width, dtype=y_true.dtype)
    height = tf.cast(height, dtype=y_true.dtype)
    # split the boxes into center and width height
    anchor_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]

    # scale thhe boxes
    anchors = tf.convert_to_tensor(anchors, dtype=y_true.dtype)
    anchors_x = anchors[..., 0] / width
    anchors_y = anchors[..., 1] / height
    anchors = tf.stack([anchors_x, anchors_y], axis=-1)
    k = tf.shape(anchors)[0]
    # build a matrix of anchor boxes of shape [num_anchors, num_boxes, 4]
    anchors = tf.transpose(anchors, perm=[1, 0])

    anchor_xy = tf.repeat(
        tf.expand_dims(anchor_xy, axis=-1), tf.shape(anchors)[-1], axis=-1)
    anchors = tf.repeat(
        tf.expand_dims(anchors, axis=0), tf.shape(anchor_xy)[1], axis=0)
    anchors = tf.repeat(
        tf.expand_dims(anchors, axis=0), tf.shape(anchor_xy)[0], axis=0)

    # stack the xy so, each anchor is asscoaited once with each center from
    # the ground truth input
    anchors = K.concatenate([anchor_xy, anchors], axis=2)
    anchors = tf.transpose(anchors, perm=[0, 3, 1, 2])

    # # copy the gt n times so that each anchor from above can be compared to
    # # input ground truth to shape: [num_anchors, num_boxes, 4]
    truth_comp = tf.tile(
        tf.expand_dims(y_true[..., 0:4], axis=-1),
        [1, 1, 1, tf.shape(anchors)[1]])
    truth_comp = tf.transpose(truth_comp, perm=[0, 3, 1, 2])

    # compute intersection over union of the boxes, and take the argmax of
    # comuted iou for each box. thus each box is associated with the
    # largest interection over union
    iou_raw = box_ops.compute_iou(truth_comp, anchors)
    values, indexes = tf.math.top_k(
        tf.transpose(iou_raw, perm=[0, 2, 1]),
        k=tf.cast(k, dtype=tf.int32),
        sorted=True)
    ind_mask = tf.cast(values > 0.213, dtype=indexes.dtype)

    # pad the indexs such that all values less than the thresh are -1
    # add one, multiply the mask to zeros all the bad locations
    # subtract 1 makeing all the bad locations 0.
    iou_index = tf.concat([
        K.expand_dims(indexes[..., 0], axis=-1),
        ((indexes[..., 1:] + 1) * ind_mask[..., 1:]) - 1
    ],
                          axis=-1)
    iou_index = iou_index[..., :6]

  return tf.cast(iou_index, dtype=tf.float32)


def build_grided_gt(y_true, mask, size, num_classes, dtype, use_tie_breaker):
  """
    convert ground truth for use in loss functions
    Args:
      y_true: tf.Tensor[] ground truth
        [box coords[0:4], classes_onehot[0:-1], best_fit_anchor_box]
      mask: list of the anchor boxes choresponding to the output,
        ex. [1, 2, 3] tells this layer to predict only the first 3
        anchors in the total.
      size: the dimensions of this output, for regular, it progresses
        from 13, to 26, to 52
      num_classes: `integer` for the number of classes
      dtype: expected output datatype
      use_tie_breaker: boolean value for wether or not to use
        the tie_breaker

    Return:
      tf.Tensor[] of shape [size, size, #of_anchors, 4, 1, num_classes]
  """
  # unpack required components from the input ground truth
  boxes = tf.cast(y_true['bbox'], dtype)
  classes = tf.expand_dims(tf.cast(y_true['classes'], dtype=dtype), axis=-1)
  anchors = tf.cast(y_true['best_anchors'], dtype)

  # get the number of boxes in the ground truth boxs
  num_boxes = tf.shape(boxes)[0]
  # get the number of anchor boxes used for this anchor scale
  len_masks = tf.shape(mask)[0]

  # init a fixed memeory size grid for this prediction scale
  # [size, size, # of anchors, 1 + 1 + number of anchors per scale]
  full = tf.zeros([size, size, len_masks, 6], dtype=dtype)
  # init a grid to use to track which locations have already
  # been used before (for the tie breaker)
  depth_track = tf.zeros((size, size, len_masks), dtype=tf.int32)

  # rescale the x and y centers to the size of the grid [size, size]
  x = tf.cast(boxes[..., 0] * tf.cast(size, dtype=dtype), dtype=tf.int32)
  y = tf.cast(boxes[..., 1] * tf.cast(size, dtype=dtype), dtype=tf.int32)

  # init all the tensorArrays to be used in storeing the index
  # and the values to be used to update both depth_track and full
  update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  update = tf.TensorArray(dtype, size=0, dynamic_size=True)

  # init constants and match data types before entering loop
  i = 0
  anchor_id = 0
  const = tf.cast(tf.convert_to_tensor([1.]), dtype=dtype)
  mask = tf.cast(mask, dtype=dtype)
  rand_update = 0.0

  for box_id in range(num_boxes):
    # if the width or height of the box is zero, skip it
    if K.all(tf.math.equal(boxes[box_id, 2:4], 0)):
      continue
    # after pre processing, if the box is not in the i
    # image bounds anymore skip the box
    if K.any(tf.math.less(boxes[box_id, 0:2], 0.0)) or K.any(
        tf.math.greater_equal(boxes[box_id, 0:2], 1.0)):
      continue
    if use_tie_breaker:
      for anchor_id in range(tf.shape(anchors)[-1]):
        index = tf.math.equal(anchors[box_id, anchor_id], mask)
        if K.any(index):
          # using the boolean index mask to determine exactly which
          # anchor box was used
          p = tf.cast(K.argmax(tf.cast(index, dtype=tf.int32)), dtype=tf.int32)
          # determine if the index was used or not
          used = depth_track[y[box_id], x[box_id], p]
          # defualt used upadte value
          uid = 1

          # if anchor_id is 0, this is the best matched anchor for this box
          # with the highest IOU
          if anchor_id == 0:
            # write the box to the update list
            # create random numbr to trigger a replacment if the cell
            # is used already
            if tf.math.equal(used, 1):
              rand_update = tf.random.uniform([], maxval=1)
            else:
              rand_update = 1.0

            if rand_update > 0.5:
              # write the box to the update list
              update_index = update_index.write(i, [y[box_id], x[box_id], p])
              value = K.concatenate([boxes[box_id], const, classes[box_id]])
              update = update.write(i, value)

          # if used is 2, this cell is filled with a non-optimal box
          # if used is 0, the cell in the ground truth is not yet consumed
          # in either case you can replace that cell with a new box, as long
          # as it is not consumed by an optimal box with anchor_id = 0
          elif tf.math.equal(used, 2) or tf.math.equal(used, 0):
            uid = 2
            # write the box to the update list
            update_index = update_index.write(i, [y[box_id], x[box_id], p])
            value = K.concatenate([boxes[box_id], const, classes[box_id]])
            update = update.write(i, value)

          depth_track = tf.tensor_scatter_nd_update(depth_track,
                                                    [(y[box_id], x[box_id], p)],
                                                    [uid])
          i += 1
    else:
      index = tf.math.equal(anchors[box_id, 0], mask)
      # if any there is an index match
      if K.any(index):
        # find the index
        p = tf.cast(K.argmax(tf.cast(index, dtype=tf.int32)), dtype=tf.int32)
        # update the list of used boxes
        update_index = update_index.write(i, [y[box_id], x[box_id], p])
        value = K.concatenate([boxes[box_id], const, classes[box_id]])
        update = update.write(i, value)
        i += 1

  # if the size of the update list is not 0, do an update, other wise,
  # no boxes and pass an empty grid
  if tf.math.greater(update_index.size(), 0):
    update_index = update_index.stack()
    update = update.stack()
    full = tf.tensor_scatter_nd_update(full, update_index, update)
  return full


def build_batch_grided_gt(y_true, mask, size, num_classes, dtype,
                          use_tie_breaker):
  """
    convert ground truth for use in loss functions
    Args:
      y_true: tf.Tensor[] ground truth
        [batch, box coords[0:4], classes_onehot[0:-1], best_fit_anchor_box]
      mask: list of the anchor boxes choresponding to the output,
        ex. [1, 2, 3] tells this layer to predict only the first 3 anchors
        in the total.
      size: the dimensions of this output, for regular, it progresses from
        13, to 26, to 52
      num_classes: `integer` for the number of classes
      dtype: expected output datatype
      use_tie_breaker: boolean value for wether or not to use the tie
        breaker

    Return:
      tf.Tensor[] of shape [batch, size, size, #of_anchors, 4, 1, num_classes]
  """
  # unpack required components from the input ground truth
  boxes = tf.cast(y_true['bbox'], dtype)
  classes = tf.expand_dims(tf.cast(y_true['classes'], dtype=dtype), axis=-1)
  anchors = tf.cast(y_true['best_anchors'], dtype)

  # get the batch size
  batches = tf.shape(boxes)[0]
  # get the number of boxes in the ground truth boxs
  num_boxes = tf.shape(boxes)[1]
  # get the number of anchor boxes used for this anchor scale
  len_masks = tf.shape(mask)[0]

  # init a fixed memeory size grid for this prediction scale
  # [batch, size, size, # of anchors, 1 + 1 + number of anchors per scale]
  full = tf.zeros([batches, size, size, len_masks, 1 + 4 + 1], dtype=dtype)
  # init a grid to use to track which locations have already
  # been used before (for the tie breaker)
  depth_track = tf.zeros((batches, size, size, len_masks), dtype=tf.int32)

  # rescale the x and y centers to the size of the grid [size, size]
  x = tf.cast(boxes[..., 0] * tf.cast(size, dtype=dtype), dtype=tf.int32)
  y = tf.cast(boxes[..., 1] * tf.cast(size, dtype=dtype), dtype=tf.int32)

  # init all the tensorArrays to be used in storeing the index and the values
  # to be used to update both depth_track and full
  update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  update = tf.TensorArray(dtype, size=0, dynamic_size=True)

  # init constants and match data types before entering loop
  i = 0
  anchor_id = 0
  const = tf.cast(tf.convert_to_tensor([1.]), dtype=dtype)
  mask = tf.cast(mask, dtype=dtype)
  rand_update = 0.0

  for batch in range(batches):
    for box_id in range(num_boxes):
      # if the width or height of the box is zero, skip it
      if K.all(tf.math.equal(boxes[batch, box_id, 2:4], 0)):
        continue
      # after pre processing, if the box is not in the image bounds anymore
      # skip the box
      if K.any(tf.math.less(boxes[batch, box_id, 0:2], 0.0)) or K.any(
          tf.math.greater_equal(boxes[batch, box_id, 0:2], 1.0)):
        continue
      if use_tie_breaker:
        for anchor_id in range(tf.shape(anchors)[-1]):
          index = tf.math.equal(anchors[batch, box_id, anchor_id], mask)
          if K.any(index):
            # using the boolean index mask to determine exactly which anchor
            #  box was used
            p = tf.cast(
                K.argmax(tf.cast(index, dtype=tf.int32)), dtype=tf.int32)
            # determine if the index was used or not
            used = depth_track[batch, y[batch, box_id], x[batch, box_id], p]
            # defualt used upadte value
            uid = 1

            # if anchor_id is 0, this is the best matched anchor for this box
            # with the highest IOU
            if anchor_id == 0:
              # create random number to trigger a replacment if the cell
              # is used already
              if tf.math.equal(used, 1):
                rand_update = tf.random.uniform([], maxval=1)
              else:
                rand_update = 1.0

              if rand_update > 0.5:
                # write the box to the update list
                update_index = update_index.write(
                    i, [batch, y[batch, box_id], x[batch, box_id], p])
                value = K.concatenate(
                    [boxes[batch, box_id], const, classes[batch, box_id]])
                update = update.write(i, value)

            # if used is 2, this cell is filled with a non-optimal box
            # if used is 0, the cell in the ground truth is not yet consumed
            # in either case you can replace that cell with a new box, as long
            # as it is not consumed by an optimal box with anchor_id = 0
            elif tf.math.equal(used, 2) or tf.math.equal(used, 0):
              uid = 2
              # write the box to the update list
              update_index = update_index.write(
                  i, [batch, y[batch, box_id], x[batch, box_id], p])
              value = K.concatenate(
                  [boxes[batch, box_id], const, classes[batch, box_id]])
              update = update.write(i, value)

            # update the used index for where and how the box was placed
            depth_track = tf.tensor_scatter_nd_update(
                depth_track, [(batch, y[batch, box_id], x[batch, box_id], p)],
                [uid])
            i += 1
      else:
        index = tf.math.equal(anchors[batch, box_id, 0], mask)
        if K.any(index):
          # if any there is an index match
          p = tf.cast(K.argmax(tf.cast(index, dtype=tf.int32)), dtype=tf.int32)
          # write the box to the update list
          update_index = update_index.write(
              i, [batch, y[batch, box_id], x[batch, box_id], p])
          value = K.concatenate(
              [boxes[batch, box_id], const, classes[batch, box_id]])
          update = update.write(i, value)
          i += 1

  # if the size of the update list is not 0, do an update, other wise,
  # no boxes and pass an empty grid
  if tf.math.greater(update_index.size(), 0):
    update_index = update_index.stack()
    update = update.stack()
    full = tf.tensor_scatter_nd_update(full, update_index, update)
  return full


def unpad_tensor(input_tensor, padding_value=0):
  if tf.rank(input_tensor) == 3:
    abs_sum_tensor = tf.reduce_sum(tf.abs(input_tensor), -1)
    padding_vector = tf.ones(shape=(1, 1)) * padding_value
    mask = abs_sum_tensor != padding_vector
    return input_tensor[mask]
  elif tf.rank(input_tensor) == 2:
    return input_tensor[input_tensor != padding_value]
