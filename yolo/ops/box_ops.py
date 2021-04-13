""" bounding box utils file """

# import libraries
import tensorflow as tf
import tensorflow.keras.backend as K
from typing import Tuple, Union
import math
from yolo.ops import math_ops


def yxyx_to_xcycwh(box: tf.Tensor):
  """Converts boxes from ymin, xmin, ymax, xmax to x_center, y_center, width, height.
    Args:
      box: a `Tensor` whose last dimension is 4 representing the coordinates of boxes
        in ymin, xmin, ymax, xmax.
    Returns:
      box: a `Tensor` whose shape is the same as `box` in new format.
    """
  with tf.name_scope('yxyx_to_xcycwh'):
    ymin, xmin, ymax, xmax = tf.split(box, 4, axis=-1)
    x_center = (xmax + xmin) / 2
    y_center = (ymax + ymin) / 2
    width = xmax - xmin
    height = ymax - ymin
    box = tf.concat([x_center, y_center, width, height], axis=-1)
  return box


def xcycwh_to_yxyx(box: tf.Tensor, split_min_max: bool = False):
  """Converts boxes from x_center, y_center, width, height to ymin, xmin, ymax, xmax.
    Args:
      box: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
        x_center, y_center, width, height.
    Returns:
      box: a `Tensor` whose shape is the same as `box` in new format.
    """
  with tf.name_scope('xcycwh_to_yxyx'):
    xy, wh = tf.split(box, 2, axis=-1)
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    x_min, y_min = tf.split(xy_min, 2, axis=-1)
    x_max, y_max = tf.split(xy_max, 2, axis=-1)
    box = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
    if split_min_max:
      box = tf.split(box, 2, axis=-1)
  return box


# IOU
def intersect_and_union(box1, box2, yxyx=False):
  if not yxyx:
    box1 = xcycwh_to_yxyx(box1)
    box2 = xcycwh_to_yxyx(box2)

  b1mi, b1ma = tf.split(box1, 2, axis=-1)
  b2mi, b2ma = tf.split(box2, 2, axis=-1)
  intersect_mins = tf.math.maximum(b1mi, b2mi)
  intersect_maxes = tf.math.minimum(b1ma, b2ma)
  intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins, 0.0)
  intersection = tf.reduce_prod(intersect_wh, axis=-1)

  box1_area = tf.reduce_prod(b1ma - b1mi, axis=-1)
  box2_area = tf.reduce_prod(b2ma - b2mi, axis=-1)
  union = box1_area + box2_area - intersection
  return intersection, union


def smallest_encompassing_box(box1, box2, yxyx=False):
  if not yxyx:
    box1 = xcycwh_to_yxyx(box1)
    box2 = xcycwh_to_yxyx(box2)

  b1mi, b1ma = tf.split(box1, 2, axis=-1)
  b2mi, b2ma = tf.split(box2, 2, axis=-1)

  bcmi = tf.math.minimum(b1mi, b2mi)
  bcma = tf.math.maximum(b1ma, b2ma)

  bca = tf.reduce_prod(bcma - bcmi, keepdims=True, axis=-1)
  box_c = tf.concat([bcmi, bcma], axis=-1)

  if not yxyx:
    box_c = yxyx_to_xcycwh(box_c)

  box_c = tf.where(bca == 0.0, tf.zeros_like(box_c), box_c)
  return box_c


def compute_iou(box1, box2, yxyx=False):
  """Calculates the intersection of union between box1 and box2.
    Args:
        box1: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
        box2: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
    Returns:
        iou: a `Tensor` who represents the intersection over union.
    """
  # get box corners
  with tf.name_scope('iou'):
    intersection, union = intersect_and_union(box1, box2, yxyx=yxyx)
    iou = math_ops.divide_no_nan(intersection, union)
    iou = math_ops.rm_nan_inf(iou, val=0.0)
  return iou


def compute_giou(box1, box2, yxyx=False):
  """Calculates the generalized intersection of union between box1 and box2.
    Args:
        box1: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
        box2: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
    Returns:
        iou: a `Tensor` who represents the generalized intersection over union.
    """
  with tf.name_scope('giou'):
    # get IOU
    if not yxyx:
      box1 = xcycwh_to_yxyx(box1)
      box2 = xcycwh_to_yxyx(box2)
      yxyx = True

    intersection, union = intersect_and_union(box1, box2, yxyx=yxyx)
    iou = math_ops.divide_no_nan(intersection, union)
    iou = math_ops.rm_nan_inf(iou, val=0.0)

    # find the smallest box to encompase both box1 and box2
    boxc = smallest_encompassing_box(box1, box2, yxyx=yxyx)
    if yxyx:
      boxc = yxyx_to_xcycwh(boxc)
    cxcy, cwch = tf.split(boxc, 2, axis=-1)
    c = tf.math.reduce_prod(cwch, axis=-1)

    # compute giou
    regularization = math_ops.divide_no_nan((c - union), c)
    giou = iou - regularization
    giou = tf.clip_by_value(giou, clip_value_min=-1.0, clip_value_max=1.0)
  return iou, giou


def compute_diou(box1, box2, beta=1.0, yxyx=False):
  """Calculates the distance intersection of union between box1 and box2.
    Args:
        box1: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
        box2: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
    Returns:
        iou: a `Tensor` who represents the distance intersection over union.
    """
  with tf.name_scope('diou'):
    # compute center distance
    if not yxyx:
      box1 = xcycwh_to_yxyx(box1)
      box2 = xcycwh_to_yxyx(box2)
      yxyx = True

    intersection, union = intersect_and_union(box1, box2, yxyx=yxyx)
    boxc = smallest_encompassing_box(box1, box2, yxyx=yxyx)

    iou = math_ops.divide_no_nan(intersection, union)
    iou = math_ops.rm_nan_inf(iou, val=0.0)
    if yxyx:
      boxc = yxyx_to_xcycwh(boxc)
      box1 = yxyx_to_xcycwh(box1)
      box2 = yxyx_to_xcycwh(box2)

    b1xy, b1wh = tf.split(box1, 2, axis=-1)
    b2xy, b2wh = tf.split(box2, 2, axis=-1)
    bcxy, bcwh = tf.split(boxc, 2, axis=-1)

    center_dist = tf.reduce_sum((b1xy - b2xy)**2, axis=-1)
    c_diag = tf.reduce_sum(bcwh**2, axis=-1)

    regularization = math_ops.divide_no_nan(center_dist, c_diag)
    diou = iou - regularization**beta
    diou = tf.clip_by_value(diou, clip_value_min=-1.0, clip_value_max=1.0)
  return iou, diou


def compute_ciou(box1, box2, yxyx=False):
  """Calculates the complete intersection of union between box1 and box2.
    Args:
        box1: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
        box2: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
    Returns:
        iou: a `Tensor` who represents the complete intersection over union.
    """
  with tf.name_scope('ciou'):
    # compute DIOU and IOU
    iou, diou = compute_diou(box1, box2, yxyx=yxyx)

    if yxyx:
      box1 = yxyx_to_xcycwh(box1)
      box2 = yxyx_to_xcycwh(box2)

    b1x, b1y, b1w, b1h = tf.split(box1, 4, axis=-1)
    b2x, b2y, b2w, b2h = tf.split(box1, 4, axis=-1)

    # computer aspect ratio consistency
    terma = tf.cast(math_ops.divide_no_nan(b1w, b1h), tf.float32)
    termb = tf.cast(math_ops.divide_no_nan(b2w, b2h), tf.float32)
    arcterm = tf.square(tf.math.atan(terma) - tf.math.atan(termb))
    v = tf.squeeze(4 * arcterm / (math.pi**2), axis=-1)
    v = tf.cast(v, b1w.dtype)

    a = tf.stop_gradient(math_ops.divide_no_nan(v, ((1 - iou) + v)))
    ciou = diou - (v * a)
    ciou = tf.clip_by_value(ciou, clip_value_min=-1.0, clip_value_max=1.0)
  return iou, ciou


# equal to bbox_overlap but far more versitile
def aggregated_comparitive_iou(boxes1,
                               boxes2=None,
                               iou_type=0,
                               beta=0.6,
                               xyxy=True):

  # if boxes2 is not None:
  #   k1 = tf.shape(boxes1)[-2]
  #   k2 = tf.shape(boxes2)[-2]
  # else:
  #   k1 = tf.shape(boxes1)[-2]
  #   k2 = tf.shape(boxes1)[-2]

  boxes1 = tf.expand_dims(boxes1, axis=-2)
  #boxes1 = tf.tile(boxes1, [1, 1, k2, 1])

  if boxes2 is not None:
    boxes2 = tf.expand_dims(boxes2, axis=-3)
    #boxes2 = tf.tile(boxes2, [1, k1, 1, 1])
  else:
    boxes2 = tf.transpose(boxes1, perm=(0, 2, 1, 3))

  if iou_type == 0:  #diou
    _, iou = compute_diou(boxes1, boxes2, beta=beta, yxyx=True)
  elif iou_type == 1:  #giou
    _, iou = compute_giou(boxes1, boxes2, yxyx=True)
  elif iou_type == 2:  #ciou
    _, iou = compute_ciou(boxes1, boxes2, yxyx=True)
  else:
    iou = compute_iou(boxes1, boxes2, yxyx=True)
  return iou
