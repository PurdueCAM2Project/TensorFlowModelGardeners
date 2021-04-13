import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K

from yolo.ops.loss_utils import GridGenerator
from yolo.ops import box_ops
from yolo.ops import math_ops
import numpy as np

from functools import partial

TILE_SIZE = 10


@tf.custom_gradient
def obj_gradient_trap(y, max_delta=np.inf):

  def trap(dy):
    dy = math_ops.rm_nan_inf(dy)
    delta = tf.cast(max_delta, dy.dtype)
    dy = tf.clip_by_value(dy, -delta, delta)
    # tf.print(tf.reduce_sum(tf.square(dy)))
    return dy, 0.0

  return y, trap


@tf.custom_gradient
def box_gradient_trap(y, max_delta=np.inf):

  def trap(dy):
    # tf.print(dy[0,0,0,0,0,...])
    dy = math_ops.rm_nan_inf(dy)
    delta = tf.cast(max_delta, dy.dtype)
    dy = tf.clip_by_value(dy, -delta, delta)
    # tf.print(tf.reduce_sum(tf.square(dy)))
    return dy, 0.0

  return y, trap


@tf.custom_gradient
def class_gradient_trap(y, max_delta=np.inf):

  def trap(dy):
    dy = math_ops.rm_nan_inf(dy)
    delta = tf.cast(max_delta, dy.dtype)
    dy = tf.clip_by_value(dy, -delta, delta)
    # tf.print(tf.reduce_sum(tf.square(dy)))
    return dy, 0.0

  return y, trap


def ce(values, labels):
  labels = labels + K.epsilon()
  loss = tf.reduce_mean(
      -math_ops.mul_no_nan(values, tf.math.log(labels)), axis=-1)
  loss = math_ops.rm_nan(loss, val=0.0)
  return loss


@tf.custom_gradient
def no_grad_sigmoid(values):
  vals = tf.math.sigmoid(values)

  def delta(dy):
    return dy

  return vals, delta


# conver to a static class
# no ops in this fn should have grads propagated
def scale_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points,
                max_delta, scale_xy):
  scale_xy = tf.cast(scale_xy, pred_xy.dtype)
  pred_xy = tf.math.sigmoid(pred_xy) * scale_xy - 0.5 * (scale_xy - 1)

  scaler = tf.convert_to_tensor([width, height])
  box_xy = grid_points + pred_xy / scaler
  box_wh = tf.math.exp(pred_wh) * anchor_grid
  pred_box = K.concatenate([box_xy, box_wh], axis=-1)
  # return (box_xy, box_wh, pred_box)
  return (pred_xy, box_wh, pred_box)


@tf.custom_gradient
def darknet_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points,
                  max_delta, scale_xy):
  # (box_xy, box_wh, pred_box) = scale_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points, max_delta, scale_xy)
  (pred_xy, box_wh, pred_box) = scale_boxes(pred_xy, pred_wh, width, height,
                                            anchor_grid, grid_points, max_delta,
                                            scale_xy)

  def delta(dy_xy, dy_wh, dy):
    # here we do not propgate the scaling of the prediction through the network because in back prop is
    # leads to the scaling of the gradient of a box relative to the size of the box and the gradient
    # for small boxes this will mean down scaling the gradient leading to poor convergence on small
    # objects and also very likely over fitting
    dy_xy_, dy_wh_ = tf.split(dy, 2, axis=-1)
    dy_wh += dy_wh_
    dy_xy += dy_xy_
    dy_wh *= tf.math.exp(pred_wh)

    dy_wh = math_ops.rm_nan_inf(dy_wh)
    delta = tf.cast(max_delta, dy_wh.dtype)
    dy_wh = tf.clip_by_value(dy_wh, -delta, delta)

    dy_xy = math_ops.rm_nan_inf(dy_xy)
    delta = tf.cast(max_delta, dy_xy.dtype)
    dy_xy = tf.clip_by_value(dy_xy, -delta, delta)
    return dy_xy, dy_wh, 0.0, 0.0, tf.zeros_like(anchor_grid), tf.zeros_like(
        grid_points), 0.0, 0.0

  # return (box_xy, box_wh, pred_box), delta
  return (pred_xy, box_wh, pred_box), delta


def get_predicted_box(width,
                      height,
                      unscaled_box,
                      anchor_grid,
                      grid_points,
                      scale_xy,
                      darknet=True,
                      max_delta=5.0):

  # TODO: scale_xy should not be propagated in darkent either
  pred_xy = unscaled_box[
      ..., 0:
      2]  #no_grad_sigmoid(unscaled_box[..., 0:2]) * scale_x_y - 0.5 * (scale_x_y - 1)
  pred_wh = unscaled_box[..., 2:4]

  # pred_xy = box_gradient_trap(pred_xy, max_delta)
  # pred_wh = box_gradient_trap(pred_wh, max_delta)

  # the gradient for non of this should be propagated
  if darknet:
    # box_xy, box_wh, pred_box = darknet_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points, max_delta, scale_xy)
    pred_xy, box_wh, pred_box = darknet_boxes(pred_xy, pred_wh, width, height,
                                              anchor_grid, grid_points,
                                              max_delta, scale_xy)
  else:
    # box_xy, box_wh, pred_box = scale_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points, max_delta, scale_xy)
    pred_xy, box_wh, pred_box = scale_boxes(pred_xy, pred_wh, width, height,
                                            anchor_grid, grid_points, max_delta,
                                            scale_xy)

  return pred_xy, box_wh, pred_box


# conver to a static class
# no ops in this fn should have grads propagated
def new_coord_scale_boxes(pred_xy, pred_wh, width, height, anchor_grid,
                          grid_points, max_delta, scale_xy):
  scale_xy = tf.cast(scale_xy, pred_xy.dtype)
  pred_xy = pred_xy * scale_xy - 0.5 * (scale_xy - 1)
  scaler = tf.convert_to_tensor([width, height])
  box_xy = grid_points + pred_xy / scaler
  box_wh = tf.square(2 * pred_wh) * anchor_grid
  pred_box = K.concatenate([box_xy, box_wh], axis=-1)
  # return (box_xy, box_wh, pred_box)
  return (pred_xy, box_wh, pred_box)


@tf.custom_gradient
def darknet_new_coord_boxes(pred_xy, pred_wh, width, height, anchor_grid,
                            grid_points, max_delta, scale_xy):
  # (box_xy, box_wh, pred_box) = new_coord_scale_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points, max_delta, scale_xy)
  (pred_xy, box_wh, pred_box) = new_coord_scale_boxes(pred_xy, pred_wh, width,
                                                      height, anchor_grid,
                                                      grid_points, max_delta,
                                                      scale_xy)

  def delta(dy_xy, dy_wh, dy):
    dy_xy_, dy_wh_ = tf.split(dy, 2, axis=-1)
    dy_wh += dy_wh_
    dy_xy += dy_xy_

    # gradient clipping
    dy_wh = math_ops.rm_nan_inf(dy_wh)
    delta = tf.cast(max_delta, dy_wh.dtype)
    dy_wh = tf.clip_by_value(dy_wh, -delta, delta)

    dy_xy = math_ops.rm_nan_inf(dy_xy)
    delta = tf.cast(max_delta, dy_xy.dtype)
    dy_xy = tf.clip_by_value(dy_xy, -delta, delta)
    return dy_xy, dy_wh, 0.0, 0.0, tf.zeros_like(anchor_grid), tf.zeros_like(
        grid_points), 0.0, 0.0

  # return (box_xy, box_wh, pred_box), delta
  return (pred_xy, box_wh, pred_box), delta


def get_predicted_box_newcords(width,
                               height,
                               unscaled_box,
                               anchor_grid,
                               grid_points,
                               scale_xy,
                               darknet=False,
                               max_delta=5.0):
  pred_xy = tf.math.sigmoid(
      unscaled_box[..., 0:2])  #* scale_x_y - 0.5 * (scale_x_y - 1)
  pred_wh = tf.math.sigmoid(unscaled_box[..., 2:4])

  # pred_xy = box_gradient_trap(pred_xy, max_delta)
  # pred_wh = box_gradient_trap(pred_wh, max_delta)

  if darknet:
    # box_xy, box_wh, pred_box = darknet_new_coord_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points, max_delta, scale_xy)
    pred_xy, box_wh, pred_box = darknet_new_coord_boxes(pred_xy, pred_wh, width,
                                                        height, anchor_grid,
                                                        grid_points, max_delta,
                                                        scale_xy)
  else:
    # pred_xy_ = pred_xy
    # scale_xy = tf.cast(scale_xy, pred_xy.dtype)
    # pred_xy = pred_xy * scale_xy - 0.5 * (scale_xy - 1)
    # box_xy, box_wh, pred_box = new_coord_scale_boxes(pred_xy_, pred_wh, width, height, anchor_grid, grid_points, max_delta, scale_xy)
    pred_xy, box_wh, pred_box = new_coord_scale_boxes(pred_xy, pred_wh, width,
                                                      height, anchor_grid,
                                                      grid_points, max_delta,
                                                      scale_xy)
  return pred_xy, box_wh, pred_box


class Yolo_Loss(object):

  def __init__(self,
               classes,
               mask,
               anchors,
               scale_anchors=1,
               num_extras=0,
               ignore_thresh=0.7,
               truth_thresh=1.0,
               loss_type="ciou",
               iou_normalizer=1.0,
               cls_normalizer=1.0,
               obj_normalizer=1.0,
               objectness_smooth=True,
               use_reduction_sum=False,
               iou_thresh=0.213,
               new_cords=False,
               scale_x_y=1.0,
               max_delta=10,
               nms_kind="greedynms",
               beta_nms=0.6,
               reduction=tf.keras.losses.Reduction.NONE,
               path_key=None,
               use_tie_breaker=True,
               name=None,
               **kwargs):
    """
    parameters for the loss functions used at each detection head output

    Args:
      mask: list of indexes for which anchors in the anchors list should be 
        used in prediction
      anchors: list of tuples (w, h) representing the anchor boxes to be 
        used in prediction
      num_extras: number of indexes predicted in addition to 4 for the box 
        and N + 1 for classes
      ignore_thresh: float for the threshold for if iou > threshold the 
        network has made a prediction, and should not be penealized for 
        p(object) prediction if an object exists at this location
      truth_thresh: float thresholding the groud truth to get the true mask
      loss_type: string for the key of the loss to use,
        options -> mse, giou, ciou
      iou_normalizer: float used for appropriatly scaling the iou or the 
        loss used for the box prediction error
      cls_normalizer: float used for appropriatly scaling the classification 
        error
      scale_x_y: float used to scale the predictied x and y outputs
      nms_kind: string used for filtering the output and ensuring each 
        object has only one prediction
      beta_nms: float for the thresholding value to apply in non max 
        supression(nms) -> not yet implemented

    call Return:
      float: for the average loss
    """
    self._classes = tf.constant(tf.cast(classes, dtype=tf.int32))
    self._num = tf.cast(len(mask), dtype=tf.int32)
    self._num_extras = tf.cast(num_extras, dtype=tf.int32)
    self._truth_thresh = truth_thresh
    self._ignore_thresh = ignore_thresh
    self._iou_thresh = iou_thresh
    self._masks = mask
    self._anchors = anchors

    self._use_tie_breaker = tf.cast(use_tie_breaker, tf.bool)
    if loss_type == "giou":
      self._loss_type = 1
    elif loss_type == "ciou":
      self._loss_type = 2
    else:
      self._loss_type = 0

    self._iou_normalizer = iou_normalizer
    self._cls_normalizer = cls_normalizer
    self._obj_normalizer = obj_normalizer
    self._scale_x_y = scale_x_y
    self._max_delta = max_delta

    self._label_smoothing = tf.cast(0.0, tf.float32)
    # self._objectness_smooth = objectness_smooth

    self._objectness_smooth = float(objectness_smooth)
    self._use_reduction_sum = use_reduction_sum

    # used in detection filtering
    self._beta_nms = beta_nms
    self._nms_kind = tf.cast(nms_kind, tf.string)
    self._new_cords = new_cords

    box_kwargs = dict(
        scale_xy=self._scale_x_y,
        darknet=not self._use_reduction_sum,
        max_delta=self._max_delta)

    if not self._new_cords:
      self._decode_boxes = partial(get_predicted_box, **box_kwargs)
    else:
      self._decode_boxes = partial(get_predicted_box_newcords, **box_kwargs)

    # grid comp
    self._anchor_generator = GridGenerator(
        masks=mask, anchors=anchors, scale_anchors=scale_anchors)

    # metric struff
    self._path_key = path_key
    return

  def print_error(self, pred, key):
    if tf.stop_gradient(tf.reduce_any(tf.math.is_nan(pred))):
      tf.print("\nerror: stop training ", key)

  def _scale_ground_truth_box(self, box, width, height, anchor_grid,
                              grid_points, dtype):
    xy = tf.nn.relu(box[..., 0:2] - grid_points)
    xy = K.concatenate([
        K.expand_dims(xy[..., 0] * width, axis=-1),
        K.expand_dims(xy[..., 1] * height, axis=-1)
    ],
                       axis=-1)
    wh = tf.math.log(box[..., 2:4] / anchor_grid)
    wh = math_ops.rm_nan_inf(wh)
    return tf.stop_gradient(xy), tf.stop_gradient(wh)

  def APAR(self, pred_conf, true_conf, pct=0.5):
    dets = tf.cast(tf.squeeze(pred_conf, axis=-1) > pct, dtype=true_conf.dtype)
    true_pos = tf.reduce_sum(
        math_ops.mul_no_nan(true_conf, dets), axis=(1, 2, 3))
    gt_pos = tf.reduce_sum(true_conf, axis=(1, 2, 3))
    all_pos = tf.reduce_sum(dets, axis=(1, 2, 3))

    recall = tf.reduce_mean(math_ops.divide_no_nan(true_pos, gt_pos))
    precision = tf.reduce_mean(math_ops.divide_no_nan(true_pos, all_pos))
    return tf.stop_gradient(recall), tf.stop_gradient(precision)

  def avgiou(self, iou):
    avg_iou = math_ops.divide_no_nan(
        tf.reduce_sum(iou),
        tf.cast(
            tf.math.count_nonzero(tf.cast(iou, dtype=iou.dtype)),
            dtype=iou.dtype))
    return tf.stop_gradient(avg_iou)

  def box_loss(self, true_box, pred_box):
    if self._loss_type == 1:
      iou, liou = box_ops.compute_giou(true_box, pred_box)
      loss_box = (1 - liou)
    elif self._loss_type == 2:
      iou, liou = box_ops.compute_ciou(true_box, pred_box)
      loss_box = (1 - liou)
    else:
      iou = box_ops.compute_iou(true_box, pred_box)
      liou = iou
      loss_box = 1 - iou
      # # mse loss computation :: yolo_layer.c: scale = (2-truth.w*truth.h)
      # scale = (2 - true_box[..., 2] * true_box[..., 3])
      # true_xy, true_wh = self._scale_ground_truth_box(true_box, fwidth, fheight,
      #                                                 anchor_grid, grid_points,
      #                                                 y_pred.dtype)
      # loss_xy = tf.reduce_sum(K.square(true_xy - pred_xy), axis=-1)
      # loss_wh = tf.reduce_sum(K.square(true_wh - pred_wh), axis=-1)
      # loss_box = (loss_wh + loss_xy) * scale

    return iou, liou, loss_box

  def _build_mask_body(self, pred_boxes_, pred_classes_, pred_conf,
                       pred_classes_max, boxes, classes, iou_max_, ignore_mask_,
                       conf_loss_, loss_, count, idx):
    batch_size = tf.shape(boxes)[0]
    box_slice = tf.slice(boxes, [0, idx * TILE_SIZE, 0],
                         [batch_size, TILE_SIZE, 4])
    class_slice = tf.slice(classes, [0, idx * TILE_SIZE],
                           [batch_size, TILE_SIZE])

    box_slice = tf.expand_dims(box_slice, axis=-2)
    box_slice = tf.expand_dims(box_slice, axis=1)
    box_slice = tf.expand_dims(box_slice, axis=1)

    class_slice = tf.expand_dims(class_slice, axis=1)
    class_slice = tf.expand_dims(class_slice, axis=1)
    class_slice = tf.expand_dims(class_slice, axis=1)
    class_slice = tf.one_hot(
        tf.cast(class_slice, tf.int32),
        depth=tf.shape(pred_classes_max)[-1],
        dtype=pred_classes_max.dtype)

    pred_boxes = tf.expand_dims(pred_boxes_, axis=-3)
    iou, liou, loss_box = self.box_loss(box_slice, pred_boxes)

    # cconfidence is low
    iou_mask = iou > self._ignore_thresh
    iou_mask = tf.transpose(iou_mask, perm=(0, 1, 2, 4, 3))
    matched_classes = tf.equal(class_slice, pred_classes_max)
    matched_classes = tf.reduce_any(matched_classes, axis=-1)
    full_iou_mask = tf.logical_and(iou_mask, matched_classes)
    iou_mask = tf.reduce_any(full_iou_mask, axis=-1, keepdims=False)
    ignore_mask_ = tf.logical_or(ignore_mask_, iou_mask)

    if self._objectness_smooth:
      iou_max = tf.transpose(iou, perm=(0, 1, 2, 4, 3))
      iou_max = iou_max * tf.cast(full_iou_mask, iou_max.dtype)
      iou_max = tf.reduce_max(iou_max, axis=-1, keepdims=False)
      iou_max_ = tf.maximum(iou_max, iou_max_)

    return (pred_boxes_, pred_classes_, pred_conf, pred_classes_max, boxes,
            classes, iou_max_, ignore_mask_, conf_loss_, loss_, count, idx + 1)

  def _tiled_global_box_search(self, pred_boxes, pred_classes, pred_conf, boxes,
                               classes, true_conf, fwidth, fheight, smoothed):
    num_boxes = tf.shape(boxes)[-2]
    num_tiles = num_boxes // TILE_SIZE
    base = tf.cast(tf.zeros_like(tf.reduce_sum(pred_boxes, axis=-1)), tf.bool)
    boxes = box_ops.yxyx_to_xcycwh(boxes)

    pred_classes_mask = tf.cast(pred_classes > 0.25, tf.float32)
    pred_classes_mask = tf.expand_dims(pred_classes_mask, axis=-2)

    loss_base = tf.zeros_like(tf.reduce_sum(pred_boxes, axis=-1))
    obns_base = tf.zeros_like(tf.reduce_sum(pred_boxes, axis=-1, keepdims=True))

    def _loop_cond(pred_boxes_, pred_classes_, pred_conf, pred_classes_max,
                   boxes, classes, iou_max_, ignore_mask_, conf_loss_, loss_,
                   count, idx):
      batch_size = tf.shape(boxes)[0]
      box_slice = tf.slice(boxes, [0, idx * TILE_SIZE, 0],
                           [batch_size, TILE_SIZE, 4])
      return tf.logical_and(idx < num_tiles,
                            tf.math.greater(tf.reduce_sum(box_slice), 0))

    (_, _, _, _, _, _, iou_max, iou_mask, obns_loss, truth_loss, count,
     idx) = tf.while_loop(
         _loop_cond,
         self._build_mask_body, [
             pred_boxes, pred_classes, pred_conf, pred_classes_mask, boxes,
             classes, loss_base, base, obns_base, loss_base, obns_base,
             tf.constant(0)
         ],
         parallel_iterations=20)

    # tf.print(tf.reduce_mean(tf.reduce_sum(truth_loss, axis = (1, 2, 3))))
    ignore_mask = tf.logical_not(iou_mask)
    ignore_mask = tf.stop_gradient(tf.cast(ignore_mask, true_conf.dtype))
    iou_max = tf.stop_gradient(iou_max)

    # tf.print(tf.reduce_sum(true_conf), tf.reduce_sum(tf.maximum(tf.cast(iou_mask, tf.float32),true_conf))))
    if not smoothed:
      obj_mask = true_conf + (1 - true_conf) * ignore_mask
    else:
      obj_mask = tf.ones_like(true_conf)
      iou_ = (1 - self._objectness_smooth) + self._objectness_smooth * iou_max
      iou_ = tf.where(iou_max > 0, iou_, tf.zeros_like(iou_))
      true_conf = tf.where(
          tf.logical_and(true_conf == 1.0, iou_ > self._ignore_thresh), iou_,
          true_conf)
      true_conf = tf.where(true_conf == 0.0, iou_, true_conf)

    obj_mask = tf.stop_gradient(obj_mask)
    true_conf = tf.stop_gradient(true_conf)
    obns_loss = tf.stop_gradient(obns_loss)
    truth_loss = tf.stop_gradient(truth_loss)
    count = tf.stop_gradient(count)
    return ignore_mask, obns_loss, truth_loss, count, true_conf, obj_mask

  def build_grid(self, indexes, truths, preds, ind_mask, update=False):
    num_flatten = tf.shape(preds)[-1]
    bhep = tf.reduce_max(tf.ones_like(indexes), axis=-1, keepdims=True)
    bhep = tf.math.cumsum(bhep, axis=0) - 1
    indexes = tf.concat([bhep, indexes], axis=-1)
    indexes = math_ops.mul_no_nan(tf.cast(ind_mask, indexes.dtype), indexes)

    indexes = tf.reshape(indexes, [-1, 4])
    truths = tf.reshape(truths, [-1, num_flatten])

    grid = tf.zeros_like(preds)
    truths = math_ops.rm_nan_inf(truths)

    if update:
      grid = tf.tensor_scatter_nd_update(grid, indexes, truths)
    else:
      grid = tf.tensor_scatter_nd_max(grid, indexes, truths)

    # grid = tf.scatter_nd(indexes, truths, tf.shape(preds))
    grid = tf.clip_by_value(grid, 0.0, 1.0)
    return tf.stop_gradient(grid)

  def call_pytorch(self, true_counts, inds, y_true, boxes, classes, y_pred):
    # 1. generate and store constants and format output
    shape = tf.shape(true_counts)
    batch_size, width, height, num = shape[0], shape[1], shape[2], shape[3]
    fwidth = tf.cast(width, tf.float32)
    fheight = tf.cast(height, tf.float32)

    boxes = tf.stop_gradient(tf.cast(boxes, tf.float32))
    classes = tf.stop_gradient(tf.cast(classes, tf.float32))
    y_true = tf.stop_gradient(tf.cast(y_true, tf.float32))
    true_counts = tf.stop_gradient(tf.cast(true_counts, tf.float32))
    true_conf = tf.stop_gradient(tf.clip_by_value(true_counts, 0.0, 1.0))
    grid_points, anchor_grid = self._anchor_generator(
        width, height, batch_size, dtype=tf.float32)

    (true_box, ind_mask, true_class, best_iou_match, num_reps) = tf.split(
        y_true, [4, 1, 1, 1, 1], axis=-1)
    true_conf = tf.squeeze(true_conf, axis=-1)
    true_class = tf.squeeze(true_class, axis=-1)
    grid_mask = true_conf

    y_pred = tf.cast(
        tf.reshape(y_pred, [batch_size, width, height, num, -1]), tf.float32)
    pred_box, pred_conf, pred_class = tf.split(y_pred, [4, 1, -1], axis=-1)

    sigmoid_class = tf.sigmoid(pred_class)
    pred_class = class_gradient_trap(pred_class, np.inf)
    sigmoid_conf = tf.sigmoid(pred_conf)
    pred_conf = obj_gradient_trap(pred_conf, np.inf)
    pred_xy, pred_wh, pred_box = self._decode_boxes(fwidth, fheight, pred_box,
                                                    anchor_grid, grid_points)

    # num_objs = tf.cast(
    #     tf.reduce_sum(grid_mask, axis=(1, 2, 3)), dtype=y_pred.dtype)
    num_objs = tf.cast(tf.reduce_sum(ind_mask, axis=(1, 2)), dtype=y_pred.dtype)

    if self._objectness_smooth <= 0.0:
      (mask_loss, thresh_conf_loss, thresh_loss, thresh_counts, true_conf,
       obj_mask) = self._tiled_global_box_search(
           pred_box,
           sigmoid_class,
           sigmoid_conf,
           boxes,
           classes,
           true_conf,
           fwidth,
           fheight,
           smoothed=False)

    true_class = tf.one_hot(
        tf.cast(true_class, tf.int32),
        depth=tf.shape(pred_class)[-1],
        dtype=pred_class.dtype)
    true_class = math_ops.mul_no_nan(ind_mask, true_class)

    # counts = true_counts
    # counts = tf.reduce_sum(counts, axis=-1, keepdims=True)
    # reps = tf.gather_nd(counts, inds, batch_dims=1)
    # reps = tf.squeeze(reps, axis=-1)
    # reps = tf.where(reps == 0.0, tf.ones_like(reps), reps)

    # scale boxes
    scale = tf.convert_to_tensor([fheight, fwidth])
    # pred_xy = pred_xy * scale - scale
    pred_wh = pred_wh * scale
    pred_box = tf.concat([pred_xy, pred_wh], axis=-1)

    true_xy, true_wh = tf.split(true_box, 2, axis=-1)
    ind_y, ind_x, ind_a = tf.split(inds, 3, axis=-1)
    ind_xy = tf.concat([ind_x, ind_y], axis=-1)
    ind_xy = tf.cast(ind_xy, true_xy.dtype)

    true_xy = (true_xy * scale) - tf.cast(ind_xy, true_xy.dtype)
    true_wh = true_wh * scale
    true_box = tf.concat([true_xy, true_wh], axis=-1)

    pred_box = math_ops.mul_no_nan(ind_mask,
                                   tf.gather_nd(pred_box, inds, batch_dims=1))
    true_box = math_ops.mul_no_nan(ind_mask, true_box)
    iou, liou, box_loss = self.box_loss(true_box, pred_box)
    # box_loss = math_ops.divide_no_nan(box_loss, reps)
    box_loss = math_ops.mul_no_nan(tf.squeeze(ind_mask, axis=-1), box_loss)
    box_loss = tf.cast(tf.reduce_sum(box_loss, axis=1), dtype=y_pred.dtype)
    box_loss = math_ops.divide_no_nan(box_loss, num_objs)

    if self._objectness_smooth > 0.0:
      iou_ = (1 - self._objectness_smooth) + self._objectness_smooth * iou
      iou_ = math_ops.mul_no_nan(ind_mask, tf.expand_dims(iou_, axis=-1))
      true_conf = self.build_grid(inds, iou_, pred_conf, ind_mask, update=False)
      true_conf = tf.squeeze(true_conf, axis=-1)
      obj_mask = tf.ones_like(true_conf)

    pred_class = math_ops.mul_no_nan(
        ind_mask, tf.gather_nd(pred_class, inds, batch_dims=1))
    class_loss = ks.losses.binary_crossentropy(
        K.expand_dims(true_class, axis=-1),
        K.expand_dims(pred_class, axis=-1),
        label_smoothing=self._label_smoothing,
        from_logits=True)
    class_loss = math_ops.mul_no_nan(ind_mask, class_loss)
    class_loss = tf.cast(
        tf.reduce_mean(class_loss, axis=(2)), dtype=y_pred.dtype)
    # class_loss = math_ops.divide_no_nan(class_loss, reps)
    class_loss = tf.cast(
        tf.reduce_sum(class_loss, axis=(1)), dtype=y_pred.dtype)
    class_loss = math_ops.divide_no_nan(class_loss, num_objs)

    # pred_conf = math_ops.rm_nan_inf(pred_conf, val = -1000.0)
    bce = ks.losses.binary_crossentropy(
        K.expand_dims(true_conf, axis=-1), pred_conf, from_logits=True)
    conf_loss = math_ops.mul_no_nan(obj_mask, bce)
    conf_loss = tf.cast(
        tf.reduce_mean(conf_loss, axis=(1, 2, 3)), dtype=y_pred.dtype)

    box_loss *= self._iou_normalizer
    class_loss *= self._cls_normalizer
    conf_loss *= self._obj_normalizer

    loss = box_loss + class_loss + conf_loss
    loss = tf.reduce_sum(loss)

    box_loss = tf.reduce_mean(box_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    class_loss = tf.reduce_mean(class_loss)

    recall50, precision50 = self.APAR(sigmoid_conf, grid_mask, pct=0.5)
    avg_iou = self.avgiou(iou * tf.gather_nd(grid_mask, inds, batch_dims=1))
    avg_obj = self.avgiou(tf.squeeze(sigmoid_conf, axis=-1) * grid_mask)
    return loss, box_loss, conf_loss, class_loss, avg_iou, avg_obj, recall50, precision50

  def call_darknet(self, true_counts, inds, y_true, boxes, classes, y_pred):
    # 1. generate and store constants and format output
    shape = tf.shape(true_counts)
    batch_size, width, height, num = shape[0], shape[1], shape[2], shape[3]
    fwidth = tf.cast(width, tf.float32)
    fheight = tf.cast(height, tf.float32)

    boxes = tf.stop_gradient(tf.cast(boxes, tf.float32))
    classes = tf.stop_gradient(tf.cast(classes, tf.float32))
    y_true = tf.stop_gradient(tf.cast(y_true, tf.float32))
    true_counts = tf.stop_gradient(tf.cast(true_counts, tf.float32))
    true_conf = tf.stop_gradient(tf.clip_by_value(true_counts, 0.0, 1.0))
    grid_points, anchor_grid = self._anchor_generator(
        width, height, batch_size, dtype=tf.float32)

    (true_box, ind_mask, true_class, best_iou_match, num_reps) = tf.split(
        y_true, [4, 1, 1, 1, 1], axis=-1)
    true_conf = tf.squeeze(true_conf, axis=-1)
    true_class = tf.squeeze(true_class, axis=-1)
    grid_mask = true_conf

    y_pred = tf.cast(
        tf.reshape(y_pred, [batch_size, width, height, num, -1]), tf.float32)
    pred_box, pred_conf, pred_class = tf.split(y_pred, [4, 1, -1], axis=-1)

    sigmoid_class = tf.sigmoid(pred_class)
    pred_class = class_gradient_trap(sigmoid_class, np.inf)

    sigmoid_conf = tf.sigmoid(pred_conf)
    sigmoid_conf = math_ops.rm_nan_inf(sigmoid_conf, val=0.0)
    pred_conf = obj_gradient_trap(sigmoid_conf, np.inf)
    pred_xy, pred_wh, pred_box = self._decode_boxes(fwidth, fheight, pred_box,
                                                    anchor_grid, grid_points)

    (mask_loss, thresh_conf_loss, thresh_loss, thresh_counts, true_conf,
     obj_mask) = self._tiled_global_box_search(
         pred_box,
         sigmoid_class,
         sigmoid_conf,
         boxes,
         classes,
         true_conf,
         fwidth,
         fheight,
         smoothed=self._objectness_smooth > 0)

    true_class = tf.one_hot(
        tf.cast(true_class, tf.int32),
        depth=tf.shape(pred_class)[-1],
        dtype=pred_class.dtype)
    true_class = math_ops.mul_no_nan(ind_mask, true_class)

    true_class = self.build_grid(
        inds, true_class, pred_class, ind_mask, update=False)
    counts = true_class
    counts = tf.reduce_sum(counts, axis=-1, keepdims=True)
    reps = tf.gather_nd(counts, inds, batch_dims=1)
    reps = tf.squeeze(reps, axis=-1)
    reps = tf.where(reps == 0.0, tf.ones_like(reps), reps)

    pred_box = math_ops.mul_no_nan(ind_mask,
                                   tf.gather_nd(pred_box, inds, batch_dims=1))
    iou, liou, box_loss = self.box_loss(true_box, pred_box)
    box_loss = math_ops.mul_no_nan(tf.squeeze(ind_mask, axis=-1), box_loss)
    box_loss = math_ops.divide_no_nan(box_loss, reps)
    box_loss = tf.cast(tf.reduce_sum(box_loss, axis=1), dtype=y_pred.dtype)

    class_loss = ks.losses.binary_crossentropy(
        K.expand_dims(true_class, axis=-1),
        K.expand_dims(pred_class, axis=-1),
        label_smoothing=self._label_smoothing,
        from_logits=False)
    class_loss = tf.reduce_sum(class_loss, axis=-1)
    class_loss = math_ops.mul_no_nan(grid_mask, class_loss)

    class_loss = math_ops.rm_nan_inf(class_loss, val=0.0)
    class_loss = tf.cast(
        tf.reduce_sum(class_loss, axis=(1, 2, 3)), dtype=y_pred.dtype)

    # bce = ks.losses.binary_crossentropy(
    #     K.expand_dims(true_conf, axis=-1), pred_conf, from_logits=False)
    bce = tf.where(true_conf > 0.0,
                   ce(K.expand_dims(true_conf, axis=-1), pred_conf),
                   ce(1 - K.expand_dims(true_conf, axis=-1), 1 - pred_conf))
    conf_loss = math_ops.mul_no_nan(obj_mask, bce)
    conf_loss = tf.cast(
        tf.reduce_sum(conf_loss, axis=(1, 2, 3)), dtype=y_pred.dtype)

    box_loss *= self._iou_normalizer
    class_loss *= self._cls_normalizer
    conf_loss *= self._obj_normalizer

    loss = box_loss + class_loss + conf_loss
    loss = tf.reduce_mean(loss)

    box_loss = tf.reduce_mean(box_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    class_loss = tf.reduce_mean(class_loss)

    recall50, precision50 = self.APAR(sigmoid_conf, grid_mask, pct=0.5)
    avg_iou = self.avgiou(iou * tf.gather_nd(grid_mask, inds, batch_dims=1))
    avg_obj = self.avgiou(tf.squeeze(sigmoid_conf, axis=-1) * grid_mask)
    return loss, box_loss, conf_loss, class_loss, avg_iou, avg_obj, recall50, precision50

  def __call__(self, true_counts, inds, y_true, boxes, classes, y_pred):
    if self._use_reduction_sum:
      return self.call_pytorch(true_counts, inds, y_true, boxes, classes,
                               y_pred)
    else:
      return self.call_darknet(true_counts, inds, y_true, boxes, classes,
                               y_pred)
