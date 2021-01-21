"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

from yolo.ops import loss_utils
from yolo.ops import box_ops as box_utils
from yolo.losses.yolo_loss import Yolo_Loss
from yolo.ops import nms_ops


@ks.utils.register_keras_serializable(package='yolo')
class YoloLayer(ks.Model):

  def __init__(self,
               masks,
               anchors,
               classes,
               thresh=0.5,
               cls_thresh=0.5,
               ignore_thresh=0.7,
               loss_type='ciou',
               use_tie_breaker=True,
               max_boxes=200,
               path_scale=None,
               scale_xy=None,
               use_nms=True,
               **kwargs):
    super().__init__(**kwargs)
    self._masks = masks
    self._anchors = anchors
    self._thresh = thresh
    self._cls_thresh = cls_thresh
    self._ignore_thresh = ignore_thresh
    self._max_boxes = max_boxes
    self._classes = classes
    self._loss_type = loss_type
    self._use_tie_breaker = use_tie_breaker
    self._keys = list(masks.keys())
    self._len_keys = len(self._keys)
    self._path_scale = path_scale
    self._use_nms = use_nms
    self._scale_xy = scale_xy or {key: 1.0 for key, _ in masks.items()}
    self._generator = {}
    self._len_mask = {}
    for key in self._keys:
      anchors = [self._anchors[mask] for mask in self._masks[key]]
      self._generator[key] = self.get_generators(anchors, self._path_scale[key],
                                                 key)
      self._len_mask[key] = len(self._masks[key])
    return

  def get_generators(self, anchors, path_scale, path_key):
    anchor_generator = loss_utils.GridGenerator(
        anchors, scale_anchors=path_scale)
    return anchor_generator

  def parse_yolo_box_predictions(self,
                                 unscaled_box,
                                 width,
                                 height,
                                 anchor_grid,
                                 grid_points,
                                 scale_x_y=1.0):
    # with tf.name_scope("decode_box_predictions_yolo"):
    ubxy, pred_wh = tf.split(unscaled_box, 2, axis=-1)
    pred_xy = tf.math.sigmoid(ubxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    x, y = tf.split(pred_xy, 2, axis=-1)
    box_xy = tf.concat([x / width, y / height], axis=-1) + grid_points
    box_wh = tf.math.exp(pred_wh) * anchor_grid
    pred_box = K.concatenate([box_xy, box_wh], axis=-1)
    return pred_xy, pred_wh, pred_box

  def parse_prediction_path(self, generator, len_mask, scale_xy, inputs):
    shape = tf.shape(inputs)
    # reshape the yolo output to (batchsize, width, height, number_anchors, remaining_points)
    data = tf.reshape(inputs, [shape[0], shape[1], shape[2], len_mask, -1])
    centers, anchors = generator(shape[1], shape[2], shape[0], dtype=data.dtype)

    # compute the true box output values
    ubox, obns, classifics = tf.split(data, [4, 1, -1], axis=-1)
    classes = tf.shape(classifics)[-1]
    obns = tf.squeeze(obns, axis=-1)
    _, _, boxes = self.parse_yolo_box_predictions(
        ubox,
        tf.cast(shape[1], data.dtype),
        tf.cast(shape[2], data.dtype),
        anchors,
        centers,
        scale_x_y=scale_xy)
    box = box_utils.xcycwh_to_yxyx(boxes)

    # computer objectness and generate grid cell mask for where objects are located in the image
    objectness = tf.expand_dims(tf.math.sigmoid(obns), axis=-1)
    scaled = tf.math.sigmoid(classifics) * objectness

    # compute the mask of where objects have been located
    mask_check = tf.fill(
        tf.shape(objectness), tf.cast(self._thresh, dtype=objectness.dtype))
    sub = tf.math.ceil(tf.nn.relu(objectness - mask_check))
    num_dets = tf.reduce_sum(sub, axis=(1, 2, 3))

    box = box * sub
    scaled = scaled * sub
    objectness = objectness * sub

    mask = tf.cast(tf.ones_like(sub), dtype=tf.bool)
    mask = tf.reduce_any(mask, axis=(0, -1))

    # reduce the dimentions of the predictions to (batch size, max predictions, -1)
    box = tf.boolean_mask(box, mask, axis=1)
    classifications = tf.boolean_mask(scaled, mask, axis=1)
    objectness = tf.squeeze(tf.boolean_mask(objectness, mask, axis=1), axis=-1)

    objectness, box, classifications = nms_ops.sort_drop(
        objectness, box, classifications, self._max_boxes)
    return objectness, box, classifications, num_dets

  def call(self, inputs):
    confidence, boxes, classifs, num_dets = self.parse_prediction_path(
        self._generator[self._keys[0]], self._len_mask[self._keys[0]],
        self._scale_xy[self._keys[0]], inputs[str(self._keys[0])])
    i = 1
    while i < self._len_keys:
      key = self._keys[i]
      conf, b, c, nd = self.parse_prediction_path(self._generator[key],
                                                  self._len_mask[key],
                                                  self._scale_xy[key],
                                                  inputs[str(key)])
      boxes = K.concatenate([boxes, b], axis=1)
      classifs = K.concatenate([classifs, c], axis=1)
      confidence = K.concatenate([confidence, conf], axis=1)
      num_dets += nd
      i += 1

    num_dets = tf.cast(tf.squeeze(num_dets, axis=-1), tf.float32)

    if self._use_nms:
      boxes = tf.cast(boxes, dtype=tf.float32)
      classifs = tf.cast(classifs, dtype=tf.float32)
      nms = tf.image.combined_non_max_suppression(
          tf.expand_dims(boxes, axis=2), classifs, self._max_boxes,
          self._max_boxes, self._thresh, self._cls_thresh)
      return {
          'bbox': nms.nmsed_boxes,
          'classes': tf.cast(nms.nmsed_classes, tf.int32),
          'confidence': nms.nmsed_scores,
          'num_dets': num_dets
      }

    boxes, classifs, confidence = nms_ops.nms(
        boxes,
        classifs,
        confidence,
        self._max_boxes,
        2.5,
        0.45,
        sorted=False,
        one_hot=True)

    return {
        'bbox': boxes,
        'classes': classifs,
        'confidence': confidence,
        'num_dets': num_dets
    }

  @property
  def losses(self):
    loss_dict = {}
    for key in self._keys:
      loss_dict[key] = Yolo_Loss(
          classes=self._classes,
          anchors=self._anchors,
          ignore_thresh=self._ignore_thresh,
          loss_type=self._loss_type,
          path_key=key,
          mask=self._masks[key],
          scale_anchors=self._path_scale[key],
          scale_x_y=self._scale_xy[key],
          use_tie_breaker=self._use_tie_breaker)
    return loss_dict

  @property
  def key_dict(self):
    return self._scale_xy

  def get_config(self):
    return {
        'masks': dict(self._masks),
        'anchors': [list(a) for a in self._anchors],
        'thresh': self._thresh,
        'cls_thresh': self._cls_thresh,
        'max_boxes': self._max_boxes,
    }


@ks.utils.register_keras_serializable(package='yolo')
class YoloGTFilter(ks.Model):

  def __init__(self, max_boxes=200, **kwargs):
    super().__init__(**kwargs)

    self._max_boxes = max_boxes
    return

  def parse_prediction_path(self, generator, len_mask, scale_xy, inputs):
    shape = tf.shape(inputs)
    # reshape the yolo output to (batchsize, width, height, number_anchors, remaining_points)
    data = tf.reshape(inputs, [shape[0], shape[1], shape[2], len_mask, -1])

    # compute the true box output values
    _, obns, classifics = tf.split(data, [4, 1, -1], axis=-1)
    scaled = tf.shape(classifics)[-1]
    objectness = tf.squeeze(obns, axis=-1)
    box = box_utils.xcycwh_to_yxyx(boxes)

    # compute the mask of where objects have been located
    num_dets = tf.reduce_sum(objectness, axis=(1, 2, 3))

    mask = tf.cast(tf.ones_like(sub), dtype=tf.bool)
    mask = tf.reduce_any(mask, axis=(0, -1))

    # reduce the dimentions of the predictions to (batch size, max predictions, -1)
    box = tf.boolean_mask(box, mask, axis=1)
    classifications = tf.boolean_mask(scaled, mask, axis=1)
    objectness = tf.squeeze(tf.boolean_mask(objectness, mask, axis=1), axis=-1)

    objectness, box, classifications = nms_ops.sort_drop(
        objectness, box, classifications, self._max_boxes)
    return objectness, box, classifications, num_dets

  def call(self, inputs):
    confidence, boxes, classifs, num_dets = self.parse_prediction_path(
        self._generator[self._keys[0]], self._len_mask[self._keys[0]],
        self._scale_xy[self._keys[0]], inputs[str(self._keys[0])])
    i = 1
    while i < self._len_keys:
      key = self._keys[i]
      conf, b, c, nd = self.parse_prediction_path(self._generator[key],
                                                  self._len_mask[key],
                                                  self._scale_xy[key],
                                                  inputs[str(key)])
      boxes = K.concatenate([boxes, b], axis=1)
      classifs = K.concatenate([classifs, c], axis=1)
      confidence = K.concatenate([confidence, conf], axis=1)
      num_dets += nd
      i += 1

    num_dets = tf.cast(tf.squeeze(num_dets, axis=-1), tf.float32)

    if self._use_nms:
      boxes = tf.cast(boxes, dtype=tf.float32)
      classifs = tf.cast(classifs, dtype=tf.float32)
      nms = tf.image.combined_non_max_suppression(
          tf.expand_dims(boxes, axis=2), classifs, self._max_boxes,
          self._max_boxes, self._thresh, self._cls_thresh)
      return {
          'bbox': nms.nmsed_boxes,
          'classes': tf.cast(nms.nmsed_classes, tf.int32),
          'confidence': nms.nmsed_scores,
          'num_dets': num_dets
      }

    confidence, boxes, classifs = nms_ops.sort_drop(confidence, boxes, classifs,
                                                    self._max_boxes)

    return {
        'bbox': boxes,
        'classes': tf.math.argmax(classifs, axis=-1),
        'confidence': confidence,
        'num_dets': num_dets
    }

  @property
  def losses(self):
    loss_dict = {}
    for key in self._keys:
      loss_dict[key] = Yolo_Loss(
          classes=self._classes,
          anchors=self._anchors,
          ignore_thresh=self._ignore_thresh,
          loss_type=self._loss_type,
          path_key=key,
          mask=self._masks[key],
          scale_anchors=self._path_scale[key],
          scale_x_y=self._scale_xy[key],
          use_tie_breaker=self._use_tie_breaker)
    return loss_dict

  @property
  def key_dict(self):
    return self._scale_xy

  def get_config(self):
    return {
        'masks': dict(self._masks),
        'anchors': [list(a) for a in self._anchors],
        'thresh': self._thresh,
        'cls_thresh': self._cls_thresh,
        'max_boxes': self._max_boxes,
    }
