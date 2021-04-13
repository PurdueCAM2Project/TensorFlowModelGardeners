import tensorflow as tf

from yolo.ops import box_ops as box_ops
from official.vision.beta.ops import box_ops as box_utils

NMS_TILE_SIZE = 512


def sort_drop(objectness, box, classificationsi, k):
  objectness, ind = tf.math.top_k(objectness, k=k)

  ind_m = tf.ones_like(ind) * tf.expand_dims(
      tf.range(0,
               tf.shape(objectness)[0]), axis=-1)
  bind = tf.stack([tf.reshape(ind_m, [-1]), tf.reshape(ind, [-1])], axis=-1)

  box = tf.gather_nd(box, bind)
  classifications = tf.gather_nd(classificationsi, bind)

  bsize = tf.shape(ind)[0]
  box = tf.reshape(box, [bsize, k, -1])
  classifications = tf.reshape(classifications, [bsize, k, -1])
  return objectness, box, classifications


def segment_nms(boxes, classes, confidence, k, iou_thresh):
  mrange = tf.range(k)
  mask_x = tf.tile(
      tf.transpose(tf.expand_dims(mrange, axis=-1), perm=[1, 0]), [k, 1])
  mask_y = tf.tile(tf.expand_dims(mrange, axis=-1), [1, k])
  mask_diag = tf.expand_dims(mask_x > mask_y, axis=0)

  iou = box_ops.aggregated_comparitive_iou(boxes, iou_type=0)

  # duplicate boxes
  iou_mask = iou >= iou_thresh
  iou_mask = tf.logical_and(mask_diag, iou_mask)
  iou *= tf.cast(iou_mask, iou.dtype)

  can_suppress_others = 1 - tf.cast(
      tf.reduce_any(iou_mask, axis=-2), boxes.dtype)

  # iou_sum = tf.reduce_sum(iou, [1])

  # option 1
  # can_suppress_others = tf.expand_dims(can_suppress_others, axis=-1)
  # supressed_i = can_suppress_others * iou
  # supressed = tf.reduce_max(supressed_i, -2) <= 0.5
  # raw = tf.cast(supressed, boxes.dtype)

  # option 2
  raw = tf.cast(can_suppress_others, boxes.dtype)

  boxes *= tf.expand_dims(raw, axis=-1)
  confidence *= tf.cast(raw, confidence.dtype)
  classes *= tf.cast(raw, classes.dtype)

  return boxes, classes, confidence


class TiledNMS():
  IOU_TYPES = {'diou': 0, 'giou': 1, 'ciou': 2, 'iou': 3}

  def __init__(self, iou_type='diou', beta=0.6):
    self._iou_type = TiledNMS.IOU_TYPES[iou_type]
    self._beta = beta

  def _self_suppression(self, iou, _, iou_sum):
    batch_size = tf.shape(iou)[0]
    can_suppress_others = tf.cast(
        tf.reshape(tf.reduce_max(iou, 1) <= 0.5, [batch_size, -1, 1]),
        iou.dtype)
    iou_suppressed = tf.reshape(
        tf.cast(tf.reduce_max(can_suppress_others * iou, 1) <= 0.5, iou.dtype),
        [batch_size, -1, 1]) * iou
    iou_sum_new = tf.reduce_sum(iou_suppressed, [1, 2])
    return [
        iou_suppressed,
        tf.reduce_any(iou_sum - iou_sum_new > 0.5), iou_sum_new
    ]

  def _cross_suppression(self, boxes, box_slice, iou_threshold, inner_idx):
    batch_size = tf.shape(boxes)[0]
    new_slice = tf.slice(boxes, [0, inner_idx * NMS_TILE_SIZE, 0],
                         [batch_size, NMS_TILE_SIZE, 4])
    #iou = box_ops.bbox_overlap(new_slice, box_slice)
    iou = box_ops.aggregated_comparitive_iou(
        new_slice, box_slice, beta=self._beta, iou_type=self._iou_type)
    ret_slice = tf.expand_dims(
        tf.cast(tf.reduce_all(iou < iou_threshold, [1]), box_slice.dtype),
        2) * box_slice
    return boxes, ret_slice, iou_threshold, inner_idx + 1

  def _suppression_loop_body(self, boxes, iou_threshold, output_size, idx):
    """Process boxes in the range [idx*NMS_TILE_SIZE, (idx+1)*NMS_TILE_SIZE).

    Args:
      boxes: a tensor with a shape of [batch_size, anchors, 4].
      iou_threshold: a float representing the threshold for deciding whether boxes
        overlap too much with respect to IOU.
      output_size: an int32 tensor of size [batch_size]. Representing the number
        of selected boxes for each batch.
      idx: an integer scalar representing induction variable.

    Returns:
      boxes: updated boxes.
      iou_threshold: pass down iou_threshold to the next iteration.
      output_size: the updated output_size.
      idx: the updated induction variable.
    """
    num_tiles = tf.shape(boxes)[1] // NMS_TILE_SIZE
    batch_size = tf.shape(boxes)[0]

    # Iterates over tiles that can possibly suppress the current tile.
    box_slice = tf.slice(boxes, [0, idx * NMS_TILE_SIZE, 0],
                         [batch_size, NMS_TILE_SIZE, 4])
    _, box_slice, _, _ = tf.while_loop(
        lambda _boxes, _box_slice, _threshold, inner_idx: inner_idx < idx,
        self._cross_suppression,
        [boxes, box_slice, iou_threshold,
         tf.constant(0)])

    # Iterates over the current tile to compute self-suppression.
    # iou = box_ops.bbox_overlap(box_slice, box_slice)
    iou = box_ops.aggregated_comparitive_iou(
        box_slice, box_slice, beta=self._beta, iou_type=self._iou_type)
    mask = tf.expand_dims(
        tf.reshape(tf.range(NMS_TILE_SIZE), [1, -1]) > tf.reshape(
            tf.range(NMS_TILE_SIZE), [-1, 1]), 0)
    iou *= tf.cast(tf.logical_and(mask, iou >= iou_threshold), iou.dtype)
    suppressed_iou, _, _ = tf.while_loop(
        lambda _iou, loop_condition, _iou_sum: loop_condition,
        self._self_suppression,
        [iou, tf.constant(True),
         tf.reduce_sum(iou, [1, 2])])
    suppressed_box = tf.reduce_sum(suppressed_iou, 1) > 0
    box_slice *= tf.expand_dims(1.0 - tf.cast(suppressed_box, box_slice.dtype),
                                2)

    # Uses box_slice to update the input boxes.
    mask = tf.reshape(
        tf.cast(tf.equal(tf.range(num_tiles), idx), boxes.dtype), [1, -1, 1, 1])
    boxes = tf.tile(tf.expand_dims(
        box_slice, [1]), [1, num_tiles, 1, 1]) * mask + tf.reshape(
            boxes, [batch_size, num_tiles, NMS_TILE_SIZE, 4]) * (1 - mask)
    boxes = tf.reshape(boxes, [batch_size, -1, 4])

    # Updates output_size.
    output_size += tf.reduce_sum(
        tf.cast(tf.reduce_any(box_slice > 0, [2]), tf.int32), [1])
    return boxes, iou_threshold, output_size, idx + 1

  def _sorted_non_max_suppression_padded(self, scores, boxes, max_output_size,
                                         iou_threshold):
    """A wrapper that handles non-maximum suppression.

    Assumption:
      * The boxes are sorted by scores unless the box is a dot (all coordinates
        are zero).
      * Boxes with higher scores can be used to suppress boxes with lower scores.

    The overal design of the algorithm is to handle boxes tile-by-tile:

    boxes = boxes.pad_to_multiply_of(tile_size)
    num_tiles = len(boxes) // tile_size
    output_boxes = []
    for i in range(num_tiles):
      box_tile = boxes[i*tile_size : (i+1)*tile_size]
      for j in range(i - 1):
        suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
        iou = bbox_overlap(box_tile, suppressing_tile)
        # if the box is suppressed in iou, clear it to a dot
        box_tile *= _update_boxes(iou)
      # Iteratively handle the diagnal tile.
      iou = _box_overlap(box_tile, box_tile)
      iou_changed = True
      while iou_changed:
        # boxes that are not suppressed by anything else
        suppressing_boxes = _get_suppressing_boxes(iou)
        # boxes that are suppressed by suppressing_boxes
        suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
        # clear iou to 0 for boxes that are suppressed, as they cannot be used
        # to suppress other boxes any more
        new_iou = _clear_iou(iou, suppressed_boxes)
        iou_changed = (new_iou != iou)
        iou = new_iou
      # remaining boxes that can still suppress others, are selected boxes.
      output_boxes.append(_get_suppressing_boxes(iou))
      if len(output_boxes) >= max_output_size:
        break

    Args:
      scores: a tensor with a shape of [batch_size, anchors].
      boxes: a tensor with a shape of [batch_size, anchors, 4].
      max_output_size: a scalar integer `Tensor` representing the maximum number
        of boxes to be selected by non max suppression.
      iou_threshold: a float representing the threshold for deciding whether boxes
        overlap too much with respect to IOU.

    Returns:
      nms_scores: a tensor with a shape of [batch_size, anchors]. It has same
        dtype as input scores.
      nms_proposals: a tensor with a shape of [batch_size, anchors, 4]. It has
        same dtype as input boxes.
    """
    batch_size = tf.shape(boxes)[0]
    num_boxes = tf.shape(boxes)[1]
    pad = tf.cast(
        tf.math.ceil(tf.cast(num_boxes, tf.float32) / NMS_TILE_SIZE),
        tf.int32) * NMS_TILE_SIZE - num_boxes
    boxes = tf.pad(tf.cast(boxes, tf.float32), [[0, 0], [0, pad], [0, 0]])
    scores = tf.pad(
        tf.cast(scores, tf.float32), [[0, 0], [0, pad]], constant_values=-1)
    num_boxes += pad

    def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
      return tf.logical_and(
          tf.reduce_min(output_size) < max_output_size,
          idx < num_boxes // NMS_TILE_SIZE)

    selected_boxes, _, output_size, _ = tf.while_loop(
        _loop_cond, self._suppression_loop_body, [
            boxes, iou_threshold,
            tf.zeros([batch_size], tf.int32),
            tf.constant(0)
        ])
    idx = num_boxes - tf.cast(
        tf.nn.top_k(
            tf.cast(tf.reduce_any(selected_boxes > 0, [2]), tf.int32) *
            tf.expand_dims(tf.range(num_boxes, 0, -1), 0), max_output_size)[0],
        tf.int32)
    idx = tf.minimum(idx, num_boxes - 1)
    idx = tf.reshape(
        idx + tf.reshape(tf.range(batch_size) * num_boxes, [-1, 1]), [-1])
    boxes = tf.reshape(
        tf.gather(tf.reshape(boxes, [-1, 4]), idx),
        [batch_size, max_output_size, 4])
    boxes = boxes * tf.cast(
        tf.reshape(tf.range(max_output_size), [1, -1, 1]) < tf.reshape(
            output_size, [-1, 1, 1]), boxes.dtype)
    scores = tf.reshape(
        tf.gather(tf.reshape(scores, [-1, 1]), idx),
        [batch_size, max_output_size])
    scores = scores * tf.cast(
        tf.reshape(tf.range(max_output_size), [1, -1]) < tf.reshape(
            output_size, [-1, 1]), scores.dtype)

    return scores, boxes

  def sorted_non_max_suppression_padded(self, scores, boxes, classes,
                                        max_output_size, iou_threshold):
    """A wrapper that handles non-maximum suppression.

    Assumption:
      * The boxes are sorted by scores unless the box is a dot (all coordinates
        are zero).
      * Boxes with higher scores can be used to suppress boxes with lower scores.

    The overal design of the algorithm is to handle boxes tile-by-tile:

    boxes = boxes.pad_to_multiply_of(tile_size)
    num_tiles = len(boxes) // tile_size
    output_boxes = []
    for i in range(num_tiles):
      box_tile = boxes[i*tile_size : (i+1)*tile_size]
      for j in range(i - 1):
        suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
        iou = bbox_overlap(box_tile, suppressing_tile)
        # if the box is suppressed in iou, clear it to a dot
        box_tile *= _update_boxes(iou)
      # Iteratively handle the diagnal tile.
      iou = _box_overlap(box_tile, box_tile)
      iou_changed = True
      while iou_changed:
        # boxes that are not suppressed by anything else
        suppressing_boxes = _get_suppressing_boxes(iou)
        # boxes that are suppressed by suppressing_boxes
        suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
        # clear iou to 0 for boxes that are suppressed, as they cannot be used
        # to suppress other boxes any more
        new_iou = _clear_iou(iou, suppressed_boxes)
        iou_changed = (new_iou != iou)
        iou = new_iou
      # remaining boxes that can still suppress others, are selected boxes.
      output_boxes.append(_get_suppressing_boxes(iou))
      if len(output_boxes) >= max_output_size:
        break

    Args:
      scores: a tensor with a shape of [batch_size, anchors].
      boxes: a tensor with a shape of [batch_size, anchors, 4].
      max_output_size: a scalar integer `Tensor` representing the maximum number
        of boxes to be selected by non max suppression.
      iou_threshold: a float representing the threshold for deciding whether boxes
        overlap too much with respect to IOU.

    Returns:
      nms_scores: a tensor with a shape of [batch_size, anchors]. It has same
        dtype as input scores.
      nms_proposals: a tensor with a shape of [batch_size, anchors, 4]. It has
        same dtype as input boxes.
    """
    batch_size = tf.shape(boxes)[0]
    num_boxes = tf.shape(boxes)[1]
    num_classes = tf.shape(classes)[-1]
    pad = tf.cast(
        tf.math.ceil(tf.cast(num_boxes, tf.float32) / NMS_TILE_SIZE),
        tf.int32) * NMS_TILE_SIZE - num_boxes
    boxes = tf.pad(tf.cast(boxes, tf.float32), [[0, 0], [0, pad], [0, 0]])
    scores = tf.pad(
        tf.cast(scores, tf.float32), [[0, 0], [0, pad]], constant_values=-1)
    classes = tf.pad(
        tf.cast(classes, tf.float32), [[0, 0], [0, pad], [0, 0]],
        constant_values=-1)
    num_boxes += pad

    def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
      return tf.logical_and(
          tf.reduce_min(output_size) < max_output_size,
          idx < num_boxes // NMS_TILE_SIZE)

    selected_boxes, _, output_size, _ = tf.while_loop(
        _loop_cond, self._suppression_loop_body, [
            boxes, iou_threshold,
            tf.zeros([batch_size], tf.int32),
            tf.constant(0)
        ])
    idx = num_boxes - tf.cast(
        tf.nn.top_k(
            tf.cast(tf.reduce_any(selected_boxes > 0, [2]), tf.int32) *
            tf.expand_dims(tf.range(num_boxes, 0, -1), 0), max_output_size)[0],
        tf.int32)
    idx = tf.minimum(idx, num_boxes - 1)
    idx = tf.reshape(
        idx + tf.reshape(tf.range(batch_size) * num_boxes, [-1, 1]), [-1])
    boxes = tf.reshape(
        tf.gather(tf.reshape(boxes, [-1, 4]), idx),
        [batch_size, max_output_size, 4])
    boxes = boxes * tf.cast(
        tf.reshape(tf.range(max_output_size), [1, -1, 1]) < tf.reshape(
            output_size, [-1, 1, 1]), boxes.dtype)
    scores = tf.reshape(
        tf.gather(tf.reshape(scores, [-1, 1]), idx),
        [batch_size, max_output_size])
    scores = scores * tf.cast(
        tf.reshape(tf.range(max_output_size), [1, -1]) < tf.reshape(
            output_size, [-1, 1]), scores.dtype)

    classes = tf.reshape(
        tf.gather(tf.reshape(classes, [-1, num_classes]), idx),
        [batch_size, max_output_size, num_classes])
    return boxes, classes, scores

  def _select_top_k_scores(self, scores_in, pre_nms_num_detections):
    # batch_size, num_anchors, num_class = scores_in.get_shape().as_list()
    scores_shape = tf.shape(scores_in)
    batch_size, num_anchors, num_class = scores_shape[0], scores_shape[
        1], scores_shape[2]
    scores_trans = tf.transpose(scores_in, perm=[0, 2, 1])
    scores_trans = tf.reshape(scores_trans, [-1, num_anchors])

    top_k_scores, top_k_indices = tf.nn.top_k(
        scores_trans, k=pre_nms_num_detections, sorted=True)

    top_k_scores = tf.reshape(top_k_scores,
                              [batch_size, num_class, pre_nms_num_detections])
    top_k_indices = tf.reshape(top_k_indices,
                               [batch_size, num_class, pre_nms_num_detections])

    return tf.transpose(top_k_scores,
                        [0, 2, 1]), tf.transpose(top_k_indices, [0, 2, 1])

  def _single_iter(self, boxes, scores, indices, nmsed_boxes, nmsed_classes,
                   nmsed_scores, max_num_detections, num_classes_for_box,
                   pre_nms_score_threshold, nms_iou_threshold, i):

    boxes_shape = tf.shape(boxes)
    batch_size, _, num_classes_for_box, _ = boxes_shape[0], boxes_shape[
        1], boxes_shape[2], boxes_shape[3]  #boxes.get_shape().as_list()

    boxes_i = boxes[:, :, tf.math.minimum(num_classes_for_box - 1, i), :]
    scores_i = scores[:, :, i]

    # Obtains pre_nms_top_k before running NMS.
    boxes_i = tf.gather(boxes_i, indices[:, :, i], batch_dims=1, axis=1)

    # Filter out scores.
    boxes_i, scores_i = box_utils.filter_boxes_by_scores(
        boxes_i,
        scores_i,
        min_score_threshold=tf.cast(pre_nms_score_threshold, boxes_i.dtype))

    (nmsed_scores_i, nmsed_boxes_i) = self._sorted_non_max_suppression_padded(
        tf.cast(scores_i, tf.float32),
        tf.cast(boxes_i, tf.float32),
        max_num_detections,
        iou_threshold=tf.cast(nms_iou_threshold, tf.float32))
    nmsed_classes_i = tf.fill([batch_size, max_num_detections], i)

    nmsed_boxes = nmsed_boxes.write(i,
                                    tf.transpose(nmsed_boxes_i, perm=(1, 0, 2)))
    nmsed_scores = nmsed_scores.write(i,
                                      tf.transpose(nmsed_scores_i, perm=(1, 0)))
    nmsed_classes = nmsed_classes.write(
        i, tf.transpose(nmsed_classes_i, perm=(1, 0)))
    return (boxes, scores, indices, nmsed_boxes, nmsed_classes, nmsed_scores,
            max_num_detections, num_classes_for_box, pre_nms_score_threshold,
            nms_iou_threshold, i + 1)

  def complete_nms(self,
                   boxes,
                   scores,
                   pre_nms_top_k=5000,
                   pre_nms_score_threshold=0.05,
                   nms_iou_threshold=0.5,
                   max_num_detections=100):

    with tf.name_scope('nms'):
      boxes_shape = tf.shape(boxes)
      batch_size, _, num_classes_for_box, _ = boxes_shape[0], boxes_shape[
          1], boxes_shape[2], boxes_shape[3]  #boxes.get_shape().as_list()

      scores_shape = tf.shape(scores)
      _, total_anchors, num_classes = scores_shape[0], scores_shape[1], scores_shape[2]  #.get_shape().as_list()

      nmsed_boxes = tf.TensorArray(tf.float32, size=num_classes)
      nmsed_classes = tf.TensorArray(tf.int32, size=num_classes)
      nmsed_scores = tf.TensorArray(tf.float32, size=num_classes)

      scores, indices = self._select_top_k_scores(
          scores, tf.math.minimum(total_anchors, pre_nms_top_k))

      def _loop_cond(boxes, scores, indices, nmsed_boxes, nmsed_classes,
                     nmsed_scores, max_num_detections, num_classes_for_box,
                     pre_nms_score_threshold, nms_iou_threshold, idx):
        return idx < num_classes

      _, _, _, nmsed_boxes, nmsed_classes, nmsed_scores, _, _, _, _, _ = tf.while_loop(
          _loop_cond,
          self._single_iter, [
              boxes, scores, indices, nmsed_boxes, nmsed_classes, nmsed_scores,
              max_num_detections, num_classes_for_box, pre_nms_score_threshold,
              nms_iou_threshold,
              tf.constant(0)
          ],
          parallel_iterations=20)

      nmsed_boxes = nmsed_boxes.concat()
      nmsed_scores = nmsed_scores.concat()
      nmsed_classes = nmsed_classes.concat()

      nmsed_boxes = tf.transpose(nmsed_boxes, perm=(1, 0, 2))
      nmsed_classes = tf.transpose(nmsed_classes, perm=(1, 0))
      nmsed_scores = tf.transpose(nmsed_scores, perm=(1, 0))

      nmsed_scores, indices = tf.nn.top_k(
          nmsed_scores, k=max_num_detections, sorted=True)
      nmsed_boxes = tf.gather(nmsed_boxes, indices, batch_dims=1, axis=1)
      nmsed_classes = tf.gather(nmsed_classes, indices, batch_dims=1)
      valid_detections = tf.reduce_sum(
          input_tensor=tf.cast(tf.greater(nmsed_scores, -1), tf.int32), axis=1)
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


BASE_NMS = TiledNMS(iou_type='diou', beta=0.6)


def sorted_non_max_suppression_padded(scores, boxes, classes, max_output_size,
                                      iou_threshold):

  return BASE_NMS.sorted_non_max_suppression_padded(scores, boxes, classes,
                                                    max_output_size,
                                                    iou_threshold)


def nms(boxes,
        classes,
        confidence,
        k,
        pre_nms_thresh,
        nms_thresh,
        prenms_top_k=500,
        limit_pre_thresh=False,
        use_classes=True):

  # boxes_ = tf.expand_dims(boxes, axis = -2)
  # boxes, confidence, classes, valid = BASE_NMS.complete_nms(boxes_, classes,
  #                                 pre_nms_top_k = 5000,
  #                                 pre_nms_score_threshold=pre_nms_thresh,
  #                                 nms_iou_threshold= nms_thresh,
  #                                 max_num_detections= 200)

  # if use_classes:
  confidence = tf.reduce_max(classes, axis=-1)
  confidence, boxes, classes = sort_drop(confidence, boxes, classes,
                                         prenms_top_k)
  # classes = tf.cast(tf.argmax(classes, axis=-1), tf.float32)
  # boxes, classes, confidence = segment_nms(boxes, classes, confidence,
  #                                          prenms_top_k, nms_thresh)
  boxes, classes, confidence = sorted_non_max_suppression_padded(
      confidence, boxes, classes, prenms_top_k, nms_thresh)

  class_confidence, class_ind = tf.math.top_k(
      classes, k=tf.shape(classes)[-1], sorted=True)
  mask = tf.fill(
      tf.shape(class_confidence),
      tf.cast(pre_nms_thresh, dtype=class_confidence.dtype))
  mask = tf.math.ceil(tf.nn.relu(class_confidence - mask))
  class_confidence = tf.cast(class_confidence, mask.dtype) * mask
  class_ind = tf.cast(class_ind, mask.dtype) * mask

  top_n = tf.math.minimum(100, tf.shape(classes)[-1])
  classes = class_ind[..., :top_n]
  confidence = class_confidence[..., :top_n]

  boxes = tf.expand_dims(boxes, axis=-2)
  boxes = tf.tile(boxes, [1, 1, top_n, 1])

  shape = tf.shape(boxes)
  boxes = tf.reshape(boxes, [shape[0], -1, 4])
  classes = tf.reshape(classes, [shape[0], -1])
  confidence = tf.reshape(confidence, [shape[0], -1])

  confidence, boxes, classes = sort_drop(confidence, boxes, classes, k)

  mask = tf.fill(
      tf.shape(confidence), tf.cast(pre_nms_thresh, dtype=confidence.dtype))
  mask = tf.math.ceil(tf.nn.relu(confidence - mask))
  confidence = confidence * mask
  mask = tf.expand_dims(mask, axis=-1)
  boxes = boxes * mask
  classes = classes * mask

  classes = tf.squeeze(classes, axis=-1)
  return boxes, classes, confidence
