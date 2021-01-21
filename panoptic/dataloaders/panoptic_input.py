# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Data parser and processing for segmentation datasets."""

import tensorflow as tf
from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import preprocess_ops


class Decoder(decoder.Decoder):
  """A tf.Example decoder for segmentation task."""

  def __init__(self):
    self._keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/width': tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value='')
    }

  def decode(self, serialized_example):
    return tf.io.parse_single_example(
        serialized_example, self._keys_to_features)


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors.
  """

  def __init__(self,
               output_size,
               train_on_crops=False,
               resize_eval_groundtruth=True,
               groundtruth_padded_size=None,
               ignore_label=255,
               aug_rand_hflip=False,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.
    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      train_on_crops: `bool`, if True, a training crop of size output_size
        is returned. This is useful for cropping original images during training
        while evaluating on original image sizes.
      resize_eval_groundtruth: `bool`, if True, eval groundtruth masks are
        resized to output_size.
      groundtruth_padded_size: `Tensor` or `list` for [height, width]. When
        resize_eval_groundtruth is set to False, the groundtruth masks are
        padded to this size.
      ignore_label: `int` the pixel with ignore label will not used for training
        and evaluation.
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
    """
    self._output_size = output_size
    self._train_on_crops = train_on_crops
    self._resize_eval_groundtruth = resize_eval_groundtruth
    if (not resize_eval_groundtruth) and (groundtruth_padded_size is None):
      raise ValueError('groundtruth_padded_size ([height, width]) needs to be'
                       'specified when resize_eval_groundtruth is False.')
    self._groundtruth_padded_size = groundtruth_padded_size
    self._ignore_label = ignore_label

    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max

    # dtype.
    self._dtype = dtype

  def _prepare_image_and_label(self, data):
    """Prepare normalized image and label."""
    image = tf.io.decode_image(data['image/encoded'], channels=3)
    label = tf.io.decode_image(data['image/segmentation/class/encoded'],
                               channels=1)
    height = data['image/height']
    width = data['image/width']
    image = tf.reshape(image, (height, width, 3))

    label = tf.reshape(label, (1, height, width))
    label = tf.cast(label, tf.float32)
    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image)
    return image, label

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""
    image, label = self._prepare_image_and_label(data)

    if self._train_on_crops:
      label = tf.reshape(label, [data['image/height'], data['image/width'], 1])
      image_mask = tf.concat([image, label], axis=2)
      image_mask_crop = tf.image.random_crop(image_mask,
                                             self._output_size + [4])
      image = image_mask_crop[:, :, :-1]
      label = tf.reshape(image_mask_crop[:, :, -1], [1] + self._output_size)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      image, _, label = preprocess_ops.random_horizontal_flip(
          image, masks=label)

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        self._output_size,
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)

    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]

    # Pad label and make sure the padded region assigned to the ignore label.
    # The label is first offset by +1 and then padded with 0.
    label += 1
    label = tf.expand_dims(label, axis=3)
    label = preprocess_ops.resize_and_crop_masks(
        label, image_scale, self._output_size, offset)
    label -= 1
    label = tf.where(tf.equal(label, -1),
                     self._ignore_label * tf.ones_like(label), label)
    label = tf.squeeze(label, axis=0)
    valid_mask = tf.not_equal(label, self._ignore_label)
    labels = {
        'masks': label,
        'valid_masks': valid_mask,
        'image_info': image_info,
    }

    # Cast image as self._dtype
    image = tf.cast(image, dtype=self._dtype)

    return image, labels

  def _parse_eval_data(self, data):
    """Parses data for training and evaluation."""
    image, label = self._prepare_image_and_label(data)
    # The label is first offset by +1 and then padded with 0.
    label += 1
    label = tf.expand_dims(label, axis=3)

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image, self._output_size, self._output_size)

    if self._resize_eval_groundtruth:
      # Resizes eval masks to match input image sizes. In that case, mean IoU
      # is computed on output_size not the original size of the images.
      image_scale = image_info[2, :]
      offset = image_info[3, :]
      label = preprocess_ops.resize_and_crop_masks(label, image_scale,
                                                   self._output_size, offset)
    else:
      label = tf.image.pad_to_bounding_box(
          label, 0, 0, self._groundtruth_padded_size[0],
          self._groundtruth_padded_size[1])

    label -= 1
    label = tf.where(tf.equal(label, -1),
                     self._ignore_label * tf.ones_like(label), label)
    label = tf.squeeze(label, axis=0)

    valid_mask = tf.not_equal(label, self._ignore_label)
    labels = {
        'masks': label,
        'valid_masks': valid_mask,
        'image_info': image_info
    }

    # Cast image as self._dtype
    image = tf.cast(image, dtype=self._dtype)

    return image, labels


from official.vision.beta.dataloaders import parser
from official.vision.beta.dataloaders import utils
from official.vision.beta.ops import anchor
from official.vision.beta.ops import box_ops
from official.vision.beta.ops import preprocess_ops


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               min_level,
               max_level,
               num_scales,
               aspect_ratios,
               anchor_size,
               rpn_match_threshold=0.7,
               rpn_unmatched_threshold=0.3,
               rpn_batch_size_per_im=256,
               rpn_fg_fraction=0.5,
               aug_rand_hflip=False,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               skip_crowd_during_training=True,
               max_num_instances=100,
               include_mask=False,
               mask_crop_size=112,
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.
    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      min_level: `int` number of minimum level of the output feature pyramid.
      max_level: `int` number of maximum level of the output feature pyramid.
      num_scales: `int` number representing intermediate scales added
        on each level. For instances, num_scales=2 adds one additional
        intermediate anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: `list` of float numbers representing the aspect raito
        anchors added on each level. The number indicates the ratio of width to
        height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
        on each scale level.
      anchor_size: `float` number representing the scale of size of the base
        anchor to the feature stride 2^level.
      rpn_match_threshold:
      rpn_unmatched_threshold:
      rpn_batch_size_per_im:
      rpn_fg_fraction:
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      skip_crowd_during_training: `bool`, if True, skip annotations labeled with
        `is_crowd` equals to 1.
      max_num_instances: `int` number of maximum number of instances in an
        image. The groundtruth data will be padded to `max_num_instances`.
      include_mask: a bool to indicate whether parse mask groundtruth.
      mask_crop_size: the size which groundtruth mask is cropped to.
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
    """

    self._max_num_instances = max_num_instances
    self._skip_crowd_during_training = skip_crowd_during_training

    # Anchor.
    self._output_size = output_size
    self._min_level = min_level
    self._max_level = max_level
    self._num_scales = num_scales
    self._aspect_ratios = aspect_ratios
    self._anchor_size = anchor_size

    # Target assigning.
    self._rpn_match_threshold = rpn_match_threshold
    self._rpn_unmatched_threshold = rpn_unmatched_threshold
    self._rpn_batch_size_per_im = rpn_batch_size_per_im
    self._rpn_fg_fraction = rpn_fg_fraction

    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max

    # Mask.
    self._include_mask = include_mask
    self._mask_crop_size = mask_crop_size

    # Image output dtype.
    self._dtype = dtype

  def _parse_train_data(self, data):
    """Parses data for training.
    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.
    Returns:
      image: image tensor that is preproessed to have normalized value and
        dimension [output_size[0], output_size[1], 3]
      labels: a dictionary of tensors used for training. The following describes
        {key: value} pairs in the dictionary.
        image_info: a 2D `Tensor` that encodes the information of the image and
          the applied preprocessing. It is in the format of
          [[original_height, original_width], [scaled_height, scaled_width],
        anchor_boxes: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, 4] representing anchor boxes at each level.
        rpn_score_targets: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, anchors_per_location]. The height_l and
          width_l represent the dimension of class logits at l-th level.
        rpn_box_targets: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, anchors_per_location * 4]. The height_l and
          width_l represent the dimension of bounding box regression output at
          l-th level.
        gt_boxes: Groundtruth bounding box annotations. The box is represented
           in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled
           image that is fed to the network. The tennsor is padded with -1 to
           the fixed dimension [self._max_num_instances, 4].
        gt_classes: Groundtruth classes annotations. The tennsor is padded
          with -1 to the fixed dimension [self._max_num_instances].
        gt_masks: groundtrugh masks cropped by the bounding box and
          resized to a fixed size determined by mask_crop_size.
    """
    classes = data['groundtruth_classes']
    boxes = data['groundtruth_boxes']
    if self._include_mask:
      masks = data['groundtruth_instance_masks']

    is_crowds = data['groundtruth_is_crowd']
    # Skips annotations with `is_crowd` = True.
    if self._skip_crowd_during_training:
      num_groundtruths = tf.shape(classes)[0]
      with tf.control_dependencies([num_groundtruths, is_crowds]):
        indices = tf.cond(
            tf.greater(tf.size(is_crowds), 0),
            lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
            lambda: tf.cast(tf.range(num_groundtruths), tf.int64))
      classes = tf.gather(classes, indices)
      boxes = tf.gather(boxes, indices)
      if self._include_mask:
        masks = tf.gather(masks, indices)

    # Gets original image and its size.
    image = data['image']
    image_shape = tf.shape(image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      if self._include_mask:
        image, boxes, masks = preprocess_ops.random_horizontal_flip(
            image, boxes, masks)
      else:
        image, boxes, _ = preprocess_ops.random_horizontal_flip(
            image, boxes)

    # Converts boxes from normalized coordinates to pixel coordinates.
    # Now the coordinates of boxes are w.r.t. the original image.
    boxes = box_ops.denormalize_boxes(boxes, image_shape)

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=preprocess_ops.compute_padded_size(
            self._output_size, 2 ** self._max_level),
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)
    image_height, image_width, _ = image.get_shape().as_list()

    # Resizes and crops boxes.
    # Now the coordinates of boxes are w.r.t the scaled image.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = preprocess_ops.resize_and_crop_boxes(
        boxes, image_scale, image_info[1, :], offset)

    # Filters out ground truth boxes that are all zeros.
    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    if self._include_mask:
      masks = tf.gather(masks, indices)
      # Transfer boxes to the original image space and do normalization.
      cropped_boxes = boxes + tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
      cropped_boxes /= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
      cropped_boxes = box_ops.normalize_boxes(cropped_boxes, image_shape)
      num_masks = tf.shape(masks)[0]
      masks = tf.image.crop_and_resize(
          tf.expand_dims(masks, axis=-1),
          cropped_boxes,
          box_indices=tf.range(num_masks, dtype=tf.int32),
          crop_size=[self._mask_crop_size, self._mask_crop_size],
          method='bilinear')
      masks = tf.squeeze(masks, axis=-1)

    # Assigns anchor targets.
    # Note that after the target assignment, box targets are absolute pixel
    # offsets w.r.t. the scaled image.
    input_anchor = anchor.build_anchor_generator(
        min_level=self._min_level,
        max_level=self._max_level,
        num_scales=self._num_scales,
        aspect_ratios=self._aspect_ratios,
        anchor_size=self._anchor_size)
    anchor_boxes = input_anchor(image_size=(image_height, image_width))
    anchor_labeler = anchor.RpnAnchorLabeler(
        self._rpn_match_threshold,
        self._rpn_unmatched_threshold,
        self._rpn_batch_size_per_im,
        self._rpn_fg_fraction)
    rpn_score_targets, rpn_box_targets = anchor_labeler.label_anchors(
        anchor_boxes, boxes,
        tf.cast(tf.expand_dims(classes, axis=-1), dtype=tf.float32))

    # Casts input image to self._dtype
    image = tf.cast(image, dtype=self._dtype)

    # Packs labels for model_fn outputs.
    labels = {
        'anchor_boxes':
            anchor_boxes,
        'image_info':
            image_info,
        'rpn_score_targets':
            rpn_score_targets,
        'rpn_box_targets':
            rpn_box_targets,
        'gt_boxes':
            preprocess_ops.clip_or_pad_to_fixed_size(boxes,
                                                     self._max_num_instances,
                                                     -1),
        'gt_classes':
            preprocess_ops.clip_or_pad_to_fixed_size(classes,
                                                     self._max_num_instances,
                                                     -1),
    }
    if self._include_mask:
      labels['gt_masks'] = preprocess_ops.clip_or_pad_to_fixed_size(
          masks, self._max_num_instances, -1)

    return image, labels

  def _parse_eval_data(self, data):
    """Parses data for evaluation.
    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.
    Returns:
      A dictionary of {'images': image, 'labels': labels} where
        image: image tensor that is preproessed to have normalized value and
          dimension [output_size[0], output_size[1], 3]
        labels: a dictionary of tensors used for training. The following
          describes {key: value} pairs in the dictionary.
          source_ids: Source image id. Default value -1 if the source id is
            empty in the groundtruth annotation.
          image_info: a 2D `Tensor` that encodes the information of the image
            and the applied preprocessing. It is in the format of
            [[original_height, original_width], [scaled_height, scaled_width],
          anchor_boxes: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, 4] representing anchor boxes at each
            level.
    """
    # Gets original image and its size.
    image = data['image']
    image_shape = tf.shape(image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image)

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=preprocess_ops.compute_padded_size(
            self._output_size, 2 ** self._max_level),
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image_height, image_width, _ = image.get_shape().as_list()

    # Casts input image to self._dtype
    image = tf.cast(image, dtype=self._dtype)

    # Converts boxes from normalized coordinates to pixel coordinates.
    boxes = box_ops.denormalize_boxes(data['groundtruth_boxes'], image_shape)

    # Compute Anchor boxes.
    input_anchor = anchor.build_anchor_generator(
        min_level=self._min_level,
        max_level=self._max_level,
        num_scales=self._num_scales,
        aspect_ratios=self._aspect_ratios,
        anchor_size=self._anchor_size)
    anchor_boxes = input_anchor(image_size=(image_height, image_width))

    labels = {
        'image_info': image_info,
        'anchor_boxes': anchor_boxes,
    }

    groundtruths = {
        'source_id': data['source_id'],
        'height': data['height'],
        'width': data['width'],
        'num_detections': tf.shape(data['groundtruth_classes']),
        'boxes': boxes,
        'classes': data['groundtruth_classes'],
        'areas': data['groundtruth_area'],
        'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32),
    }
    groundtruths['source_id'] = utils.process_source_id(
        groundtruths['source_id'])
    groundtruths = utils.pad_groundtruths_to_fixed_size(
        groundtruths, self._max_num_instances)
    labels['groundtruths'] = groundtruths
    return image, labels