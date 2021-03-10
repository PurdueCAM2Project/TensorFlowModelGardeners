from tensorflow import keras as ks
import tensorflow as tf

class CenterNetLayer(ks.Model):
  
  def __init__(self,
               max_detections=100,
               detect_per_channel=False,
               center_thresh=0.1,
               **kwargs):

    super().__init__(**kwargs)
    self._max_detections = max_detections
    self._detect_per_channel = detect_per_channel
    self._center_thresh = center_thresh
  
  def process_heatmap(self, 
                      feature_map,
                      kernel_size=3,
                      peak_error=1e-6):
    feature_map = tf.math.sigmoid(feature_map)
    if not kernel_size or kernel_size == 1:
      feature_map_peaks = feature_map
    else:
      feature_map_max_pool = tf.nn.max_pool(
          feature_map, ksize=kernel_size, strides=1, padding='SAME')

      feature_map_peak_mask = tf.math.abs(
          feature_map - feature_map_max_pool) < peak_error

      # Zero out everything that is not a peak.
      feature_map_peaks = (
          feature_map * tf.cast(feature_map_peak_mask, tf.float32))
    
    return feature_map_peaks
  
  def get_row_col_channel_indices_from_flattened_indices(self,
                                                         indices, 
                                                         num_cols,
                                                         num_channels):
    """Computes row, column and channel indices from flattened indices.
    Args:
      indices: An integer tensor of any shape holding the indices in the flattened
        space.
      num_cols: Number of columns in the image (width).
      num_channels: Number of channels in the image.
    Returns:
      row_indices: The row indices corresponding to each of the input indices.
        Same shape as indices.
      col_indices: The column indices corresponding to each of the input indices.
        Same shape as indices.
      channel_indices. The channel indices corresponding to each of the input
        indices.
    """
    # Avoid using mod operator to make the ops more easy to be compatible with
    # different environments, e.g. WASM.
    row_indices = (indices // num_channels) // num_cols
    col_indices = (indices // num_channels) - row_indices * num_cols
    channel_indices_temp = indices // num_channels
    channel_indices = indices - channel_indices_temp * num_channels

    return row_indices, col_indices, channel_indices

  def multi_range(self,
                   limit,
                   value_repetitions=1,
                   range_repetitions=1,
                   dtype=tf.int32):
    """Creates a sequence with optional value duplication and range repetition.
    As an example (see the Args section for more details),
    _multi_range(limit=2, value_repetitions=3, range_repetitions=4) returns:
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    Args:
      limit: A 0-D Tensor (scalar). Upper limit of sequence, exclusive.
      value_repetitions: Integer. The number of times a value in the sequence is
        repeated. With value_repetitions=3, the result is [0, 0, 0, 1, 1, 1, ..].
      range_repetitions: Integer. The number of times the range is repeated. With
        range_repetitions=3, the result is [0, 1, 2, .., 0, 1, 2, ..].
      dtype: The type of the elements of the resulting tensor.
    Returns:
      A 1-D tensor of type `dtype` and size
        [`limit` * `value_repetitions` * `range_repetitions`] that contains the
        specified range with given repetitions.
    """
    return tf.reshape(
        tf.tile(
          tf.expand_dims(tf.range(limit, dtype=dtype), axis=-1),
          multiples=[range_repetitions, value_repetitions]), [-1])
  
  def get_top_k_peaks(self,
                      feature_map_peaks,
                      k=100,
                      per_channel=False):
    (batch_size, _, width, num_channels) = feature_map_peaks.get_shape()

    if per_channel:
      if k == 1:
        feature_map_flattened = tf.reshape(
            feature_map_peaks, [batch_size, -1, num_channels])
        scores = tf.math.reduce_max(feature_map_flattened, axis=1)
        peak_flat_indices = tf.math.argmax(
            feature_map_flattened, axis=1, output_type=tf.dtypes.int32)
        peak_flat_indices = tf.expand_dims(peak_flat_indices, axis=-1)
      else:
        # Perform top k over batch and channels.
        feature_map_peaks_transposed = tf.transpose(feature_map_peaks,
                                                    perm=[0, 3, 1, 2])
        feature_map_peaks_transposed = tf.reshape(
            feature_map_peaks_transposed, [batch_size, num_channels, -1])
        scores, peak_flat_indices = tf.math.top_k(
            feature_map_peaks_transposed, k=k)
      # Convert the indices such that they represent the location in the full
      # (flattened) feature map of size [batch, height * width * channels].
      channel_idx = tf.range(num_channels)[tf.newaxis, :, tf.newaxis]
      peak_flat_indices = num_channels * peak_flat_indices + channel_idx
      scores = tf.reshape(scores, [batch_size, -1])
      peak_flat_indices = tf.reshape(peak_flat_indices, [batch_size, -1])
    else:
      if k == 1:
        feature_map_peaks_flat = tf.reshape(feature_map_peaks, [batch_size, -1])
        scores = tf.math.reduce_max(feature_map_peaks_flat, axis=1, keepdims=True)
        peak_flat_indices = tf.expand_dims(tf.math.argmax(
            feature_map_peaks_flat, axis=1, output_type=tf.dtypes.int32), axis=-1)
      else:
        feature_map_peaks_flat = tf.reshape(feature_map_peaks, [batch_size, -1])
        scores, peak_flat_indices = tf.math.top_k(feature_map_peaks_flat, k=k)

    # Get x, y and channel indices corresponding to the top indices in the flat
    # array.
    y_indices, x_indices, channel_indices = (
        self.get_row_col_channel_indices_from_flattened_indices(
            peak_flat_indices, width, num_channels))
    return scores, y_indices, x_indices, channel_indices

  def get_boxes(self, 
                detection_scores, 
                y_indices, 
                x_indices,
                channel_indices, 
                height_width_predictions,
                offset_predictions):
    batch_size, num_boxes = y_indices.get_shape()

    # TF Lite does not support tf.gather with batch_dims > 0, so we need to use
    # tf_gather_nd instead and here we prepare the indices for that.
    combined_indices = tf.stack([
        self.multi_range(batch_size, value_repetitions=num_boxes),
        tf.reshape(y_indices, [-1]),
        tf.reshape(x_indices, [-1])
    ], axis=1)
    new_height_width = tf.gather_nd(height_width_predictions, combined_indices)
    new_height_width = tf.reshape(new_height_width, [batch_size, num_boxes, -1])

    new_offsets = tf.gather_nd(offset_predictions, combined_indices)
    offsets = tf.reshape(new_offsets, [batch_size, num_boxes, -1])

    y_indices = tf.cast(y_indices, dtype=tf.float32)
    x_indices = tf.cast(x_indices, dtype=tf.float32)

    height_width = tf.maximum(new_height_width, 0)
    heights, widths = tf.unstack(height_width, axis=2)
    y_offsets, x_offsets = tf.unstack(offsets, axis=2)

    detection_classes = channel_indices

    num_detections = tf.reduce_sum(tf.cast(detection_scores > 0, dtype=tf.int32), axis=1)

    boxes = tf.stack([y_indices + y_offsets - heights / 2.0,
                      x_indices + x_offsets - widths / 2.0,
                      y_indices + y_offsets + heights / 2.0,
                      x_indices + x_offsets + widths / 2.0], axis=2)

    return boxes, detection_classes, detection_scores, num_detections

  def call(self, inputs):
    # Get heatmaps from decoded ourputs via final hourglass stack output
    ct_heatmaps = inputs['ct_heatmaps'][-1]
    ct_sizes = inputs['ct_size'][-1]
    ct_offsets = inputs['ct_offset'][-1]

    batch_size = ct_heatmaps.get_shape()[0]

    # Process heatmaps using 3x3 convolution and applying sigmoid
    peaks = self.process_heatmap(ct_heatmaps)
    
    scores, y_indices, x_indices, channel_indices = self.get_top_k_peaks(peaks, k=self._max_detections, per_channel=self._detect_per_channel)

    boxes, classes, scores, num_det = self.get_boxes(scores, 
      y_indices, x_indices, channel_indices, ct_sizes, ct_offsets)
    
    boxes = boxes / ct_heatmaps.get_shape()[1]

    boxes = tf.expand_dims(boxes, axis=-2)
    multi_class_scores = tf.gather_nd(
        peaks, tf.stack([y_indices, x_indices], -1), batch_dims=1)

    boxes, scores, _, num_det = tf.image.combined_non_max_suppression(boxes=boxes, 
      scores=multi_class_scores, max_output_size_per_class=self._max_detections, 
      max_total_size=self._max_detections, score_threshold=0.1)

    return {
      'bbox': boxes,
      'classes': classes,
      'confidence': scores,
      'num_dets': num_det
    }