# Moved groundtruth.py here
import tensorflow as tf
from yolo.ops import preprocessing_ops


LARGE_NUM = 1. / tf.keras.backend.epsilon()

@tf.function
def _smallest_positive_root(a, b, c) -> tf.Tensor:
  """
    Returns the smallest positive root of a quadratic equation.
    This implements the fixed version in https://github.com/princeton-vl/CornerNet.
  """

  discriminant = b ** 2 - 4 * a * c
  discriminant_sqrt = tf.sqrt(discriminant)

  root1 = (-b - discriminant_sqrt) / (2 * a)
  root2 = (-b + discriminant_sqrt) / (2 * a)

  return tf.where(tf.less(discriminant, 0), LARGE_NUM, (-b + discriminant_sqrt) / (2))
  # return tf.where(tf.less(discriminant, 0), LARGE_NUM, tf.where(tf.less(root1, 0), root2, root1))

@tf.function
def gaussian_radius(det_size, min_overlap=0.7) -> int:
  """
    Given a bounding box size, returns a lower bound on how far apart the
    corners of another bounding box can lie while still maintaining the given
    minimum overlap, or IoU. Modified from implementation found in
    https://github.com/tensorflow/models/blob/master/research/object_detection/core/target_assigner.py.

    Params:
        det_size (tuple): tuple of integers representing height and width
        min_overlap (tf.float32): minimum IoU desired
    Returns:
        int representing desired gaussian radius
    """
  height, width = det_size

  # Case where detected box is offset from ground truth and no box completely
  # contains the other.

  a1  = 1
  b1  = -(height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  r1 = _smallest_positive_root(a1, b1, c1)

  # Case where detection is smaller than ground truth and completely contained
  # in it.

  a2  = 4
  b2  = -2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  r2 = _smallest_positive_root(a2, b2, c2)

  # Case where ground truth is smaller than detection and completely contained
  # in it.

  a3  = 4 * min_overlap
  b3  = 2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  r3 = _smallest_positive_root(a3, b3, c3)
  # TODO discuss whether to return scalar or tensor
  # return tf.reduce_min([r1, r2, r3], axis=0)

  print(r1, r2, r3)
  return tf.reduce_min([r1, r2, r3], axis=0)

@tf.function
def _gaussian_penalty(radius: int, dtype=tf.float32) -> tf.Tensor:
    """
    This represents the penalty reduction around a point.
    Params:
        radius (int): integer for radius of penalty reduction
        type (tf.dtypes.DType): datatype of returned tensor
    Returns:
        tf.Tensor of shape (2 * radius + 1, 2 * radius + 1).
    """
    width = 2 * radius + 1
    sigma = tf.cast(radius / 3, dtype=dtype)

    range_width = tf.range(width)
    range_width = tf.cast(range_width - tf.expand_dims(radius, axis=-1), dtype=dtype)

    x = tf.expand_dims(range_width, axis=-1)
    y = tf.expand_dims(range_width, axis=-2)

    exponent = ((-1 * (x ** 2) - (y ** 2)) / (2 * sigma ** 2))
    return tf.math.exp(exponent)

@tf.function
def cartesian_product(*tensors, repeat=1):
  """
  Equivalent of itertools.product except for TensorFlow tensors.

  Example:
    cartesian_product(tf.range(3), tf.range(4))

    array([[0, 0],
       [0, 1],
       [0, 2],
       [0, 3],
       [1, 0],
       [1, 1],
       [1, 2],
       [1, 3],
       [2, 0],
       [2, 1],
       [2, 2],
       [2, 3]], dtype=int32)>

  Params:
    tensors (list[tf.Tensor]): a list of 1D tensors to compute the product of
    repeat (int): number of times to repeat the tensors
      (https://docs.python.org/3/library/itertools.html#itertools.product)

  Returns:
    An nD tensor where n is the number of tensors
  """
  tensors = tensors * repeat
  return tf.reshape(tf.transpose(tf.stack(tf.meshgrid(*tensors, indexing='ij')),
    [*[i + 1 for i in range(len(tensors))], 0]), (-1, len(tensors)))

@tf.function
def write_all(ta, index, values):
  for i in range(tf.shape(values)[0]):
    ta = ta.write(index + i, values[i, ...])
  return ta, index + i

# scaling_factor doesn't do anything right now
@tf.function
def draw_gaussian(heatmap, blobs, scaling_factor=1, dtype=tf.float32):
    """
    Draws a gaussian heatmap around a center point given a radius.
    Params:
        heatmap (tf.Tensor): heatmap placeholder to fill
        blobs (tf.Tensor): a tensor whose last dimension is 5 integers for
          the batch index, the category of the object, center (x, y), and
          for radius of the gaussian
        scaling_factor (int): scaling factor for gaussian
    """
    blobs = tf.cast(blobs, tf.int32)
    bn_ind = blobs[..., 0]
    category = blobs[..., 1]
    x = blobs[..., 2]
    y = blobs[..., 3]
    radius = blobs[..., 4]
    num_boxes = tf.shape(radius)[0]

    diameter = 2 * radius + 1

    heatmap_shape = tf.shape(heatmap)
    height, width = heatmap_shape[-2], heatmap_shape[-1]

    left, right = tf.math.minimum(x, radius), tf.math.minimum(width - x, radius + 1)
    top, bottom = tf.math.minimum(y, radius), tf.math.minimum(height - y, radius + 1)

    print('heatmap ',heatmap)
    print(len(heatmap))
    print('category ',category)

    # TODO: make sure this replicates original functionality
    # masked_heatmap  = heatmap[0, category, y - top:y + bottom, x - left:x + right]
    # masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    # np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    update_count = tf.reduce_sum((bottom + top) * (right + left))
    masked_gaussian_ta = tf.TensorArray(dtype, size=update_count)
    heatmap_mask_ta = tf.TensorArray(tf.int32, element_shape=tf.TensorShape((4,)), size=update_count)
    i = 0
    for j in range(num_boxes):
      bn_i = bn_ind[j]
      cat = category[j]
      X = x[j]
      Y = y[j]
      R = radius[j]
      l = left[j]
      r = right[j]
      t = top[j]
      b = bottom[j]

      gaussian = _gaussian_penalty(R, dtype=dtype)
      masked_gaussian_instance = tf.reshape(gaussian[R - t:R + b, R - l:R + r], (-1,))
      heatmap_mask_instance = cartesian_product([bn_i], [cat], tf.range(Y - t, Y + b), tf.range(X - l, X + r))
      masked_gaussian_ta, _ = write_all(masked_gaussian_ta, i, masked_gaussian_instance)
      heatmap_mask_ta, i = write_all(heatmap_mask_ta, i, heatmap_mask_instance)
    masked_gaussian = masked_gaussian_ta.concat()
    heatmap_mask = heatmap_mask_ta.concat()
    print(masked_gaussian, heatmap_mask)
    heatmap_mask = tf.reshape(heatmap_mask, (-1, 4))
    masked_gaussian_ta.close()
    heatmap_mask_ta.close()
    # masked_gaussian = tf.concat(masked_gaussian, axis = 0)
    # heatmap_mask = tf.concat(heatmap_mask, axis = 0)
    heatmap = tf.tensor_scatter_nd_max(heatmap, heatmap_mask, masked_gaussian * scaling_factor)
    return heatmap

# def draw_gaussian(heatmap, category, center, radius, scaling_factor=1):
#     """
#     Draws a gaussian heatmap around a center point given a radius.
#     Params:
#         heatmap (tf.Tensor): heatmap placeholder to fill
#         center (int): integer for center of gaussian
#         radius (int): integer for radius of gaussian
#         scaling_factor (int): scaling factor for gaussian
#     """

#     diameter = 2 * radius + 1
#     gaussian = _gaussian_penalty(radius)

#     x, y = center

#     height, width = heatmap.shape[0:2]

#     left, right = min(x, radius), min(width - x, radius + 1)
#     top, bottom = min(y, radius), min(height - y, radius + 1)

#     print('heatmap ',heatmap)
#     print(len(heatmap))
#     print('category ',category)

#     heatmap_category = heatmap[0][category, ...]

#     print('heatmap_category ',heatmap_category)

#     masked_heatmap  = heatmap_category[y - top:y + bottom, x - left:x + right]
#     masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
#     # TODO: make sure this replicates original functionality
#     # np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
#     masked_heatmap = tf.math.maximum(masked_heatmap, masked_gaussian * scaling_factor)
#     return masked_heatmap