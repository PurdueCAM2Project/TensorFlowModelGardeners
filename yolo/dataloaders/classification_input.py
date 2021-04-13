"""Classification parser."""

# Import libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import preprocess_ops
from yolo.ops import preprocessing_ops


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               num_classes,
               aug_rand_saturation=True,
               aug_rand_brightness=True,
               aug_rand_zoom=True,
               aug_rand_rotate=True,
               aug_rand_hue=True,
               aug_rand_aspect=True,
               scale=[128, 448],
               seed=10,
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.
    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      num_classes: `float`, number of classes.
      aug_rand_saturation: `bool`, if True, augment training with random
        saturation.
      aug_rand_brightness: `bool`, if True, augment training with random
        brightness.
      aug_rand_zoom: `bool`, if True, augment training with random
        zoom.
      aug_rand_rotate: `bool`, if True, augment training with random
        rotate.
      aug_rand_hue: `bool`, if True, augment training with random
        hue.
      aug_rand_aspect: `bool`, if True, augment training with random
        aspect.
      scale: 'list', `Tensor` or `list` for [low, high] of the bounds of the 
        random scale.
      seed: an `int` for the seed used by tf.random
    """
    self._output_size = output_size
    self._aug_rand_saturation = aug_rand_saturation
    self._aug_rand_brightness = aug_rand_brightness
    self._aug_rand_zoom = aug_rand_zoom
    self._aug_rand_rotate = aug_rand_rotate
    self._aug_rand_hue = aug_rand_hue
    self._num_classes = num_classes
    self._aug_rand_aspect = aug_rand_aspect

    scaler = 256 if self._output_size[0] == None else self._output_size[0]
    self._scale = tf.cast(((tf.convert_to_tensor(scale) / 256) * scaler),
                          tf.int32)

    self._seed = seed
    if dtype == 'float32':
      self._dtype = tf.float32
    elif dtype == 'float16':
      self._dtype = tf.float16
    elif dtype == 'bfloat16':
      self._dtype = tf.bfloat16
    else:
      raise ValueError('dtype {!r} is not supported!'.format(dtype))

  def _parse_train_data(self, decoded_tensors):
    """Generates images and labels that are usable for model training.
     Args:
       decoded_tensors: a dict of Tensors produced by the decoder.
     Returns:
       images: the image tensor.
       labels: a dict of Tensors that contains labels.
    """
    image = tf.io.decode_image(decoded_tensors['image/encoded'])
    image.set_shape((None, None, 3))
    image = tf.cast(image, tf.float32) / 255
    w = tf.cast(tf.shape(image)[0], tf.float32)
    h = tf.cast(tf.shape(image)[1], tf.int32)

    # slow as balls 20 second addition at batch size 128
    # image = tf.image.rgb_to_hsv(image)
    # i_h, i_s, i_v = tf.split(image, 3, axis=-1)

    # if self._aug_rand_hue:
    #   delta = 0.1 #preprocessing_ops.rand_uniform_strong(-0.1, 0.1)
    #   i_h = i_h + delta  # Hue
    #   i_h = tf.clip_by_value(i_h, 0.0, 1.0)
    # if self._aug_rand_saturation:
    #   delta = 0.75 #preprocessing_ops.rand_scale(0.75)
    #   i_s = i_s * delta
    # if self._aug_rand_brightness:
    #   delta = 0.75 #preprocessing_ops.rand_scale(0.75)
    #   i_v = i_v * delta

    # image = tf.concat([i_h, i_s, i_v], axis=-1)
    # image = tf.image.hsv_to_rgb(image)

    if self._aug_rand_hue:
      delta = preprocessing_ops.rand_uniform_strong(-0.1, 0.1)
      image = tf.image.adjust_hue(image, delta)
    if self._aug_rand_saturation:
      delta = preprocessing_ops.rand_scale(0.75)
      image = tf.image.adjust_saturation(image, delta)
    if self._aug_rand_brightness:
      delta = preprocessing_ops.rand_scale(0.75)
      image *= delta
    image = tf.clip_by_value(image, 0.0, 1.0)

    # stddev = tf.random.uniform([],
    #                            minval=0,
    #                            maxval=40 / 255,
    #                            seed=self._seed,
    #                            dtype=tf.float32)
    # noise = tf.random.normal(
    #     shape=tf.shape(image), mean=0.0, stddev=stddev, seed=self._seed)
    # noise = tf.math.minimum(noise, 0.5)
    # noise = tf.math.maximum(noise, 0)
    # image += noise
    # image = tf.clip_by_value(image, 0.0, 1.0)

    if self._aug_rand_aspect:
      aspect = preprocessing_ops.rand_scale(0.75)
      nh = tf.cast(w / aspect, dtype=tf.int32)
      nw = tf.cast(w, dtype=tf.int32)
      image = tf.image.resize(image, size=(nw, nh))
    image = tf.image.random_flip_left_right(image, seed=self._seed)

    image = tf.image.resize_with_pad(
        image,
        target_width=self._output_size[0],
        target_height=self._output_size[1])

    if self._aug_rand_rotate:
      deg = tf.random.uniform([],
                              minval=-7,
                              maxval=7,
                              seed=self._seed,
                              dtype=tf.float32)
      deg = deg * 3.14 / 360.
      deg.set_shape(())
      image = tfa.image.rotate(image, deg, interpolation='BILINEAR')

    if self._aug_rand_zoom:
      scale = tf.random.uniform([],
                                minval=self._scale[0],
                                maxval=self._scale[1],
                                seed=self._seed,
                                dtype=tf.int32)
      if scale > self._output_size[0]:
        image = tf.image.resize_with_crop_or_pad(
            image, target_height=scale, target_width=scale)
      else:
        image = tf.image.random_crop(image, (scale, scale, 3))

    image = tf.image.resize(image, (self._output_size[0], self._output_size[1]))

    label = decoded_tensors['image/class/label']
    return image, label

  def _parse_eval_data(self, decoded_tensors):
    """Generates images and labels that are usable for model evaluation.
    Args:
      decoded_tensors: a dict of Tensors produced by the decoder.
    Returns:
      images: the image tensor.
      labels: a dict of Tensors that contains labels.
    """
    image = tf.io.decode_image(decoded_tensors['image/encoded'])
    image.set_shape((None, None, 3))
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(
        image,
        target_width=self._output_size[0],
        target_height=self._output_size[1])  # Final Output Shape
    image = image / 255.  # Normalize
    #label = tf.one_hot(decoded_tensors['image/class/label'], self._num_classes)
    label = decoded_tensors['image/class/label']
    return image, label
