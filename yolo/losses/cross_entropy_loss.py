import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K


def smooth_labels(y_true, y_pred, label_smoothing):
  num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
  return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)


def ce_loss(y_true, y_pred, label_smoothing):
  y_true = smooth_labels(y_true, y_pred, label_smoothing)\

  loss = -tf.math.xlogy(y_true, y_pred + tf.keras.backend.epsilon())
  loss = tf.where(tf.math.is_nan(loss), 1.0, loss)
  return tf.math.reduce_sum(loss, axis=-1)


# def ce_loss(y_true, y_pred, label_smoothing):
#   y_true = smooth_labels(y_true, y_pred, label_smoothing)\
#   loss = -y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())
#   return tf.math.reduce_sum(loss, axis = -1)


class CrossEntropyLoss(tf.keras.losses.Loss):
  """Wraps a loss function in the `Loss` class."""

  def __init__(self,
               label_smoothing=0.0,
               reduction=tf.keras.losses.Reduction.SUM,
               name=None,
               **kwargs):
    """Initializes `LossFunctionWrapper` class.
    Args:
      fn: The loss function to wrap, with signature `fn(y_true, y_pred,
        **kwargs)`.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: (Optional) name for the loss.
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
    super(CrossEntropyLoss, self).__init__(reduction=reduction, name=name)
    self._label_smoothing = label_smoothing

  def call(self, y_true, y_pred):
    """Invokes the `LossFunctionWrapper` instance.
    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
    Returns:
      Loss values per sample.
    """
    y_true = smooth_labels(y_true, y_pred, self._label_smoothing)
    tf.print(y_true)
    return -tf.math.xlogy(y_true, y_pred)


if __name__ == '__main__':
  loss = CrossEntropyLoss(label_smoothing=0.1)

  y = tf.ones((1, 2))
  x = tf.ones((1, 2))

  print(loss(x, y))
