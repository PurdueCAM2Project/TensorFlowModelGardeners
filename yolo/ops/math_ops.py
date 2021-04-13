import tensorflow as tf
import tensorflow.keras.backend as K


def rm_nan_inf(x, val=0.0):
  cond = tf.math.logical_or(tf.math.is_nan(x), tf.math.is_inf(x))
  val = tf.cast(val, dtype=x.dtype)
  # safe_x = tf.where(cond, tf.cast(0.0, dtype=x.dtype) , x)
  x = tf.where(cond, val, x)
  return x


def rm_nan(x, val=0.0):
  cond = tf.math.is_nan(x)
  val = tf.cast(val, dtype=x.dtype)
  # safe_x = tf.where(cond, tf.cast(0.0, dtype=x.dtype) , x)
  x = tf.where(cond, val, x)
  return x


def divide_no_nan(a, b):
  zero = tf.cast(0.0, b.dtype)
  return tf.where(b == zero, zero, a / b)


def mul_no_nan(x, y):
  return tf.where(x == 0, tf.cast(0, x.dtype), x * y)
