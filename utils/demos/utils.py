import tensorflow as tf
import tensorflow.keras.backend as K
import socket
import struct
from typing import Callable
import numpy as np
import colorsys
import cv2
from contextlib import closing
import multiprocessing as mp
import os


def get_device(policy):
  if not isinstance(policy, str):
    return policy
  elif policy == "mirrored":
    strat = tf.distribute.MirroredStrategy()
    return strat.scope()
  elif policy == "oneDevice":
    strat = tf.distribute.OneDeviceStrategy()
    return strat.scope()
  elif policy == "tpu":
    strat = tf.distribute.TPUStrategy()
    return strat.scope()
  elif "GPU" in policy or "gpu" in policy:
    return tf.device(policy)
  return tf.device("/CPU:0")


def get_wait_time(wait_time, batch_size):
  if wait_time is None:
    return 0.001 * batch_size
  return wait_time


def set_policy(policy_name):
  if policy_name is None:
    return tf.float32
  from tensorflow.keras.mixed_precision import experimental as mixed_precision
  policy = mixed_precision.Policy(policy_name)
  mixed_precision.set_policy(policy)
  dtype = policy.compute_dtype
  return dtype


def get_run_fn(model, signature="serving_default") -> Callable:
  if ("saved_model" in str(type(model))):
    try:
      return model.signatures[signature]
    except BaseException:
      print("WARNING: signature not found in model")
      return model
  elif (hasattr(model, "predict")):
    return model.predict
  else:
    return model


def udp_socket(address, port, server=False):
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  if server:
    sock.bind((address, port))
  return sock


def tcp_socket(address, port, server=False):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  if server:
    sock.bind((address, port))
    sock.listen(1)
  return sock


def clear_buffer(s):
  while True:
    seg, addr = s.recvfrom(2**16)
    print(seg[0])
    if struct.unpack("B", seg[0:1])[0] == 1:
      print("finish emptying buffer")
      break


def gen_colors(max_classes):
  hue = np.linspace(start=0, stop=1, num=max_classes)
  np.random.shuffle(hue)
  colors = []
  for val in hue:
    colors.append(colorsys.hsv_to_rgb(val, 0.75, 1.0))
  return colors


def draw_box(image, boxes, classes, conf, draw_fn):
  i = 0
  for i in range(boxes.shape[0]):
    if draw_fn(image, boxes[i], classes[i], conf[i]):
      i += 1
    else:
      return i
  return i


def get_draw_fn(self, colors, label_names, display_name):

  def draw_box_name(image, box, classes, conf):
    if box[3] == 0:
      return False
    cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors[classes], 1)
    cv2.putText(image, "%s, %0.3f" % (label_names[classes], conf),
                (box[0], box[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                colors[classes], 1)
    return True

  def draw_box(image, box, classes, conf):
    if box[3] == 0:
      return False
    cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors[classes], 1)
    return True

  if display_name or label_names is None:
    return draw_box_name
  else:
    return draw_box


def get_draw_fn(colors, label_names, display_name):

  def draw_box_name(image, box, classes, conf):
    if box[3] == 0:
      return False
    cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors[classes], 1)
    cv2.putText(image, "%s, %0.3f" % (label_names[classes], conf),
                (box[0], box[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                colors[classes], 1)
    return True

  def draw_box(image, box, classes, conf):
    if box[3] == 0:
      return False
    cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors[classes], 1)
    return True

  if display_name or label_names is None:
    return draw_box_name
  else:
    return draw_box


def int_scale_boxes(boxes, classes, width, height):
  boxes = K.stack([
      tf.cast(boxes[..., 1] * width, dtype=tf.int32),
      tf.cast(boxes[..., 3] * width, dtype=tf.int32),
      tf.cast(boxes[..., 0] * height, dtype=tf.int32),
      tf.cast(boxes[..., 2] * height, dtype=tf.int32)
  ],
                  axis=-1)
  classes = tf.cast(classes, dtype=tf.int32)
  return boxes, classes

def shared_to_numpy(shared_arr, dtype, shape):
    """Get a NumPy array from a shared memory buffer, with a given dtype and shape.
    No copy is involved, the array reflects the underlying shared buffer."""
    return np.frombuffer(shared_arr, dtype=dtype).reshape(shape)

def create_shared_array(dtype, shape):
    """Create a new shared array. Return the shared array pointer, and a NumPy array view to it.
    Note that the buffer values are not initialized.
    """
    dtype = np.dtype(dtype)
    # Get a ctype type from the NumPy dtype.
    cdtype = np.ctypeslib.as_ctypes_type(dtype)
    # Create the RawArray instance.
    shared_arr = mp.RawArray(cdtype, sum(shape))
    # Get a NumPy array view.
    arr = shared_to_numpy(shared_arr, dtype, shape)
    return shared_arr, arr

class DrawBoxes(object):
  def __init__(self, classes=80, labels=None, display_names=True, thickness=2):

    self._classes = classes
    self._colors = gen_colors(classes)
    self._labels = labels
    self._display_names = display_names
    self._thickness = thickness
    self._draw_fn = self._get_draw_fn(self._colors, self._labels,
                                      self._display_names)
    return

  def _get_draw_fn(self, colors, label_names, display_name):

    def draw_box_name(image, box, classes, conf):
      if box[3] == 0:
        return False
      cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors[classes],
                    self._thickness)

      if conf is not None:
        cv2.putText(image, "%s, %0.3f" % (label_names[classes], conf),
                    (box[0], box[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[classes], self._thickness)
      else:
        cv2.putText(image, "%s, %0.3f" % (label_names[classes]),
                    (box[0], box[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[classes], self._thickness)
      return True

    def draw_box(image, box, classes, conf):
      if box[3] == 0:
        return False
      cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors[classes],1)
      return True

    if display_name and label_names is not None:
      return draw_box_name
    else:
      return draw_box

  # def _draw(self, image, boxes, classes, conf):
  #   i = 0
  #   for i in range(boxes.shape[0]):
  #     if self._draw_fn(image, boxes[i], classes[i], conf[i]):
  #       i += 1
  #     else:
  #       return i
  #   return i

  def _init_proc(shared_arr_, boxes_, classes_, conf_):
    # The shared array pointer is a global variable so that it can be accessed by the
    # child processes. It is a tuple (pointer, dtype, shape).
    global shared_arr, boxes, classes, conf
    shared_arr = shared_arr_
    boxes = boxes_
    classifcs = classes_ 
    confens = conf_

  def _draw_parallel(index):
    box = boxes[index]
    classes = classifcs[index]
    conf = confens[index]
    image = shared_to_numpy(*shared_arr)
    image = self._draw_fn(image, box, classes, conf)
    return 

  def _draw(self, image, boxes, classes, conf):
    n_processes = os.cpu_count()
    index_map = list(range(boxes.shape[0])) 
    shared_arr, arr = create_shared_array(image.dtype, image.shape)
    arr.flat[:] = image.flat[:]
    print(f"multiprocess draw {n_processes}")
    with closing(mp.Pool(
                  n_processes, 
                  initializer=self._init_proc, 
                  initargs=((shared_arr, image.dtype, image.shape), boxes, classes, conf, ))) as p:

      p.map(parallel_function, index_map)
    return n_processes

  def __call__(self, image, results):
    """ expectoed format = {bbox: , classes: , "confidence": }"""

    boxes = results["bbox"]
    classes = results["classes"]

    try:
      conf = results["confidence"]
    except:
      conf = results["classes"]

    if not isinstance(image, list):
      ndims = len(image.shape)
      if ndims == 4:
        boxes, classes = int_scale_boxes(boxes, classes, image.shape[2],
                                         image.shape[1])
      elif ndims == 3:
        boxes, classes = int_scale_boxes(boxes, classes, image.shape[1],
                                         image.shape[0])
    else:
      img = image[0]
      boxes, classes = int_scale_boxes(boxes, classes, img.shape[1],
                                       img.shape[0])

    if hasattr(image, "numpy"):
      image = image.numpy()
    if hasattr(boxes, "numpy"):
      boxes = boxes.numpy()
    if hasattr(classes, "numpy"):
      classes = classes.numpy()
    if not isinstance(conf, type(None)) and hasattr(conf, "numpy"):
      conf = conf.numpy()

    if not isinstance(image, list):
      if ndims == 4:
        images = []
        for i, im in enumerate(image):
          self._draw(im, boxes[i], classes[i], conf[i])
          images.append(im)
        image = np.stack(images, axis=0)
      elif ndims == 3:
        if len(boxes.shape) == 2:
          self._draw(image, boxes, classes, conf)
        else:
          self._draw(image, boxes[0], classes[0], conf[0])
    else:
      images = []
      for i, im in enumerate(image):
        if hasattr(im, "numpy"):
          im = im.numpy()
        self._draw(im, boxes[i], classes[i], conf[i])
        images.append(im)
      image = np.stack(images, axis=0)
    return image


def int_scale_boxes(boxes, classes, width, height):
  boxes = K.stack([
      tf.cast(boxes[..., 1] * width, dtype=tf.int32),
      tf.cast(boxes[..., 3] * width, dtype=tf.int32),
      tf.cast(boxes[..., 0] * height, dtype=tf.int32),
      tf.cast(boxes[..., 2] * height, dtype=tf.int32)
  ],
                  axis=-1)
  classes = tf.cast(classes, dtype=tf.int32)
  return boxes, classes
