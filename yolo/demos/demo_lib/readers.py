import threading as t
from queue import Queue
import time
import cv2
import abc


class Reader(object):

  @abc.abstractclassmethod
  def read(self):
    ...

  @abc.abstractclassmethod
  def run(self):
    ...

  @abc.abstractclassmethod
  def close(self):
    ...

  @abc.abstractclassmethod
  @property
  def running(self):
    ...


class VideoReader(Reader):

  def __init__(self, file=0, output_reslution=720):

    self._file = file
    self._cap = cv2.VideoCapture(file)
    if not self._cap.isOpened():
      raise IOError("video file was not found")

    print(f"frame dims: {self._cap.get(3)}, {self._cap.get(4)}")
    self._original_height = int(self._cap.get(4))
    self._height = int(self._cap.get(4)) if disp_h is None else disp_h
    self._width = int(self._cap.get(3) * (self._height / self._original_height))
    self._id = 0
    return

  def read(self):
    suc, image = self._cap.read()
    if suc:
      image = cv2.resize(
          image, (self._width, self._height), interpolation=cv2.INTER_AREA)
    return suc, {"image": image, "id": self._id}

  @property
  def running(self):
    return self._cap.isOpened()

  def start(self):
    return

  def close(self):
    return


# TODO: File List Reader
