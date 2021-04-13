import threading as t
from queue import Queue
import time
import cv2
import abc


class Writer(object):

  @abc.abstractclassmethod
  def run(self):
    ...

  @abc.abstractclassmethod
  def close(self):
    ...

  @abc.abstractclassmethod
  def write(self):
    ...

  @abc.abstractclassmethod
  @property
  def running(self):
    ...


class DisplayWriter(Writer):

  def __init__():
    self._display_fps = 0
    return

  def run(self):
    return

  def close(self):
    return

  def write(self, image):
    cv2.imshow('frame', image)
    # compute the display fps
    l += 1
    if time.time() - start - tick >= 1:
      tick += 1
      # store the fps to diplayed to the user
      self._display_fps = l
      l = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    return self.display_fps

  @property
  def running(self):
    return
