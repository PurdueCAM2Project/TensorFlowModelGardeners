import cv2
import datetime
import colorsys
import numpy as np
import time

import threading as t
from queue import Queue

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

from yolo.utils.run_utils import support_windows
from yolo.utils.demos.coco import draw_box
from yolo.utils.demos.coco import get_draw_fn
from yolo.utils.demos.coco import gen_colors
from yolo.utils.demos.coco import get_coco_names
from yolo.utils.demos.coco import int_scale_boxes
from yolo.utils.demos import utils
# from utils.demos import utils
from yolo.utils.run_utils import prep_gpu
from yolo.configs import yolo as exp_cfg
from yolo.tasks.yolo import YoloTask


class FastVideo(object):

  def __init__(self,
               reader,
               processor,
               drawer,
               writer,
               max_batch=None,
               wait_time=None,
               scale_que=10):

    support_windows()

    self._reader = reader()
    self._preprocessor = processor()
    self._drawer = drawer()
    self._writer = writer()

    self._batch_size = max_batch
    # self._wait_time = wait_time

    self._load_que = Queue(self._batch_size * scale_que)
    self._load_lock = t.Lock()
    self._load_empty = t.Condition(self._load_lock)
    self._load_full = t.Condition(self._load_lock)

    self._display_que = Queue(1 * scale_que)
    self._display_lock = t.Lock()
    self._display_empty = t.Condition(self._load_lock)
    self._display_full = t.Condition(self._load_lock)

    self._load_thread = t.Thread(target=self.read, args=())
    self._display_thread = t.Thread(target=self.display, args=())
    self._running = False
    return

  def read(self, lock=None):
    # not_full = lambda: not self._load_que.full()
    while (self._reader.running and self._running):
      with self._load_full:
        # self._load_full.wait_for(not_full)
        self._load_full.wait()
      suc, image = self._reader.read()
      if suc:
        with self._load_empty:
          # put in a frame notify all waiting FNs
          self._load_que.put(image)
          self._load_empty.notify_all()
      # time.sleep(self._wait_time)

    self._running = False
    return

  def display(self):
    # not_full = lambda: not self._load_que.full()
    while (self._writer.running and self._running):
      with self._display_empty:
        # self._load_full.wait_for(not_full)
        self._display_empty.wait()
      suc, image = self._reader.read()
      if suc:
        with self._load_empty:
          # put in a frame notify all waiting FNs
          self._load_que.put(image)
          self._load_empty.notify_all()
      # time.sleep(self._wait_time)
    self._running = False
    return

  def run(self):
    self._running = True
    self._load_thread.start()
    self._display_thread.start()

    self._load_thread.join()
    self._display_thread.join()
    self._running = False
    return


if __name__ == '__main__':
  draw_fn = utils.DrawBoxes(
      classes=80, labels=self._labels, display_names=print_conf, thickness=1)
#   config = [os.path.abspath('yolo/configs/experiments/yolov4-eval.yaml')]
#   model_dir = "" #os.path.abspath("../checkpoints/yolo_dt8_norm_iou")

#   task, model, params = run.load_model(experiment='yolo_custom', config_path=config, model_dir=model_dir)

#   cap = FastVideo(
#       "../videos/nyc.mp4",
#       model=model,
#       process_width=416,
#       process_height=416,
#       preprocess_with_gpu=True,
#       print_conf=True,
#       max_batch=5,
#       disp_h=416,
#       scale_que=1,
#       wait_time='dynamic')
#   cap.run()
