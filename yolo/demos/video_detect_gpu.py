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
import yolo.demos.three_servers.video_server as video_t


class FastVideo(object):
  """
    program for faster object detection in TensorFlow. Algorithm was tested on RTX 2070 Super with Intel Core i5-4500k CPU. The first 2-3 seconds of
    the video diplayed may be too fast or too slow as the algorithem is designed to converge to the most optimal FPS based on the input source. This also
    may only be noticeable on webcam input. Think of it as configuration time, or the set up time. This algorithem requires GPU to work correctly.

    using a default model:
      cap = BufferVideo("test.mp4", model = "regular", process_width=416, process_height=416)
      cap.run()

    using a non-default model:
      prep_gpu()
      with tf.device("/GPU:0"):
        model = build_model(name = "regular", w = 416, h = 416)
      cap = BufferVideo(0, model = model, process_width=416, process_height=416)
      cap.run()


    Args:
        file_name: a string for the video file you would like tensorflow to process, or an integer for the webcam youd would like to use
        model: a string in {"regular", "tiny", "spp"} for the yolo model you would like to use. if None, regular will be used as defualt.
          if a non standard model is used, you may also pass in a regular keras model, but the implemented error checking is not garunteed to function
        preprocess_function: python function to preprocess images input into the model, if none out default preprocessing function is used
        process_width: the integer width that the model should reshape the input to prior to input, 416 by default
        process_height: the integer height that the model should reshape the input to prior to input, 416 by default
        classes: integer number of possible classes that could be predicted, if none is provided, 80 will be used as the default from the COCO dataset
        labels: a List[string] of the name of the class associated with each neural net prediction, if nothing is provided, the default COCO labels will be used
        preprocess_with_gpu: boolean for wether you would like to do the pre processing on the gpu or cpu. some devices lack the memory capability to
          run 2 piplines on one graphics device, so the default is set to the CPU, but the program could work faster by using the GPU,
          by about 2 to 3 fps at high resolutions (1080p or higher)
        gpu_device: string for the device you would like to use to run the model, if the model you pass in is not standard make sure you prep
          the model on the same device that you pass in, by default /GPU:0
        preprocess_gpu: the gpu device you would like to use to preprocess the image if you have multiple. by default use the first /GPU:0

    Raises:
        IOError: the video file you would like to use is not found
        Exception: the model you input is a string and is not in the list of supported models

    """

  def __init__(self,
               file_name,
               model=None,
               preprocess_function=None,
               process_width=416,
               process_height=416,
               disp_h=720,
               classes=80,
               labels=None,
               print_conf=True,
               max_batch=None,
               wait_time=None,
               preprocess_with_gpu=False,
               scale_que=1,
               gpu_device='/GPU:0',
               preprocess_gpu='/GPU:0'):

    file_name = 0 if file_name is None else file_name
    try:
      file_name = int(file_name)
    except BaseException:
      print(file_name)
    self._cap = cv2.VideoCapture(file_name)
    if not self._cap.isOpened():
      raise IOError('video file was not found')

    # support for ANSI cahracters in windows
    support_windows()

    self._file = file_name
    self._fps = 120000000

    self._gpu_device = gpu_device
    if preprocess_with_gpu:
      self._pre_process_device = preprocess_gpu
    else:
      self._pre_process_device = '/CPU:0'

    # self._cap.set(3, int(960))
    # self._cap.set(4, int(720))
    # self._cap.set(cv2.CAP_PROP_FPS, int(30))
    print(self._cap.get(3), self._cap.get(4))

    self._preprocess_function = preprocess_function
    self._height = int(self._cap.get(4)) if disp_h is None else disp_h
    self._og_height = int(self._cap.get(4))
    self._width = int(self._cap.get(3) * (self._height / self._og_height))
    self._classes = classes
    self._p_width = process_width
    self._p_height = process_height
    self._model = model

    # fast but as close to one 2 one as possible
    if max_batch is None:
      if file_name == 0:
        self._batch_size = 5  # 40 fps more conistent frame to frame
      else:
        # faster but more potential for delay from input to output
        if tf.keras.mixed_precision.experimental.global_policy(
        ).name == 'mixed_float16' or tf.keras.mixed_precision.experimental.global_policy(
        ).name == 'float16':
          # self._batch_size = 9 # 45 fps faster but less frame to frame consistent, it will remain consistant, but there is opertunity for more frames to be loaded than
          self._batch_size = 5
        else:
          self._batch_size = 3

        if process_width > 416 or process_height > 416:
          self._batch_size = 3
    else:
      self._batch_size = max_batch

    self._colors = gen_colors(self._classes)

    if labels is None and classes == 80:
      self._labels = get_coco_names(
          path='yolo/dataloaders/dataset_specs/coco.names')
    elif labels is None and classes == 91:
      self._labels = get_coco_names(
          path='yolo/dataloaders/dataset_specs/coco-91.names')
    else:
      self._labels = labels

    self._draw_fn = utils.DrawBoxes(
        classes=classes,
        labels=self._labels,
        display_names=print_conf,
        thickness=1)
    #get_draw_fn(self._colors, self._labels, print_conf)

    self._load_que = Queue(self._batch_size * scale_que)
    self._display_que = Queue(1 * scale_que)
    self._running = True

    self._dynamic_wt = (wait_time == 'dynamic' or wait_time is None)
    if not self._dynamic_wt:
      self._wait_time = utils.get_wait_time(wait_time, max_batch)
    else:
      self._wait_time = 0.0001

    self._read_fps = 1
    self._prev_display_fps = 1
    self._display_fps = 1
    self._latency = -1
    self._batch_proc = 1
    self._frames = 1
    self._obj_detected = -1

    self._display = video_t.DisplayThread()
    return

  def _preprocess(self, image):
    if type(self._file) == int:
      image = tf.image.flip_left_right(image)
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255
    return image

  def read(self, lock=None):
    """ read video frames in a thread """
    start = time.time()
    l = 0
    tick = 0
    timeout = 0
    process_device = self._pre_process_device
    if self._preprocess_function is None:
      preprocess = self._preprocess
    else:
      preprocess = self._preprocess_function
    try:
      while (self._cap.isOpened() and self._running):
        # wait for que to get some space, if waiting too long, error occured and return
        while self._load_que.full() and self._running:
          timeout += 1
          time.sleep(self._wait_time * 5)
          if not self._running:
            return
          if timeout >= self._fps:
            print(
                '[EXIT] an error has occured, frames are not being pulled from que'
            )
            return
        if not self._running:
          return

        # with the CPU load and process an image
        timeout = 0
        success, image = self._cap.read()
        if not success:
          time.sleep(self._wait_time)
          continue
        e = datetime.datetime.now()
        image = tf.cast(tf.convert_to_tensor(image), tf.float16)
        self._load_que.put(image)
        f = datetime.datetime.now()

        # compute the reading FPS
        l += 1
        if time.time() - start - tick >= 1:
          tick += 1
          # store the reading FPS so it can be printed clearly
          self._read_fps = l
          l = 0
        # sleep for default 0.01 seconds, to allow other functions the time to catch up or keep pace
        time.sleep(self._wait_time)
    except ValueError:
      self._running = False
      raise
    except Exception as e:
      self._running = False
      raise e
    self._running = False
    return

  def display(self):
    """
        display the processed images in a thread, for models expected output format see tf.image.combined_non_max_suppression
        https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
        """
    # init the starting variables to calculate FPS
    self._display.start()
    try:
      start = time.time()
      l = 0
      tick = 0
      while (self._cap.isOpened() and self._running):
        # if the display que is empty, nothing has been processed, just wait for an image to arrive
        # do not timeout ever as too long does not garuntee an error
        while self._display_que.empty() and self._running:
          time.sleep(self._wait_time)
          if not self._running:
            return
        if not self._running:
          return

        # get the images, the predictions placed on the que via the run function (the model)
        image, pred = self._display_que.get()
        image = self._draw_fn(image, pred)
        # del image
        # del pred

        # there is potential for the images to be processed in batches, so for each image in the batch draw the boxes and the predictions and the confidence
        for im in iter(image):
          while not self._display.put(im):
            time.sleep(self._wait_time)
          self._display_fps = self._display.fps

    except ValueError:
      self._running = False
      self._display.close()
      raise
    except Exception as e:
      #print ("display", e)
      self._running = False
      self._display.close()
      # time.sleep(10)
      raise e
    self._running = False
    self._display.close()
    return

  def run(self):
    # init the model
    model = self._model
    gpu_device = self._gpu_device

    # print processing information
    print(f"capture (width, height): ({self._width},{self._height})")
    print(f"Yolo Possible classes: {self._classes}")

    if self._preprocess_function is None:
      preprocess = self._preprocess
    else:
      preprocess = self._preprocess_function

    # get one frame and put it inthe process que to get the process started
    if self._cap.isOpened():
      success, image = self._cap.read()
      e = datetime.datetime.now()
      self._load_que.put(image)
      f = datetime.datetime.now()
      if hasattr(model, 'predict'):
        predfunc = model.predict
        print('using pred function')
      else:
        predfunc = model
        print('using call function')
    else:
      return

    # start a thread to load frames
    load_thread = t.Thread(target=self.read, args=())
    load_thread.start()

    # start a thread to display frames
    display_thread = t.Thread(target=self.display, args=())
    display_thread.start()

    try:
      # while the captue is open start processing frames
      while (self._cap.isOpened()):
        # in case the load que has many frames in it, load one batch
        proc = []
        for i in range(self._batch_size):
          if self._load_que.empty():
            break
          value = self._load_que.get()
          proc.append(value)

        # if the que was empty the model is ahead, and take abreak to let the other threads catch up
        if len(proc) == 0:
          time.sleep(self._wait_time)
          continue

        # log time and process the batch loaded in the for loop above
        a = datetime.datetime.now()
        with tf.device(self._gpu_device):
          image = tf.stack(proc, axis=0)
          image = preprocess(image)
          pimage = tf.image.resize(image, (self._p_width, self._p_height))
          pred = predfunc(pimage)
          if image.shape[1] != self._height:
            image = tf.image.resize(pimage, (self._height, self._width))
        b = datetime.datetime.now()

        # computation latency to see how much delay between input and output
        if self._frames >= 1000:
          self._frames = 0
          self._latency = -1

        # compute the latency
        if self._latency != -1:
          self._latency += (b - a)
        else:
          self._latency = (b - a)

        # compute the number of frames processed, used to compute the moving average of latency
        self._frames += image.shape[0]
        self._batch_proc = image.shape[0]
        timeout = 0

        # if the display que is full, do not put anything, just wait for a ms
        while self._display_que.full() and not self._running:
          time.sleep(self._wait_time)

        # put processed frames on the display que
        self._display_que.put((image, pred))

        if self._dynamic_wt:
          if self._prev_display_fps > self._display_fps:
            self._wait_time = self._wait_time + 0.01 * self._wait_time
            self._prev_display_fps = self._display_fps * 0.1 + 0.9 * self._prev_display_fps
          else:
            self._wait_time = self._wait_time - 0.02 * self._wait_time
            self._prev_display_fps = self._display_fps * 0.1 + 0.9 * self._prev_display_fps

        # print everything
        self.print_opt()
        if not self._running:
          raise

      # join the laoding thread and diplay thread
      load_thread.join()
      display_thread.join()

      # close the video capture and destroy all open windows
      self._cap.release()
      cv2.destroyAllWindows()
    except KeyboardInterrupt:
      # arbitrary keyboard input
      self.print_opt()
      print('\n\n\n\n\n::: Video File Stopped -> KeyBoard Interrupt :::')
      self._running = False

      # join the laoding thread and diplay thread
      load_thread.join()
      display_thread.join()

      # close the video capture and destroy all open windows
      self._cap.release()
      cv2.destroyAllWindows()
    except Exception as e:
      # arbitrary keyboard input
      self.print_opt()
      print(f"\n\n\n\n\n::: Video File Complete ::: or error  -> -> ->{e}")
      self._running = False
      time.sleep(5)

      # join the laoding thread and diplay thread
      load_thread.join()
      display_thread.join()

      # close the video capture and destroy all open windows
      self._cap.release()
      cv2.destroyAllWindows()
    return

  def print_opt(self):
    # make everything print pretty using ANSI
    print(
        f"                                \rlatency:, \033[1;32;40m{self._latency/self._frames * 1000} \033[0m ms",
        end='\n')
    print(
        '                                 \rread fps: \033[1;34;40m%d\033[0m ' %
        (self._read_fps),
        end='\n')
    print(
        '                                 \rdisplay fps: \033[1;34;40m%d\033[0m'
        % (self._display_fps),
        end='\n')
    print(
        '                                 \rbatch processed: \033[1;37;40m%d\033[0m'
        % (self._batch_proc),
        end='\n')
    print(
        '                                 \rwait time: \033[1;37;40m%0.15f\033[0m'
        % (self._wait_time),
        end='\n')
    print(
        '                                 \rfps avg: \033[1;37;40m%0.5f\033[0m'
        % (self._prev_display_fps),
        end='\n')
    print(
        '                                 \rmemuse: \033[1;37;40m%0.5f\033[0m GB'
        % (tf.config.experimental.get_memory_usage(self._gpu_device) / 1e9),
        end='\n')

    print('\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F', end='\n')
    return


def func(inputs):
  boxes = inputs['bbox']
  classifs = inputs['confidence']
  nms = tf.image.combined_non_max_suppression(
      tf.expand_dims(boxes, axis=2), classifs, 200, 200, 0.5, 0.5)
  return {
      'bbox': nms.nmsed_boxes,
      'classes': nms.nmsed_classes,
      'confidence': nms.nmsed_scores,
  }


def get_model(model):

  @tf.function
  def ret(image):
    outs = model.call(image, training=False)

    with tf.device('cpu:0'):
      boxes = tf.cast(outs['bbox'], tf.float32)
      classifs = tf.cast(outs['confidence'], tf.float32)
      nms = tf.image.combined_non_max_suppression(
          tf.expand_dims(boxes, axis=2), classifs, 200, 200, 0.5, 0.5)

      fouts = {
          'bbox': tf.cast(nms.nmsed_boxes, tf.float32),
          'classes': tf.cast(nms.nmsed_classes, tf.float32),
          'confidence': tf.cast(nms.nmsed_scores, tf.float32),
      }
    return fouts

  return ret


# import yolo.utils.export.tensor_rt as trt
# model(tf.ones((1, 416, 416, 3), dtype = tf.float32))
# name = "saved_models/v3/tflite-tiny-no-nms"
# model.save(name, overwrite=True)
# new_name = f"{name}_tensorrt"
# model = trt.TensorRT(saved_model=name, save_new_path=new_name, max_workspace_size_bytes=4000000000, max_batch_size=5)#, precision_mode="INT8", use_calibration=True)
# model.convertModel()
# model.compile()
# model.summary()
# model.set_postprocessor_fn(func)

# #name = "saved_models/v4/tflite-regualr-no-nms"
# name = "saved_models/v4/tflite-tiny-no-nms"
# # name = "saved_models/v3/tflite-regular-no-nms"
# # name = "saved_models/v3/tflite-tiny-no-nms"
# new_name = f"{name}_tensorrt"
# model = trt.TensorRT(saved_model=new_name, save_new_path=new_name, max_workspace_size_bytes=4000000000, max_batch_size=5)
# model.compile()
# model.summary()
# model.set_postprocessor_fn(func)

# if __name__ == '__main__':
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
