from yolo.utils.run_utils import prep_gpu, expand_gpu

expand_gpu()
try:
  prep_gpu()
except BaseException:
  print("GPUs ready")
