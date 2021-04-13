def prep_gpu(distribution=None):
  import tensorflow as tf
  try:
    from tensorflow.config import list_physical_devices, list_logical_devices
  except ImportError:
    from tensorflow.config.experimental import list_physical_devices, list_logical_devices
  print(f"\n!--PREPPING GPU--! ")
  if distribution is None:
    gpus = list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = list_logical_devices('GPU')
          print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        raise
  return


def expand_gpu(distribution=None):
  import tensorflow as tf
  physical_devices = tf.config.list_physical_devices('GPU')
  try:
    tf.config.set_logical_device_configuration(physical_devices[0], [
        tf.config.LogicalDeviceConfiguration(memory_limit=5000),
        tf.config.LogicalDeviceConfiguration(memory_limit=5000)
    ])

    logical_devices = tf.config.list_logical_devices('GPU')
    print(logical_devices)
  except:
    pass
  return


def support_windows():
  import platform
  if platform.system().lower() == 'windows':
    from ctypes import windll, c_int, byref
    stdout_handle = windll.kernel32.GetStdHandle(c_int(-11))
    mode = c_int(0)
    windll.kernel32.GetConsoleMode(c_int(stdout_handle), byref(mode))
    mode = c_int(mode.value | 4)
    windll.kernel32.SetConsoleMode(c_int(stdout_handle), mode)
  return


def change_policy(policy):
  from tensorflow.keras.mixed_precision import experimental as mixed_precision
  mixed_precision.set_policy(policy)
  return
