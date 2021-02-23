"""
import importlib as _imp
import sys as _sys

_dict = {
    'DarkNet53': 'yolo.modeling.DarkNet',
    'CSPDarkNet53': 'yolo.modeling.CSPDarkNet',
    'Yolov3': 'yolo.modeling.Yolov3',
    'Yolov4': 'yolo.modeling.Yolov4',
}

__all__ = list(_dict)

if _sys.version_info < (3, 7):
  from types import ModuleType as _ModuleType

  class _Module(_ModuleType):

    def __getattr__(self, name):
      try:
        module = _dict[name]
        module = _imp.import_module(module)
        return getattr(module, name)
      except KeyError:
        raise AttributeError

    def __dir__(self):
      return [*_sys.modules[__name__].__dict__, *_dict]

  _sys.modules[__name__].__class__ = _Module
else:

  def __getattr__(name):
    try:
      module = _dict[name]
      module = _imp.import_module(module)
      return getattr(module, name)
    except KeyError:
      raise AttributeError

  def __dir__():
    return [*_sys.modules[__name__].__dict__, *_dict]
"""
