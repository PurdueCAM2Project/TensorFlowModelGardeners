from yolo.utils.run_utils import prep_gpu
from yolo.configs import yolo as exp_cfg
from yolo.tasks.yolo import YoloTask
import yolo.utils.export.tensor_rt as trt
import tensorflow as tf 

prep_gpu()

from tensorflow.keras.mixed_precision import experimental as mixed_precision
mixed_precision.set_policy("mixed_float16")
# mixed_precision.set_policy("float32")

config = exp_cfg.YoloTask(model=exp_cfg.Yolo(base='v4', 
                        min_level=3, 
                        norm_activation = exp_cfg.common.NormActivation(activation="mish"), 
                        #norm_activation = exp_cfg.common.NormActivation(activation="leaky"), 
                        #_boxes = ['(10, 14)', '(23, 27)', '(37, 58)', '(81, 82)', '(135, 169)', '(344, 319)'],
                        #_boxes = ["(10, 13)", "(16, 30)", "(33, 23)","(30, 61)", "(62, 45)", "(59, 119)","(116, 90)", "(156, 198)", "(373, 326)"],
                        _boxes = ['(12, 16)', '(19, 36)', '(40, 28)', '(36, 75)','(76, 55)', '(72, 146)', '(142, 110)', '(192, 243)','(459, 401)'],
                        filter = exp_cfg.YoloLossLayer(use_nms=False)),
                        load_darknet_weights = False,
                        darknet_load_decoder = False) 

# config = exp_cfg.YoloTask(model=exp_cfg.Yolo(base='v3', 
#                     min_level=3, 
#                     #norm_activation = exp_cfg.common.NormActivation(activation="mish"), 
#                     norm_activation = exp_cfg.common.NormActivation(activation="leaky"), 
#                     #_boxes = ['(10, 14)', '(23, 27)', '(37, 58)', '(81, 82)', '(135, 169)', '(344, 319)'],
#                     #_boxes = ["(10, 13)", "(16, 30)", "(33, 23)","(30, 61)", "(62, 45)", "(59, 119)","(116, 90)", "(156, 198)", "(373, 326)"],
#                     _boxes = ['(12, 16)', '(19, 36)', '(40, 28)', '(36, 75)','(76, 55)', '(72, 146)', '(142, 110)', '(192, 243)','(459, 401)'],
#                     filter = exp_cfg.YoloLossLayer(use_nms=False)
#                     ))   
task = YoloTask(config)
model = task.build_model()
task.initialize(model)


model(tf.ones((1, 416, 416, 3), tf.float32))