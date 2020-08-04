import tensorflow as tf
import tensorflow.keras as ks
from yolo.modeling.building_blocks import DarkConv
from yolo.modeling.building_blocks import DarkRouteProcess
from yolo.modeling.building_blocks import DarkUpsampleRoute
# for testing
from yolo.modeling.backbones.backbone_builder import Backbone_Builder

from . import configs

import importlib

#@ks.utils.register_keras_serializable(package='yolo')
class Yolov3Head(tf.keras.Model):
    def __init__(self, model="regular", classes=80, boxes=9, cfg_dict = None, **kwargs):
        """
        construct a detection head for an arbitrary back bone following the Yolo style 

        config format:
            back bones can out put a head with many outputs that will be processed, 
            for each backbone output yolo makes predictions for objects and N boxes 

            back bone our pur should be a dictionary, that is what allow this config style to work

            {<backbone_output_name>:{
                                        "depth":<number of output channels>, 
                                        "upsample":None or a layer takes in 2 tensors and returns 1,
                                        "upsample_conditions":{dict conditions for layer above},
                                        "processor": Layer that will process output with this key and return 2 tensors, 
                                        "processor_conditions": {dict conditions for layer above},
                                        "output_conditions": {dict conditions for detection or output layer},
                                        "output-extras": integer for the number of things to predict in addiction to the 
                                                         4 items for  the bounding box, so maybe you want to do pose estimation, 
                                                         and need the head to output 10 addictional values, place that here and for 
                                                         each bounding box, we will predict 10 more values, be sure to modify the loss 
                                                         function template to handle this modification, and calculate a loss for the
                                                         additional values. 
                                    }
                ...
                <backbone_output_name>:{ ... }
            }

        Args:
            model: to generate a standard yolo head, we have 3 string key words that can accomplish this
                    regular -> corresponds to yolov3
                    spp -> corresponds to yolov3-spp
                    tiny -> corresponds to yolov3-tiny

                if you construct a custom backbone config, name it as follows:
                    yolov3_<name>.py and we will be able to find the model automaticially 
                    in this case model corresponds to the value of <name>
            
            classes: integer for the number of classes in the prediction 
            boxes: integer for the total number of anchor boxes, this will be devided by the number of paths 
                   in the detection head config
            cfg_dict: dict, suppose you do not have the model_head saved config file in the configs folder, 
                      you can provide us with a dictionary that will be used to generate the model head, 
                      be sure to follow the correct format. 

            
        """
        self._cfg_dict = cfg_dict
        self._classes = classes
        self._boxes = boxes 
        self._model_name = model

        self._cfg_dict = self.load_dict_cfg(model)
        self._layer_keys = list(self._cfg_dict.keys())
        self._conv_depth = boxes//len(self._layer_keys) * (classes + 5)

        inputs, input_shapes, routes, upsamples, prediction_heads = self._get_attributes()
        outputs = self._connect_layers(routes, upsamples, prediction_heads, inputs)
        super().__init__(inputs=inputs, outputs=outputs, name=model, **kwargs)
        self._input_shape = input_shapes
        return

    def load_dict_cfg(self, model):
        """ find the config file and load it for use"""
        if self._cfg_dict != None:
            return self._cfg_dict
        try:
            return importlib.import_module('.yolov3_' + model, package=configs.__package__).head
        except ModuleNotFoundError as e:
            if e.name == configs.__package__ + '.yolov3_' + model:
                raise ValueError(f"Invlid head '{name}'") from e
            else:
                raise

    def _get_attributes(self):
        """ use config dictionary to generate all important attributes for head construction """
        inputs = dict()
        input_shapes = dict()
        routes = dict()
        upsamples = dict()
        prediction_heads = dict()

        for key in self._layer_keys:
            path_keys = self._cfg_dict[key]

            inputs[key] = ks.layers.Input(shape=[None, None, path_keys["depth"]])
            input_shapes[key] = tf.TensorSpec([None, None, None, path_keys["depth"]])

            if type(path_keys["upsample"]) != type(None):
                args = path_keys["upsample_conditions"]
                layer = path_keys["upsample"]
                upsamples[key] = layer(**args)

            args = path_keys["processor_conditions"]
            layer = path_keys["processor"]
            routes[key] = layer(**args)

            args = path_keys["output_conditions"]
            prediction_heads[key] = DarkConv(filters=self._conv_depth + path_keys["output-extras"],**args)
        return inputs, input_shapes, routes, upsamples, prediction_heads

    def _connect_layers(self, routes, upsamples, prediction_heads, inputs):
        """ connect all attributes the yolo way, if you want a different method of construction use something else """
        outputs = dict()
        layer_in = inputs[self._layer_keys[0]]
        for i in range(len(self._layer_keys)):
            x = routes[self._layer_keys[i]](layer_in)
            if i + 1 < len(self._layer_keys):
                x_next = inputs[self._layer_keys[i + 1]]
                layer_in = upsamples[self._layer_keys[i + 1]]([x[0], x_next])

            if type(x) == list or type(x) == tuple:
                outputs[self._layer_keys[i]] = prediction_heads[self._layer_keys[i]](x[1])
            else:
                outputs[self._layer_keys[i]] = prediction_heads[self._layer_keys[i]](x)
        return outputs
    
    def get_config():
        layer_config = {"cfg_dict": self._cfg_dict, 
                        "classes": self._classes, 
                        "boxes": self._boxes, 
                        "model": self._model_name}
        layer_config.update(super().get_config())
        return layer_config
