import tensorflow as tf
import tensorflow.keras as ks
from typing import *

from yolo.modeling.backbones.darknet import Darknet
from yolo.modeling.decoders.yolo_decoder import YoloDecoder
from yolo.modeling.heads.yolo_head import YoloHead
from yolo.modeling.layers.detection_generator import YoloLayer


class Yolo(ks.Model):

  def __init__(self,
               backbone=None,
               decoder=None,
               head=None,
               filter=None,
               **kwargs):
    super().__init__(**kwargs)
    #model components
    self._backbone = backbone
    self._decoder = decoder
    self._head = head
    self._filter = filter
    return

  def build(self, input_shape):
    self._backbone.build(input_shape)
    nshape = self._backbone.output_shape
    self._decoder.build(nshape)
    super().build(input_shape)
    return

  def call(self, inputs, training=False):
    maps = self._backbone(inputs)
    decoded_maps = self._decoder(maps)
    raw_predictions = self._head(decoded_maps)
    if training:
      return {"raw_output": raw_predictions}
    else:
      predictions = self._filter(raw_predictions)
      predictions.update({"raw_output": raw_predictions})
      return predictions

  @property
  def backbone(self):
    return self._backbone

  @property
  def decoder(self):
    return self._decoder

  @property
  def head(self):
    return self._head

  @property
  def filter(self):
    return self._filter


from yolo.modeling.backbones.darknet import build_darknet
from official.vision.beta.modeling.backbones import factory
from official.core import registry
from yolo.configs import yolo


def build_yolo_decoder(input_specs, model_config: yolo.Yolo, l2_regularization):
  activation = model_config.decoder_activation if model_config.decoder_activation != "same" else model_config.norm_activation.activation
  if model_config.decoder.version == None:  #custom yolo
    model = YoloDecoder(
        embed_spp=model_config.decoder.embed_spp,
        embed_fpn=model_config.decoder.embed_fpn,
        fpn_path_len=model_config.decoder.fpn_path_len,
        path_process_len=model_config.decoder.path_process_len,
        max_level_process_len=model_config.decoder.max_level_process_len,
        xy_exponential=model_config.decoder.xy_exponential,
        activation=activation,
        use_sync_bn=model_config.norm_activation.use_sync_bn,
        norm_momentum=model_config.norm_activation.norm_momentum,
        norm_epsilon=model_config.norm_activation.norm_epsilon,
        kernel_regularizer=l2_regularization)
    model.build(input_specs)
    return model

  if model_config.decoder.type == None or model_config.decoder.type == "regular":  # defaut regular
    if model_config.decoder.version == "v4":
      model = YoloDecoder(
          embed_spp=False,
          embed_fpn=True,
          max_level_process_len=None,
          path_process_len=6,
          activation=activation,
          use_sync_bn=model_config.norm_activation.use_sync_bn,
          norm_momentum=model_config.norm_activation.norm_momentum,
          norm_epsilon=model_config.norm_activation.norm_epsilon,
          kernel_regularizer=l2_regularization)
    if model_config.decoder.version == "v3":
      model = YoloDecoder(
          embed_spp=False,
          embed_fpn=False,
          max_level_process_len=None,
          path_process_len=6,
          activation=activation,
          use_sync_bn=model_config.norm_activation.use_sync_bn,
          norm_momentum=model_config.norm_activation.norm_momentum,
          norm_epsilon=model_config.norm_activation.norm_epsilon,
          kernel_regularizer=l2_regularization)
  elif model_config.decoder.type == "tiny":
    model = YoloDecoder(
        embed_spp=False,
        embed_fpn=False,
        max_level_process_len=2,
        path_process_len=1,
        activation=activation,
        use_sync_bn=model_config.norm_activation.use_sync_bn,
        norm_momentum=model_config.norm_activation.norm_momentum,
        norm_epsilon=model_config.norm_activation.norm_epsilon,
        kernel_regularizer=l2_regularization)
  elif model_config.decoder.type == "spp":
    model = YoloDecoder(
        embed_spp=True,
        embed_fpn=False,
        max_level_process_len=None,
        path_process_len=6,
        activation=activation,
        use_sync_bn=model_config.norm_activation.use_sync_bn,
        norm_momentum=model_config.norm_activation.norm_momentum,
        norm_epsilon=model_config.norm_activation.norm_epsilon,
        kernel_regularizer=l2_regularization)
  else:
    raise Exception(
        "unsupported model_key please select from {v3, v4, v3spp, v3tiny, v4tiny}, \n\n or specify a custom decoder config using YoloDecoder in you yaml"
    )
  model.build(input_specs)
  return model


def build_yolo_filter(model_config: yolo.Yolo, decoder: YoloDecoder, masks,
                      xy_scales, path_scales):
  model = YoloLayer(
      masks=masks,
      classes=model_config.num_classes,
      anchors=model_config.boxes,
      thresh=model_config.filter.iou_thresh,
      cls_thresh=model_config.filter.class_thresh,
      max_boxes=model_config.filter.max_boxes,
      path_scale=path_scales,
      scale_xy=xy_scales,
      use_nms=model_config.filter.use_nms,
      loss_type=model_config.filter.loss_type,
      ignore_thresh=model_config.filter.ignore_thresh)
  return model


def build_yolo_head(input_specs, model_config: yolo.Yolo, l2_regularization):
  head = YoloHead(
      classes=model_config.num_classes,
      boxes_per_level=model_config.boxes_per_scale,
      xy_exponential=False,  #model_config.decoder.version == 'v4',
      norm_momentum=model_config.norm_activation.norm_momentum,
      norm_epsilon=model_config.norm_activation.norm_epsilon,
      kernel_regularizer=l2_regularization)
  head.build(input_specs)
  return head


def build_yolo(input_specs, model_config, l2_regularization, masks, xy_scales,
               path_scales):
  print(model_config.as_dict())
  print(input_specs)
  print(l2_regularization)

  backbone = factory.build_backbone(input_specs, model_config,
                                    l2_regularization)
  decoder = build_yolo_decoder(backbone.output_specs, model_config,
                               l2_regularization)
  head = build_yolo_head(backbone.output_specs, model_config, l2_regularization)
  filter = build_yolo_filter(model_config, head, masks, xy_scales, path_scales)

  model = Yolo(backbone=backbone, decoder=decoder, head=head, filter=filter)
  model.build(input_specs.shape)

  losses = filter.losses
  return model, losses
