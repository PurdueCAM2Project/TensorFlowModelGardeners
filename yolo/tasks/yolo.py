import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from absl import logging
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from yolo.configs import yolo as exp_cfg

from official.vision.beta.evaluation import coco_evaluator
from official.vision.beta.dataloaders import tf_example_decoder
from official.vision.beta.dataloaders import tfds_detection_decoders
from official.vision.beta.dataloaders import tf_example_label_map_decoder

from yolo.dataloaders import yolo_input
from yolo.ops import mosaic
from yolo.ops.kmeans_anchors import BoxGenInputReader
from yolo.ops.box_ops import xcycwh_to_yxyx

from official.vision.beta.ops import box_ops, preprocess_ops
from yolo.modeling.layers import detection_generator
from collections import defaultdict


class ListMetrics(object):

  def __init__(self, metric_names, name="ListMetrics", **kwargs):
    # super(ListMetrics, self).__init__(name=name, **kwargs)
    self.name = name
    self._metric_names = metric_names
    self._metrics = self.build_metric()
    return

  def build_metric(self):
    metric_names = self._metric_names
    metrics = []
    for name in metric_names:
      metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))
    return metrics

  def update_state(self, loss_metrics):
    metrics = self._metrics
    for m in metrics:
      m.update_state(loss_metrics[m.name])
    return

  def result(self):
    logs = dict()
    metrics = self._metrics
    for m in metrics:
      logs.update({m.name: m.result()})
    return logs

  def reset_states(self):
    # super().reset_states()
    metrics = self._metrics
    for m in metrics:
      m.reset_states()
    return


@task_factory.register_task_cls(exp_cfg.YoloTask)
class YoloTask(base_task.Task):
  """A single-replica view of training procedure.
  RetinaNet task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """

  def __init__(self, params, logging_dir: str = None):
    super().__init__(params, logging_dir)
    self._loss_dict = None
    self._num_boxes = None
    self._anchors_built = False
    # self._get_gt_boxes_used = None #YoloGTFilter()

    self._masks = None
    self._path_scales = None
    self._x_y_scales = None
    self.coco_metric = None
    self._metric_names = []
    self._metrics = []

    self._dfilter = detection_generator.YoloFilter(
        classes=self._task_config.model.num_classes)

    return

  def build_model(self):
    """get an instance of Yolo v3 or v4"""
    from yolo.modeling.Yolo import build_yolo
    params = self.task_config.train_data
    model_base_cfg = self.task_config.model
    l2_weight_decay = self.task_config.weight_decay / 2.0

    masks, path_scales, xy_scales = self._get_masks()
    print(xy_scales, l2_weight_decay)

    self._get_boxes(gen_boxes=params.is_training)

    input_specs = tf.keras.layers.InputSpec(shape=[None] +
                                            model_base_cfg.input_size)
    l2_regularizer = (
        tf.keras.regularizers.l2(l2_weight_decay) if l2_weight_decay else None)

    model, losses = build_yolo(input_specs, model_base_cfg, l2_regularizer,
                               masks, xy_scales, path_scales)
    self._loss_dict = losses
    return model

  def get_decoder(self, params):
    if params.tfds_name:
      if params.tfds_name in tfds_detection_decoders.TFDS_ID_TO_DECODER_MAP:
        decoder = tfds_detection_decoders.TFDS_ID_TO_DECODER_MAP[
            params.tfds_name]()
      else:
        raise ValueError('TFDS {} is not supported'.format(params.tfds_name))
    else:
      decoder_cfg = params.decoder.get()
      if params.decoder.type == 'simple_decoder':
        decoder = tf_example_decoder.TfExampleDecoder(
            regenerate_source_id=decoder_cfg.regenerate_source_id)
      elif params.decoder.type == 'label_map_decoder':
        decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
            label_map=decoder_cfg.label_map,
            regenerate_source_id=decoder_cfg.regenerate_source_id)
      else:
        raise ValueError('Unknown decoder type: {}!'.format(
            params.decoder.type))
    return decoder

  def build_inputs(self, params, input_context=None):
    """Build input dataset."""

    decoder = self.get_decoder(params)
    model = self.task_config.model

    masks, path_scales, xy_scales = self._get_masks()
    anchors = self._get_boxes(gen_boxes=params.is_training)

    print(xy_scales)

    sample_fn = mosaic.Mosaic(
        output_size=params.parser.mosaic.output_size,
        mosaic_frequency=params.parser.mosaic.mosaic_frequency,
        crop_area=params.parser.mosaic.crop_area,
        random_crop=params.parser.mosaic.random_crop,
        crop_area_mosaic=params.parser.mosaic.crop_area_mosaic,
        random_crop_mosaic=params.parser.mosaic.random_crop_mosaic,
    )

    parser = yolo_input.Parser(
        image_w=params.parser.image_w,
        image_h=params.parser.image_h,
        min_level=model.min_level,
        max_level=model.max_level,
        num_classes=model.num_classes,
        batch_size=params.global_batch_size,
        masks=masks,
        anchors=anchors,
        fixed_size=params.parser.fixed_size,
        letter_box=params.parser.letter_box,
        use_tie_breaker=params.parser.use_tie_breaker,
        random_flip=params.parser.random_flip,
        jitter_im=params.parser.jitter_im,
        jitter_boxes=params.parser.jitter_boxes,
        aug_rand_transalate=params.parser.aug_rand_translate,
        aug_rand_saturation=params.parser.aug_rand_saturation,
        aug_rand_brightness=params.parser.aug_rand_brightness,
        aug_rand_zoom=params.parser.aug_rand_zoom,
        aug_rand_hue=params.parser.aug_rand_hue,
        aug_rand_angle=params.parser.aug_rand_angle,
        min_process_size=params.parser.min_process_size,
        max_num_instances=params.parser.max_num_instances,
        pct_rand=params.parser.pct_rand,
        scale_xy=xy_scales,
        use_scale_xy=params.parser.use_scale_xy,
        anchor_t=params.parser.anchor_thresh,
        dtype=params.dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        sample_fn=sample_fn.mosaic_fn(is_training=params.is_training),
        parser_fn=parser.parse_fn(params.is_training),
        postprocess_fn=parser.postprocess_fn(params.is_training))
    dataset = reader.read(input_context=input_context)

    print(dataset)
    return dataset

  def build_losses(self,
                   outputs,
                   labels,
                   num_replicas=1,
                   scale_replicas=1,
                   aux_losses=None):
    metric_dict = defaultdict(dict)
    loss_val = 0
    metric_dict['global']['total_loss'] = 0

    grid = labels['true_conf']
    inds = labels['inds']
    upds = labels['upds']

    scale = tf.cast(3 / len(list(outputs.keys())), tf.float32)
    for key in outputs.keys():
      (_loss, _loss_box, _loss_conf, _loss_class, _avg_iou, _avg_obj, _recall50,
       _precision50) = self._loss_dict[key](grid[key], inds[key], upds[key],
                                            labels['bbox'], labels['classes'],
                                            outputs[key])
      metric_dict['global']['total_loss'] += _loss
      metric_dict[key]['conf_loss'] = _loss_conf / scale_replicas
      metric_dict[key]['box_loss'] = _loss_box / scale_replicas
      metric_dict[key]['class_loss'] = _loss_class / scale_replicas
      metric_dict[key]["recall50"] = tf.stop_gradient(_recall50 /
                                                      scale_replicas)
      metric_dict[key]["precision50"] = tf.stop_gradient(_precision50 /
                                                         scale_replicas)
      metric_dict[key]["avg_iou"] = tf.stop_gradient(_avg_iou / scale_replicas)
      metric_dict[key]["avg_obj"] = tf.stop_gradient(_avg_obj / scale_replicas)
      loss_val += _loss * scale / num_replicas

    return loss_val, metric_dict

  def build_metrics(self, training=True):
    metrics = []
    metric_names = self._metric_names

    for i, key in enumerate(metric_names.keys()):
      metrics.append(ListMetrics(metric_names[key], name=key))

    self._metrics = metrics

    if not training:
      self.coco_metric = coco_evaluator.COCOEvaluator(
          annotation_file=self.task_config.annotation_file,
          include_mask=False,
          need_rescale_bboxes=False,
          per_category_metrics=self._task_config.per_category_metrics)

    return metrics

  def train_step(self, inputs, model, optimizer, metrics=None):
    # get the data point
    image, label = inputs

    scale_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    if self._task_config.model.filter.use_reduction_sum:
      num_replicas = 1
    else:
      num_replicas = scale_replicas

    with tf.GradientTape() as tape:
      # compute a prediction
      # cast to float32
      y_pred = model(image, training=True)
      y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred)
      scaled_loss, loss_metrics = self.build_losses(
          y_pred['raw_output'],
          label,
          num_replicas=num_replicas,
          scale_replicas=1)
      # scaled_loss = loss / num_replicas

      # scale the loss for numerical stability
      if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
        # for key in scaled_loss.keys():
        #   scaled_loss[key] = optimizer.get_scaled_loss(scaled_loss[key])
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    # compute the gradient
    train_vars = model.trainable_variables
    gradients = tape.gradient(scaled_loss, train_vars)

    # get unscaled loss if the scaled_loss was used
    if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
      gradients = optimizer.get_unscaled_gradients(gradients)

    if self.task_config.gradient_clip_norm > 0.0:
      gradients, _ = tf.clip_by_global_norm(gradients,
                                            self.task_config.gradient_clip_norm)

    optimizer.apply_gradients(zip(gradients, train_vars))

    logs = {self.loss: loss_metrics['global']['total_loss']}
    if metrics:
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})

      # for m in metrics:
      #   m.update_state(loss_metrics[m.name])
      #   logs.update({m.name: m.result()})

    tf.print(logs, end='\n')
    ret = '\033[F' * (len(logs.keys()) * 7 - 14 + 2 + 1)
    tf.print(ret, end='\n')
    return logs

  def validation_step(self, inputs, model, metrics=None):
    # get the data point
    image, label = inputs

    scale_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    if self._task_config.model.filter.use_reduction_sum:
      num_replicas = 1
    else:
      num_replicas = scale_replicas

    y_pred = model(image, training=False)
    y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred)
    loss, loss_metrics = self.build_losses(
        y_pred['raw_output'],
        label,
        num_replicas=num_replicas,
        scale_replicas=1)
    logs = {self.loss: loss_metrics['global']['total_loss']}

    image_shape = tf.shape(image)[1:-1]

    label['boxes'] = box_ops.denormalize_boxes(
        tf.cast(label['bbox'], tf.float32), image_shape)
    del label['bbox']

    coco_model_outputs = {
        'detection_boxes':
            box_ops.denormalize_boxes(
                tf.cast(y_pred['bbox'], tf.float32), image_shape),
        'detection_scores':
            y_pred['confidence'],
        'detection_classes':
            y_pred['classes'],
        'num_detections':
            y_pred['num_detections'],
        'source_id':
            label['source_id'],
    }

    logs.update({self.coco_metric.name: (label, coco_model_outputs)})

    # if metrics:
    #   for m in metrics:
    #     m.update_state(loss_metrics[m.name])
    #     logs.update({m.name: m.result()})

    if metrics:
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})

      # for i, key in enumerate(self._metric_names.keys()):
      #   if metrics[i] is not None:
      #     logs[key] = dict()
      #     for m in metrics[i]:
      #       m.update_state(loss_metrics[key][m.name])
      #       logs[key].update({m.name: m.result()})

    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    # return super().aggregate_logs(state=state, step_outputs=step_outputs)

    if not state:
      self.coco_metric.reset_states()
      state = self.coco_metric
    self.coco_metric.update_state(step_outputs[self.coco_metric.name][0],
                                  step_outputs[self.coco_metric.name][1])
    return state

  def reduce_aggregated_logs(self, aggregated_logs):
    # return super().reduce_aggregated_logsI(aggregated_logs)
    return self.coco_metric.result()

  @property
  def anchors(self):
    return self.task_config.model.boxes

  def _get_boxes(self, gen_boxes=True):

    if gen_boxes and self.task_config.model.boxes is None and not self._anchors_built:
      # must save the boxes!
      params = self.task_config.train_data
      decoder = self.get_decoder(params)
      model_base_cfg = self.task_config.model
      self._num_boxes = (model_base_cfg.max_level - model_base_cfg.min_level +
                         1) * model_base_cfg.boxes_per_scale
      reader = BoxGenInputReader(
          params,
          decoder_fn=decoder.decode,
          transform_and_batch_fn=lambda x, y: x,
          parser_fn=None)
      anchors = reader.read(
          k=9, image_width=params.parser.image_w, input_context=None)
      self.task_config.model.set_boxes(anchors)
      self._anchors_built = True
      del reader

    return self.task_config.model.boxes

  def _get_masks(self,
                 xy_exponential=True,
                 exp_base=2,
                 xy_scale_base='default_value'):
    start = 0
    boxes = {}
    path_scales = {}
    scale_x_y = {}

    if xy_scale_base == 'default_base':
      xy_scale_base = 0.05
      xy_scale_base = xy_scale_base / (
          self._boxes_per_level * (self._max_level - self._min_level + 1) - 1)
    elif xy_scale_base == 'default_value':
      xy_scale_base = 0.00625

    params = self.task_config.model

    if self._masks is None or self._path_scales is None or self._x_y_scales is None:
      for i in range(params.min_level, params.max_level + 1):
        boxes[str(i)] = list(range(start, params.boxes_per_scale + start))
        start += params.boxes_per_scale

      self._masks = boxes
      self._path_scales = params.filter.path_scales.as_dict()
      self._x_y_scales = params.filter.scale_xy.as_dict()

    metric_names = defaultdict(list)
    for key in self._masks.keys():
      metric_names[key].append(f'box_loss')
      metric_names[key].append(f'class_loss')
      metric_names[key].append(f'conf_loss')
      metric_names[key].append(f"recall50")
      metric_names[key].append(f"precision50")
      metric_names[key].append(f"avg_iou")
      metric_names[key].append(f"avg_obj")

    metric_names['global'].append(f'total_loss')

    print(metric_names)

    # metric_names.append('darknet_loss')
    self._metric_names = metric_names

    return self._masks, self._path_scales, self._x_y_scales

  def initialize(self, model: tf.keras.Model):

    if self.task_config.load_darknet_weights:
      from yolo.utils import DarkNetConverter
      from yolo.utils._darknet2tf.load_weights import split_converter
      from yolo.utils._darknet2tf.load_weights2 import load_weights_backbone
      from yolo.utils._darknet2tf.load_weights2 import load_weights_decoder
      from yolo.utils._darknet2tf.load_weights2 import load_weights_prediction_layers
      from yolo.utils.downloads.file_manager import download

      weights_file = self.task_config.model.darknet_weights_file
      config_file = self.task_config.model.darknet_weights_cfg

      if ('cache' not in weights_file and 'cache' not in config_file):
        list_encdec = DarkNetConverter.read(config_file, weights_file)
      else:
        import os
        path = os.path.abspath('cache')
        if (not os.path.isdir(path)):
          os.mkdir(path)

        cfg = f"{path}/cfg/{config_file.split('/')[-1]}"
        if not os.path.isfile(cfg):
          download(config_file.split('/')[-1])

        wgt = f"{path}/weights/{weights_file.split('/')[-1]}"
        if not os.path.isfile(wgt):
          download(weights_file.split('/')[-1])

        list_encdec = DarkNetConverter.read(cfg, wgt)

      splits = model.backbone._splits
      if 'neck_split' in splits.keys():
        encoder, neck, decoder = split_converter(list_encdec,
                                                 splits['backbone_split'],
                                                 splits['neck_split'])
      else:
        encoder, decoder = split_converter(list_encdec,
                                           splits['backbone_split'])
        neck = None

      load_weights_backbone(model.backbone, encoder)
      #model.backbone.trainable = False

      if self.task_config.darknet_load_decoder:
        cfgheads = load_weights_decoder(
            model.decoder, [neck, decoder],
            csp=self._task_config.model.base.decoder.type == 'csp')
        load_weights_prediction_layers(cfgheads, model.head)
        #model.head.trainable = False

    else:
      """Loading pretrained checkpoint."""
      if not self.task_config.init_checkpoint:
        return

      ckpt_dir_or_file = self.task_config.init_checkpoint
      if tf.io.gfile.isdir(ckpt_dir_or_file):
        ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

      # Restoring checkpoint.
      if self.task_config.init_checkpoint_modules == 'all':
        ckpt = tf.train.Checkpoint(model=model)
        # status = ckpt.restore(ckpt_dir_or_file)
        # status.assert_consumed()
        # optimizer = self.create_optimizer(params.trainer.optimizer_config,
        #                               params.runtime)
        # optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.SGD(), dynamic = True)
        # ckpt = tf.train.Checkpoint(backbone = model.backbone, decoder = model.decoder, head = model.head) #, optimizer=optimizer)
        status = ckpt.restore(ckpt_dir_or_file)
        status.expect_partial().assert_existing_objects_matched()

      elif self.task_config.init_checkpoint_modules == 'backbone':
        ckpt = tf.train.Checkpoint(backbone=model.backbone)
        status = ckpt.restore(ckpt_dir_or_file)
        # if self.task_config.model.subdivisions == 1:
        #   try:
        status.expect_partial().assert_existing_objects_matched()
        #except:
        #print("this checkpoint could not assert all components consumed")
      else:
        assert "Only 'all' or 'backbone' can be used to initialize the model."

      logging.info('Finished loading pretrained checkpoint from %s',
                   ckpt_dir_or_file)


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from yolo.utils.run_utils import prep_gpu
  prep_gpu()

  config = exp_cfg.YoloTask(model=exp_cfg.Yolo(base='v3'))
  task = YoloTask(config)
  model = task.build_model()
  model.summary()
  task.initialize(model)

  train_data = task.build_inputs(config.train_data)
  # test_data = task.build_inputs(config.task.validation_data)

  for l, (i, j) in enumerate(train_data):
    preds = model(i, training=False)
    boxes = xcycwh_to_yxyx(j['bbox'])

    i = tf.image.draw_bounding_boxes(i, boxes, [[1.0, 0.0, 0.0]])

    i = tf.image.draw_bounding_boxes(i, preds['bbox'], [[0.0, 1.0, 0.0]])
    plt.imshow(i[0].numpy())
    plt.show()

    if l > 2:
      break
