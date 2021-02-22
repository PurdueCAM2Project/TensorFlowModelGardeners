import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from absl import logging
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from yolo.configs import yolo as exp_cfg

from official.vision.beta.evaluation import coco_evaluator

from yolo.dataloaders import yolo_input
from yolo.dataloaders.decoders import tfds_coco_decoder
from yolo.ops.kmeans_anchors import BoxGenInputReader
from yolo.ops.box_ops import xcycwh_to_yxyx

from official.vision.beta.ops import box_ops, preprocess_ops
from yolo.modeling.layers.detection_generator import YoloGTFilter


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
  
    return

  def build_model(self):
    """get an instance of Yolo v3 or v4"""
    from yolo.modeling.Yolo import build_yolo
    params = self.task_config.train_data
    model_base_cfg = self.task_config.model
    l2_weight_decay = self.task_config.weight_decay / 2.0

    masks, path_scales, xy_scales = self._get_masks()
    self._get_boxes(gen_boxes=params.is_training)

    input_specs = tf.keras.layers.InputSpec(shape=[None] +
                                            model_base_cfg.input_size)
    l2_regularizer = (
        tf.keras.regularizers.l2(l2_weight_decay) if l2_weight_decay else None)

    model, losses = build_yolo(input_specs, model_base_cfg, l2_regularizer,
                               masks, xy_scales, path_scales)
    self._loss_dict = losses
    model.summary()
    return model

  def build_inputs(self, params, input_context=None):
    """Build input dataset."""
    decoder = tfds_coco_decoder.MSCOCODecoder()
    """
    decoder_cfg = params.decoder.get()
    if params.decoder.type == 'simple_decoder':
        decoder = tf_example_decoder.TfExampleDecoder(
            regenerate_source_id=decoder_cfg.regenerate_source_id)
    elif params.decoder.type == 'label_map_decoder':
        decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
            label_map=decoder_cfg.label_map,
            regenerate_source_id=decoder_cfg.regenerate_source_id)
    else:
        raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))
    """

    model = self.task_config.model

    masks, path_scales, xy_scales = self._get_masks()
    anchors = self._get_boxes(gen_boxes=params.is_training)

    
    if params.is_training and params.parser.mosaic:
      # gbs = params.global_batch_size
      params.global_batch_size = 4 * params.global_batch_size

    parser = yolo_input.Parser(
        image_w=params.parser.image_w,
        image_h=params.parser.image_h,
        num_classes=model.num_classes,
        min_level=model.min_level,
        max_level=model.max_level,
        fixed_size=params.parser.fixed_size,
        jitter_im=params.parser.jitter_im,
        jitter_boxes=params.parser.jitter_boxes,
        masks=masks,
        letter_box=params.parser.letter_box,
        cutmix=params.parser.cutmix,
        mosaic=params.parser.mosaic, 
        use_tie_breaker=params.parser.use_tie_breaker,
        min_process_size=params.parser.min_process_size,
        max_process_size=params.parser.max_process_size,
        max_num_instances=params.parser.max_num_instances,
        random_flip=params.parser.random_flip,
        pct_rand=params.parser.pct_rand,
        seed=params.parser.seed,
        aug_rand_saturation=params.parser.aug_rand_saturation,
        aug_rand_brightness=params.parser.aug_rand_brightness,
        aug_rand_zoom=params.parser.aug_rand_zoom,
        aug_rand_hue=params.parser.aug_rand_hue,
        anchors=anchors,
        dtype=params.dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training),
        postprocess_fn=parser.postprocess_fn(params.is_training))
    dataset = reader.read(input_context=input_context)


    # if params.is_training and params.parser.mosaic:
    #   dataset = dataset.unbatch()
    #   dataset = dataset.batch(gbs)

    return dataset

  def build_losses(self, outputs, labels, aux_losses=None):
    loss = 0.0
    loss_box = 0.0
    loss_conf = 0.0
    loss_class = 0.0
    metric_dict = dict()

    grid = labels['grid_form']
    for key in outputs.keys():
      # _loss, _loss_box, _loss_conf, _loss_class, _avg_iou, _recall50 = self._loss_dict[key](labels, outputs[key])
      _loss, _loss_box, _loss_conf, _loss_class, _avg_iou, _recall50 = self._loss_dict[
          key](grid[key], outputs[key])
      #_loss, _loss_box, _loss_conf, _loss_class, _avg_iou, _recall50 = self._loss_dict[key](labels[key], outputs[key])
      loss += _loss
      loss_box += _loss_box
      loss_conf += _loss_conf
      loss_class += _loss_class
      metric_dict[f"recall50_{key}"] = tf.stop_gradient(_recall50)
      metric_dict[f"avg_iou_{key}"] = tf.stop_gradient(_avg_iou)

    metric_dict['box_loss'] = loss_box
    metric_dict['conf_loss'] = loss_conf
    metric_dict['class_loss'] = loss_class
    metric_dict['total_loss'] = loss

    return loss, metric_dict

  def build_metrics(self, training=True):
    metrics = []
    metric_names = self._metric_names

    for name in metric_names:
      metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))

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
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape(persistent=True) as tape:
      # compute a prediction
      # cast to float32
      y_pred = model(image, training=True)
      loss, loss_metrics = self.build_losses(y_pred['raw_output'], label)
      scaled_loss = loss / num_replicas

      # scale the loss for numerical stability
      if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
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

    delta = tape.gradient(loss, y_pred['raw_output'])
    loss_val = 0
    for value in delta.keys():
      loss_val += tf.reduce_sum(tf.square(delta[value]))
    # custom metrics
    logs = {'loss': loss}
    loss_metrics['darknet_loss'] = loss_val
    if metrics:
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})

    tf.print(logs, end='\n')

    ret = '\033[F' * (len(logs.keys()) + 1)
    tf.print(ret, end='\n')
    return logs

  def validation_step(self, inputs, model, metrics=None):
    # get the data point
    image, label = inputs

    # computer detivative and apply gradients
    with tf.GradientTape() as tape:
      y_pred = model(image, training=False)
      loss, loss_metrics = self.build_losses(y_pred['raw_output'], label)

    gradients = tape.gradient(loss, y_pred['raw_output'])

    loss_val = 0
    for value in gradients.keys():
      loss_val += tf.reduce_sum(tf.square(gradients[value]))

    # #custom metrics
    logs = {'loss': loss}
    # loss_metrics.update(metrics)
    loss_metrics['darknet_loss'] = loss_val
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
            tf.shape(y_pred['bbox'])[:-1],
        'source_id':
            label['source_id'],
    }

    logs.update({self.coco_metric.name: (label, coco_model_outputs)})

    if metrics:
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    # return super().aggregate_logs(state=state, step_outputs=step_outputs)

    if not state:
      # for metric in self._metrics:
      #   metric.reset_states()
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
    # gen_boxes = params.is_training
    if gen_boxes and self.task_config.model.boxes is None and not self._anchors_built:
      # must save the boxes!
      model_base_cfg = self.task_config.model
      self._num_boxes = (model_base_cfg.max_level - model_base_cfg.min_level +
                         1) * model_base_cfg.boxes_per_scale
      decoder = tfds_coco_decoder.MSCOCODecoder()
      reader = BoxGenInputReader(
          params,
          dataset_fn=tf.data.TFRecordDataset,
          decoder_fn=decoder.decode,
          parser_fn=None)
      anchors = reader.read(
          k=9, image_width=params.parser.image_w, input_context=input_context)
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
        path_scales[str(i)] = 2**i
        if xy_exponential:
          scale_x_y[str(i)] = 1.0 + xy_scale_base * (exp_base**i)
        else:
          scale_x_y[str(i)] = 1.0
        start += params.boxes_per_scale

      self._masks = boxes
      self._path_scales = path_scales
      self._x_y_scales = scale_x_y
      # self._get_gt_boxes_used = YoloGTFilter(self._masks)

    metric_names = []
    for key in self._masks.keys():
      metric_names.append(f"recall50_{key}")
      metric_names.append(f"avg_iou_{key}")

    metric_names.append('box_loss')
    metric_names.append('conf_loss')
    metric_names.append('class_loss')
    metric_names.append('total_loss')
    metric_names.append('darknet_loss')
    self._metric_names = metric_names

    return self._masks, self._path_scales, self._x_y_scales

  def initialize(self, model: tf.keras.Model):
    if self.task_config.load_darknet_weights:
      from yolo.utils import DarkNetConverter
      from yolo.utils._darknet2tf.load_weights import split_converter
      from yolo.utils._darknet2tf.load_weights2 import load_weights_backbone
      from yolo.utils._darknet2tf.load_weights2 import load_weights_neck
      from yolo.utils._darknet2tf.load_weights2 import load_head
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
        # if neck is not None:
        #   load_weights_neck(model.decoder, neck)
        #   #model.decoder.neck.trainable = False
        cfgheads = load_head(model.decoder, decoder)
        #model.decoder.head.trainable = False
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
        ckpt = tf.train.Checkpoint(**model.checkpoint_items)
        status = ckpt.restore(ckpt_dir_or_file)
        status.assert_consumed()
      elif self.task_config.init_checkpoint_modules == 'backbone':
        ckpt = tf.train.Checkpoint(backbone=model.backbone)
        status = ckpt.restore(ckpt_dir_or_file)
        status.expect_partial().assert_existing_objects_matched()
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
