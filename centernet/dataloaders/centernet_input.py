import tensorflow as tf
from official.vision.beta.dataloaders import parser

from centernet.ops import preprocessing_ops

class CenterNetParser(parser.Parser):
    def __init__(
        self,
        num_classes: int,
        max_num_instances: int,
        gaussian_iou: float
    ):
        self._num_classes = num_classes
        self._max_num_instances = max_num_instances
        self._gaussian_iou = gaussian_iou

    def _generate_heatmap(self, boxes, output_size, input_size):
      batch_size = tf.shape(boxes)[0]
      tl_heatmaps = tf.zeros((batch_size, self._num_classes, output_size[0], output_size[1]), dtype=tf.float32)
      br_heatmaps = tf.zeros((batch_size, self._num_classes, output_size[0], output_size[1]), dtype=tf.float32)
      ct_heatmaps = tf.zeros((batch_size, self._num_classes, output_size[0], output_size[1]), dtype=tf.float32)
      tl_offset = tf.zeros((batch_size, self._max_num_instances, 2), dtype=tf.float32)
      br_offset = tf.zeros((batch_size, self._max_num_instances, 2), dtype=tf.float32)
      ct_offset = tf.zeros((batch_size, self._max_num_instances, 2), dtype=tf.float32)
      tl_size = tf.zeros((batch_size, self._max_num_instances), dtype=tf.int64)
      br_size = tf.zeros((batch_size, self._max_num_instances), dtype=tf.int64)
      ct_size = tf.zeros((batch_size, self._max_num_instances), dtype=tf.int64)
      tag_masks = tf.zeros((batch_size, self._max_num_instances), dtype=tf.uint8)

      width_ratio = output_size[1] / input_size[1]
      height_ratio = output_size[0] / input_size[0]

      for tag_ind, detection in enumerate(boxes):
        category = int(detection[-1]) - 1
        category = 0

        xtl, ytl = detection[0], detection[1]
        xbr, ybr = detection[2], detection[3]

        xct, yct = (
            (detection[2] + detection[0]) / 2,
            (detection[3] + detection[1]) / 2
        )

        fxtl = (xtl * width_ratio)
        fytl = (ytl * height_ratio)
        fxbr = (xbr * width_ratio)
        fybr = (ybr * height_ratio)
        fxct = (xct * width_ratio)
        fyct = (yct * height_ratio)

        xtl = int(fxtl)
        ytl = int(fytl)
        xbr = int(fxbr)
        ybr = int(fybr)
        xct = int(fxct)
        yct = int(fyct)
        # temporarily set to true
        gaussian_bump=True
        gaussian_rad=1

        if gaussian_bump:
          width = detection[2] - detection[0]
          height = detection[3] - detection[1]

          width = tf.math.ceil(width * width_ratio)
          height = tf.math.ceil(height * height_ratio)

          if gaussian_rad == -1:
            radius = preprocessing_ops.gaussian_radius((height, width), self._gaussian_iou)
            radius = max(0, int(radius))
          else:
            radius = gaussian_rad
        
        # test
        #   tl_heatmaps = preprocessing_ops.draw_gaussian(tl_heatmaps[b_ind, category], category, [xtl, ytl], radius)
        # inputs heatmap, center, radius, k=1
          tl_heatmaps = preprocessing_ops.draw_gaussian(tl_heatmaps, category, [xtl, ytl], radius)
          br_heatmaps = preprocessing_ops.draw_gaussian(br_heatmaps, category, [xbr, ybr], radius)
          ct_heatmaps = preprocessing_ops.draw_gaussian(ct_heatmaps, category, [xct, yct], radius, scaling_factor=5)

        else:
          # tl_heatmaps[category, ytl, xtl] = 1
          # br_heatmaps[category, ybr, xbr] = 1
          # ct_heatmaps[category, yct, xct] = 1
          tl_heatmaps = tf.tensor_scatter_nd_update(tl_heatmaps, [[category, ytl, xtl]], [1])
          br_heatmaps = tf.tensor_scatter_nd_update(br_heatmaps, [[category, ytl, xtl]], [1])
          ct_heatmaps = tf.tensor_scatter_nd_update(ct_heatmaps, [[category, ytl, xtl]], [1])

        # tl_offset[tag_ind, :] = [fxtl - xtl, fytl - ytl]
        # br_offset[tag_ind, :] = [fxbr - xbr, fybr - ybr]
        # ct_offset[tag_ind, :] = [fxct - xct, fyct - yct]
        # tl_size[tag_ind] = ytl * output_size[1] + xtl
        # br_size[tag_ind] = ybr * output_size[1] + xbr
        # ct_size[tag_ind] = yct * output_size[1] + xct
        tl_offset = tf.tensor_scatter_nd_update(tl_offset, [[tag_ind, 0], [tag_ind, 1]], [fxtl - xtl, fytl - ytl])
        br_offset = tf.tensor_scatter_nd_update(br_offset, [[tag_ind, 0], [tag_ind, 1]], [fxbr - xbr, fybr - ybr])
        ct_offset = tf.tensor_scatter_nd_update(ct_offset, [[tag_ind, 0], [tag_ind, 1]], [fxct - xct, fyct - yct])
        tl_size = tf.tensor_scatter_nd_update(tl_size, [[tag_ind]], [ytl * output_size[1] + xtl])
        br_size = tf.tensor_scatter_nd_update(br_size, [[tag_ind]], [ybr * output_size[1] + xbr])
        ct_size = tf.tensor_scatter_nd_update(ct_size, [[tag_ind]], [yct * output_size[1] + xct])

      labels = {
          'tl_size': tl_size,
          'br_size': br_size,
          'ct_size': ct_size,
          'tl_heatmaps': tl_heatmaps,
          'br_heatmaps': br_heatmaps,
          'ct_heatmaps': ct_heatmaps,
          'tag_masks': tag_masks,
          'tl_offset': tl_offset,
          'br_offset': br_offset,
          'ct_offset': ct_offset,
      }
      return labels

    def _parse_train_data(self, decoded_tensors):
        """Generates images and labels that are usable for model training.

        Args:
            decoded_tensors: a dict of Tensors produced by the decoder.

        Returns:
            images: the image tensor.
            labels: a dict of Tensors that contains labels.
        """
        # TODO: input size, output size
        image = decoded_tensors["image"]
        labels = self._generate_heatmap(
            decoded_tensors["groundtruth_boxes"],
            output_size, input_size
        )
        return image, labels

    def _parse_eval_data(self, data):
        image = decoded_tensors["image"]
        labels = self._generate_heatmap(
            decoded_tensors["groundtruth_boxes"],
            output_size, input_size
        )
        return image, labels

class ObjectDetectionTest(tf.test.TestCase):
    def generate_heatmaps(self, dectections):
      detections = [[
        (10, 30, 15, 17, 0)
      ]]
      
      tl_heatmaps, br_heatmaps, ct_heatmaps = generate_heatmaps(1, 2, (416, 416), detections)
      pass

if __name__ == '__main__':
  # This code is for visualization
  import matplotlib.pyplot as plt
  detections = [[
    (10, 300, 15, 370, 0),
    (100, 300, 150, 370, 0),
    (200, 100, 15, 170, 0),
  ],
  # more images can go here if you like
  ]

  tl_heatmaps, br_heatmaps, ct_heatmaps =  CenterNetParser(2, 200, 0.7)._generate_heatmap(tf.constant([
    [10, 300, 15, 370, 0],
    [100, 300, 150, 370, 0],
    [200, 100, 15, 170, 0],
    ], dtype=tf.float32), [416, 416], [416, 416])

#   tl_heatmaps, br_heatmaps, ct_heatmaps = generate_heatmaps(1, 2, (416, 416), detections)
  # ct_heatmaps[batch_id, class_id, ...]
  plt.imshow(ct_heatmaps[0, 0, ...])
  plt.show()
  # This is to run the test
  # tf.test.main()
