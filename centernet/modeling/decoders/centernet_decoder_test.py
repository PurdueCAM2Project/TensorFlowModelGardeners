from absl.testing import parameterized
import tensorflow as tf
import numpy as np

from centernet.configs import centernet as cfg
from centernet.modeling.decoders import centernet_decoder


class CenterNetDecoderTest(tf.test.TestCase, parameterized.TestCase):

  def test_create_decoder(self):
    decoder = centernet_decoder.build_centernet_decoder(model_config=cfg.CenterNet2D())
    config = decoder.get_config()
    self.assertEqual(len(config), 2)
    self.assertEqual(config['heatmap_bias'], -2.19)

  def test_decoder_shape(self):
    decoder = centernet_decoder.build_centernet_decoder(model_config=cfg.CenterNet2D())

    # Output shape tests
    outputs = decoder(np.zeros((2, 128, 128, 256), dtype=np.float32))
    self.assertEqual(len(outputs), 3)
    self.assertEqual(outputs['heatmap'].shape, (2, 128, 128, 91))
    self.assertEqual(outputs['local_offset'].shape, (2, 128, 128, 2))
    self.assertEqual(outputs['object_size'].shape, (2, 128, 128, 2))

    # Weight initialization tests
    hm_bias_vector = np.asarray(decoder.out_layers['heatmap'].weights[-1])
    off_bias_vector = np.asarray(decoder.out_layers['local_offset'].weights[-1])
    size_bias_vector = np.asarray(decoder.out_layers['object_size'].weights[-1])

    self.assertArrayNear(hm_bias_vector, 
      np.repeat(-2.19, repeats=91), err=1.00e-6)
    self.assertArrayNear(off_bias_vector, 
      np.repeat(0, repeats=2), err=1.00e-6)
    self.assertArrayNear(size_bias_vector, 
      np.repeat(0, repeats=2), err=1.00e-6)    

if __name__ == '__main__':
  tf.test.main()