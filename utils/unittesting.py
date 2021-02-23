import sys
import unittest
loader = unittest.TestLoader()
suite = loader.discover(sys.argv[1], '*_test.py', top_level_dir='.')
runner = unittest.TextTestRunner()
runner.run(suite)
