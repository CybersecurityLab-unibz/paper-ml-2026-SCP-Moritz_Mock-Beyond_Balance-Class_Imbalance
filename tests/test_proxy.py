import unittest
from nlbse_jex.proxy import Proxy
from nlbse_jex.utils import parse_args
from unittest.mock import patch
import os

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.path_prefix = os.getcwd() if os.getcwd().split("/")[-1] == "tests" else os.path.join(os.getcwd(), "tests")
        self.default_config_path = f'{self.path_prefix}/dummy_data/dummy_config.yml'
        self.patcher = patch('sys.argv', [
            'program_name',
            '--weighted-loss-strategy', f'{self.path_prefix}/dummy_data/test_function.py',
            '--train', f'{self.path_prefix}/dummy_data/pathon_train.csv',
            '--test', f'{self.path_prefix}/dummy_data/pathon_test.csv',
            '--column-key-text', 'combo',
            '--column-key-label', 'labels',
            '--config-path', self.default_config_path,
        ])

        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_proxy(self):
        args = parse_args().parse_args()
        p = Proxy(args)
        p.create_model()

