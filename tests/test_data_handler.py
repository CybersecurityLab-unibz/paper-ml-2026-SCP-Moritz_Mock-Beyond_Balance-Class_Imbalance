import unittest
from nlbse_jex.utils import read_config, str2bool
from nlbse_jex.utils import read_files_weighted_loss_strategy
from nlbse_jex.utils import find_function_callable
from nlbse_jex.utils import get_first_function_name
from nlbse_jex.utils import compute_metrics
from nlbse_jex.utils import compute_metrics_setfit
from nlbse_jex.utils import copy_all_files
from nlbse_jex.utils import recursively_delete_all_files
from nlbse_jex.utils import parse_args
from nlbse_jex.utils import generate_base_config
from nlbse_jex.utils import load_yaml_config
from nlbse_jex.utils import import_function_from_path
from unittest.mock import patch
import os
import filecmp

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.path_prefix = os.getcwd() if os.getcwd().split("/")[-1] == "tests" else os.path.join(os.getcwd(), "tests")
        self.default_config_path = f'{self.path_prefix}/dummy_data/dummy_config.yml'
        self.patcher = patch('sys.argv', [
            'program_name',
            '--weighted-loss-strategy', f'{self.path_prefix}/dummy_data/test_function.py',
            '--train', 'dummy_data/pathon_train.csv',
            '--test', 'dummy_data/pathon_test.csv',
            '--column-key-text', 'combo',
            '--column-key-label', 'labels',
            '--config-path', self.default_config_path,
        ])

        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        generate_base_config(self.default_config_path, {"epoch": [1, 2, 3, 4, 5]})