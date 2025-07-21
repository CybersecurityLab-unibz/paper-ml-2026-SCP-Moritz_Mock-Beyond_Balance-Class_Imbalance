import unittest
from nlbse_jex.linear_weights import LinearWeights
from nlbse_jex.utils import parse_args, read_files_weighted_loss_strategy
from unittest.mock import patch
import os
import pandas as pd
from datasets import Dataset
import numpy as np

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.path_prefix = os.getcwd() if os.getcwd().split("/")[-1] == "tests" else os.path.join(os.getcwd(), "tests")
        self.default_config_path = f'{self.path_prefix}/dummy_data/dummy_config.yml'
        self.patcher = patch('sys.argv', [
            'program_name',
            '--weighted-loss-strategy', f'{self.path_prefix}/dummy_data/test_function.py',
            '--train', f'{self.path_prefix}/dummy_data/dataset/python_train.csv',
            '--test', f'{self.path_prefix}/dummy_data/dataset/python_test.csv',
            '--column-key-text', 'combo',
            '--column-key-label', 'labels',
            '--config-path', self.default_config_path,
        ])

        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_proxy(self):
        self.args = parse_args().parse_args()
        train = pd.read_csv(self.args.train)
        val = pd.read_csv(self.args.val) if self.args.val is not None else None
        test = pd.read_csv(self.args.test)

        self.data_train = Dataset.from_pandas(train)
        self.data_val = Dataset.from_pandas(val) if val is not None else None
        self.data_test = Dataset.from_pandas(test)

        self.data_train = self.data_train.rename_columns(
            {self.args.column_key_text: "text", self.args.column_key_label: "label"})
        self.data_val = self.data_val.rename_columns({self.args.column_key_text: "text",
                                                      self.args.column_key_label: "label"}) if self.data_val is not None else None
        self.data_test = self.data_test.rename_columns(
            {self.args.column_key_text: "text", self.args.column_key_label: "label"})

        self.data_train = self.data_train.map(
            lambda ex: {'label': np.fromstring(ex['label'].strip('[]'), sep=' ', dtype=int)})
        self.data_val = self.data_val.map(lambda ex: {
            'label': np.fromstring(ex['label'].strip('[]'), sep=' ', dtype=int)}) if self.data_val is not None else None
        self.data_test = self.data_test.map(
            lambda ex: {'label': np.fromstring(ex['label'].strip('[]'), sep=' ', dtype=int)})

        lw = LinearWeights("EW", self.data_train["label"])
        w = lw.get_class_weight_dicts()

        self.assertEqual(w, [{1: 1, 0: 0}, {1: 1, 0: 0}, {1: 1, 0: 0}, {1: 1, 0: 0}, {1: 1, 0: 0}])

        lw = LinearWeights("ICF", self.data_train["label"])
        w = lw.get_class_weight_dicts()
        self.assertEqual(w, [{1: 0.044, 0: 0.956}, {1: 0.19, 0: 0.81}, {1: 0.103, 0: 0.897}, {1: 0.379, 0: 0.621}, {1: 0.284, 0: 0.716}])

        lw = LinearWeights("RBF", self.data_train["label"])
        w = lw.get_class_weight_dicts()
        self.assertEqual(w, [{1: 0.06, 0: 0.94}, {1: 0.12, 0: 0.88}, {1: 0.08, 0: 0.92}, {1: 0.52, 0: 0.48}, {1: 0.22, 0: 0.78}])

        self.args = read_files_weighted_loss_strategy(self.args)

        lw = LinearWeights(self.args.weighted_loss_strategy[0], self.data_train["label"])
        w = lw.get_class_weight_dicts()
        print(w)
        self.assertEqual(w, [{1: 0.1, 0: 0.9}, {1: 0.1, 0: 0.9}, {1: 0.1, 0: 0.9}, {1: 0.1, 0: 0.9}, {1: 0.1, 0: 0.9}])

        with self.assertRaises(ValueError):
            LinearWeights("NOT_VALID", self.data_train["label"])