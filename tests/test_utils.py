import unittest

import numpy as np

from nlbse_jex.utils import read_config, str2bool
from nlbse_jex.utils import read_files_weighted_loss_strategy
from nlbse_jex.utils import find_function_callable
from nlbse_jex.utils import get_first_function_name
from nlbse_jex.utils import compute_metrics
from nlbse_jex.utils import compute_metrics_setfit
from nlbse_jex.utils import compute_metrics_roberta
from nlbse_jex.utils import copy_all_files
from nlbse_jex.utils import recursively_delete_all_files
from nlbse_jex.utils import parse_args
from nlbse_jex.utils import generate_base_config
from nlbse_jex.utils import load_yaml_config
from nlbse_jex.utils import import_function_from_path
from unittest.mock import patch
import os
import filecmp

def compare_folders(dir1, dir2):
    if not os.path.isdir(dir2):
        return False

    comparison = filecmp.dircmp(dir1, dir2)

    if comparison.left_only or comparison.right_only or comparison.diff_files or comparison.funny_files:
        return False
    else:
        for subdir in comparison.common_dirs:
            new_dir1 = os.path.join(dir1, subdir)
            new_dir2 = os.path.join(dir2, subdir)
            if not compare_folders(new_dir1, new_dir2):
                return False
        return True


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

    def test_write_yml(self):
        data = {"epoch": [1]}
        generate_base_config(self.default_config_path, data)
        yml = load_yaml_config(self.default_config_path)
        self.assertEqual(data, yml)


    def test_read_config_does_not_exists(self):
        parser = parse_args()
        args = parser.parse_args()
        args.config_path = f'{self.path_prefix}/dummy_data/does_not_exists.yml'
        with self.assertRaises(FileNotFoundError):
            read_config(args, parser)

    def test_read_config_empty_config(self):
        path = f'{self.path_prefix}/dummy_data/dummy_config_empty.yml'
        open(path, 'w').close()
        parser = parse_args()
        args = parser.parse_args()
        args.config_path = path
        yml = read_config(args, parser)
        self.assertEqual(args, yml)
        os.remove(path)

    def test_read_config_invalid_config(self):
        data = {"epochs": [1]}
        generate_base_config(self.default_config_path, data)
        parser = parse_args()
        args = parser.parse_args()
        with self.assertRaises(ValueError):
            read_config(args, parser)

    def test_read_config_invalid_value(self):
        data = {"epoch": [-1]}
        generate_base_config(self.default_config_path, data)
        parser = parse_args()
        args = parser.parse_args()
        with self.assertRaises(ValueError):
            read_config(args, parser)

        data = {"epoch": 1}
        generate_base_config(self.default_config_path, data)
        parser = parse_args()
        args = parser.parse_args()
        with self.assertRaises(ValueError):
            read_config(args, parser)

        data = {"maximise_model": True}
        generate_base_config(self.default_config_path, data)
        parser = parse_args()
        args = parser.parse_args()
        yml = read_config(args, parser)
        self.assertEqual(yml.maximise_model, True)

        data = {"maximise_model": "yes"}
        generate_base_config(self.default_config_path, data)
        parser = parse_args()
        args = parser.parse_args()
        yml = read_config(args, parser)
        self.assertEqual(yml.maximise_model, True)


        data = {"maximise_model": "TURE"}
        generate_base_config(self.default_config_path, data)
        parser = parse_args()
        args = parser.parse_args()
        with self.assertRaises(ValueError):
            read_config(args, parser)

    def test_read_config(self):
        parser = parse_args()
        args = parser.parse_args()
        yml = read_config(args, parser)
        args.epoch = [1, 2, 3, 4, 5]

        self.assertEqual(args, yml)

    def test_str2bool(self):
        self.assertEqual(True, str2bool(True))
        self.assertEqual(True, str2bool("yes"))
        self.assertEqual(True, str2bool("true"))
        self.assertEqual(True, str2bool("t"))
        self.assertEqual(True, str2bool("y"))
        self.assertEqual(True, str2bool("1"))
        self.assertEqual(False, str2bool(False))
        self.assertEqual(False, str2bool("no"))
        self.assertEqual(False, str2bool("False"))
        self.assertEqual(False, str2bool("n"))
        self.assertEqual(False, str2bool("0"))
        with self.assertRaises(ValueError):
            str2bool("hello")

    def test_import_function_dynamically(self):
        args = parse_args().parse_args()
        function = import_function_from_path(args.weighted_loss_strategy[0])
        data = [[0, 1, 0]]
        weights = function(data)
        self.assertEqual(weights, [{1: 0.1, 0: 0.9}, {1: 0.1, 0: 0.9}, {1: 0.1, 0: 0.9}])

    def test_read_files_weighted_loss_strategy(self):
        args = parse_args().parse_args()
        args = read_files_weighted_loss_strategy(args)
        function = find_function_callable(args.weighted_loss_strategy, "my_function")
        self.assertEqual(function, args.weighted_loss_strategy[0])
        with self.assertRaises(ValueError):
            find_function_callable([], "my_function")
        with self.assertRaises(ValueError):
            find_function_callable(args.weighted_loss_strategy, "own_function")

    def test_import_function_error(self):
        path = f'{self.path_prefix}/dummy_data/my_function.py'
        open(path, 'w').write("")
        with self.assertRaises(ValueError):
            get_first_function_name(path)

        os.remove(path)

    def test_compute_metrics(self):
        labels = np.array([[0, 1], [0, 1]])
        preds = np.array([[0, 1], [1, 0]])
        metrics = compute_metrics(preds, labels, 2)

        self.assertEqual(0, metrics["class_0_true_positives"])
        self.assertEqual(1, metrics["class_0_false_positives"])
        self.assertEqual(1, metrics["class_0_true_negatives"])
        self.assertEqual(0, metrics["class_0_false_negatives"])
        self.assertEqual(0, metrics["class_0_precision"])
        self.assertEqual(0, metrics["class_0_recall"])
        self.assertEqual(0, metrics["class_0_f1"])
        self.assertEqual(1, metrics["class_1_true_positives"])
        self.assertEqual(0, metrics["class_1_false_positives"])
        self.assertEqual(0, metrics["class_1_true_negatives"])
        self.assertEqual(1, metrics["class_1_false_negatives"])
        self.assertEqual(1, metrics["class_1_precision"])
        self.assertEqual(0.5, metrics["class_1_recall"])
        self.assertEqual(0.6666666666666666, metrics["class_1_f1"])


    def test_compute_metrics_setfit(self):
        labels = [[0, 1], [0, 1]]
        preds = [1, 0]
        metrics = compute_metrics_setfit(preds, labels)

        self.assertEqual(0, metrics["class_0_true_positives"])
        self.assertEqual(1, metrics["class_0_false_positives"])
        self.assertEqual(1, metrics["class_0_true_negatives"])
        self.assertEqual(0, metrics["class_0_false_negatives"])
        self.assertEqual(0, metrics["class_0_precision"])
        self.assertEqual(0, metrics["class_0_recall"])
        self.assertEqual(0, metrics["class_0_f1"])
        self.assertEqual(1, metrics["class_1_true_positives"])
        self.assertEqual(0, metrics["class_1_false_positives"])
        self.assertEqual(0, metrics["class_1_true_negatives"])
        self.assertEqual(1, metrics["class_1_false_negatives"])
        self.assertEqual(1, metrics["class_1_precision"])
        self.assertEqual(0.5, metrics["class_1_recall"])
        self.assertEqual(0.6666666666666666, metrics["class_1_f1"])

    def test_compute_metrics_roberta(self):
        labels = np.array([[0, 1], [0, 1]])
        logits = np.array([[-0.1, 1], [1, -0.1]]) # sigmoid converts into [[0.4750, 0.7311], [0.7311, 0.4750]], which is translated to [[0, 1], [1, 0]] using the threshold of 0.5
        metrics = compute_metrics_roberta((logits, labels))

        self.assertEqual(0, metrics["class_0_true_positives"])
        self.assertEqual(1, metrics["class_0_false_positives"])
        self.assertEqual(1, metrics["class_0_true_negatives"])
        self.assertEqual(0, metrics["class_0_false_negatives"])
        self.assertEqual(0, metrics["class_0_precision"])
        self.assertEqual(0, metrics["class_0_recall"])
        self.assertEqual(0, metrics["class_0_f1"])
        self.assertEqual(1, metrics["class_1_true_positives"])
        self.assertEqual(0, metrics["class_1_false_positives"])
        self.assertEqual(0, metrics["class_1_true_negatives"])
        self.assertEqual(1, metrics["class_1_false_negatives"])
        self.assertEqual(1, metrics["class_1_precision"])
        self.assertEqual(0.5, metrics["class_1_recall"])
        self.assertEqual(0.6666666666666666, metrics["class_1_f1"])

    def test_copy_all_files(self):
        print(self.path_prefix)
        path_source = f'{self.path_prefix}/dummy_data'
        path_target = f'{self.path_prefix}/depth_level/dummy_data_copy'

        copy_all_files(path_source, path_target)

        self.assertEqual(True, compare_folders(path_source, path_target))

        recursively_delete_all_files(path_target)

        self.assertEqual(False, compare_folders(path_source, path_target))
