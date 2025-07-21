import unittest
from nlbse_jex.utils import read_config
from nlbse_jex.utils import parse_args
from unittest.mock import patch
import os
import argparse

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.patcher = patch('sys.argv', [
            'program_name',
            '--train', 'dummy_data/pathon_train.csv',
            '--test', 'dummy_data/pathon_test.csv',
            '--column-key-text', 'combo',
            '--column-key-label', 'labels',
            '--config-path', 'tests/dummy_data/dummy_config.yml'
        ])

        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_read_config_does_not_exists(self):
        parser = parse_args()
        args = parser.parse_args()
        args.config_path = "tests/dummy_data/does_not_exists.yml"
        with self.assertRaises(FileNotFoundError):
            read_config(args, parser)

    def test_read_config_empty_config(self):
        path = "tests/dummy_data/dummy_config_empty.yml"
        open(path, 'w').close()
        parser = parse_args()
        args = parser.parse_args()
        args.config_path = path
        yml = read_config(args, parser)
        self.assertEqual(args, yml)
        os.remove(path)

    def test_read_config_invalid_config(self):
        parser = parse_args()
        args = parser.parse_args()
        args.config_path = "tests/dummy_data/dummy_config_invalid_key.yml"
        with self.assertRaises(ValueError):
            read_config(args, parser)

    def test_read_config_invalid_value(self):
        parser = parse_args()
        args = parser.parse_args()
        args.config_path = "tests/dummy_data/dummy_config_invalid_value.yml"
        with self.assertRaises(ValueError):
            read_config(args, parser)

    def test_read_config(self):
        parser = parse_args()
        args = parser.parse_args()
        yml = read_config(args, parser)
        args.epoch = [1, 2, 3, 4, 5]

        self.assertEqual(args, yml)

