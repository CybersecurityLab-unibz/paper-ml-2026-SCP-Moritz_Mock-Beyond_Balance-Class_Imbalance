import re
import unittest
import os
from nlbse_jex.data_handler import DataHandler
from nlbse_jex.utils import parse_args
from nlbse_jex.utils import compute_metrics
from nlbse_jex.utils import recursively_delete_all_files
from unittest.mock import patch
from datasets import Dataset
import numpy as np
import json


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.path_prefix = os.getcwd() if os.getcwd().split("/")[-1] == "tests" else os.path.join(os.getcwd(), "tests")
        self.patcher = patch('sys.argv', [
            'program_name',
            '--weighted-loss-strategy', f'{self.path_prefix}/dummy_data/test_function.py',
            '--train', f'{self.path_prefix}/dummy_data/dataset/python_train.csv',
            '--test', f'{self.path_prefix}/dummy_data/dataset/python_test.csv',
            '--column-key-text', 'combo',
            '--column-key-label', 'labels',
            '--weighted-loss-strategy', 'EW',
            '--result-path', f'{self.path_prefix}/results'
        ])

        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        recursively_delete_all_files(f'{self.path_prefix}/results')

    def test_data_handler_default(self):
        args = parse_args().parse_args()
        args.batch_size = args.batch_size[0]
        args.epoch = args.epoch[0]
        args.lr = args.lr[0]
        args.weight_decay = args.weight_decay[0]
        args.weighted_loss_strategy = args.weighted_loss_strategy[0]
        dh = DataHandler(args)
        self.assertEqual(dh.csv_path, f'{self.path_prefix}/results/sentence_transformers_paraphrase_MiniLM_L6_v2/{re.sub("/", "_", f'{self.path_prefix}/dummy_data/dataset')}/final.csv')
        self.assertEqual(dh.best_model, f'{self.path_prefix}/results/sentence_transformers_paraphrase_MiniLM_L6_v2/{re.sub("/", "_", f'{self.path_prefix}/dummy_data/dataset')}/best_model')
        self.assertEqual(dh.logs_path, f'{self.path_prefix}/results/sentence_transformers_paraphrase_MiniLM_L6_v2/{re.sub("/", "_", f'{self.path_prefix}/dummy_data/dataset')}/32_5_0.01_0_EW_False/logs')
        self.assertEqual(dh.output_path, f'{self.path_prefix}/results/sentence_transformers_paraphrase_MiniLM_L6_v2/{re.sub("/", "_", f'{self.path_prefix}/dummy_data/dataset')}/32_5_0.01_0_EW_False/checkpoints')

    def test_data_handler_maximise_model(self):
        args = parse_args().parse_args()
        args.batch_size = args.batch_size[0]
        args.epoch = args.epoch[0]
        args.lr = args.lr[0]
        args.weight_decay = args.weight_decay[0]
        args.weighted_loss_strategy = args.weighted_loss_strategy[0]
        args.maximise_model = True
        dh = DataHandler(args)
        self.assertEqual(dh.output_path, f'{self.path_prefix}/results/sentence_transformers_paraphrase_MiniLM_L6_v2/{re.sub("/", "_", f'{self.path_prefix}/dummy_data/dataset')}/maximise_model/checkpoints')
        self.assertEqual(dh.logs_path, f'{self.path_prefix}/results/sentence_transformers_paraphrase_MiniLM_L6_v2/{re.sub("/", "_", f'{self.path_prefix}/dummy_data/dataset')}/maximise_model/logs')

    def test_data_handler_calculate_weights(self):
        args = parse_args().parse_args()
        args.batch_size = args.batch_size[0]
        args.epoch = args.epoch[0]
        args.lr = args.lr[0]
        args.weight_decay = args.weight_decay[0]
        args.weighted_loss_strategy = args.weighted_loss_strategy[0]
        dh = DataHandler(args)
        dh.calculate_weights()

        self.assertEqual([{1: 1, 0: 0}, {1: 1, 0: 0}, {1: 1, 0: 0}, {1: 1, 0: 0}, {1: 1, 0: 0}], dh.class_weight_dicts)


    def test_data_handler_summarize_metrics(self):
        args = parse_args().parse_args()
        args.batch_size = args.batch_size[0]
        args.epoch = args.epoch[0]
        args.lr = args.lr[0]
        args.weight_decay = args.weight_decay[0]
        args.weighted_loss_strategy = args.weighted_loss_strategy[0]
        dh = DataHandler(args)

        labels = np.array([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]])
        preds = np.array([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
        metrics = compute_metrics(preds, labels, 5)
        metrics = json.loads(json.dumps(metrics, default=lambda o: float(o) if isinstance(o, (np.integer, np.floating)) else o))

        dh.calculate_weights()
        with self.assertLogs('nlbse_jex.data_handler', level='INFO') as cm:
            dh.summarize_metrics(metrics)

        self.assertEqual(cm.output[0], "INFO:nlbse_jex.data_handler:Class 0: F1=0.000, Precision=0.000, Recall=0.000")
        self.assertEqual(cm.output[1], "INFO:nlbse_jex.data_handler:Class 1: F1=0.667, Precision=1.000, Recall=0.500")
        self.assertEqual(cm.output[2], "INFO:nlbse_jex.data_handler:Class 2: F1=0.000, Precision=0.000, Recall=0.000")
        self.assertEqual(cm.output[3], "INFO:nlbse_jex.data_handler:Class 3: F1=0.000, Precision=0.000, Recall=0.000")
        self.assertEqual(cm.output[4], "INFO:nlbse_jex.data_handler:Class 4: F1=0.000, Precision=0.000, Recall=0.000")

        self.assertEqual(True, os.path.isdir(dh.output_path))
        self.assertEqual(False, os.path.isfile(dh.csv_path))
        self.assertEqual(False, dh.run_already_exists())
        dh.update_csv(metrics)
        self.assertEqual(True, dh.run_already_exists())
        self.assertEqual(False, os.path.isdir(dh.output_path))
        self.assertEqual(True, os.path.isfile(dh.csv_path))

        args = parse_args().parse_args()
        args.batch_size = 100
        args.epoch = args.epoch[0]
        args.lr = args.lr[0]
        args.weight_decay = args.weight_decay[0]
        args.weighted_loss_strategy = args.weighted_loss_strategy[0]
        dh = DataHandler(args)
        labels = np.array([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]])
        preds = np.array([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
        metrics = compute_metrics(preds, labels, 5)
        metrics = json.loads(
            json.dumps(metrics, default=lambda o: float(o) if isinstance(o, (np.integer, np.floating)) else o))

        dh.calculate_weights()
        dh.summarize_metrics(metrics)

        self.assertEqual(False, dh.run_already_exists())

        dh.find_best_args()

        self.assertEqual('EW', dh.args.weighted_loss_strategy)
        self.assertEqual(32, dh.args.batch_size)
        self.assertEqual(5, dh.args.epoch)
        self.assertEqual(0.01, dh.args.lr)
        self.assertEqual(0.0, dh.args.weight_decay)
        self.assertEqual(False, dh.args.freeze_embeddings)

        dh.merge_dataset()

        self.assertEqual(dh.data_val, None)
        self.assertEqual(dh.data_test, None)
        self.assertEqual(type(dh.data_train), Dataset)

        open(os.path.join(dh.output_path, "hello.txt"), "w").write("Hello, world")

        dh.update_best_model()

        self.assertEqual(False, not os.listdir(dh.best_model))
