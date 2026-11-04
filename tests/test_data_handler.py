import re
import unittest
import os
from beyondBalance.data_handler import DataHandler
from beyondBalance.utils import parse_args
from beyondBalance.utils import compute_metrics
from beyondBalance.utils import recursively_delete_all_files
from unittest.mock import patch
from datasets import Dataset
import numpy as np
import json
import pandas as pd
from tempfile import TemporaryDirectory


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.path_prefix = (
            os.getcwd()
            if os.getcwd().split("/")[-1] == "tests"
            else os.path.join(os.getcwd(), "tests")
        )
        self.dataset_name = re.sub(r"[/-]", "_", f"{self.path_prefix}/dummy_data/dataset")
        self.patcher = patch(
            "sys.argv",
            [
                "program_name",
                "--weighted-loss-strategy",
                f"{self.path_prefix}/dummy_data/test_function.py",
                "--train",
                f"{self.path_prefix}/dummy_data/dataset/python_train.csv",
                "--test",
                f"{self.path_prefix}/dummy_data/dataset/python_test.csv",
                "--column-key-text",
                "combo",
                "--column-key-label",
                "labels",
                "--weighted-loss-strategy",
                "EW",
                "--result-path",
                f"{self.path_prefix}/results",
            ],
        )

        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        recursively_delete_all_files(f"{self.path_prefix}/results")

    def test_non_positive_sample_size_rejected_before_setup(self):
        args = parse_args().parse_args()

        for sample_size in (-1, 0):
            args.sample_size = sample_size
            with (
                self.subTest(sample_size=sample_size),
                patch.object(DataHandler, "read_data") as read_data,
                patch.object(DataHandler, "create_paths") as create_paths,
                patch.object(DataHandler, "set_seed") as set_seed,
                patch("beyondBalance.data_handler.get_device") as get_device,
            ):
                with self.assertRaisesRegex(ValueError, "sample_size must be greater"):
                    DataHandler(args)

                read_data.assert_not_called()
                create_paths.assert_not_called()
                set_seed.assert_not_called()
                get_device.assert_not_called()

    def test_data_handler_default(self):
        args = parse_args().parse_args()
        args.batch_size = args.batch_size[0]
        args.epoch = args.epoch[0]
        args.lr = args.lr[0]
        args.weight_decay = args.weight_decay[0]
        args.weighted_loss_strategy = args.weighted_loss_strategy[0]
        dh = DataHandler(args)
        self.assertEqual(
            dh.csv_path,
            f'{self.path_prefix}/results/sentence_transformers_paraphrase_MiniLM_L6_v2/{self.dataset_name}/seed_42/final.csv',
        )
        self.assertEqual(
            dh.best_model,
            f'{self.path_prefix}/results/sentence_transformers_paraphrase_MiniLM_L6_v2/{self.dataset_name}/seed_42/best_model',
        )
        self.assertEqual(
            dh.logs_path,
            f'{self.path_prefix}/results/sentence_transformers_paraphrase_MiniLM_L6_v2/{self.dataset_name}/seed_42/32_5_0.01_0_EW_False/logs',
        )
        self.assertEqual(
            dh.output_path,
            f'{self.path_prefix}/results/sentence_transformers_paraphrase_MiniLM_L6_v2/{self.dataset_name}/seed_42/32_5_0.01_0_EW_False/checkpoints',
        )

    def test_data_handler_maximise_model(self):
        args = parse_args().parse_args()
        args.batch_size = args.batch_size[0]
        args.epoch = args.epoch[0]
        args.lr = args.lr[0]
        args.weight_decay = args.weight_decay[0]
        args.weighted_loss_strategy = args.weighted_loss_strategy[0]
        args.maximise_model = True
        dh = DataHandler(args)
        self.assertEqual(
            dh.output_path,
            f'{self.path_prefix}/results/sentence_transformers_paraphrase_MiniLM_L6_v2/{self.dataset_name}/seed_42/maximise_model/checkpoints',
        )
        self.assertEqual(
            dh.logs_path,
            f'{self.path_prefix}/results/sentence_transformers_paraphrase_MiniLM_L6_v2/{self.dataset_name}/seed_42/maximise_model/logs',
        )

    def test_data_handler_in_depth(self):
        args = parse_args().parse_args()
        args.batch_size = args.batch_size[0]
        args.epoch = args.epoch[0]
        args.lr = args.lr[0]
        args.weight_decay = args.weight_decay[0]
        args.weighted_loss_strategy = args.weighted_loss_strategy[0]
        dh = DataHandler(args)
        dh.calculate_weights()

        self.assertEqual(
            [{1: 1, 0: 0}, {1: 1, 0: 0}, {1: 1, 0: 0}, {1: 1, 0: 0}, {1: 1, 0: 0}],
            dh.class_weight_dicts,
        )

        labels = np.array([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]])
        preds = np.array([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
        metrics = compute_metrics(preds, labels, 5)
        metrics = json.loads(
            json.dumps(
                metrics,
                default=lambda o: (
                    float(o) if isinstance(o, (np.integer, np.floating)) else o
                ),
            )
        )

        dh.calculate_weights()
        with self.assertLogs("beyondBalance.data_handler", level="INFO") as cm:
            dh.summarize_metrics(metrics)

        self.assertEqual(
            cm.output[0],
            "INFO:beyondBalance.data_handler:Class 0: F1=0.000, Precision=0.000, Recall=0.000",
        )
        self.assertEqual(
            cm.output[1],
            "INFO:beyondBalance.data_handler:Class 1: F1=0.667, Precision=1.000, Recall=0.500",
        )
        self.assertEqual(
            cm.output[2],
            "INFO:beyondBalance.data_handler:Class 2: F1=0.000, Precision=0.000, Recall=0.000",
        )
        self.assertEqual(
            cm.output[3],
            "INFO:beyondBalance.data_handler:Class 3: F1=0.000, Precision=0.000, Recall=0.000",
        )
        self.assertEqual(
            cm.output[4],
            "INFO:beyondBalance.data_handler:Class 4: F1=0.000, Precision=0.000, Recall=0.000",
        )

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
            json.dumps(
                metrics,
                default=lambda o: (
                    float(o) if isinstance(o, (np.integer, np.floating)) else o
                ),
            )
        )

        dh.calculate_weights()
        dh.summarize_metrics(metrics)

        self.assertEqual(False, dh.run_already_exists())

        dh.find_best_args()

        self.assertEqual("EW", dh.args.weighted_loss_strategy)
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


class TestSampleSizeTracking(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def make_handler(self, sample_size=None, maximise_model=False, seed=42):
        args = parse_args().parse_args(
            [
                "--train", "data/train.csv",
                "--test", "data/test.csv",
                "--column-key-text", "text",
                "--column-key-label", "label",
                "--result-path", self.directory.name,
                "--seed", str(seed),
            ]
        )
        for name in ("batch_size", "epoch", "lr", "weight_decay", "weighted_loss_strategy"):
            setattr(args, name, getattr(args, name)[0])
        args.sample_size = sample_size
        args.maximise_model = maximise_model
        with patch.object(DataHandler, "read_data"):
            return DataHandler(args)

    def record_run(self, handler, score=0.5):
        metrics = {
            "class_0_precision": score,
            "class_0_recall": score,
            "class_0_f1": score,
        }
        with (
            patch.object(handler, "update_best_model") as update_best_model,
            patch.object(handler, "delete_path"),
        ):
            handler.update_csv(metrics)
        return update_best_model

    def test_paths_distinguish_sample_sizes(self):
        handlers = [self.make_handler(size) for size in (None, 2, 3)]
        self.assertEqual(len({handler.csv_path for handler in handlers}), 1)
        for attribute in ("output_path", "logs_path", "best_model"):
            self.assertEqual(len({getattr(handler, attribute) for handler in handlers}), 3)
        self.assertIn("samples_2", handlers[1].output_path)

        maximised = [self.make_handler(size, maximise_model=True) for size in (None, 2, 3)]
        for attribute in ("output_path", "logs_path"):
            self.assertEqual(len({getattr(handler, attribute) for handler in maximised}), 3)

    def test_paths_distinguish_seeds(self):
        for sample_size in (None, 2):
            for maximise_model in (False, True):
                with self.subTest(sample_size=sample_size, maximise_model=maximise_model):
                    first = self.make_handler(sample_size, maximise_model, seed=42)
                    second = self.make_handler(sample_size, maximise_model, seed=7)
                    repeated = self.make_handler(sample_size, maximise_model, seed=42)
                    attributes = ["csv_path", "output_path", "logs_path"]
                    if not maximise_model:
                        attributes.append("best_model")
                    for attribute in attributes:
                        first_path = getattr(first, attribute)
                        second_path = getattr(second, attribute)
                        self.assertIn("seed_42", first_path.split(os.sep))
                        self.assertIn("seed_7", second_path.split(os.sep))
                        self.assertNotEqual(first_path, second_path)
                        self.assertEqual(first_path, getattr(repeated, attribute))
                        self.assertTrue(os.path.isdir(os.path.dirname(first_path)))

    def test_run_history_is_separate_for_each_seed(self):
        first = self.make_handler(2, seed=42)
        second = self.make_handler(2, seed=7)
        self.record_run(first, score=0.9).assert_called_once()
        self.assertTrue(first.run_already_exists())
        self.assertFalse(second.run_already_exists())

        second.args.epoch = 9
        self.record_run(second, score=0.5).assert_called_once()
        self.assertTrue(second.run_already_exists())
        first.find_best_args()
        second.find_best_args()
        self.assertEqual(first.args.epoch, 5)
        self.assertEqual(second.args.epoch, 9)
        self.assertEqual(pd.read_csv(first.csv_path, sep=";")["Average F1"].tolist(), [0.9])
        self.assertEqual(pd.read_csv(second.csv_path, sep=";")["Average F1"].tolist(), [0.5])

    def test_duplicate_detection_and_csv_distinguish_sample_sizes(self):
        subset = self.make_handler(2)
        other_subset = self.make_handler(3)
        full_dataset = self.make_handler()
        self.record_run(subset)

        self.assertTrue(subset.run_already_exists())
        self.assertTrue(self.make_handler(2).run_already_exists())
        self.assertFalse(other_subset.run_already_exists())
        self.assertFalse(full_dataset.run_already_exists())

        self.record_run(full_dataset)
        self.assertTrue(full_dataset.run_already_exists())
        self.assertTrue(subset.run_already_exists())
        self.assertFalse(other_subset.run_already_exists())
        results = pd.read_csv(subset.csv_path, sep=";")
        self.assertEqual(results["Sample Size"].tolist(), ["2", "all"])

    def test_best_models_compared_within_sample_size(self):
        subset = self.make_handler(2)
        self.record_run(subset, score=0.9).assert_called_once()
        self.record_run(self.make_handler(), score=0.5).assert_called_once()
        self.record_run(self.make_handler(3), score=0.3).assert_called_once()
        self.record_run(subset, score=0.1).assert_not_called()

    def test_find_best_args_uses_matching_sample_size(self):
        subset = self.make_handler(2)
        subset.args.epoch = 2
        self.record_run(subset, score=0.5)
        subset.args.epoch = 3
        self.record_run(subset, score=0.7)
        other_subset = self.make_handler(3)
        other_subset.args.epoch = 99
        self.record_run(other_subset, score=0.9)
        full_dataset = self.make_handler()
        full_dataset.args.epoch = 100
        self.record_run(full_dataset, score=1.0)

        for size, expected_epoch in ((2, 3), (3, 99), (None, 100)):
            with self.subTest(sample_size=size):
                handler = self.make_handler(size, maximise_model=True)
                handler.find_best_args()
                self.assertEqual(handler.args.epoch, expected_epoch)

    def test_find_best_args_without_matching_size(self):
        self.record_run(self.make_handler(2))
        with self.assertRaisesRegex(ValueError, "sample_size=3"):
            self.make_handler(3, maximise_model=True).find_best_args()

    def test_legacy_rows_preserved_with_unknown_sample_sizes(self):
        subset = self.make_handler(2)
        self.record_run(subset, score=0.9)
        legacy = pd.read_csv(subset.csv_path, sep=";").drop(columns=["Sample Size"])
        legacy.to_csv(subset.csv_path, index=False, sep=";")

        self.assertFalse(subset.run_already_exists())
        self.assertFalse(self.make_handler().run_already_exists())
        with self.assertRaisesRegex(ValueError, "sample_size=2"):
            subset.find_best_args()

        self.record_run(subset, score=0.5).assert_called_once()
        results = pd.read_csv(subset.csv_path, sep=";")
        self.assertEqual(results["Sample Size"].tolist(), ["unknown", "2"])
        self.assertEqual(results["Average F1"].tolist(), [0.9, 0.5])
        self.assertTrue(subset.run_already_exists())
        full_dataset = self.make_handler()
        self.assertFalse(full_dataset.run_already_exists())
        self.record_run(full_dataset, score=0.4).assert_called_once()
        self.assertTrue(full_dataset.run_already_exists())
