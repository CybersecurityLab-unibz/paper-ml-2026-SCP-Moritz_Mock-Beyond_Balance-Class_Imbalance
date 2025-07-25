import re
import types
from datasets import Dataset, concatenate_datasets
import os
import logging
import pandas as pd
import numpy as np
import random
import torch

from nlbse_jex.linear_weights import LinearWeights
from nlbse_jex.utils import copy_all_files
from nlbse_jex.utils import recursively_delete_all_files
from nlbse_jex.utils import find_function_callable

dataset_id = 'NLBSE/nlbse25-code-comment-classification'

class DataHandler():
    def __init__(self, args):
        self.logger = logging.getLogger('nlbse_jex.data_handler')
        self.args = args
        self.num_classes = None
        self.class_weight_dicts = None
        self.labels = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.output_path = None
        self.logs_path = None
        self.best_model = None
        self.csv_path = None
        self.dataset_name = None
        self.dynamc = self.args.weighted_loss_strategy in ["FAMO", "COBA"]
        self.read_data()
        self.create_paths()
        self.set_seed(42)


    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    def create_paths(self):
        base_path = self.args.result_path
        model_path = re.sub(r"[/-]", "_", self.args.model_path)
        self.dataset_name = re.sub("-", "_", "_".join(self.args.train.split("/")[:-1]))

        self.csv_path = os.path.join(base_path, model_path, self.dataset_name, "final.csv")

        if self.args.maximise_model:
            self.output_path = os.path.join(base_path, model_path, self.dataset_name, "maximise_model", "checkpoints")
            self.logs_path = os.path.join(base_path, model_path, self.dataset_name, "maximise_model", "logs")
            os.makedirs(self.output_path, exist_ok=True)
            os.makedirs(self.logs_path, exist_ok=True)
        else:
            parameters = "_".join([
                str(self.args.batch_size),
                str(self.args.epoch),
                str(self.args.lr),
                str(self.args.weight_decay),
                self.args.weighted_loss_strategy if not isinstance(self.args.weighted_loss_strategy, types.FunctionType) else self.args.weighted_loss_strategy.__name__,
                str(self.args.freeze_embeddings)
            ])

            self.output_path = os.path.join(base_path, model_path, self.dataset_name, parameters, "checkpoints")
            self.logs_path = os.path.join(base_path, model_path, self.dataset_name, parameters, "logs")
            self.best_model = os.path.join(base_path, model_path, self.dataset_name, "best_model")

            os.makedirs(self.best_model, exist_ok=True)
            os.makedirs(self.output_path, exist_ok=True)
            os.makedirs(self.logs_path, exist_ok=True)

    def summarize_metrics(self, metrics):
        for i in range(self.num_classes):
            f1 = metrics.get(f"class_{i}_f1")
            precision = metrics.get(f"class_{i}_precision")
            recall = metrics.get(f"class_{i}_recall")
            self.logger.info(f"Class {i}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

    def update_csv(self, metric):
        if not os.path.exists(self.csv_path):
            columns = [
                "Model",
                "Dataset",
                "Weighted Loss Strategy",
                "Batch Size",
                "Epoch",
                "Learning Rate",
                "Weight Decay",
                "Freeze Embeddings",
                "Average Precision",
                "Average Recall",
                "Average F1"
            ]
            columns = columns + list(metric.keys())
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_path, index=False, sep=";")

        avg_precision = np.mean([float(v) for k, v in metric.items() if k.endswith("_precision")])
        avg_recall = np.mean([float(v) for k, v in metric.items() if k.endswith("_recall")])
        avg_f1 = np.mean([float(v) for k, v in metric.items() if k.endswith("_f1")])

        df = pd.read_csv(self.csv_path, sep=";")

        row = [
                  self.args.model_path,
                  self.dataset_name,
                  self.args.weighted_loss_strategy if not isinstance(self.args.weighted_loss_strategy, types.FunctionType) else self.args.weighted_loss_strategy.__name__,
                  self.args.batch_size,
                  self.args.epoch,
                  self.args.lr,
                  self.args.weight_decay,
                  self.args.freeze_embeddings,
                  avg_precision,
                  avg_recall,
                  avg_f1
              ] + list(metric.values())

        df.loc[len(df)] = row
        df.to_csv(self.csv_path, index=False, sep=";")

        if len(df[df["Average F1"]>avg_f1]) == 0:
            self.logger.info("New best average F1 found!")
            self.update_best_model()

        self.delete_path()


    def delete_path(self):
        recursively_delete_all_files("/".join(self.output_path.split("/")[:-1]))


    def update_best_model(self):
        last_checkpoint_number = -1
        pattern = re.compile(r'^checkpoint-(\d+)$')

        try:
            for name in os.listdir(self.output_path):
                full_path = os.path.join(self.output_path, name)
                if os.path.isdir(full_path):
                    match = pattern.match(name)
                    if match:
                        number = int(match.group(1))
                        if number > last_checkpoint_number:
                            last_checkpoint_number = number
        except Exception as e:
            self.logger.info(f"Error while deleting file - error message -> {e}")

        copy_all_files(self.output_path, self.best_model)


    def run_already_exists(self):
        if not os.path.exists(self.csv_path):
            return False

        df = pd.read_csv(self.csv_path, sep=";")
        df = df[
            (df["Model"] == self.args.model_path) &
            (df["Dataset"] == self.dataset_name) &
            (df["Weighted Loss Strategy"] == (self.args.weighted_loss_strategy if not isinstance(self.args.weighted_loss_strategy, types.FunctionType) else self.args.weighted_loss_strategy.__name__)) &
            (df["Batch Size"] == self.args.batch_size) &
            (df["Epoch"] == self.args.epoch) &
            (df["Learning Rate"] == self.args.lr) &
            (df["Weight Decay"] == self.args.weight_decay) &
            (df["Freeze Embeddings"].astype(str).str.lower().map({"true": True, "false": False}) == self.args.freeze_embeddings)
        ]
        
        self.logger.info(df)

        return len(df) != 0


    def read_data(self):
        train = pd.read_csv(self.args.train)
        val = pd.read_csv(self.args.val) if self.args.val is not None else None
        test = pd.read_csv(self.args.test)

        self.data_train = Dataset.from_pandas(train)
        self.data_val = Dataset.from_pandas(val) if val is not None else None
        self.data_test = Dataset.from_pandas(test)

        self.data_train = self.data_train.rename_columns({self.args.column_key_text: "text", self.args.column_key_label: "label"})
        self.data_val = self.data_val.rename_columns({self.args.column_key_text: "text", self.args.column_key_label: "label"}) if self.data_val is not None else None
        self.data_test = self.data_test.rename_columns({self.args.column_key_text: "text", self.args.column_key_label: "label"})

        self.data_train = self.data_train.map(lambda ex: {'label': np.fromstring(ex['label'].strip('[]'), sep=' ', dtype=int)})
        self.data_val = self.data_val.map(lambda ex: {'label': np.fromstring(ex['label'].strip('[]'), sep=' ', dtype=int)}) if self.data_val is not None else None
        self.data_test = self.data_test.map(lambda ex: {'label': np.fromstring(ex['label'].strip('[]'), sep=' ', dtype=int)})

    def merge_dataset(self):
        splits = [self.data_train, self.data_val, self.data_test]
        splits = [split for split in splits if split is not None]

        self.data_train = concatenate_datasets(splits)
        self.data_val = None
        self.data_test = None

        self.logger.info(
            self.data_train
        )

    def find_best_args(self):
        df = pd.read_csv(self.csv_path, sep=";")
        self.logger.info(df)
        best_run = df.loc[df["Average F1"].idxmax()]
        self.args.weighted_loss_strategy = best_run["Weighted Loss Strategy"] if best_run["Weighted Loss Strategy"] in ["EW", "ICF", "RBF"] else find_function_callable(self.args.weighted_loss_strategy, best_run["Weighted Loss Strategy"])
        self.args.batch_size = int(best_run["Batch Size"])
        self.args.epoch = int(best_run["Epoch"])
        self.args.lr = float(best_run["Learning Rate"])
        self.args.weight_decay = float(best_run["Weight Decay"])
        self.args.freeze_embeddings = bool(best_run["Freeze Embeddings"])


    def calculate_weights(self):
        if not self.dynamc:
            self.num_classes = len(self.data_train["label"][0])
            self.class_weight_dicts = []
            self.logger.info(self.args)
            linear_weights = LinearWeights(self.args.weighted_loss_strategy, self.data_train["label"])
            self.class_weight_dicts = linear_weights.get_class_weight_dicts()

            self.logger.info(self.class_weight_dicts)
