from datasets import load_dataset
import numpy as np
import logging

from nlbse_jex.linear_weights import LinearWeights

dataset_id = 'NLBSE/nlbse25-code-comment-classification'

class DataHandler():
    def __init__(self, args):
        self.logger = logging.getLogger()
        self.args = args
        self.num_classes = None
        self.class_weight_dicts = None
        self.labels = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.read_data()

    def summarize_metrics(self, metrics):
        for i in range(7):  # num classes
            f1 = metrics.get(f"class_{i}_f1")
            prec = metrics.get(f"class_{i}_precision")
            rec = metrics.get(f"class_{i}_recall")
            self.logger.info(f"Class {i}: F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")

    def read_data(self):
        # TODO make this dynamic, such that the dataset is read from the args.
        reference_dataset = load_dataset(dataset_id)
        self.logger.info(reference_dataset)
        reference_dataset['pharo_train'] = reference_dataset['pharo_train'].rename_columns({self.args.column_key_text: "text", self.args.column_key_label: "label"})
        reference_dataset['pharo_test'] = reference_dataset['pharo_test'].rename_columns({self.args.column_key_text: "text", self.args.column_key_label: "label"})
        self.logger.info(reference_dataset)
        self.data_train = reference_dataset['pharo_train']
        self.data_val = reference_dataset['pharo_train']
        self.data_test = reference_dataset['pharo_test']

    def calculate_weights(self):
        self.logger.info(self.data_train)
        self.logger.info(len(self.data_train["label"][0]))
        self.num_classes = len(self.data_train["label"][0])
        self.class_weight_dicts = []
        self.logger.info(self.args)
        linear_weights = LinearWeights(self.args.weighted_loss_strategy, self.data_train["label"])
        self.class_weight_dicts = linear_weights.get_class_weight_dicts()

        self.logger.info(self.class_weight_dicts)
