from datasets import load_dataset
import numpy as np

dataset_id = 'NLBSE/nlbse25-code-comment-classification'

class DataHandler():
    def __init__(self):
        self.num_classes = None
        self.class_weight_dicts = None
        self.labels = None
        self.read_data()

    def summarize_metrics(self, metrics):
        for i in range(7):  # num classes
            f1 = metrics.get(f"class_{i}_f1")
            prec = metrics.get(f"class_{i}_precision")
            rec = metrics.get(f"class_{i}_recall")
            print(f"Class {i}: F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")

    def read_data(self):
        # TODO make this dynamic, such that the dataset is read from the args.
        self.reference_dataset = load_dataset(dataset_id)
        self.labels = np.array(self.reference_dataset['pharo_train']['labels'])

    def calculate_weights(self):
        total = self.labels.shape[0]

        # Convert to dict: {class_index: weight}
        self.num_classes = self.labels.shape[1]
        self.class_weight_dicts = []

        # TODO make this in a way that the args control the type of weights are calculated
        for i in range(self.num_classes):
            positives = self.labels[:, i].sum()
            negatives = total - positives
            weight_pos = positives / total
            weight_neg = negatives / total
            self.class_weight_dicts.append({1: weight_neg, 0: weight_pos})
