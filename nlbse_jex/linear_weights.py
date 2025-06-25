import logging
import numpy as np
import types

class LinearWeights():

    def __init__(self, type, labels):
        self.logger = logging.getLogger()
        self.type = type
        self.labels = labels
        self.class_weight_dicts = []
        self.calculate_linear_weights()

    def calculate_linear_weights(self):
        if self.type == "EW":
            self._EW()
        if self.type == "ICF":
            self._ICF()
        if self.type == "RBF":
            self._RBF()
        if isinstance(self.type, types.FunctionType):
            self.class_weight_dicts = self.type(self.labels)
        else:
            self.logger.error("Not defined type of liner weights!")

    def _EW(self):
        num_classes = self.labels.shape[1]
        for i in range(num_classes):
            self.class_weight_dicts.append({1: 0.5, 0: 0.5})

    def _ICF(self):
        labels_array = np.array(self.labels)
        class_counts = labels_array.sum(axis=0)
        total_samples = len(self.labels)
        class_frequencies = (class_counts / total_samples) * 100
        result = []

        for i, freq in enumerate(class_frequencies):
            result.append(1 / freq)

        result = np.array(result)

        result = result * (1 / np.sum(result))

        for i in range(labels_array.shape[1]):
            self.class_weight_dicts.append({
                1: round(float(result[i]), 3),
                0: round(float(1 - result[i]), 3)
            })

    def _RBF(self):
        labels_array = np.array(self.labels)
        class_counts = labels_array.sum(axis=0)
        total_samples = len(self.labels)
        class_frequencies = (class_counts / total_samples)

        sorted_values = sorted(range(len(class_frequencies)), key=lambda x: class_frequencies[x], reverse=True)
        sorted_frequencies = sorted(class_frequencies)

        result = [0] * len(class_frequencies)
        for i, idx in enumerate(sorted_values):
            result[idx] = round(float(sorted_frequencies[i]), 3)

        for i in range(labels_array.shape[1]):
            self.class_weight_dicts.append({
                1: result[i],
                0: round(float(1 - result[i]), 3)
            })

    def get_class_weight_dicts(self):
        return self.class_weight_dicts