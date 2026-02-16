import logging
import numpy as np
import types


class LinearWeights:

    def __init__(self, type, labels):
        """
        Initializes the LinearWeights object for computing class weights based on the specified strategy.

        This constructor:
            - Sets up logging for the weight calculation module
            - Stores the weighting strategy type and label data
            - Determines the number of classes from the label format
            - Initializes an empty list for storing class weight dictionaries
            - Automatically computes the class weights using the specified strategy

        Parameters:
            type (str or Callable): The weighting strategy to apply. Can be one of the predefined strategies ("EW", "ICF", "RBF")
                                    or a custom callable that accepts `labels` and returns class weight dictionaries.
            labels (List[List[int]]): A list of multi-class or multi-label vectors representing the class labels for each sample.

        Returns:
            None
        """

        self.logger = logging.getLogger("beyondBalance.linear_weights")
        self.type = type
        self.labels = labels
        self.num_classes = len(labels[0])
        self.class_weight_dicts = []
        self.calculate_linear_weights()

    def calculate_linear_weights(self):
        """
        Calculates linear class weights based on the specified type.

        Supported types:
            - "EW": Equal Weights
            - "ICF": Inverse Class Frequency
            - "RBF": Radial Basis Function
            - Callable: A custom function that accepts `labels` and returns class weight dictionaries

        Raises:
            ValueError: If the specified type is not recognized or supported.

        Returns:
            None
        """

        if self.type == "EW":
            self._EW()
        elif self.type == "ICF":
            self._ICF()
        elif self.type == "RBF":
            self._RBF()
        elif isinstance(self.type, types.FunctionType):
            self.class_weight_dicts = self.type(self.labels)
        else:
            raise ValueError("Not defined type of liner weights!")

    def _EW(self):
        """
        Applies Equal Weights (EW) strategy by assigning a fixed weight of 1 to the positive class
        and 0 to the negative class for each class.

        Returns:
            None
        """

        for i in range(self.num_classes):
            self.class_weight_dicts.append({1: 1, 0: 0})

    def _ICF(self):
        """
        Applies the Inverse Class Frequency (ICF) strategy to compute class weights.

        This method calculates weights inversely proportional to the frequency of each class
        across all samples, then normalizes the weights so they sum to 1. Each class receives
        a weight dictionary with separate weights for the positive (1) and negative (0) class labels.

        Returns:
            None
        """

        labels_array = np.array(self.labels)
        class_counts = labels_array.sum(axis=0)
        total_samples = len(self.labels)
        class_frequencies = (class_counts / total_samples) * 100
        result = []

        for i, freq in enumerate(class_frequencies):
            result.append(1 / freq)

        result = np.array(result)

        result = result * (1 / np.sum(result))

        for i in range(self.num_classes):
            self.class_weight_dicts.append(
                {1: round(float(result[i]), 3), 0: round(float(1 - result[i]), 3)}
            )

    def _RBF(self):
        """
        Applies the Radial Basis Function (RBF)-inspired strategy to compute class weights.

        This method ranks classes by frequency and reassigns their frequencies in reverse order
        to promote less frequent classes. Each class receives a weight dictionary with weights
        for the positive (1) and negative (0) class labels.

        Returns:
            None
        """

        labels_array = np.array(self.labels)
        class_counts = labels_array.sum(axis=0)
        total_samples = len(self.labels)
        class_frequencies = class_counts / total_samples

        sorted_values = sorted(
            range(len(class_frequencies)),
            key=lambda x: class_frequencies[x],
            reverse=True,
        )
        sorted_frequencies = sorted(class_frequencies)

        result = [0] * len(class_frequencies)
        for i, idx in enumerate(sorted_values):
            result[idx] = round(float(sorted_frequencies[i]), 3)

        for i in range(self.num_classes):
            self.class_weight_dicts.append(
                {1: result[i], 0: round(float(1 - result[i]), 3)}
            )

    def get_class_weight_dicts(self):
        """
        Returns the list of class weight dictionaries computed by the selected weighting strategy.

        Each dictionary in the list corresponds to a class and contains weights for:
            - 1: the positive class
            - 0: the negative class

        Returns:
            List[Dict[int, float]]: A list of dictionaries mapping class labels to their weights.
        """

        return self.class_weight_dicts
