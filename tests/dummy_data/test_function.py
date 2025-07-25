import numpy as np

def my_function(labels):
    labels_array = np.array(labels)
    class_weight_dicts = []

    for i in range(len(labels_array[0])):
        class_weight_dicts.append({
            1: 0.1,
            0: 0.9
        })

    return class_weight_dicts
