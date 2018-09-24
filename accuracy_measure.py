import numpy as np


def ber(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels, number = np.unique(y_true, return_counts=True)
    acc = [np.sum(np.logical_and([labels[i] == y_true], [labels[i] == y_pred]))/number[i] for i in range(len(labels))]
    return np.mean(acc)
