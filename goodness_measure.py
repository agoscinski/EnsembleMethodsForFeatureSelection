from data_sets import DataSets
from robustness_measure import Measure
from abc import ABCMeta, abstractmethod
import numpy as np


class RankData:
    def __init__(self, features_rank, n_significant_features, n_indices):
        self.features_rank = features_rank
        self.sorted_indices = np.argsort(features_rank)[::-1]
        self.n_significant = n_significant_features
        self.n_indices = n_indices

    def __len__(self):
        return len(self.features_rank)

    @property
    def true_positive(self):
        return (self.sorted_indices[:self.n_indices] < self.n_significant).sum()

    @property
    def false_positive(self):
        return (self.sorted_indices[:self.n_indices] >= self.n_significant).sum()

    @property
    def true_negative(self):
        return (self.sorted_indices[self.n_indices:] >= self.n_significant).sum()

    @property
    def false_negative(self):
        return (self.sorted_indices[self.n_indices:] < self.n_significant).sum()


class GoodnessMeasure(Measure, metaclass=ABCMeta):
    def __init__(self, data_set_name, n_indices=None):
        super().__init__()
        feature_probe_labels = DataSets.load_features_labels(data_set_name)
        if feature_probe_labels is None:
            self.n_significant_features = None
        else:
            self.n_significant_features = np.sum([feature_probe_labels == 1])
        self.n_indices = self.n_significant_features if n_indices is None else n_indices

    def measures(self, features_ranks):
        if not self.n_significant_features:
            return 0

        goodness = []
        for i in range(features_ranks.shape[1]):
            goodness.append(self.goodness(
                RankData(features_ranks[:, i].T, self.n_significant_features, self.n_indices)
            ))
        return np.array(goodness)

    @abstractmethod
    def goodness(self, data: RankData):
        pass


class Dummy(GoodnessMeasure):
    def __init__(self, *args, n_significant_features=None, **kwargs):
        super().__init__(*args, **kwargs)
        if n_significant_features is not None:
            self.n_significant_features = n_significant_features

    def goodness(self, data: RankData):
        return data.features_rank[0]


class Accuracy(GoodnessMeasure):
    def goodness(self, data: RankData):
        return (data.true_negative + data.true_positive) / len(data)


class Precision(GoodnessMeasure):
    def goodness(self, data: RankData):
        return data.true_positive / data.n_significant


class XPrecision(GoodnessMeasure):
    def goodness(self, data: RankData):
        p = 0
        alpha = 0.5
        n = data.n_significant
        for i in range(data.sorted_indices.shape[0] // n):
            positives = (data.sorted_indices[n * i: n * (i+1)] < n).sum() / n
            p += alpha ** i * positives
        return p


