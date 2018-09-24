import numpy as np
from sklearn.cross_validation import KFold, ShuffleSplit
from abc import ABCMeta, abstractmethod
import multiprocessing
import ctypes
from feature_selector import FeatureSelector
from robustness_measure import Measure, JaccardIndex
from sklearn.base import clone as clone_classifier
from collections import Iterable
from accuracy_measure import ber


class Benchmark(metaclass=ABCMeta):
    feature_selector = None

    def generate_features_selection(self, data, labels):
        if not isinstance(self.feature_selector, FeatureSelector):
            raise TypeError("feature_selector needs to be defined")

        return self.feature_selector.generate(data, labels, self.cv(labels.shape[0]), "rank")

    @staticmethod
    def cv(sample_size):
        pass

    # Returns mean results for each measure
    def run(self, *args, **kwargs):
        mean_results = []
        for i, measure_results in enumerate(self.run_raw_result(*args, **kwargs)):
            mean_results.append(np.mean(measure_results))

        return np.array(mean_results)

    @abstractmethod
    # Returns an Iterable, one item per measures with all the results associated with it
    def run_raw_result(self, data, labels, features_selection=None) -> Iterable:
        pass

    @abstractmethod
    def get_measures(self):
        pass


class MeasureBenchmark(Benchmark):
    def __init__(self, measure, feature_selector: FeatureSelector = None):
        self.feature_selector = feature_selector

        if not isinstance(measure, list):
            measure = [measure]

        for robustness_measure in measure:
            if not isinstance(robustness_measure, Measure):
                raise ValueError("At least one robustness measure does not inherit RobustnessMeasure")

        self.measures = measure

    def run_raw_result(self, data, labels, features_selection=None):
        if features_selection is None:
            features_selection = self.generate_features_selection(data, labels)

        features_selection = np.array(features_selection).T

        measures_results = multiprocessing.Manager().dict()

        processes = []
        for i in range(len(self.measures)):
            p = multiprocessing.Process(
                target=self.measures[i].run_and_set_in_results,
                kwargs={
                    'features_selection': features_selection,
                    'results': measures_results,
                    'result_index': i
                }
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        return [measure_results for _, measure_results in measures_results.items()]

    @staticmethod
    def cv(sample_length):
        return ShuffleSplit(sample_length, n_iter=10, test_size=0.1)

    def get_measures(self):
        return self.measures


class ClassifierWrapper:
    def __init__(self, classifier, accuracy_measure):
        self.classifier = classifier
        self.__name__ = type(classifier).__name__
        self.accuracy_measure = accuracy_measure

    def run_and_set_in_results(self, data, labels, train_index, test_index, results, result_index):
        np.random.seed()
        classifier = clone_classifier(self.classifier)

        classifier.fit(
            data[:, train_index].T,
            labels[train_index]
        )

        results[result_index] = self.accuracy_measure(
            labels[test_index],
            classifier.predict(data[:, test_index].T)
        )


class AccuracyBenchmark(Benchmark):
    percentage_of_features = 0.01
    n_fold = 10

    def __init__(self, classifiers, feature_selector: FeatureSelector = None, percentage_of_features=None,
                 accuracy_measure=ber
                 ):
        self.feature_selector = feature_selector

        if percentage_of_features is not None:
            self.percentage_of_features = percentage_of_features

        if not isinstance(classifiers, list):
            classifiers = [classifiers]

        self.classifiers = [ClassifierWrapper(c, accuracy_measure) for c in classifiers]

    def run_raw_result(self, data, labels, features_selection=None):
        if features_selection is None:
            features_selection = self.generate_features_selection(data, labels)

        features_indexes = {}
        for i, ranking in enumerate(features_selection):
            features_indexes[i] = self.highest_percent(ranking, self.percentage_of_features)

        shape = (len(self.classifiers), AccuracyBenchmark.n_fold)
        shared_array_base = multiprocessing.Array(ctypes.c_double, shape[0] * shape[1])
        classification_accuracies = np.ctypeslib.as_array(shared_array_base.get_obj())
        classification_accuracies = classification_accuracies.reshape(shape)

        processes = []
        for i, classifier in enumerate(self.classifiers):
            for j, (train_index, test_index) in enumerate(self.cv(labels.shape[0])):
                p = multiprocessing.Process(
                    target=classifier.run_and_set_in_results,
                    kwargs={
                        'data': data[features_indexes[j], :],
                        'labels': labels,
                        'train_index': train_index,
                        'test_index': test_index,
                        'results': classification_accuracies,
                        'result_index': (i, j)
                    }
                )
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

        return classification_accuracies

    @staticmethod
    def cv(sample_length):
        return KFold(sample_length, n_folds=AccuracyBenchmark.n_fold)

    # 1% best features
    @staticmethod
    def highest_percent(features_selection, percentage):
        if percentage == 100:
            return np.arange(features_selection.size)
        size = 1 + int(features_selection.size * percentage)
        return np.argsort(features_selection)[:-size:-1]

    def get_measures(self):
        return self.classifiers


class FMeasureBenchmark:
    def __init__(self, classifiers, feature_selector: FeatureSelector = None, jaccard_percentage=0.01, beta=1):
        self.robustness_benchmark = MeasureBenchmark(
            JaccardIndex(percentage=jaccard_percentage),
            feature_selector=feature_selector
        )
        self.accuracy_benchmark = AccuracyBenchmark(
            classifiers,
            feature_selector=feature_selector,
            percentage_of_features=jaccard_percentage
        )
        self.beta = beta

    def run(self, data, labels, robustness_features_selection=None, accuracy_features_selection=None):
        return np.mean(self.f_measure(
            self.robustness_benchmark.run(data, labels, robustness_features_selection),
            self.accuracy_benchmark.run(data, labels, accuracy_features_selection),
            self.beta
        ))

    @staticmethod
    def f_measure(robustness, accuracy, beta=1):
        return ((beta ** 2 + 1) * robustness * accuracy) / (beta ** 2 * robustness + accuracy)
