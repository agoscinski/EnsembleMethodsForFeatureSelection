from benchmarks import MeasureBenchmark, AccuracyBenchmark, Benchmark, FMeasureBenchmark
from feature_selector import DataSetFeatureSelector
from tabulate import tabulate
import numpy as np
import csv
from data_sets import DataSets
from io_utils import mkdir


class Experiment:
    results = np.array([])
    row_labels = []
    col_labels = []

    @staticmethod
    def results_table(rows_label, cols_label, results):
        rows = [
            ["Measure"] + cols_label
        ]
        for i in range(results.shape[0]):
            row = [rows_label[i]]
            row += map(lambda i: "{:.2%}".format(i), results[i, :].tolist())
            rows.append(row)

        return rows

    @staticmethod
    def raw_results_table(rows_label, cols_label, mean, std):
        rows = [
            ["Measure"] + cols_label
        ]
        std /= 2
        for i in range(mean.shape[0]):
            row = [rows_label[i]]
            row += map(lambda m, s: "{:.2%} ± {:.2%}".format(m, s), mean[i, :].tolist(), std[i, :].tolist())
            rows.append(row)

        return rows

    def print_results(self):
        table = self.results_table(self.row_labels, self.col_labels, self.results)
        print(tabulate(table[1:len(table)], table[0], tablefmt='pipe'))
        print()

    def save_results(self, file_name="output.csv", append=False):
        root_dir = DataSets.root_dir + "/results/" + type(self).__name__
        table = self.results_table(self.row_labels, self.col_labels, self.results)

        mkdir(root_dir)

        with open(root_dir + "/" + file_name, 'a' if append else 'w') as f:
            writer = csv.writer(f)
            writer.writerows(table)


class MeasureExperiment(Experiment):
    def __init__(self, feature_selectors, measures):
        if not isinstance(measures, list):
            measures = [measures]

        if not isinstance(feature_selectors, list):
            feature_selectors = [feature_selectors]

        self.measures = measures
        self.feature_selectors = feature_selectors
        self.results = np.zeros((len(measures), len(feature_selectors)))

        self.row_labels = [type(r).__name__ for r in self.measures]
        self.col_labels = [f.__name__ for f in self.feature_selectors]

    def run(self, data, labels):
        for i in range(self.results.shape[1]):
            benchmark = MeasureBenchmark(
                measure=self.measures,
                feature_selector=self.feature_selectors[i]
            )
            self.results[:, i] = benchmark.run(data, labels)

        return self.results

    def print_results(self):
        print("Robustness Experiment : ")
        super().print_results()


class AccuracyExperiment(Experiment):
    measure_name = "classifiers"

    def __init__(self, feature_selectors, classifiers):
        if not isinstance(classifiers, list):
            classifiers = [classifiers]

        if not isinstance(feature_selectors, list):
            feature_selectors = [feature_selectors]

        results_shape = (len(classifiers), len(feature_selectors))

        self.classifiers = classifiers
        self.feature_selectors = feature_selectors
        self.results = np.zeros(results_shape)
        self.row_labels = [type(c).__name__ for c in self.classifiers]
        self.col_labels = [f.__name__ for f in self.feature_selectors]

    def run(self, data, labels):
        for i in range(self.results.shape[1]):
            benchmark = AccuracyBenchmark(
                classifiers=self.classifiers,
                feature_selector=self.feature_selectors[i]
            )
            self.results[:, i] = benchmark.run(data, labels)

        return self.results

    def print_results(self):
        print("Accuracy Experiment : ")
        super().print_results()


class DataSetExperiment:
    root_dir = DataSets.root_dir + "/results/RAW"

    def __init__(self, benchmark: Benchmark, data_set_feature_selectors):
        self.benchmark = benchmark

        if not isinstance(data_set_feature_selectors, list):
            data_set_feature_selectors = [data_set_feature_selectors]

        for data_set_feature_selector in data_set_feature_selectors:
            if not isinstance(data_set_feature_selector, DataSetFeatureSelector):
                raise ValueError("Only DataSetFeatureSelector can be used")

        self.feature_selectors = data_set_feature_selectors
        self.results = None
        self.data_sets = None

        self.row_labels = [m.__name__ for m in self.benchmark.get_measures()] + ["Mean"]
        self.col_labels = [f.__name__ for f in self.feature_selectors]

    def run(self, data_sets):
        self.results = []
        self.data_sets = [data_sets] if isinstance(data_sets, str) else data_sets
        bc_name = type(self.benchmark).__name__

        for i, data_set in enumerate(self.data_sets):
            data, labels = DataSets.load(data_set)
            result = []

            for feature_selector in self.feature_selectors:
                print("{}: {} [{}]".format(
                    bc_name,
                    data_set,
                    feature_selector.__name__
                ))

                result.append(self.benchmark.run_raw_result(
                    data,
                    labels,
                    feature_selector.rank_data_set(data_set, self.benchmark.cv)
                ))

            result = np.array(result)
            self.results.append(result)

            print("\n{}".format(data_set.upper()))
            self._print_result(result)

        print("{} done".format(bc_name))

        self.results = np.array(self.results)

        return self.results

    def _print_result(self, result):
        table = Experiment.raw_results_table(
            self.row_labels,
            self.col_labels,
            np.vstack((result.mean(axis=-1).T, result.mean(axis=(-1, -2)))),
            np.vstack((result.std(axis=-1).T, result.std(axis=(-1, -2)))),
        )
        print(tabulate(table[1:len(table)], table[0], tablefmt='pipe'))
        print()

    def save_results(self, filename=None):
        if filename is None:
            filename = type(self.benchmark).__name__

        mkdir(self.root_dir)

        np.save("{}/{}.npy".format(self.root_dir, filename), self.results)

        self.__write_dim_info(filename, 0, self.data_sets)
        self.__write_dim_info(filename, 1, [f.__name__ for f in self.feature_selectors])
        self.__write_dim_info(filename, 2, [m.__name__ for m in self.benchmark.get_measures()])

    def __write_dim_info(self, filename, dim, data):
        with open("{}/{}_{}.txt".format(self.root_dir, filename, dim), "w") as f:
            for d in data:
                f.write(d + "\n")


class EnsembleFMeasureExperiment(Experiment):
    def __init__(self, classifiers, data_set_feature_selectors, jaccard_percentage=0.01, beta=1):
        if not isinstance(data_set_feature_selectors, list):
            data_set_feature_selectors = [data_set_feature_selectors]

        for data_set_feature_selector in data_set_feature_selectors:
            if not isinstance(data_set_feature_selector, DataSetFeatureSelector):
                raise ValueError("Only DataSetFeatureSelector can be used")

        self.feature_selectors = data_set_feature_selectors
        self.classifiers = classifiers
        self.jaccard_percentage = jaccard_percentage
        self.beta = beta
        self.results = None

    def run(self, data_sets):
        self.results = np.zeros((len(data_sets) + 1, len(self.feature_selectors)))

        benchmark = FMeasureBenchmark(
            classifiers=self.classifiers,
            jaccard_percentage=self.jaccard_percentage,
            beta=self.beta,
        )

        len_fs = len(self.feature_selectors)
        size = len(data_sets) * len_fs

        for i, data_set in enumerate(data_sets):
            data, labels = DataSets.load(data_set)

            for j, feature_selector in enumerate(self.feature_selectors):
                print("Progress: {:.2%}".format((i * len_fs + j)/size))
                self.results[i, j] = benchmark.run(
                    data,
                    labels,
                    robustness_features_selection=feature_selector.rank_data_set(
                        data_set,
                        benchmark.robustness_benchmark.cv,
                    ),
                    accuracy_features_selection=feature_selector.rank_data_set(
                        data_set,
                        benchmark.accuracy_benchmark.cv
                    )
                )

        self.results[-1, :] = self.results[:-1].mean(axis=0)

        order = np.argsort(self.results[-1])[::-1]
        self.results = self.results[:, order]

        self.row_labels = data_sets + ["Mean"]
        self.col_labels = []
        for i in order:
            self.col_labels.append(self.feature_selectors[i].__name__)

        return self.results

    def print_results(self):
        print("Ensemble Method with {:.0%} features and beta={}".format(self.jaccard_percentage, self.beta))
        super().print_results()
