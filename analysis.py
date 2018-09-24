from experiments import DataSetExperiment
from benchmarks import MeasureBenchmark, AccuracyBenchmark
from sklearn.neighbors import KNeighborsClassifier
from sklearn_utilities import SVC_Grid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
import robustness_measure
import goodness_measure
from feature_selector import DummyFeatureSelector

default_classifiers = [
    KNeighborsClassifier(3),
    SVC_Grid(kernel="linear"),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    LogisticRegressionCV(penalty='l1', solver='liblinear')
]


def run(data_sets, feature_selectors, jaccard_percentage=0.01, classifiers=None,
        measures=None, save=True, prefix=""):
    if isinstance(data_sets, str):
        data_sets = [data_sets]

    if classifiers is None:
        classifiers = default_classifiers

    if measures is None:
        measures = [
            robustness_measure.JaccardIndex(percentage=jaccard_percentage)
        ]

    if len(prefix) > 0:
        prefix += "_"

    robustness_exp = DataSetExperiment(
        MeasureBenchmark(measures),
        feature_selectors
    )

    accuracy_exp = DataSetExperiment(
        AccuracyBenchmark(classifiers, percentage_of_features=jaccard_percentage),
        feature_selectors
    )

    jcp = int(jaccard_percentage * 1e3)
    robustness_exp.run(data_sets)
    if save:
        robustness_exp.save_results(prefix + "jc{}_robustness".format(jcp))

    accuracy_exp.run(data_sets)
    if save:
        accuracy_exp.save_results(prefix + "jc{}_accuracy".format(jcp))


def artificial(feature_selectors, jaccard_percentage=0.01, save=True, classifiers=None):
    if classifiers is None:
        classifiers = default_classifiers

    robustness_exp = DataSetExperiment(
        MeasureBenchmark([
            robustness_measure.JaccardIndex(percentage=jaccard_percentage)
        ]),
        feature_selectors
    )

    precision_exp = DataSetExperiment(
        MeasureBenchmark([
            goodness_measure.Precision("artificial", 100),
            goodness_measure.Precision("artificial", 200),
            goodness_measure.Precision("artificial", 300),
            goodness_measure.Precision("artificial", 400),
            goodness_measure.Precision("artificial", 500),
            goodness_measure.Precision("artificial", 600),
            goodness_measure.Precision("artificial", 700),
            goodness_measure.Precision("artificial", 800),
            goodness_measure.Precision("artificial", 900),
            goodness_measure.Precision("artificial", 1000),
            goodness_measure.XPrecision("artificial")
        ]),
        feature_selectors
    )

    accuracy_exp = DataSetExperiment(
        AccuracyBenchmark(classifiers, percentage_of_features=jaccard_percentage),
        feature_selectors
    )

    jcp = int(jaccard_percentage * 1e3)
    robustness_exp.run("artificial")
    if save:
        robustness_exp.save_results("artificial_jc{}_robustness".format(jcp))

    precision_exp.run("artificial")
    if save:
        precision_exp.save_results("artificial_jc{}_precision".format(jcp))

    accuracy_exp.run("artificial")
    if save:
        accuracy_exp.save_results("artificial_jc{}_accuracy".format(jcp))


def accuracy_with_all_features(data_sets, classifiers=None):
    if classifiers is None:
        classifiers = default_classifiers

    if isinstance(data_sets, str):
        data_sets = [data_sets]

    accuracy_exp = DataSetExperiment(
        AccuracyBenchmark(classifiers, percentage_of_features=100),
        DummyFeatureSelector()
    )

    accuracy_exp.run(data_sets)
    accuracy_exp.save_results("all_features")
