from feature_selector import SymmetricalUncertainty, Relief, SVM_RFE, LassoFeatureSelector, Random, FeatureSelector
import ensemble_methods
import analysis
import artificial_data
import numpy as np
import matplotlib.pyplot as plt
from data_sets import DataSets
import itertools
from experiments import Experiment

import warnings

warnings.filterwarnings('ignore')


# GENERATION OF DATA SET

# total_features = 1e4
# n_significant_features = 100
# DataSets.save_artificial(
#     *artificial_data.generate(
#         n_samples=300,
#         n_features=total_features,
#         n_significant_features=n_significant_features,
#         feature_distribution=artificial_data.multiple_distribution(
#             distributions=[
#                 artificial_data.multivariate_normal(
#                     mean=artificial_data.constant(0),
#                     cov=artificial_data.uniform(0, 1)
#                 ),
#                 artificial_data.normal(-1, 1)
#             ],
#             shares=[0.5, 0.5]
#         ),
#         insignificant_feature_distribution=artificial_data.multiple_distribution(
#             distributions=[
#                 artificial_data.multivariate_normal(
#                     mean=artificial_data.constant(0),
#                     cov=artificial_data.uniform(0, 1)
#                 ),
#                 artificial_data.normal(0, 1)
#             ],
#             shares=[0.5, 0.5]
#         ),
#         labeling=artificial_data.linear_labeling(weights=np.ones(n_significant_features))
#     )
# )

# data, _ = DataSets.load("artificial")
# cov = np.cov(data[:200])
# plt.imshow(cov, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()
# plt.clf()


def compare_feature_selectors(p):
    feature_selectors = [
        SymmetricalUncertainty(),
        Relief(),
        SVM_RFE(percentage_features_to_select=p),
    ]

    e_methods = [
        ensemble_methods.Mean(data_set_feature_selectors=feature_selectors),
        ensemble_methods.MeanNormalizedSum(data_set_feature_selectors=feature_selectors),
        ensemble_methods.MeanWithClassifier(data_set_feature_selectors=feature_selectors, classifiers=analysis.default_classifiers),
    ]

    fs = feature_selectors + [LassoFeatureSelector(), Random()] + e_methods

    data_sets = ["colon", "arcene", "dexter", "gisette"]

    # analysis.artificial(fs, jaccard_percentage=p)

    analysis.run(data_sets, fs, jaccard_percentage=p)


def combinations():
    fs = [
        SymmetricalUncertainty(),
        Relief(),
        SVM_RFE(),
        LassoFeatureSelector(),
    ]

    e_methods = [
        ensemble_methods.Mean(data_set_feature_selectors=fs),
        ensemble_methods.Influence(data_set_feature_selectors=fs),
        ensemble_methods.MeanNormalizedSum(data_set_feature_selectors=fs),
        ensemble_methods.MeanWithClassifier(
            data_set_feature_selectors=fs,
            classifiers=analysis.default_classifiers
        ),
        ensemble_methods.InfluenceWithClassifier(
            data_set_feature_selectors=fs,
            classifiers=analysis.default_classifiers
        ),
        ensemble_methods.MeanNormWithClassifier(
            data_set_feature_selectors=fs,
            classifiers=analysis.default_classifiers
        ),
    ]

    for comb in itertools.combinations(list(range(4)), 3):
        comb_fs = [fs[i] for i in comb]
        e_methods.extend([
            ensemble_methods.Mean(data_set_feature_selectors=comb_fs),
            ensemble_methods.Influence(data_set_feature_selectors=comb_fs),
            ensemble_methods.MeanNormalizedSum(data_set_feature_selectors=comb_fs),
            ensemble_methods.MeanWithClassifier(
                data_set_feature_selectors=comb_fs,
                classifiers=analysis.default_classifiers
            ),
            ensemble_methods.InfluenceWithClassifier(
                data_set_feature_selectors=comb_fs,
                classifiers=analysis.default_classifiers
            ),
            ensemble_methods.MeanNormWithClassifier(
                data_set_feature_selectors=comb_fs,
                classifiers=analysis.default_classifiers
            ),
        ])

    for comb in itertools.combinations(list(range(4)), 2):
        comb_fs = [fs[i] for i in comb]
        e_methods.extend([
            ensemble_methods.Mean(data_set_feature_selectors=comb_fs),
            ensemble_methods.Influence(data_set_feature_selectors=comb_fs),
            ensemble_methods.MeanNormalizedSum(data_set_feature_selectors=comb_fs),
            ensemble_methods.MeanWithClassifier(
                data_set_feature_selectors=comb_fs,
                classifiers=analysis.default_classifiers
            ),
            ensemble_methods.InfluenceWithClassifier(
                data_set_feature_selectors=comb_fs,
                classifiers=analysis.default_classifiers
            ),
            ensemble_methods.MeanNormWithClassifier(
                data_set_feature_selectors=comb_fs,
                classifiers=analysis.default_classifiers
            ),
        ])

    data_sets = ["artificial", "colon", "arcene", "dexter", "gisette"]

    analysis.run(data_sets, fs + e_methods, prefix="combinations")


def combination_plot():
    from tabulate import tabulate

    accuracy = np.load('../results/RAW/combinations_jc10_accuracy.npy')
    robustness = np.load('../results/RAW/combinations_jc10_robustness.npy')
    labels = []
    with open('../results/RAW/combinations_jc10_accuracy_1.txt') as f:
        for label in f:
            labels.append(label.strip())

    beta = 2 * accuracy.mean(axis=-1) * robustness.mean(axis=-1) / (robustness.mean(axis=-1) + accuracy.mean(axis=-1))
    beta = beta.reshape(4, 70, 1, beta.shape[-1])

    methods_label = [
        "SU",
        "RLF",
        "SVM",
        "LSO",
        "Avg",
        "Inf",
        "AvgNormed",
        "Avg_C",
        "Inf_C",
        "AvgNormed_C",
    ]

    combinations_label = [
        "SU",
        "RLF",
        "SVM",
        "LSO",
        "SU+RLF+SVM+LSO",
        "SU+RLF+SVM",
        "SU+RLF+LSO",
        "SU+SVM+LSO",
        "RLF+SVM+LSO",
        "SU+RLF",
        "SU+SVM",
        "SU+LSO",
        "RLF+SVM",
        "RLF+LSO",
        "SVM+LSO"
    ]

    data = [
        accuracy,
        robustness,
        beta
    ]

    mean = list(map(
        lambda a: a.mean(axis=(0, -2, -1)),
        data
    ))

    mean_error = list(map(
        lambda a: a.std(axis=(-1)).mean(axis=(0, -1)),
        data
    ))

    comb = list(map(
        lambda a: np.hstack((
            a[:, :4].mean(axis=(0, -2, -1)),
            a[:, 4:].reshape(4, 11, 6, a.shape[-2], a.shape[-1]).mean(axis=(0, -3, -2, -1))
        )),
        data
    ))

    comb_error = list(map(
        lambda a: np.hstack((
            a[:, :4].std(axis=(-1)).mean(axis=(0, -1)),
            a[:, 4:].reshape(4, 11, 6, a.shape[-2], a.shape[-1]).std(axis=(-1)).mean(axis=(0, -2, -1))
        )),
        data
    ))

    meth = list(map(
        lambda a: np.hstack((
            a[:, :4].mean(axis=(0, -2, -1)),
            a[:, 4:].reshape(4, 11, 6, a.shape[-2], a.shape[-1])[:, [1, 5, 6, 8]].mean(axis=(0, 1, -2, -1))
        )),
        data
    ))

    meth_error = list(map(
        lambda a: np.hstack((
            a[:, :4].std(axis=(-1)).mean(axis=(0, -1)),
            a[:, 4:].reshape(4, 11, 6, a.shape[-2], a.shape[-1])[:, [1, 5, 6, 8]].std(axis=(-1)).mean(axis=(0, 1, -1))
        )),
        data
    ))

    def sprint(order, mean, std, header):
        rows = [
            ["accuracy"] + list(map(lambda m, s: "{:.2%} ± {:.2%}".format(m, s), mean[0][order].tolist(), std[0][order].tolist())),
            ["robustness"] + list(map(lambda m, s: "{:.2%} ± {:.2%}".format(m, s), mean[1][order].tolist(), std[1][order].tolist())),
            ["beta"] + list(map(lambda m, s: "{:.2%} ± {:.2%}".format(m, s), mean[2][order].tolist(), std[2][order].tolist())),
        ]
        print(tabulate(rows, ["Measure"] + [header[i] for i in order], tablefmt='pipe'))
        print()

    def table(mean, std, header):
        order = list(map(
            lambda a: np.argsort(a)[::-1],
            mean
        ))

        print("SORTED BY ACCURACY")
        sprint(order[0], mean, std, header)
        print("SORTED BY ROBUSTNESS")
        sprint(order[1], mean, std, header)
        print("SORTED BY BETA")
        sprint(order[2], mean, std, header)

        x = np.arange(mean[0].shape[0])
        for i in range(3):
            plt.figure(num=["accuracy", "robustness", "beta"][i])
            plt.errorbar(x, mean[i][order[i]], yerr=std[i][order[i]])
            plt.xticks(x, [header[k] for k in order[i]], rotation=-90)
            axes = plt.gca()
            axes.set_xlim([-1, x.size])
            vals = axes.get_yticks()
            axes.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])
            plt.tight_layout()

    # print("DATA\n")
    # table(mean, mean_error, labels)
    # print("\nCOMBINATIONS\n")
    # table(comb, comb_error, combinations_label)
    print("\nMETHODS\n")
    table(meth, meth_error, methods_label)

    plt.figure()
    plt.hist(mean[0], label="accuracy")
    plt.hist(mean[1], bins=20, label="robustness")
    plt.hist(mean[2], bins=20, label="beta")
    plt.legend()
    plt.tight_layout()
    plt.show()

