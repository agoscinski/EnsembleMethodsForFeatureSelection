import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from data_sets import Analysis, PreComputedData, DataSets
import pandas as pd
import numpy as np
from feature_selector import FeatureSelector
from sklearn.cross_validation import ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import errno


class AnalyseBenchmarkResults():
    def __init__(self, feature_selector: FeatureSelector = None):
        self.feature_selector = feature_selector

        if not isinstance(feature_selector, list):
            self.feature_selectors = [feature_selector]
        else:
            self.feature_selectors = feature_selector

    def run(self, data_set, features_to_filter=0.01, save_to_file=False):
        for i in range(len(self.feature_selectors)):
            analysis = AnalyseFeatureSelection(self.feature_selectors[i], features_to_filter, save_to_file)
            analysis.generate(
                data_set,
                self.cv())

    @staticmethod
    def cv():
        return ShuffleSplit(0)


class AnalyseFeatureSelection:
    def __init__(self, feature_selector: FeatureSelector, features_to_filter, save_to_file=False):
        self.feature_selector = feature_selector
        self.features_to_filter = features_to_filter
        self.save_to_file = save_to_file

    def generate(self, data_set, cv):
        data, labels = DataSets.load(data_set)
        weights = PreComputedData.load(data_set, cv, "weight", self.feature_selector)
        ranks = PreComputedData.load(data_set, cv, "rank", self.feature_selector)
        stats, fig_hist_and_box, _, _ = AnalyseWeights.analyse_weights(weights.T)
        #fig_pca, fig_tsne = Analyse2D.analyse_2d(data, labels, ranks, self.features_to_filter)

        self.update_weights_plots(stats, fig_hist_and_box)
        #self.update_pca_plot(fig_pca)
        #self.update_tsne_plot(fig_tsne)
        plt.show()
        print(stats)

        if self.save_to_file:
            file_name = Analysis.file_name(data_set, cv, "weight", self.feature_selector)
            AnalyseFeatureSelection.create_directory(Analysis.dir_name(data_set, cv, "weight"))
            AnalyseFeatureSelection.save_weights_data(stats, fig_hist_and_box, file_name)

    def update_weights_plots(self, stats, fig):
        fig.suptitle("Weight analysis for " + self.feature_selector.__name__, fontsize=14, fontweight='bold')
        fig.subplots_adjust(top=0.9)

    def update_pca_plot(self, fig):
        fig.suptitle("PCA for " + self.feature_selector.__name__, fontsize=14, fontweight='bold')
        fig.subplots_adjust(top=0.95)

    def update_tsne_plot(self, fig):
        fig.suptitle("TSNE for " + self.feature_selector.__name__, fontsize=14, fontweight='bold')
        fig.subplots_adjust(top=0.95)

    @staticmethod
    def save_weights_data(stats, fig, file_name):
        fig.savefig(file_name + '.png')
        stats.to_csv(file_name + '.csv')

    @staticmethod
    def create_directory(directory):
        try:
            os.makedirs(directory)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise


class AnalyseWeights:
    # shape: (features x samples)
    @staticmethod
    def analyse_weights(weights):
        column_names = ['S' + str(s) for s in range(weights.shape[1])]
        weights_df = pd.DataFrame(weights, columns=column_names)
        weights_mean = weights_df.T.mean()

        stats = AnalyseWeights.weights_stats(weights_df, weights_mean)
        fig_hist_and_box = AnalyseWeights.weights_hist(weights_df, weights_mean)
        fig_plot = AnalyseWeights.weights_plot(weights, weights_mean, sorted=False)
        fig_plot_sorted = AnalyseWeights.weights_plot(weights, weights_mean, sorted=True)
        return stats, fig_hist_and_box, fig_plot, fig_plot_sorted
    @staticmethod
    def weights_stats(weights, weights_mean):
        weights_mean_df = pd.DataFrame(weights_mean, columns=['mean'])
        stats_df = pd.concat([weights, weights_mean_df], axis=1)
        stats_matrix = stats_df.as_matrix()
        n_unique_values = [len(np.unique(stats_matrix[:, i])) for i in range(stats_matrix.shape[1])]
        unique_df = pd.DataFrame(n_unique_values, columns=['unique']).T
        unique_df.columns = ['S' + str(s) for s in range(weights.shape[1])] + ['mean']
        stats = stats_df.describe().append(unique_df)
        return stats

    @staticmethod
    def weights_hist(weights, weights_mean):
        fig = plt.figure(figsize=(15, 10))
        sample_size = weights.shape[1]
        gs = GridSpec(round(sample_size / 3 + 0.5), 6)

        ax = fig.add_subplot(gs[:3, 0:3])
        AnalyseWeights.plot_boxplot(weights, ax)

        for i in range(sample_size):
            ax = fig.add_subplot(gs[int(i / 3), 3 + (i % 3)])
            AnalyseWeights.plot_hist(weights, weights_mean, ax, i)

        fig.tight_layout()
        return fig

    @staticmethod
    def weights_plot(weights, weights_mean, sorted):
        fig = plt.figure(figsize=(15, 20))
        plot_rows = len(weights.T) + 2 // 2
        weights_mean_processed = np.sort(weights_mean) if sorted else weights_mean
        for i in range(len(weights.T)):
            weights_processed = np.sort(weights.T[i]) if sorted else weights.T[i]
            ax = plt.subplot(plot_rows , 2, i+1)
            if not sorted:
                ax.plot(weights_processed,  markersize='4', marker='o', linestyle='None')
            else:
                ax.plot(weights_processed, linewidth=2)


        ax = plt.subplot(plot_rows, 2, i + 2)
        if not sorted:
            ax.plot(weights_mean_processed, c='orange', markersize='4', marker='o', linestyle='None')
        else:
            ax.plot(weights_mean_processed, c='orange', linewidth='2')

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_boxplot(weights, ax):
        meanlineprops = dict(linestyle='-', linewidth=1.5, color='purple')
        weights.boxplot(ax=ax, return_type='axes', meanprops=meanlineprops,
                        meanline=True, showmeans=True, notch=True, showfliers=False)

    @staticmethod
    def plot_hist(weights, weights_mean, ax, sample_index):
        n_bins = 50
        max_xticks = 4
        max_yticks = 5

        weights['S' + str(sample_index)].plot.hist(ax=ax, color='green', alpha=0.5, bins=n_bins)
        weights_mean.plot.hist(ax=ax, color='orange', alpha=0.5, bins=n_bins)
        ax.set_ylabel('')
        xloc = plt.MaxNLocator(max_xticks)
        ax.xaxis.set_major_locator(xloc)
        yloc = plt.MaxNLocator(max_yticks)
        ax.yaxis.set_major_locator(yloc)


class Analyse2D:
    @staticmethod
    def analyse_2d(data, labels, ranks, features_to_filter):
        fig_pca = Analyse2D.pca_plot(data, labels, ranks, features_to_filter)
        fig_tsne = Analyse2D.tsne_plot(data, labels, ranks, features_to_filter)
        return fig_pca, fig_tsne

    @staticmethod
    def select_p_features(data, ranks, p):
        ranks_args_sorted_descending = np.argsort(ranks)[::-1]
        num_features_to_select = int(len(ranks) * p)
        data_filtered = data[ranks_args_sorted_descending[:num_features_to_select]]
        return data_filtered

    @staticmethod
    def pca_plot(data, labels, ranks, features_to_filter):
        pca = PCA()
        transformed_data = pca.fit_transform(data.T).T[:2]

        number_of_plots = len(ranks) + 1

        fig = plt.figure(figsize=(15, 20))
        ax = plt.subplot(round(number_of_plots / 2.), 2, 1)
        ax.set_title("With all features")
        ax.scatter(*transformed_data, c=labels, cmap="viridis")

        for i in range(len(ranks)):
            data_filtered = Analyse2D.select_p_features(data, ranks[i], p=features_to_filter)
            transformed_data_filtered = pca.fit_transform(data_filtered.T).T[:2]

            ax = plt.subplot(round(number_of_plots / 2.), 2, i + 2)
            ax.set_title("With filtered features using ranks from S" + str(i))
            ax.scatter(*transformed_data_filtered, c=labels, cmap="viridis")
        return fig

    @staticmethod
    def tsne_plot(data, labels, ranks, features_to_filter):
        tsne = TSNE()
        transformed_data = tsne.fit_transform(data.T).T[:2]

        number_of_plots = len(ranks) + 1

        fig = plt.figure(figsize=(15, 20))
        ax = plt.subplot(round(number_of_plots / 2.), 2, 1)
        ax.set_title("With all features")
        ax.scatter(*transformed_data, c=labels, cmap="viridis")

        for i in range(len(ranks)):
            data_filtered = Analyse2D.select_p_features(data, ranks[i], p=features_to_filter)
            transformed_data_filtered = tsne.fit_transform(data_filtered.T).T[:2]

            ax = plt.subplot(round(number_of_plots / 2.), 2, i + 2)
            ax.set_title("With filtered features using ranks from S" + str(i))
            ax.scatter(*transformed_data_filtered, c=labels, cmap="viridis")
        return fig
