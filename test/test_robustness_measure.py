from robustness_measure import *
import numpy as np


class TestSpearman:
    def test_spearman(self):
        features_rank = np.array([
            [1, 2, 3, 4],
            [2, 4, 1, 3],
            [1, 3, 4, 2]
        ])

        n = features_rank.shape[1]
        normalization_term = float(n * (n ** 2 - 1))
        s01 = ((features_rank[0] - features_rank[1]) ** 2 / normalization_term).sum()
        s02 = ((features_rank[0] - features_rank[2]) ** 2 / normalization_term).sum()
        s12 = ((features_rank[1] - features_rank[2]) ** 2 / normalization_term).sum()

        expected_spearman = (1 - 6 * np.array([s01, s02, s12])).tolist()

        spearman = Spearman()

        assert np.allclose(expected_spearman, spearman.measures(features_rank.T))


class TestJaccardIndex:
    def test_measures(self):
        jaccard = JaccardIndex(percentage=0.3)
        features_ranks = np.array([
            [1, 2, 3, 4, 5, 6, 7, 10, 9, 8],
            [2, 4, 3, 1, 10, 5, 7, 9, 8, 6],
            [1, 3, 4, 2, 7, 5, 9, 10, 6, 8]
        ])

        expected_result = [1/2, 1/2, 1/5]

        assert np.allclose(expected_result, jaccard.measures(features_ranks.T))
