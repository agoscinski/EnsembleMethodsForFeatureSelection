from feature_selector import *
import math
import numpy as np


class TestFeatureSelector:
    def test_normalize(self):
        a = np.arange(5)
        expected_result = np.arange(5) / 4

        assert expected_result.tolist() == FeatureSelector.normalize(a).tolist()

    def test_rank_weights(self):
        weights = np.array([0.5, 0.1, 0.5, 0.6])
        ranked_weights = FeatureSelector.rank_weights(weights).tolist()

        assert [3., 1., 2., 4.] == ranked_weights or [2., 1., 3., 4.] == ranked_weights


class TestSymmetricalUncertainty:
    data = np.array([[1, 2, 3],
                     [0, 0, 0],
                     [1, 0, 1]])
    classes = np.array([1, 0, 1])

    def test_weight_features(self):
        h_f = -3 * 1. / 3 * math.log(1. / 3)
        h_c = -1. / 3 * math.log(1. / 3) - 2. / 3 * math.log(2. / 3)
        h_fc = 2 * 1. / 3 * math.log((2. / 3) / (1. / 3)) + 1. / 3 * math.log((1. / 3) / (1. / 3))
        su = 2 * (h_f - h_fc) / (h_f + h_c)
        su_ranker = SymmetricalUncertainty()
        assert np.allclose(su, su_ranker.weight(self.data, self.classes)[0])


# class TestRelief:

class TestSVM_RFE:
    reversed_ranks = np.array([1, 3, 2, 1, 1, 3, 4])

    # TODO def test_find_hyperparameter_with_grid_search_cv(self):

    def test_reverse_order(self):
        svm_rfe = SVM_RFE()
        correctly_ordered_ranks = svm_rfe.reverse_order(
            self.reversed_ranks)
        assert np.allclose([4, 2, 3, 4, 4, 2, 1], correctly_ordered_ranks)
