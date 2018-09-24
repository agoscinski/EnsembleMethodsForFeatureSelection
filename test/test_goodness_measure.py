from goodness_measure import *


class TestRankData:
    data = [8, 0, 1, 2, 3, 4, 5, 6, 7]
    rank_data = RankData(data, 4, 4)

    def test_true_positive(self):
        assert 1 == self.rank_data.true_positive

    def test_false_positive(self):
        assert 3 == self.rank_data.false_positive

    def test_true_negative(self):
        assert 2 == self.rank_data.true_negative

    def test_false_negative(self):
        assert 3 == self.rank_data.false_negative


class TestGoodnessMeasure:
    goodness_measure = Dummy("", n_significant_features=2)

    def test_measures(self):
        features_ranks = np.array([
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
            [3, 4, 1, 2, 0]
        ]).T

        expected_result = [0, 4, 3]

        assert expected_result == self.goodness_measure.measures(features_ranks).tolist()


class TestAccuracy:
    accuracy = Accuracy(4)

    def test_goodness(self):
        rank_data = RankData([8, 0, 1, 2, 3, 4, 5, 6, 7], 4, 4)
        expected_result = (1 + 2)/9

        assert expected_result == self.accuracy.goodness(rank_data)
