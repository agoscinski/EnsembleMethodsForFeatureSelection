import warnings
import numpy as np
from sklearn.utils import check_X_y, safe_sqr
from sklearn.base import clone
import sklearn.feature_selection
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC


class SVC_Grid(SVC):
    """
        SVC from scikit-learn with integrated Grid Search
    """
    def fit(self, data, labels, sample_weight=None):
        grid_search = GridSearchCV(
            SVC(),
            {
                "C": [1, 10, 100, 1000]
            },
            cv=5,
            scoring='precision'
        )
        grid_search.fit(data, labels)
        self.C = grid_search.best_params_["C"]

        super(SVC, self).fit(data, labels, sample_weight)


class RFE(sklearn.feature_selection.RFE):
    """
        RFE from scikit-learn with stepwise feature selection.

        If enabled:
        At each iteration (step * count of remaining features) are discarded
        instead of (step * total count of features)
    """
    def __init__(self, *args, stepwise_selection=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.stepwise_selection = stepwise_selection

    def _fit(self, X, y, step_score=None):
        X, y = check_X_y(X, y, "csc")
        # Initialization
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select

        if 0.0 < self.step < 1.0:
            if not self.stepwise_selection:
                step = int(max(1, self.step * n_features))
            else:
                step = self.step    
        else:
            if self.stepwise_selection:
                warnings.warn("The parameter 'stepwise_selection' is true but "
                              "a fixed step size is given. Procedure will "
                              " continue as if 'stepwise_selection' is false",
                              RuntimeWarning)
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        if self.estimator_params is not None:
            warnings.warn("The parameter 'estimator_params' is deprecated as "
                          "of version 0.16 and will be removed in 0.18. The "
                          "parameter is no longer necessary because the value "
                          "is set via the estimator initialisation or "
                          "set_params method.", DeprecationWarning)

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        # Elimination
        while np.sum(support_) > n_features_to_select:
            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            estimator = clone(self.estimator)
            if self.estimator_params:
                estimator.set_params(**self.estimator_params)
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))

            estimator.fit(X[:, features], y)

            # Get coefs
            if hasattr(estimator, 'coef_'):
                coefs = estimator.coef_
            elif hasattr(estimator, 'feature_importances_'):
                coefs = estimator.feature_importances_
            else:
                raise RuntimeError('The classifier does not expose '
                                   '"coef_" or "feature_importances_" '
                                   'attributes')

            # Get ranks
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
            else:
                ranks = np.argsort(safe_sqr(coefs))

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            if self.stepwise_selection and 0.0 < step < 1.0:
                current_step_size = int(np.sum(support_) * step)
            else:
                current_step_size = step 
            threshold = min(current_step_size, np.sum(support_) - n_features_to_select)

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        if self.estimator_params:
            self.estimator_.set_params(**self.estimator_params)
        self.estimator_.fit(X[:, features], y)

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self
