import numpy as np
from numpy.linalg import norm
from slope.utils import dual_norm_slope
from scipy import stats, sparse
from sklearn import preprocessing, feature_selection

from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "SLOPE"
    parameters = {'reg': [0.1, 0.01, 0.001],
                  'q': [0.2, 0.1, 0.05],
                  'standardize': [True, False]}

    def __init__(self, reg, q, standardize):
        self.q = q
        self.reg = reg
        self.standardize = standardize

    def set_data(self, X, y):
        # remove zero variance predictors
        X = feature_selection.VarianceThreshold().fit_transform(X)

        y -= np.mean(y)
        y /= np.linalg.norm(y) ** 2

        if self.standardize:
            if sparse.issparse(X):
                X = preprocessing.MaxAbsScaler().fit_transform(X).tocsc()
            else:
                X = preprocessing.StandardScaler().fit_transform(X)

                # NOTE(jolars): not sure if this is necessary, probably not
                if X.flags.c_contiguous:
                    X = np.array(X, order="F")

        self.X, self.y = X, y
        self.n_samples, self.n_features = self.X.shape
        self.alphas = self._get_lambda_seq()

    def compute(self, beta):
        # compute residuals
        X, y = self.X, self.y
        n_samples, n_features = X.shape

        diff = y - X @ beta

        # compute primal objective
        p_obj = (1. / (2 * n_samples) * diff @ diff
                 + np.sum(self.alphas * np.sort(np.abs(beta))[::-1]))

        # Compute dual
        theta = diff
        theta /= max(1, dual_norm_slope(X, theta, self.alphas))

        d_obj = (norm(y) ** 2 - norm(y - theta * n_samples) ** 2) / \
                (2 * n_samples)
        return dict(value=p_obj,
                    duality_gap=p_obj - d_obj)

    def to_dict(self):
        return dict(X=self.X, y=self.y, alphas=self.alphas)

    def _get_lambda_seq(self):
        randnorm = stats.norm(loc=0, scale=1)
        q = self.q
        alphas_seq = randnorm.ppf(
            1 - np.arange(1, self.X.shape[1] + 1) * q / (2 * self.X.shape[1]))
        alpha_max = dual_norm_slope(self.X, self.y / len(self.y), alphas_seq)

        return alpha_max * alphas_seq * self.reg
