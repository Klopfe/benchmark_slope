import numpy as np
from benchopt import BaseObjective
from numpy.linalg import norm
from scipy import stats

from slope.utils import dual_norm_slope


class Objective(BaseObjective):
    name = "SLOPE"
    parameters = {'reg': [0.1, 0.01, 0.001],
                  'q': [0.2, 0.1, 0.05],
                  'fit_intercept': [True, False]}

    def __init__(self, reg, q, fit_intercept):
        self.q = q
        self.reg = reg
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.n_samples, self.n_features = self.X.shape
        self.alphas = self._get_lambda_seq()

    def compute(self, res):
        intercept, beta = res[0], res[1:]

        # compute residuals
        X, y = self.X, self.y
        n_samples, n_features = X.shape

        diff = y - X @ beta - intercept

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
        return dict(
            X=self.X,
            y=self.y,
            alphas=self.alphas,
            fit_intercept=self.fit_intercept
        )

    def _get_lambda_seq(self):
        randnorm = stats.norm(loc=0, scale=1)
        q = self.q
        alphas_seq = randnorm.ppf(
            1 - np.arange(1, self.X.shape[1] + 1) * q / (2 * self.X.shape[1]))
        alpha_max = dual_norm_slope(
            self.X,
            (self.y - self.fit_intercept * np.mean(self.y)) / len(self.y),
            alphas_seq
        )

        return alpha_max * alphas_seq * self.reg
