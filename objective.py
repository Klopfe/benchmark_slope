import numpy as np
from numpy.linalg import norm
from scipy import stats

from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "SLOPE"
    parameters = {'reg': [0.1, 0.01, 0.001],
                  'q': [0.2, 0.1, 0.05]}

    def __init__(self, reg, q):
        self.q = q
        self.reg = reg

    def set_data(self, X, y):
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
        theta /= max(1, self._dual_norm_slope(theta, self.alphas))

        d_obj = (norm(y) ** 2 - norm(y - theta * n_samples) ** 2) / \
                (2 * n_samples)
        return dict(value=p_obj,
                    duality_gap=p_obj - d_obj)

    def to_dict(self):
        return dict(X=self.X, y=self.y, alphas=self.alphas)

    def _dual_norm_slope(self, theta, alphas):
        Xtheta = np.sort(np.abs(self.X.T @ theta))[::-1]
        taus = 1 / np.cumsum(alphas)
        return np.max(np.cumsum(Xtheta) * taus)

    def _get_lambda_seq(self):
        randnorm = stats.norm(loc=0, scale=1)
        q = self.q
        alphas_seq = randnorm.ppf(
            1 - np.arange(1, self.X.shape[1] + 1) * q / (2 * self.X.shape[1]))
        alpha_max = self._dual_norm_slope(self.y / len(self.y), alphas_seq)

        return alpha_max * alphas_seq * self.reg
