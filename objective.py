from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    from slope.sovlers import hybrid_cd


class Objective(BaseObjective):
    name = "SLOPE"
    parameters = {
        "dev_ratio": [0.01, 0.1, 0.5],
        "q": [0.2, 0.1, 0.05],
        "fit_intercept": [True, False],
    }

    def __init__(self, dev_ratio, q, fit_intercept):
        self.q = q
        self.dev_ratio = dev_ratio
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape

        lambda_min_ratio = 1e-2 if self.n_samples < self.n_features else 1e-4

        lambdas = self._get_lambda_seq()

        alphas = np.geomspace(1, lambda_min_ratio, 100)

        dfmax = (
            self.n_samples if self.n_features > self.n_samples else self.n_features + 1
        )

        r = y - np.mean(y) if self.fit_intercept else y.copy()
        null_dev = 0.5 * np.linalg.norm(r) ** 2

        dev_ratio_target = self.dev_ratio

        # if n > p, find R2 of OLS and take a fraction of that
        if self.n_samples > self.n_features:
            r2_full = LinearRegression().fit(X, y).score(X, y)
            dev_ratio_target = self.dev_ratio * r2_full

        reg = 1 - dev_ratio_target

        while True:
            w, intercept = hybrid_cd(
                X, y, lambdas * reg, tol=1e-4, fit_intercept=fit_intercept
            )[:2]

            dev = 0.5 * np.linalg.norm(y - X @ w - intercept) ** 2
            dev_ratio = 1 - dev / null_dev

            print(f"dev_ratio: {dev_ratio}")

            # stop when we are close to the target
            if abs(dev_ratio - dev_target) <= 0.001:
                break

            dev_diff = dev_ratio / dev_target

            reg *= dev_diff

        self.alphas = lambdas * reg

    def compute(self, res):
        intercept, beta = res[0], res[1:]

        X, y = self.X, self.y
        n_samples = X.shape[0]
        # compute residuals
        diff = y - X @ beta - intercept

        # compute primal
        p_obj = 1.0 / (2 * n_samples) * diff @ diff + np.sum(
            self.alphas * np.sort(np.abs(beta))[::-1]
        )

        # compute dual
        theta = diff
        theta /= max(1, self._dual_norm_slope(theta, self.alphas))
        d_obj = (norm(y) ** 2 - norm(y - theta * n_samples) ** 2) / (2 * n_samples)

        return dict(value=p_obj, duality_gap=p_obj - d_obj)

    def to_dict(self):
        return dict(
            X=self.X, y=self.y, alphas=self.alphas, fit_intercept=self.fit_intercept
        )

    def _dual_norm_slope(self, theta, alphas):
        Xtheta = np.sort(np.abs(self.X.T @ theta))[::-1]
        taus = 1 / np.cumsum(alphas)
        return np.max(np.cumsum(Xtheta) * taus)

    def _get_lambda_seq(self):
        randnorm = stats.norm(loc=0, scale=1)
        q = self.q
        alphas_seq = randnorm.ppf(
            1 - np.arange(1, self.X.shape[1] + 1) * q / (2 * self.X.shape[1])
        )

        alpha_max = self._dual_norm_slope(
            (self.y - self.fit_intercept * np.mean(self.y)) / len(self.y), alphas_seq
        )

        return alpha_max * alphas_seq
