from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    from slope.solvers import hybrid_cd


class Objective(BaseObjective):
    name = "SLOPE"
    parameters = {
        "dev_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "fit_intercept": [True, False],
        "q": [0.2, 0.1, 0.05],
    }

    def __init__(self, dev_ratio, fit_intercept, q):
        self.dev_ratio = dev_ratio
        self.fit_intercept = fit_intercept
        self.q = q

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape

        lambda_min_ratio = 1e-2 if self.n_samples < self.n_features else 1e-4

        lambdas = self._get_lambda_seq()

        alphas = np.geomspace(1, lambda_min_ratio, 100)

        r = y - np.mean(y) if self.fit_intercept else y.copy()
        null_dev = 0.5 * np.linalg.norm(r) ** 2
        dev_ratio_target = self.dev_ratio

        # if n > p, find R2 of OLS and take a fraction of that
        if self.n_samples > self.n_features:
            r2_full = (
                LinearRegression(fit_intercept=self.fit_intercept).fit(X, y).score(X, y)
            )
            dev_ratio_target = self.dev_ratio * r2_full

        w = np.zeros(self.n_features)
        intercept = 0.0

        def f(reg, w_start, intercept_start):
            w, intercept = hybrid_cd(
                X,
                y,
                lambdas * reg,
                w_start=w_start,
                intercept_start=intercept_start,
                tol=1e-4,
                fit_intercept=self.fit_intercept,
                use_reduced_X=False,
            )[:2]

            dev = 0.5 * np.linalg.norm(y - X @ w - intercept) ** 2
            dev_ratio = 1 - dev / null_dev

            return dev_ratio - dev_ratio_target, w, intercept

        for i in range(len(alphas)):
            f_i, w, intercept = f(alphas[i], w, intercept)
            if f_i >= 0:
                hi = alphas[i - 1] if i > 0 else 1.0
                lo = alphas[i]
                break

        # bisect to find dev_ratio close to dev_ratio_target
        a = lo
        b = hi

        it_max = 100
        for it in range(it_max):
            c = (a + b) / 2

            f_c, w, intercept = f(c, w, intercept)

            if abs(f_c) <= 0.005:
                reg = c
                break

            f_a, _, _ = f(a, w, intercept)

            if np.sign(f_c) == np.sign(f_a):
                a = c
            else:
                b = c

            if it == it_max - 1:
                raise ValueError("bisection did not converge")

        print(f"reg: {reg}, dev_ratio: {f_c + dev_ratio_target}")

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
