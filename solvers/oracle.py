from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from slope.solvers import oracle_cd, prox_grad


class Solver(BaseSolver):
    name = "oracle"
    install_cmd = "conda"
    requirements = ["slope"]
    references = []

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.alphas = X, y, alphas
        self.fit_intercept = fit_intercept
        self.w_star = prox_grad(
            X,
            y,
            alphas,
            max_epochs=10_000,
            tol=1e-14,
            fista=False,
            verbose=False,
            fit_intercept=False,
        )[0]
        self.run(2)

    def run(self, n_iter):
        self.coef_, self.intercept_ = oracle_cd(
            self.X,
            self.y,
            self.alphas,
            max_epochs=n_iter,
            verbose=False,
            tol=1e-12,
            w_star=self.w_star,
            fit_intercept=False,
        )[:2]

    def get_result(self):
        return np.hstack((self.intercept_, self.coef_))
