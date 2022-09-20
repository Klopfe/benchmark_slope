from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from slope.solvers import oracle_cd, prox_grad


class Solver(BaseSolver):
    name = "oracle"
    install_cmd = "conda"
    requirements = ["slope"]
    references = []
    stopping_strategy = 'callback'

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.alphas = X, y, alphas
        self.fit_intercept = fit_intercept
        self.w_star = prox_grad(
            X,
            y,
            alphas,
            max_epochs=50_000,
            tol=1e-12,
            fista=False,
            fit_intercept=self.fit_intercept,
        )[0]
        self.run(2)

    def run(self, n_iter):
        self.coef_, self.intercept_ = oracle_cd(
            self.X,
            self.y,
            self.alphas,
            max_epochs=n_iter,
            tol=1e-12,
            w_star=self.w_star,
            fit_intercept=self.fit_intercept,
        )[:2]

    def get_result(self):
        return np.hstack((self.intercept_, self.coef_))
