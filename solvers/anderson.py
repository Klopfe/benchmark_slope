from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from slope.solvers import prox_grad


class Solver(BaseSolver):
    name = "anderson"
    stopping_strategy = 'callback'
    install_cmd = "conda"
    requirements = ["slope"]
    references = []

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.alphas = X, y, alphas
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        self.coef_, self.intercept_ = prox_grad(
            self.X,
            self.y,
            self.alphas,
            fista=False,
            max_epochs=n_iter,
            tol=1e-12,
            fit_intercept=self.fit_intercept,
            anderson=True,
        )[:2]

    def get_result(self):
        return np.hstack((self.intercept_, self.coef_))
