from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from slope.solvers import prox_grad


class Solver(BaseSolver):
    name = "pgd"
    parameters = {"fista": [False, True]}
    stopping_strategy = 'callback'

    install_cmd = "conda"
    requirements = ["slope"]
    references = []

    def set_objective(self, X, y, alphas, fit_intercept, anderson=False):
        self.X, self.y, self.alphas = X, y, alphas
        self.anderson = anderson
        self.fit_intercept = fit_intercept

    def run(self, callback):
        self.coef_, self.intercept_ = prox_grad(
            self.X,
            self.y,
            self.alphas,
            fista=self.fista,
            tol=1e-12,
            fit_intercept=self.fit_intercept,
            anderson=self.anderson,
            callback=callback
        )[:2]

    def get_result(self):
        return np.hstack((self.intercept_, self.coef_))
