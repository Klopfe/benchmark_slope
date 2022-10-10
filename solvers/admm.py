from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientDescentCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from slope.solvers import admm


class Solver(BaseSolver):
    name = "admm"
    stopping_strategy = 'callback'
    parameters = {
        "adaptive_rho": [False, True],
        "rho": [1, 10, 100]}

    install_cmd = "conda"
    requirements = ["slope"]
    references = []
    stopping_criterion = SufficientDescentCriterion(eps=1e-10, patience=5)

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.alphas = X, y, alphas
        self.fit_intercept = fit_intercept

    def run(self, callback):
        if self.adaptive_rho:
            rho = 1.
        else:
            rho = self.rho
        self.coef_, self.intercept_ = admm(
            self.X,
            self.y,
            self.alphas,
            fit_intercept=self.fit_intercept,
            tol=1e-12,
            adaptive_rho=self.adaptive_rho,
            rho=rho,
            callback=callback
        )[:2]

    def get_result(self):
        return np.hstack((self.intercept_, self.coef_))
