from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from slope.solvers import newt_alm


class Solver(BaseSolver):
    name = "newt_alm"
    stopping_strategy = "iteration"

    install_cmd = "conda"
    requirements = ["slope"]
    references = []

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.alphas = X, y, alphas
        self.fit_intercept = fit_intercept

        # cache numba stuff
        self.run(2)

    def run(self, n_iter):
        self.coef_, self.intercept_ = newt_alm(
            self.X,
            self.y,
            self.alphas,
            fit_intercept=self.fit_intercept,
            tol=1e-12,
            max_epochs=n_iter,
        )[:2]

    @staticmethod
    def get_next(previous):
        # Linear growth for number of iterations
        return previous + 1

    def get_result(self):
        return np.hstack((self.intercept_, self.coef_))
