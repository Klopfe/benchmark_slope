from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from slope.solvers import newt_alm


class Solver(BaseSolver):
    name = "newt_alm"
    stopping_strategy = 'callback'

    install_cmd = "conda"
    requirements = ["slope"]
    references = []

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.alphas = X, y, alphas
        self.fit_intercept = fit_intercept

        # cache numba stuff
        _, _ = newt_alm(
            self.X,
            self.y,
            self.alphas,
            fit_intercept=self.fit_intercept,
            tol=1e-12,
            solver="auto",
            callback=None,
            max_epochs=2
        )[:2]

    def run(self, callback):
        self.coef_, self.intercept_ = newt_alm(
            self.X,
            self.y,
            self.alphas,
            fit_intercept=self.fit_intercept,
            tol=1e-12,
            solver="standard",
            callback=callback
        )[:2]

    @staticmethod
    def get_next(previous):
        # Linear growth for number of iterations
        return previous + 1

    def get_result(self):
        return np.hstack((self.intercept_, self.coef_))
