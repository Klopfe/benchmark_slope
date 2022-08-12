from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from slope.solvers import newt_alm


class Solver(BaseSolver):
    name = "newt_alm"
    stopping_strategy = "iteration"
    install_cmd = "conda"
    requirements = ["slope"]
    references = []

    def set_objective(self, X, y, alphas):
        self.X, self.y, self.alphas = X, y, alphas

    def run(self, n_iter):
        self.coef_, self.intercept_ = newt_alm(
            self.X,
            self.y,
            self.alphas,
            fit_intercept=False,
            tol=1e-12,
            max_epochs=n_iter,
        )[:2]

    def get_result(self):
        return self.coef_
