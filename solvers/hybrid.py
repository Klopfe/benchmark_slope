from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from slope.solvers.hybrid import hybrid_cd


class Solver(BaseSolver):
    name = "hybrid"
    install_cmd = "conda"
    requirements = ["slope"]
    references = []
    stopping_strategy = 'callback'

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.alphas = X, y, alphas
        self.fit_intercept = fit_intercept
        _, _ = hybrid_cd(
            self.X,
            self.y,
            self.alphas,
            tol=1e-12,
            max_epochs=2,
            cluster_updates=True,
            fit_intercept=self.fit_intercept,
            callback=None
        )[:2]

    def run(self, callback):
        self.coef_, self.intercept_ = hybrid_cd(
            self.X,
            self.y,
            self.alphas,
            tol=1e-12,
            cluster_updates=True,
            fit_intercept=self.fit_intercept,
            callback=callback
        )[:2]

    def get_result(self):
        print(self.intercept_)
        return np.hstack((self.intercept_, self.coef_))
