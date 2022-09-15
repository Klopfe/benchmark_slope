from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from slope.solvers.hybrid import hybrid_cd
    from scipy.sparse import issparse


class Solver(BaseSolver):
    name = "hybrid"
    install_cmd = "conda"
    requirements = ["slope"]
    parameters = {"cluster_updates": [True]}
    references = []

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X, self.y, self.alphas = X, y, alphas
        self.fit_intercept = fit_intercept
        self.run(2)

    def run(self, n_iter):
        # reduced_X = False if issparse(self.X) else True
        self.coef_, self.intercept_ = hybrid_cd(
            self.X,
            self.y,
            self.alphas,
            max_epochs=n_iter,
            tol=1e-12,
            cluster_updates=self.cluster_updates,
            fit_intercept=self.fit_intercept,
            use_reduced_X=False
        )[:2]

    def get_result(self):
        return np.hstack((self.intercept_, self.coef_))
