from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from slope.solvers.hybrid import hybrid_cd


class Solver(BaseSolver):
    name = 'hybrid'
    install_cmd = 'conda'
    requirements = ['slope']
    parameters = {'cluster_updates': [True]}
    references = []

    def set_objective(self, X, y, alphas):
        self.X, self.y, self.alphas = X, y, alphas
        self.run(2)

    def run(self, n_iter):
        self.coef_ = hybrid_cd(
            self.X, self.y, self.alphas, max_epochs=n_iter, verbose=False,
            tol=1e-12, cluster_updates=self.cluster_updates,
            fit_intercept=False)[0]

    def get_result(self):
        return self.coef_
