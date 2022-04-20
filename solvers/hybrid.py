from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from slope.solvers.hybrid import hybrid_cd


class Solver(BaseSolver):
    name = 'hybrid'

    install_cmd = 'conda'
    requirements = ['slope']
    references = []

    def set_objective(self, X, y, alphas):
        self.X, self.y, self.alphas = X, y, alphas
        self.run(1)

    def run(self, n_iter):
        self.coef_, _, _, _ = hybrid_cd(
            self.X, self.y, self.alphas, max_epochs=n_iter, verbose=False,
            tol=1e-12)

    def get_result(self):
        return self.coef_
