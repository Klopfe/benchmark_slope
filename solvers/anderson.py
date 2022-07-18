from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from slope.solvers import prox_grad


class Solver(BaseSolver):
    name = 'anderson'

    install_cmd = 'conda'
    requirements = ['slope']
    references = []

    def set_objective(self, X, y, alphas):
        self.X, self.y, self.alphas = X, y, alphas

    def run(self, n_iter):
        self.coef_ = prox_grad(
            self.X, self.y, self.alphas, fista=False, max_epochs=n_iter,
            tol=1e-12, fit_intercept=False, anderson=True,
            verbose=False, gap_freq=10)[0]

    def get_result(self):
        return self.coef_
