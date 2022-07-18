from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from slope.solvers import prox_grad


class Solver(BaseSolver):
    name = 'pgd'
    parameters = {'fista': [False, True]}

    install_cmd = 'conda'
    requirements = ['slope']
    references = []

    def set_objective(self, X, y, alphas, anderson=False):
        self.X, self.y, self.alphas = X, y, alphas
        self.anderson = anderson

    def run(self, n_iter):
        self.coef_ = prox_grad(
            self.X, self.y, self.alphas, fista=self.fista, max_epochs=n_iter,
            tol=1e-12, fit_intercept=False, anderson=self.anderson,
            verbose=False, gap_freq=10)[0]

    def get_result(self):
        return self.coef_
