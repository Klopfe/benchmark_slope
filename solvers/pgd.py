from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from slope.solvers.pgd import prox_grad


class Solver(BaseSolver):
    name = 'pgd'
    parameters = {'fista': [False, True]}

    install_cmd = 'conda'
    requirements = ['slope']
    references = []

    def set_objective(self, X, y, alphas, fista=False, anderson=False):
        self.X, self.y, self.alphas = X, y, alphas
        self.fista, self.anderson = fista, anderson

    def run(self, n_iter):
        self.coef_, _, _, _ = prox_grad(
            self.X, self.y, self.alphas, fista=self.fista, max_epochs=n_iter, tol=1e-12,
            anderson=self.anderson, verbose=False)

    def get_result(self):
        return self.coef_
