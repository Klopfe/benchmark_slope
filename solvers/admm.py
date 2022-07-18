from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientDescentCriterion
with safe_import_context() as import_ctx:
    from slope.solvers import admm


class Solver(BaseSolver):
    name = 'admm'
    stopping_strategy = 'iteration'
    stopping_criterion = SufficientDescentCriterion(eps=1e-10, patience=5)
    install_cmd = 'conda'
    requirements = ['slope']
    references = []

    def set_objective(self, X, y, alphas):
        self.X, self.y, self.alphas = X, y, alphas

    def run(self, n_iter):
        self.coef_ = admm(
            self.X, self.y, self.alphas, fit_intercept=False, tol=1e-12,
            max_epochs=n_iter)[0]

    def get_result(self):
        return self.coef_
