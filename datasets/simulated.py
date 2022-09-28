from benchopt import BaseDataset, safe_import_context
from benchopt.datasets import make_correlated_data

with safe_import_context() as import_ctx:
    from slope.benchmark_utils import get_reg_devratio
    from slope.utils import preprocess


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        "n_samples, n_features, density, X_density": [
            (200, 20_000, 0.001, 1.0),
            (20_000, 1_000, 0.04, 1.0),
            (200, 2_000_000, 0.00001, 0.001),
        ]
    }

    def __init__(
        self, n_samples=10, n_features=50, random_state=27, density=0.5, X_density=1.0
    ):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.density = density
        self.X_density = X_density

    def get_data(self):
        X, y, _ = make_correlated_data(
            self.n_samples,
            self.n_features,
            random_state=self.random_state,
            density=self.density,
            X_density=self.X_density,
        )

        # Standardize with mean and standard deviation for dense data and scale with
        # maximum absolute value otherwise
        X = preprocess(X)

        # These HAVE to include all the settings in objectives.py for this to work
        dev_ratios = [0.3, 0.6, 0.99]
        q = 0.2
        fit_intercept = True

        regs, _ = get_reg_devratio(
            dev_ratios, X, y, q=q, fit_intercept=fit_intercept, verbose=False
        )

        regs_dict = dict(zip(dev_ratios, regs))

        return dict(X=X, y=y, regs_dict=regs_dict)
