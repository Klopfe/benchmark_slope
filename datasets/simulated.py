from benchopt import BaseDataset, safe_import_context
from benchopt.datasets import make_correlated_data

with safe_import_context() as import_ctx:
    from scipy import sparse
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        "n_samples, n_features, density, X_density": [
            (200, 20_000, 0.001, 1.0),
            (20_000, 1_000, 0.04, 1.0),
            (200, 2_000_000, 0.00001, 0.001),
            (500, 200, 0.1, 1.)
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
        self.X, self.y, _ = make_correlated_data(
            self.n_samples,
            self.n_features,
            random_state=self.random_state,
            density=self.density,
            X_density=self.X_density,
        )

        # remove nonzero variance predictors
        self.X = VarianceThreshold().fit_transform(self.X)

        # standardize X
        if sparse.issparse(self.X):
            self.X = MaxAbsScaler().fit_transform(self.X).tocsc()
        else:
            self.X = StandardScaler().fit_transform(self.X)

        data = dict(X=self.X, y=self.y)

        return data
