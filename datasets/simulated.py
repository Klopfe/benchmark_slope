from benchopt.datasets import make_correlated_data
from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features, density, X_density': [
            (200, 20_000, 0.001, 1.),
            (20_000, 1_000, 0.04, 1.),
            (200, 2_000_000, 0.00001, 0.001)]
    }

    def __init__(
            self, n_samples=10, n_features=50, random_state=27,
            density=0.5, X_density=1.):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.density = density
        self.X_density = X_density

    def get_data(self):
        X, y, _ = make_correlated_data(
            self.n_samples, self.n_features, random_state=self.random_state,
            density=self.density, X_density=self.X_density)

        data = dict(X=X, y=y)

        return data
