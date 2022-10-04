from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm
    from scipy import sparse
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        "dataset": [
            "rcv1.binary",
            "news20.binary",
            "leukemia",
            "real-sim",
            "url",
            "YearPredictionMSD",
        ],
    }

    install_cmd = "conda"
    requirements = ["pip:libsvmdata"]

    def __init__(self, dataset="bodyfat"):
        self.dataset = dataset
        self.X, self.y = None, None

    def get_data(self):
        self.X, self.y = fetch_libsvm(self.dataset, min_nnz=3)

        # remove nonzero variance predictors
        self.X = VarianceThreshold().fit_transform(self.X)

        # standardize X
        if sparse.issparse(self.X):
            self.X = MaxAbsScaler().fit_transform(self.X).tocsc()
        else:
            self.X = StandardScaler().fit_transform(self.X)

        data = dict(X=self.X, y=self.y)

        return data
