from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from scipy import sparse
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler
    from slope.data import fetch_breheny


class Dataset(BaseDataset):

    name = "breheny"

    parameters = {
        "dataset": [
            "bcTCGA",
            "Scheetz2006",
            "Rhee2006",
            "Golub1999",
            "Singh2002",
            "Gode2011",
            "Scholtens2004",
            "pollution",
            "whoari",
            "bcTCGA",
            "Koussounadis2014",
            "Scheetz2006",
            "Ramaswamy2001",
            "Shedden2008",
            "Rhee2006",
            "Yeoh2002",
            "glc-amd",
            "glioma",
            "spam",
        ]
    }

    install_cmd = "conda"
    requirements = ["pip:slope"]

    def __init__(self, dataset="Rhee2006"):
        self.dataset = dataset
        self.X, self.y = None, None

    def get_data(self):
        self.X, self.y = fetch_breheny(self.dataset, min_nnz=3)

        # remove nonzero variance predictors
        self.X = VarianceThreshold().fit_transform(self.X)

        # standardize X
        if sparse.issparse(self.X):
            self.X = MaxAbsScaler().fit_transform(self.X).tocsc()
        else:
            self.X = StandardScaler().fit_transform(self.X)

        data = dict(X=self.X, y=self.y)

        return data
