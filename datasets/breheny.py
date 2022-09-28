from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from slope.benchmark_utils import get_reg_devratio
    from slope.data import fetch_breheny
    from slope.utils import preprocess


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
        X, y = fetch_breheny(self.dataset, min_nnz=3)

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
