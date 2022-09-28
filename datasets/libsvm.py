from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm
    from slope.benchmark_utils import get_reg_devratio
    from slope.utils import preprocess


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

        # if self.X is None:
        self.X, self.y = fetch_libsvm(self.dataset, min_nnz=3)

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
