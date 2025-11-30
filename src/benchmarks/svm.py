"""
SVM benchmark.

Adapted from https://github.com/LeoIV/BenchSuite/blob/master/benchsuite/svm.py
"""

import gzip
import logging
import os
import urllib
from pathlib import Path

import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from jaxtyping import Float
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from torch import Tensor


class SVM(SyntheticTestFunction):
    """
    SVM benchmark task.

    :param noise_std: Standard deviation of the Gaussian noise to be added to the function values.
    :param negate: If True, the function is negated (i.e. minimize rather than maximize).
    :param bounds: The bounds of the input space.
    """

    def __init__(self, noise_std: float = 0.0):
        # BoTorch initialization
        self.dim = 388
        bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))
        super().__init__(noise_std=noise_std, negate=True, bounds=bounds)

        # Load SVM data
        data_folder = os.path.join(Path(__file__).parent.parent.parent, "data", "svm")
        X_file = os.path.join(data_folder, "CT_slice_X.npy.gz")
        y_file = os.path.join(data_folder, "CT_slice_y.npy.gz")
        if not os.path.exists(os.path.join(data_folder, "CT_slice_X.npy.gz")):
            url = "https://github.com/LeoIV/BenchSuite/raw/73de8c581aacf2dc99120d9cf65b79cbfe2aaf4e/data/svm/CT_slice_X.npy.gz"
            logging.info(f"Downloading {url} to {X_file}")
            urllib.request.urlretrieve(url, X_file)
        if not os.path.exists(os.path.join(data_folder, "CT_slice_y.npy.gz")):
            url = "https://github.com/LeoIV/BenchSuite/raw/73de8c581aacf2dc99120d9cf65b79cbfe2aaf4e/data/svm/CT_slice_y.npy.gz"
            logging.info(f"Downloading {url} to {y_file}")
            urllib.request.urlretrieve(url, y_file)
        with gzip.GzipFile(X_file, "r") as fx:
            X_np = np.load(fx)
        with gzip.GzipFile(y_file, "r") as fy:
            y_np = np.load(fy)
        X: Float[np.ndarray, "n+m p"] = MinMaxScaler().fit_transform(X_np)
        y: Float[np.ndarray, " n+m"] = MinMaxScaler().fit_transform(y_np[:, None]).squeeze(-1)

        # Make train/test split
        np.random.seed(
            388
        )  # from https://github.com/hvarfner/vanilla_bo_in_highdim/blob/8174f6322d12154b4f84448a0bb54b71e56ffede/BenchSuite/benchsuite/svm.py#L30
        idxs = np.random.choice(np.arange(len(X)), min(500, len(X)), replace=False)
        half = len(idxs) // 2
        self._X_train: Float[np.ndarray, "n p"] = X[idxs[:half]]
        self._X_test: Float[np.ndarray, " n"] = X[idxs[half:]]
        self._y_train: Float[np.ndarray, "m p"] = y[idxs[:half]]
        self._y_test: Float[np.ndarray, " m"] = y[idxs[half:]]

    def _evaluate_true_nonbatch_numpy(self, hypers: Float[np.ndarray, " d"]) -> Float[np.ndarray, ""]:
        """
        Evaluate SVM error for one set of hyperparameters, stored in a numpy array.

        :param hypers: one input hyperparameter configuration
        :return: SVM error
        """
        C = 0.01 * (500 ** hypers[387])
        gamma = 0.1 * (30 ** hypers[386])
        epsilon = 0.01 * (100 ** hypers[385])
        lengthscales = np.exp(4 * hypers[:385] - 2)

        svr = SVR(gamma=gamma, epsilon=epsilon, C=C, cache_size=1500, tol=0.001)
        svr.fit(self._X_train / lengthscales, self._y_train)
        pred = svr.predict(self._X_test / lengthscales)
        error = np.sqrt(np.mean(np.square(pred - self._y_test)))

        return error

    def _evaluate_true(self, X: Float[Tensor, "... d"]) -> Float[Tensor, "..."]:
        """
        Evaluate Mopta08 benchmark for one point

        :param x: one input configuration
        :return: value with soft constraints
        """
        # Convert hypers ("X") to numpy
        hypers: Float[np.ndarray, "... d"] = X.detach().cpu().numpy()

        # Evaluate SVM for each set of hypers
        res: Float[np.ndarray, "..."] = np.vectorize(self._evaluate_true_nonbatch_numpy, signature="(d)->()")(hypers)

        # Recast results as a torch Tensor
        return torch.from_numpy(res).to(dtype=X.dtype, device=X.device)
