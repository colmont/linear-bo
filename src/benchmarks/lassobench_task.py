"""
Tasks from the LassoBench benchmark suite.

Adapted from
https://github.com/hvarfner/vanilla_bo_in_highdim/blob/main/benchmarking/lassobench_task.py
"""

import LassoBench
import torch
from botorch.test_functions.base import BaseTestProblem
from jaxtyping import Float
from torch import Tensor


class _LassoRealFunction(BaseTestProblem):
    r"""
    A benchmark task for problems defined on the LassoBench.

    :param pick_data: The name of the dataset to use.
    :param noise_std: Standard deviation of the Gaussian noise to be added to the function values.
    :param seed: Seed for random number generation.
    """

    def __init__(
        self,
        pick_data: str,
        noise_std: float = None,
        seed: int = 42,
    ):
        # For LassoBench
        self.seed = seed
        self.benchmark = LassoBench.RealBenchmark(pick_data=pick_data)

        # BoTorch initialization
        self.dim = self.benchmark.n_features
        self._bounds = [(-1.0, 1.0) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))
        super().__init__(noise_std=noise_std, negate=True)

    def _evaluate_true(self, X: Float[Tensor, "... d"]) -> Float[Tensor, "..."]:
        *batch_shape, d = X.shape
        res = torch.stack(
            [
                torch.tensor(self.benchmark.evaluate(x), dtype=X.dtype, device=X.device)
                for x in X.view(-1, d).cpu().numpy()
            ],
            dim=0,
        ).view(*batch_shape)
        return res


class LassoDna(_LassoRealFunction):
    r"""
    The Lasso benchmark task for the DNA dataset.

    :param noise_std: Standard deviation of the Gaussian noise to be added to the function values.
    :param seed: Seed for random number generation.
    """

    def __init__(self, noise_std: float = None, seed: int = 42):
        super().__init__(
            pick_data="dna",
            noise_std=noise_std,
            seed=seed,
        )
