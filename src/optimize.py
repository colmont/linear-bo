"""
Main module for running Bayesian optimization loops.
"""

import logging
from collections import OrderedDict
from contextlib import ExitStack

import botorch
import gpytorch
import hydra
import numpy as np
import torch
import tqdm as tqdm
from botorch.acquisition import LogExpectedImprovement as LogEI
from botorch.generation import gen_candidates_scipy
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.kernels.spherical_linear import SphericalLinearKernel


def initialize_model(
    X_train: Float[Tensor, "n d"],
    Y_train: Float[Tensor, "n 1"],
) -> botorch.models.SingleTaskGP:
    """
    Initialize the model for Bayesian optimization.

    :param X_train: Training input points.
    :param Y_train: Training output points.
    """
    d = X_train.size(-1)

    mean = gpytorch.means.ConstantMean()
    kernel = SphericalLinearKernel(ard_num_dims=d)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior := LogNormalPrior(loc=-4.0, scale=1.0),
        noise_constraint=GreaterThan(1e-4, initial_value=noise_prior.mode),
    )

    return botorch.models.SingleTaskGP(
        train_X=X_train,
        train_Y=Y_train,
        mean_module=mean,
        covar_module=kernel,
        likelihood=likelihood,
    )


def evaluate_y(
    test_function: botorch.test_functions.base.BaseTestProblem,
    X: Float[Tensor, "... d"],
) -> Float[Tensor, "..."]:
    r"""
    Evaluate a test function at a given (batch of) input(s) :math:`X`.

    :param test_function: The test function to evaluate.
    :param X: The input(s) at which to evaluate the function.
    """
    unnormalized_X = botorch.utils.transforms.unnormalize(X, test_function.bounds)
    Y = test_function(unnormalized_X)
    return Y


def fit_model(model) -> None:
    """
    Fits the GP to the training data.

    :param model: The model to fit.
    """
    model.train()
    botorch_optimizer = botorch.fit.fit_gpytorch_mll_scipy

    with ExitStack() as es:
        es.enter_context(gpytorch.settings.cholesky_max_tries(10))

        # log_prob=True to use the woodbury decomposition for scalability
        es.enter_context(gpytorch.settings.max_cholesky_size(float("inf")))
        es.enter_context(
            gpytorch.settings.fast_computations(log_prob=True, covar_root_decomposition=False, solves=False)
        )

        # Fit the model
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        botorch.fit.fit_gpytorch_mll(mll, optimizer=botorch_optimizer)


def run_optimization(config: DictConfig) -> None:
    r"""
    Main function to run a BO loop.

    :params config: Configuration object containing the parameters for the BO loop.
    """
    # Resolve configuration
    OmegaConf.resolve(config)
    logging.info("\n" + OmegaConf.to_yaml(config))

    # Set seeds
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)

    # Dtype and device
    dtype = torch.float64
    device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")

    # Get benchmark
    test_function = hydra.utils.instantiate(config.benchmark.fn).to(dtype=dtype, device=device)

    # Lower and upper bound for model (normalized to be in [0, 1]^d hypercube)
    lb = torch.zeros(test_function.dim, dtype=dtype, device=device)
    ub = torch.ones(test_function.dim, dtype=dtype, device=device)
    bounds = torch.stack([lb, ub], dim=-2)
    n_tot = config.benchmark.n_tot
    d = test_function.dim
    n_init = config.benchmark.n_init

    # Construct tensors with x, y values
    Xs: Float[Tensor, "n_tot d"] = torch.empty((n_tot, d), dtype=dtype, device=device)
    Ys: Float[Tensor, "n_tot 1"] = torch.empty((n_tot, 1), dtype=dtype, device=device)

    # Get initial points
    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=config.seed)
    Xs[:n_init] = sobol.draw(n=n_init).to(dtype=dtype, device=device)
    Ys[:n_init] = evaluate_y(test_function, Xs[:n_init]).unsqueeze(-1)

    # Compute current best observation
    y_max: float = Ys[:n_init, 0].max().item()

    # Construct iterator for BO loop
    tqdm_log_list = ["y_max"]
    pbar = tqdm.tqdm(
        initial=n_init,
        total=n_tot,
        desc="BO loop",
        bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        disable=(not len(tqdm_log_list)),
    )

    ###
    # BO loop
    ###
    n = n_init
    n_prev = n_init
    while n_prev < n_tot:
        n = n_prev + 1

        X = Xs[:n_prev]
        Y = Ys[:n_prev]

        # Fit model
        model = initialize_model(X_train=X, Y_train=standardize(Y))
        fit_model(model)
        model.eval()

        # Maximize acquisition function to get next Xs
        acqf = LogEI(model, standardize(Y).max())
        options = {
            "raw_samples": 512,
            "num_restarts": 4,
            "retry_on_optimization_warning": False,
            "options": {
                "nonnegative": False,
                "sample_around_best": True,
                "sample_around_best_sigma": 0.1,
                "maxiter": 300,
                "batch_limit": 64,
            },
        }
        Xs[n_prev:n], _ = optimize_acqf(
            acqf,
            bounds=bounds,
            q=1,
            gen_candidates=gen_candidates_scipy,
            **options,
        )

        # Observe next Ys
        Ys[n_prev:n] = evaluate_y(test_function, Xs[n_prev:n]).unsqueeze(-1)

        with torch.no_grad():
            # See if we have a new best observation
            y_curr = Ys[n_prev:n, 0].max().item()
            if y_curr > y_max:
                y_max = y_curr

            # Update progress bar
            iter_stats = OrderedDict(
                y_max=y_max,
                y_curr=y_curr,
            )
            pbar.set_postfix(**{stat: iter_stats.get(stat, None) for stat in tqdm_log_list})

        n_prev = n
        pbar.update(1)

    pbar.close()

    logging.info(f"Best observation: {y_max}")
