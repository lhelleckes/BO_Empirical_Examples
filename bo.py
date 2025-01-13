import numpy
import typing
import torch

import botorch
import gpytorch

import scipy

from kinetics import (
    enzyme_truth,
    symmetric_noise,
)


def generate_noisy_observations(
    x: numpy.ndarray,
    bounds: typing.Tuple[float, float],
    truth_fn: typing.Callable[
        [numpy.ndarray, typing.List[typing.Dict[str, float]]],
        typing.Tuple[numpy.ndarray, typing.List[numpy.ndarray]],
    ],
    noise_fn: typing.Callable[
        [numpy.ndarray, typing.Tuple[float, float], float, float, float, typing.Optional[int]],
        numpy.ndarray,
    ],
    truth_params: typing.List[typing.Dict[str, float]],
    noise_params: typing.Dict[str, float],
    seed: typing.Optional[int] = None,
) -> numpy.ndarray:
    """
    Generate noisy observations from a ground truth function and noise function,
    replacing negative values with samples from a half-normal distribution.

    Parameters
    ----------
    x : numpy.ndarray
        Input values.
    truth_fn : Callable
        Ground truth function to compute true values. Accepts `x` and `truth_params`.
    noise_fn : Callable
        Function to generate noise.
    truth_params : List[Dict[str, float]]
        Parameters for the ground truth function.
    noise_params : Dict[str, float]
        Dictionary of noise parameters containing `sigma_0`, `sigma_1`, and `max_noise`.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Noisy observations with non-negative values.
    """
    # Consolidate noise parameters
    required_params = ["sigma_0", "sigma_1", "max_noise"]
    for param in required_params:
        if param not in noise_params:
            raise ValueError(f"Missing required noise parameter: {param}")

    sigma_0 = noise_params["sigma_0"]
    sigma_1 = noise_params["sigma_1"]
    max_noise = noise_params["max_noise"]

    y_true, _ = truth_fn(x, truth_params)
    noise = noise_fn(x, bounds, sigma_0, sigma_1, max_noise, seed)
    y_noisy = y_true + noise

    # prevent negative observation values
    negative_indices = y_noisy < 0
    if numpy.any(negative_indices):
        rng = numpy.random.default_rng(seed)  # Modern RNG
        replacement_values = rng.normal(loc=0, scale=sigma_0, size=numpy.sum(negative_indices))
        y_noisy[negative_indices] = numpy.abs(replacement_values)

    return y_noisy


def generate_sobol_points(num_points: int, bounds: tuple, seed: int = None) -> numpy.ndarray:
    """
    Generate Sobol sequence points within a specified range.

    Parameters
    ----------
    num_points : int
        Number of points to generate.
    bounds : Tuple[float, float]
        Bounds of the parameter space.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Sobol points scaled to the specified range.
    """
    range_min = numpy.atleast_1d(bounds[0])
    range_max = numpy.atleast_1d(bounds[1])

    dimensions = len(range_min)

    rng = numpy.random.RandomState(seed)
    sobol_gen = scipy.stats.qmcSobol(d=dimensions, scramble=True, seed=rng)

    # bitwise operation to check if num_points is a power of 2
    if (num_points & (num_points - 1)) == 0:
        points = sobol_gen.random_base2(m=int(numpy.log2(num_points)))
    else:
        points = sobol_gen.random(n=num_points)

    return range_min + points.flatten() * (range_max - range_min)


def perform_EI(train_x, train_y, bounds, num_candidates):
    gp_model = fit_gp_model(train_x, train_y, bounds)
    best_f = train_y.max().item()
    bounds = torch.tensor(bounds, dtype=torch.double).view(2, -1)
    qei = botorch.acquisition.qExpectedImprovement(model=gp_model, best_f=best_f)
    candidates, _ = botorch.optim.optimize_acqf(
        acq_function=qei,
        bounds=bounds,
        q=num_candidates,
        num_restarts=10,
        raw_samples=256,
    )
    return candidates, gp_model, qei


def perform_logEI(train_x, train_y, bounds, num_candidates):
    gp_model = fit_gp_model(train_x, train_y, bounds)
    best_f = train_y.max().item()
    bounds = torch.tensor(bounds, dtype=torch.double).view(2, -1)
    qlog_ei = botorch.acquisition.qLogExpectedImprovement(model=gp_model, best_f=best_f)
    candidates, _ = botorch.optim.optimize_acqf(
        acq_function=qlog_ei,
        bounds=bounds,
        q=num_candidates,
        num_restarts=10,
        raw_samples=256,
    )
    return candidates, gp_model, qlog_ei


class EnzymeGP(botorch.models.SingleTaskGP):
    def __init__(self, train_x, train_y, bounds, mean_module=None, covar_module=None):
        """
        Initialize the GP model.

        Parameters
        ----------
        train_x : torch.Tensor
            The training inputs.
        train_y : torch.Tensor
            The training targets.
        bounds : tuple
            The bounds of the parameter space.
        mean_module : gpytorch.means, optional
            Custom mean module, defaults to ConstantMean.
        covar_module : gpytorch.kernels, optional
            Custom covariance module, defaults to  RBFKernel.
        """
        super().__init__(
            train_x,
            train_y,
            outcome_transform=botorch.models.transforms.Standardize(m=train_x.shape[-1]),
        )

        self.mean_module = mean_module if mean_module else gpytorch.means.ConstantMean()
        if isinstance(self.mean_module, gpytorch.means.ConstantMean):
            self.mean_module.constant.data.fill_(train_y.mean().item())

        self.covar_module = (
            covar_module
            if covar_module
            else gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale_prior=gpytorch.priors.LogNormalPrior(
                        loc=numpy.log(numpy.abs(bounds[0] - bounds[1])) / 3, scale=0.5
                    )
                )
            )
        )


def fit_gp_model(train_x, train_y, bounds, mean_module=None, covar_module=None):
    """
    Fit a Gaussian Process model to the training data.

    Parameters
    ----------
    train_x : torch.Tensor
        The training inputs.
    train_y : torch.Tensor
        The training targets.
    mean_module : gpytorch.means.Mean, optional
        Custom mean module. Defaults to ConstantMean.
    covar_module : gpytorch.kernels.Kernel, optional
        Custom covariance module. Defaults to a scaled RBF kernel.

    Returns
    -------
    model : CustomSingleTaskGP
        The fitted Gaussian Process model.
    """
    model = EnzymeGP(train_x, train_y, bounds, mean_module, covar_module)

    # Define the Marginal Log Likelihood (MLL)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    botorch.fit.fit_gpytorch_mll(mll)

    return model


def perform_bo(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: typing.Tuple[float, float],
    num_candidates: int,
    num_rounds: int,
    method: str,
    truth_params: typing.List[typing.Dict[str, float]],
    noise_params: typing.Dict[str, float],
    truth_fn: typing.Callable[
        [numpy.ndarray, typing.List[typing.Dict[str, float]]],
        typing.Tuple[numpy.ndarray, typing.List[numpy.ndarray]],
    ] = enzyme_truth,
    noise_fn: typing.Callable[
        [numpy.ndarray, typing.Tuple[float, float], float, float, float, typing.Optional[int]],
        numpy.ndarray,
    ] = symmetric_noise,
    seed: int = None,
):
    """
    Perform multiple rounds of Bayesian Optimization and collect data for visualization.

    Parameters
    ----------
    train_x : torch.Tensor
        Initial training inputs.
    train_y : torch.Tensor
        Initial training outputs.
    bounds : torch.Tensor
        Input domain bounds.
    num_candidates : int
        Number of candidates to propose in each round.
    num_rounds : int
        Number of BO rounds.
    method : str, optional
        Optimization method: "EI" (Expected Improvement) or "TS" (Thompson Sampling).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary containing:
        - gp_models: List of GP models from each round.
        - acquisition_fns: List of acquisition functions from each round.
        - train_x_per_round: List of training inputs for each round.
        - train_y_per_round: List of training outputs for each round.
        - candidates_per_round: List of proposed candidates for each round.
    """
    gp_models = []
    acquisition_fns = []
    train_x_per_round = [train_x]
    train_y_per_round = [train_y]
    candidates_per_round = []

    for round_idx in range(num_rounds):
        print(f"Round {round_idx + 1}/{num_rounds}")

        if method == "EI":
            candidates, gp_model, acquisition_fn = perform_EI(
                train_x, train_y, bounds, num_candidates
            )
        elif method == "logEI":
            candidates, gp_model, acquisition_fn = perform_logEI(
                train_x, train_y, bounds, num_candidates
            )
        else:
            raise ValueError("Unsupported method. Use 'EI' or 'logEI'.")

        # Simulate observing new data
        observation = generate_noisy_observations(
            x=candidates.numpy().flatten(),
            bounds=bounds,
            truth_fn=truth_fn,
            noise_fn=noise_fn,
            truth_params=truth_params,
            noise_params=noise_params,
            seed=seed + round_idx if seed is not None else None,
        )

        new_y = torch.tensor(observation, dtype=torch.double).reshape(-1, 1)

        # Update training data with new observations
        train_x = torch.cat([train_x, candidates], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

        # Collect data for visualization
        gp_models.append(gp_model)
        acquisition_fns.append(acquisition_fn)
        train_x_per_round.append(train_x.clone())
        train_y_per_round.append(train_y.clone())
        candidates_per_round.append(candidates.clone())

    return {
        "gp_models": gp_models,
        "acquisition_fns": acquisition_fns,
        "train_x_per_round": train_x_per_round,
        "train_y_per_round": train_y_per_round,
        "candidates_per_round": candidates_per_round,
    }
