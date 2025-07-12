import typing

import botorch
import gpytorch
import numpy
import scipy
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.priors import LogNormalPrior

from kinetics import enzyme_truth, symmetric_noise


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

    sobol_gen = scipy.stats.qmc.Sobol(d=dimensions, scramble=True, seed=seed)

    if (num_points & (num_points - 1)) == 0:
        m = int(numpy.log2(num_points))
        points = sobol_gen.random_base2(m=m)
    else:
        points = sobol_gen.random(n=num_points)

    scaled_points = scipy.stats.qmc.scale(points, l_bounds=range_min, u_bounds=range_max)

    if dimensions == 1:
        return scaled_points.flatten()
    else:
        return scaled_points


def perform_EI(train_x, train_y, bounds, num_candidates, ls_prior=True, noise=0.5):
    gp_model = fit_gp_model(train_x, train_y, bounds, ls_prior=ls_prior, noise=noise)
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


def perform_logEI(train_x, train_y, bounds, num_candidates, ls_prior=True, noise=0.5):
    gp_model = fit_gp_model(train_x, train_y, bounds, ls_prior, noise)
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
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        covar_module: gpytorch.kernels.Kernel,
        mean_module: gpytorch.means.Mean = None,
        input_transform=None,
        outcome_transform=None,
    ):
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
        ls_prior : boolean, optional
            Toggle informed length scale prior
        """

        mean_mod = mean_module or gpytorch.means.ConstantMean()
        super().__init__(
            train_x,
            train_y,
            covar_module=covar_module,
            mean_module=mean_mod,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )

        if isinstance(self.mean_module, gpytorch.means.ConstantMean):
            self.mean_module.constant.data.fill_(train_y.mean().item())


def fit_gp_model(
    train_x,
    train_y,
    bounds,
    *,
    lengthscale_prior: gpytorch.priors.Prior,
    noise_prior: gpytorch.priors.Prior,
    mean_module: gpytorch.means.Mean = None,
):
    base_kernel = RBFKernel(
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=gpytorch.constraints.Interval(1e-5, 1e5),
    )

    covar_mod = ScaleKernel(base_kernel)

    bounds_t = torch.tensor(bounds, dtype=train_x.dtype).view(2, -1)
    input_tf = botorch.models.transforms.Normalize(d=train_x.size(-1), bounds=bounds_t)
    outcome_tf = botorch.models.transforms.Standardize(m=train_y.shape[-1])

    model = EnzymeGP(
        train_x=train_x,
        train_y=train_y,
        covar_module=covar_mod,
        mean_module=mean_module,
        input_transform=input_tf,
        outcome_transform=outcome_tf,
    )

    model.likelihood.noise_covar.register_prior(
        "noise_prior",
        noise_prior,
        "noise",
    )

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
    gp_model_builder=None,
    seed: int = None,
) -> dict:
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
    results: dict = {
        "gp_models": [],
        "acquisition_fns": [],
        "train_x_per_round": [],
        "train_y_per_round": [],
        "candidates_per_round": [],
    }

    for round_idx in range(num_rounds):
        print(f"Round {round_idx + 1}/{num_rounds}")

        gp_model = gp_model_builder(train_x, train_y)

        if method == "EI":
            best_f = train_y.max().item()
            bounds_tensor = torch.tensor(bounds, dtype=torch.double).view(2, -1)
            acquisition_fn = botorch.acquisition.ExpectedImprovement(model=gp_model, best_f=best_f)

            # Optimize acquisition function
            candidates, _ = botorch.optim.optimize_acqf(
                acq_function=acquisition_fn,
                bounds=bounds_tensor,
                q=num_candidates,
                num_restarts=10,
                raw_samples=256,
            )
        elif method == "logEI":
            best_f = train_y.max().item()
            bounds_tensor = torch.tensor(bounds, dtype=torch.double).view(2, -1)
            acquisition_fn = botorch.acquisition.LogExpectedImprovement(
                model=gp_model, best_f=best_f
            )

            # Optimize acquisition function
            candidates, _ = botorch.optim.optimize_acqf(
                acq_function=acquisition_fn,
                bounds=bounds_tensor,
                q=num_candidates,
                num_restarts=10,
                raw_samples=256,
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

        results["train_x_per_round"].append(train_x.clone())
        results["train_y_per_round"].append(train_y.clone())

        # Update training data with new observations
        train_x = torch.cat([train_x, candidates], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

        # Collect data for visualization
        results["gp_models"].append(gp_model)
        results["acquisition_fns"].append(acquisition_fn)
        results["candidates_per_round"].append(candidates.clone())

    return results


def transform_lengthscale_prior(
    raw_lengthscale: float, bounds: tuple, scale: float = 0.5
) -> LogNormalPrior:
    low, high = bounds
    # Convert raw lengthscale to normalized [0,1] space
    normalized_ls = raw_lengthscale / (high - low)
    # LogNormalPrior is on the positive parameter itself, so loc = log(normalized_ls)
    loc = torch.log(torch.tensor(normalized_ls, dtype=torch.double))
    return LogNormalPrior(loc=loc, scale=scale)


def transform_noise_prior(
    noise_val: float,
    y_train: torch.Tensor,
    scale: float = 0.2,
) -> LogNormalPrior:
    y = y_train.flatten()
    y_min, y_max = y.min().item(), y.max().item()

    y_rep = (y_min + y_max) / 2

    # expected raw std‐dev and variance
    raw_sigma = noise_val * y_rep
    raw_var = raw_sigma**2

    # convert to standardized‐space variance
    y_std = float(y.std(unbiased=True).item())
    std_var = raw_var / (y_std**2)

    # build the LogNormal prior on variance
    loc = torch.log(torch.tensor(std_var, dtype=torch.double))
    return LogNormalPrior(loc=loc, scale=scale)
