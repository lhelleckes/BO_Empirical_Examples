import typing

import botorch
import gpytorch
import numpy
import scipy
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.priors import LogNormalPrior

from bo_examples.kinetics import enzyme_truth, symmetric_noise


def generate_noisy_observations(
    x: numpy.ndarray,
    bounds: typing.Tuple[float, float],
    truth_fn: typing.Callable[
        [numpy.ndarray, typing.List[typing.Dict[str, float]]],
        typing.Tuple[numpy.ndarray, typing.List[numpy.ndarray]],
    ],
    noise_fn: typing.Callable[
        [
            numpy.ndarray,
            typing.Tuple[float, float],
            float,
            float,
            float,
            typing.Optional[int],
        ],
        numpy.ndarray,
    ],
    truth_params: typing.List[typing.Dict[str, float]],
    noise_params: typing.Dict[str, float],
    seed: typing.Optional[int] = None,
) -> numpy.ndarray:
    """
    Generate noisy, non-negative observations from a ground truth function.
    This function adds heteroscedastic noise via `noise_fn` and replaces any negative
    values by sampling from a half-normal distribution with scale `sigma_0`.

    Parameters
    ----------
    x
        An array of input values of shape `(n,)` or `(n, d)`.
    bounds
        Tuple `(low, high)` specifying the parameter space bounds for noise computation.
    truth_fn
        Callable that calculates enzyme reaction rates for given pH and truth function parameters.
    noise_fn : Callable
        Callable mapping `(x, bounds, sigma_0, sigma_1, max_noise, seed)` to noise array.
    truth_params
       List of parameter dictionaries for `truth_fn`. Each enzyme has a corresponding dictionary.
    noise_params : Dict[str, float]
        Dictionary of noise parameters containing `sigma_0`, `sigma_1`, and `max_noise`.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        An array of noisy observations with the same shape as `y_true`

    Raises
    ------
    ValueError
        If any required key is missing in `noise_params`.
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


def generate_sobol_points(
    num_points: int, bounds: tuple[float, float], seed: typing.Optional[int] = None
) -> numpy.ndarray:
    """
    Generate Sobol sequence points within specified bounds.

    Parameters
    ----------
    num_points
        Number of points to generate.
    bounds
        Bounds of the parameter space.
    seed
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        An array of shape `(num_points,)` or `(num_points, d)` of Sobol samples.
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
        A GP model specialized for enzyme rate data.

        Parameters
        ----------
        train_x
            Tensor of shape `(n, d)` of training inputs.
        train_y
            Tensor of shape `(n, 1)` of training outputs.
        covar_module
            A GPyTorch kernel.
        mean_module
            A GPyTorch mean function; defaults to ConstantMean.
        input_transform
            A BoTorch input transform (e.g. Normalize).
        outcome_transform
            A BoTorch outcome transform (e.g. Standardize).
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
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: tuple[float, float],
    *,
    lengthscale_prior: gpytorch.priors.Prior,
    noise_prior: gpytorch.priors.Prior,
    mean_module: gpytorch.means.Mean = None,
) -> EnzymeGP:
    """
    Fit an EnzymeGP.

    Parameters
    ----------
    train_x
        Tensor `(n, d)` of inputs.
    train_y
        Tensor `(n, 1)` of targets.
    bounds
        Tuple `(low, high)` for input space normalization.
    lengthscale_prior
        Prior over kernel lengthscale.
    noise_prior
        Prior over likelihood noise.

    Returns
    -------
    EnzymeGP
        The trained GP model with registered priors.
    """
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
    gp_model_builder: typing.Callable[..., EnzymeGP],
    truth_fn: typing.Callable[
        [numpy.ndarray, typing.List[typing.Dict[str, float]]],
        typing.Tuple[numpy.ndarray, typing.List[numpy.ndarray]],
    ] = enzyme_truth,
    noise_fn: typing.Callable[
        [
            numpy.ndarray,
            typing.Tuple[float, float],
            float,
            float,
            float,
            typing.Optional[int],
        ],
        numpy.ndarray,
    ] = symmetric_noise,
    seed: typing.Optional[int] = None,
) -> dict[str, typing.Any]:
    """
    Run Bayesian Optimization for multiple rounds and collect results.

    Parameters
    ----------
    train_x
        Initial `(n, d)` tensor of inputs.
    train_y
        Initial `(n, 1)` tensor of outputs.
    bounds
        Tuple `(low, high)` for input domain.
    num_candidates
        Number of proposals per round.
    num_rounds
        Total number of BO iterations.
    method
        Acquisition: "EI" or "logEI".
    truth_params
        Parameters for ground truth function.
    noise_params
        Parameters for noise generation.
    gp_model_builder
        Optional custom GP builder; defaults to `fit_gp_model`.
    truth_fn
        Function to compute ground truth values.
    noise_fn
        Function to generate noise.
    seed
        Optional seed for reproducibility.

    Returns
    -------
    dict
        Contains lists: 'gp_models', 'acquisition_fns', 'train_x_per_round',
        'train_y_per_round', 'candidates_per_round', 'acquisition_vals'.
    """
    results: dict = {
        "gp_models": [],
        "acquisition_fns": [],
        "train_x_per_round": [],
        "train_y_per_round": [],
        "candidates_per_round": [],
        "acquisition_vals": [],
    }

    for round_idx in range(num_rounds):
        print(f"Round {round_idx + 1}/{num_rounds}")

        gp_model = gp_model_builder(train_x, train_y)

        if method == "EI":
            best_f = train_y.max().item()
            bounds_tensor = torch.tensor(bounds, dtype=torch.double).view(2, -1)
            acquisition_fn = botorch.acquisition.ExpectedImprovement(model=gp_model, best_f=best_f)

            # Optimize acquisition function
            candidates, vals = botorch.optim.optimize_acqf(
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
            candidates, vals = botorch.optim.optimize_acqf(
                acq_function=acquisition_fn,
                bounds=bounds_tensor,
                q=num_candidates,
                num_restarts=10,
                raw_samples=256,
            )

            vals = numpy.exp(vals)

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
        results["acquisition_vals"].append(vals.clone())

    return results


def transform_lengthscale_prior(
    raw_lengthscale: float, bounds: tuple[float, float], scale: float = 0.5
) -> LogNormalPrior:
    """
    Create a LogNormal prior for the kernel lengthscale normalized by the input range.

    Parameters
    ----------
    raw_lengthscale
        Informative guess of lengthscale in original units.
    bounds
        Tuple `(low, high)` of the input domain.
    scale
        Scale (stddev) of the log-normal prior.

    Returns
    -------
    LogNormalPrior
        A prior over the normalized lengthscale.
    """

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
    """
    Create a LogNormal prior for the observation noise variance.

    Parameters
    ----------
    noise_val
        Relative noise magnitude (fraction of response range).
    y_train
        Tensor `(n, 1)` of observed targets.
    scale
        Scale (stddev) of the log-normal prior.

    Returns
    -------
    LogNormalPrior
        A prior over the noise variance, standardized by empirical y std.
    """
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
