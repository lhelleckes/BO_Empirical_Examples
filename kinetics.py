import typing
import numpy

import typing


def ratkowsky_curve(
    T: typing.Union[float, numpy.ndarray],
    T_min: float = 5,
    T_max: float = 70,
    b: float = 0.1,
    c: float = 0.0002,
) -> typing.Union[float, numpy.ndarray]:
    """
    Computes the reaction rate using the extended Ratkowsky model.

    The extended Ratkowsky model accounts for enzyme activity increasing with temperature
    up to an optimal point and decreasing beyond it.

    Parameters
    ----------
    T
        Temperatures at which to compute the reaction rates (°C).
    T_min
        Minimum temperature for activity (default is 5°C).
    T_max
        Maximum temperature at which activity becomes zero (default is 70°C).
    b
        Growth constant (default is 0.1).
    c
        Decay constant for the exponential term (default is 0.0002).

    Returns
    -------
    float or numpy.ndarray
        Reaction rate(s) corresponding to the input temperatures.

    Notes
    -----
    - Reaction rates are zero for T < T_min.
    - The model combines a quadratic term and an exponential decay term.
    """
    k = b * (T - T_min) ** 2 * (1 - numpy.exp(c * (T - T_max)))
    k = numpy.where((T < T_min) | (T > T_max), 0, k)
    return k


def gaussian_vmax_pH(
    pH: typing.Union[float, numpy.ndarray],
    Vmax_base: float,
    pH_opt: float,
    sigma: float,
) -> typing.Union[float, numpy.ndarray]:
    """
    Gaussian-like model for Vmax dependence on pH.

    Parameters
    ----------
    pH
        pH values.
    Vmax_base
        Base Vmax value (peak height).
    pH_opt
        Optimal pH where the peak occurs.
    sigma
        Standard deviation controlling the peak width.

    Returns
    -------
    float or numpy.ndarray
        Adjusted Vmax values.
    """
    return Vmax_base * numpy.exp(-((pH - pH_opt) ** 2) / (2 * sigma**2))


def michaelis_menten_rate(
    S: typing.Union[float, numpy.ndarray],
    Vmax: typing.Union[float, numpy.ndarray],
    Km: typing.Union[float, numpy.ndarray],
) -> typing.Union[float, numpy.ndarray]:
    """
    Compute the Michaelis-Menten reaction rate.

    Parameters
    ----------
    S
        Substrate concentration.
    Vmax
        Maximum reaction rate.
    Km
        Michaelis constant.

    Returns
    -------
    float or numpy.ndarray
        Reaction rate.
    """
    return (Vmax * S) / (Km + S)


def enzyme_truth(
    pH: typing.Union[numpy.ndarray], enzyme_params: list[dict[str, float]]
) -> tuple[numpy.ndarray, list[numpy.ndarray]]:
    """
    Compute total and individual reaction rates for an arbitrary number of enzymes.

    Parameters
    ----------
    pH
        pH values.
    enzyme_params
        List of dictionaries, each containing the parameters for one enzyme:
        [{'Vmax': float, 'Km': float, 'pH_opt': float, 'sigma': float}, ...]

    Returns
    -------
    numpy.ndarray
        Total reaction rate and individual rates for each enzyme.
    """
    pH_arr = numpy.atleast_1d(pH)
    S: float = 1500.0
    rates: list[numpy.ndarray] = []
    for params in enzyme_params:
        v = gaussian_vmax_pH(pH_arr, params["Vmax"], params["pH_opt"], params["sigma"])
        r = michaelis_menten_rate(S, v, params["Km"])

        rates.append(numpy.atleast_1d(r))

    rates_arr = numpy.vstack(rates)
    total_rate = rates_arr.sum(axis=0)
    return total_rate, rates


def heteroscedastic_noise(
    x: numpy.ndarray,
    bounds: tuple[float, float],
    sigma_0: float = 0.03,
    sigma_1: float = 0.1,
    max_noise: float = 0.4,
    seed: typing.Optional[int] = None,
) -> numpy.ndarray:
    """
    Generate heteroscedastic noise using a polynomial variance structure,
    agnostic to the range of x-values.

    Parameters
    ----------
    x
        Inumpyut values.
    bounds
        Lower and upper bounds of the parameter space.
    sigma_0
        Base noise level.
    sigma_1
        Coefficient for quadratic term.
    max_noise
        Maximum allowable noise.

    Returns
    -------
    numpy.ndarray
        Heteroskedastic noise values for each input value.
    """
    # Normalize x to [0, 1]
    x_norm = (x - bounds[0]) / (bounds[1] - bounds[0])

    def polynomial_variance(x_norm: numpy.ndarray) -> numpy.ndarray:
        return sigma_0 + sigma_1 * x_norm**2

    # Compute standard deviation
    std_dev = numpy.clip(polynomial_variance(x_norm), 0, max_noise)
    rng = numpy.random.RandomState(seed)
    noise = rng.normal(loc=0, scale=std_dev)

    return noise


def symmetric_noise(
    x: numpy.ndarray,
    bounds: tuple[float, float],
    sigma_0: float = 0.03,
    sigma_1: float = 0.1,
    max_noise: float = 0.4,
    seed: typing.Optional[int] = None,
) -> numpy.ndarray:
    """
    Generate symmetric noise that increases towards the edges of the parameter space
    and is minimal in the center.

    Parameters
    ----------
    x
        Input values.
    bounds
        Lower and upper bounds of the parameter space.
    sigma_0
        Minimum noise level at the center.
    sigma_1
        Coefficient controlling the increase towards the edges.
    max_noise
        Maximum allowable noise.
    seed
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Symmetric noise values for each input value.
    """
    # Normalize x to [0, 1]
    x_norm = (x - bounds[0]) / (bounds[1] - bounds[0])

    def pol_variance(x_norm: numpy.ndarray) -> numpy.ndarray:
        return sigma_0 + sigma_1 * numpy.abs(2 * x_norm - 1)

    # Compute standard deviation
    std_dev = numpy.clip(pol_variance(x_norm), 0, max_noise)
    rng = numpy.random.RandomState(seed)
    noise = rng.normal(loc=0, scale=std_dev)

    return noise
