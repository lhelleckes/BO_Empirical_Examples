import numpy
import typing


def ratkowsky_curve(T, T_min=5, T_max=70, b=0.1, c=0.0002):
    """
    Computes the reaction rate using the extended Ratkowsky model.

    The extended Ratkowsky model accounts for enzyme activity increasing with temperature
    up to an optimal point and decreasing beyond it.

    Parameters
    ----------
    T : ndarray or float
        Temperatures at which to compute the reaction rates (°C).
    T_min : float, optional
        Minimum temperature for activity (default is 5°C).
    T_max : float, optional
        Maximum temperature at which activity becomes zero (default is 70°C).
    b : float, optional
        Growth constant (default is 0.1).
    c : float, optional
        Decay constant for the exponential term (default is 0.0002).

    Returns
    -------
    k : ndarray or float
        Reaction rate(s) corresponding to the input temperatures.

    Notes
    -----
    - Reaction rates are zero for T < T_min.
    - The model combines a quadratic term and an exponential decay term.
    """
    k = b * (T - T_min) ** 2 * (1 - numpy.exp(c * (T - T_max)))
    k = numpy.where((T < T_min) | (T > T_max), 0, k)
    return k


def gaussian_vmax(pH, Vmax_base, pH_opt, sigma):
    """
    Gaussian-like model for Vmax dependence on pH.

    Parameters
    ----------
    pH : float or numpy.ndarray
        pH values.
    Vmax_base : float
        Base Vmax value (peak height).
    pH_opt : float
        Optimal pH where the peak occurs.
    sigma : float
        Standard deviation controlling the peak width.

    Returns
    -------
    float or numpy.ndarray
        Adjusted Vmax values.
    """
    return Vmax_base * numpy.exp(-((pH - pH_opt) ** 2) / (2 * sigma**2))


def compute_rate(S, Vmax, Km):
    """
    Compute the Michaelis-Menten reaction rate.

    Parameters
    ----------
    S : float
        Substrate concentration.
    Vmax : float
        Maximum reaction rate.
    Km : float
        Michaelis constant.

    Returns
    -------
    float
        Reaction rate.
    """
    return (Vmax * S) / (Km + S)


def enzyme_truth(pH, enzyme_params):
    """
    Compute total and individual reaction rates for an arbitrary number of enzymes.

    Parameters
    ----------
    pH : numpy.ndarray
        pH values.
    enzyme_params : list of dict
        List of dictionaries, each containing the parameters for one enzyme:
        [{'Vmax': float, 'Km': float, 'pH_opt': float, 'sigma': float}, ...]

    Returns
    -------
    numpy.ndarray
        Total reaction rate and individual rates for each enzyme.
    """
    S = 15.0  # Fixed substrate concentration # TODO: Revaluate substrate and enzyme concentrations

    # Compute rates for all enzymes
    rates = [
        compute_rate(
            S,
            gaussian_vmax(pH, params["Vmax"], params["pH_opt"], params["sigma"]),
            params["Km"],
        )
        for params in enzyme_params
    ]
    total_rate = numpy.sum(rates, axis=0)
    return total_rate, rates


def heteroskedastic_noise(
    x: numpy.ndarray,
    bounds: typing.Tuple[float, float],
    sigma_0: float = 0.03,
    sigma_1: float = 0.1,
    max_noise: float = 0.4,
    seed: int = None,
) -> numpy.ndarray:
    """
    Generate heteroskedastic noise using a polynomial variance structure,
    agnostic to the range of x-values.

    Parameters
    ----------
    x : numpy.ndarray
        Inumpyut values.
    bounds : Tuple[float, float]
        Lower and upper bounds of the parameter space.
    sigma_0 : float
        Base noise level.
    sigma_1 : float
        Coefficient for quadratic term.
    max_noise : float
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
    bounds: typing.Tuple[float, float],
    sigma_0: float = 0.03,
    sigma_1: float = 0.1,
    max_noise: float = 0.4,
    seed: int = None,
) -> numpy.ndarray:
    """
    Generate symmetric noise that increases towards the edges of the parameter space
    and is minimal in the center.

    Parameters
    ----------
    x : numpy.ndarray
        Input values.
    bounds : Tuple[float, float]
        Lower and upper bounds of the parameter space.
    sigma_0 : float
        Minimum noise level at the center.
    sigma_1 : float
        Coefficient controlling the increase towards the edges.
    max_noise : float
        Maximum allowable noise.
    seed : int, optional
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
