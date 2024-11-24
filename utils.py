import numpy as np


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
    k = b * (T - T_min) ** 2 * (1 - np.exp(c * (T - T_max)))
    k = np.where(T < T_min, 0, k)  # Set rates to zero for T < T_min
    k = np.where(T > T_max, 0, k)  # Set rates to zero for T < T_min
    return k


def get_P_samples(
    k: float,
    S0: float = 100,
    time: np.ndarray = np.arange(0, 13, 2),
    sigma: float = 5,
) -> dict:
    """
    Computes product concentrations for given reaction rate and initial
    substrate rate.

    Parameters
    ----------
    k : float or ndarray
        Reaction rate in 1/h
    S0 : float
        initial substrate concentration in mM
    time : ndarray
        time points for measurements
    sigma : float
        std for Gaussian noise

    Returns
    -------
    P_noisy: ndarray
        P values as ndarray
    """
    if not (type(k) == float or type(k) == np.float64):
        raise ValueError(f"Expected float but got {type(k)} for k.")
    P_exact = S0 * (1 - np.exp(-k * time))  # Exact product formation
    noise = np.random.normal(
        0, sigma, size=len(P_exact)
    )  # Gaussian noise, std dev = 5 mM
    P_noisy = np.clip(
        P_exact + noise, 0, None
    )  # Add noise and clip to ensure non-negative
    return P_noisy


def extract_high_res_P_series(time: np.ndarray, k: float, S0: float = 100):
    """
    Computes the exact and high-resolution time series for a given k.

    Parameters
    ----------
    time : ndarray
        Low-resolution time samples
    k : ndarray or float
        Reaction rate(s) corresponding to the input temperatures
    S0 : float
        Initial substrate concentration

    Returns
    -------
    t_res : ndarray
        High-resolution times data.
    P_res : ndarray
        High-resolution product data.
    """
    t_res = np.linspace(min(time), max(time), 200)
    P_res = S0 * (1 - np.exp(-k * t_res))
    return (t_res, P_res)
