import numpy


def get_P_samples(
    k: float,
    S0: float = 100,
    time: numpy.ndarray = numpy.arange(0, 13, 2),
    sigma: float = 5,
    seed: int = None,
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
    seed : integer (optional)
        Set a seed for noise generation

    Returns
    -------
    P_noisy: ndarray
        P values as ndarray
    """
    if not (type(k) is float or type(k) is numpy.float64):
        raise ValueError(f"Expected float but got {type(k)} for k.")
    P_exact = S0 * (1 - numpy.exp(-k * time))  # Exact product formation
    rng = numpy.random.RandomState(seed)
    noise = rng.normal(0, sigma, size=len(P_exact))  # Gaussian noise, std dev = 5 mM
    P_noisy = numpy.clip(P_exact + noise, 0, None)  # Add noise and clip to ensure non-negative
    return P_noisy


def extract_high_res_P_series(time: numpy.ndarray, k: float, S0: float = 100):
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
    t_res = numpy.linspace(min(time), max(time), 200)
    P_res = S0 * (1 - numpy.exp(-k * t_res))
    return (t_res, P_res)
