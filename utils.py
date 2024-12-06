import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import VBox, HBox, interactive_output
from typing import Callable, Dict, Optional
from IPython.display import display


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
    k = np.where((T < T_min) | (T > T_max), 0, k)
    return k


def get_P_samples(
    k: float,
    S0: float = 100,
    time: np.ndarray = np.arange(0, 13, 2),
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
    if not (type(k) == float or type(k) == np.float64):
        raise ValueError(f"Expected float but got {type(k)} for k.")
    P_exact = S0 * (1 - np.exp(-k * time))  # Exact product formation
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, sigma, size=len(P_exact))  # Gaussian noise, std dev = 5 mM
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


# Create sliders for parameters with value labels enabled
def create_slider(min_val, max_val, step, value, description):
    return widgets.FloatSlider(
        min=min_val,
        max=max_val,
        step=step,
        value=value,
        description=description,
        style={"description_width": "initial"},
    )


def get_slider_values(sliders):
    """
    Extract current values from the sliders.

    Parameters
    ----------
    sliders : dict
        A dictionary of slider widgets.

    Returns
    -------
    dict
        Dictionary of current slider values.
    """
    return {key: slider.value for key, slider in sliders.items()}

def heteroskedastic_noise(
    x: np.ndarray,
    sigma_0: float = 0.03,
    sigma_1: float = 0.1,
    max_noise: float = 0.4,
) -> np.ndarray:
    """
    Generate heteroskedastic noise using a polynomial variance structure,
    agnostic to the range of x-values.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    sigma_0 : float
        Base noise level.
    sigma_1 : float
        Coefficient for quadratic term.
    max_noise : float
        Maximum allowable noise.

    Returns
    -------
    np.ndarray
        Heteroskedastic noise values for each input value.
    """
    # Normalize x to [0, 1]
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))

    def polynomial_variance(x_norm: np.ndarray) -> np.ndarray:
        return sigma_0 + sigma_1 * x_norm**2

    # Compute standard deviation
    std_dev = np.clip(polynomial_variance(x_norm), 0, max_noise)

    noise = np.random.normal(loc=0, scale=std_dev)

    return noise


def generate_noisy_observations(
    x: np.ndarray,
    truth_fn: Callable[[np.ndarray, Dict[str, float]], np.ndarray],
    noise_fn: Callable[[np.ndarray, float, float, float], np.ndarray],
    truth_params: Dict[str, float],
    sigma_0: Optional[float] = None,
    sigma_1: Optional[float] = None,
    max_noise: Optional[float] = None,
    noise_params: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Generate noisy observations from a ground truth function and noise function,
    replacing negative values with values from a half-normal distribution.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    truth_fn : Callable
        Ground truth function to compute true values. Accepts `x` and `truth_params`.
    noise_fn : Callable
        Function to generate heteroskedastic noise.
    truth_params : Dict[str, float]
        Parameters for the ground truth function.
    sigma_0 : float, optional
        Base noise level (default is None).
    sigma_1 : float, optional
        Coefficient for the quadratic term in the noise function (default is None).
    max_noise : float, optional
        Maximum allowable noise level (default is None).
    noise_params : Dict[str, float], optional
        Dictionary of noise parameters containing `sigma_0`, `sigma_1`, and `max_noise`.
        Overrides individual parameters if provided.

    Returns
    -------
    np.ndarray
        Noisy observations with non-negative values.
    """
    # Extract noise parameters from `noise_params` if provided
    if noise_params:
        sigma_0 = noise_params.get("sigma_0", sigma_0)
        sigma_1 = noise_params.get("sigma_1", sigma_1)
        max_noise = noise_params.get("max_noise", max_noise)

    # Ensure all parameters are set
    if sigma_0 is None or sigma_1 is None or max_noise is None:
        raise ValueError(
            "Noise parameters must be provided either individually or in `noise_params`."
        )

    # Compute ground truth and noise
    y_true = truth_fn(x, **truth_params)
    noise = noise_fn(x, sigma_0, sigma_1, max_noise)
    y_noisy = y_true + noise

    # Replace negative values with half-normal samples
    negative_indices = y_noisy < 0
    if np.any(negative_indices):
        # Scale replacement values based on local noise properties
        scale = np.maximum(np.std(noise[negative_indices]), max_noise / 2)
        replacement_values = np.abs(np.random.normal(loc=0, scale=scale, size=np.sum(negative_indices)))
        y_noisy[negative_indices] = replacement_values

    return y_noisy
