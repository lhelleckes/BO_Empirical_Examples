from typing import Callable, Dict, Optional, List, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from IPython.display import display
from ipywidgets import HBox, VBox, interactive_output, FloatSlider, Layout
from scipy.stats.qmc import Sobol


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


### First Empirical Example


def gaussian_vmax(pH, Vmax_base, pH_opt, sigma):
    """
    Gaussian-like model for Vmax dependence on pH.

    Parameters
    ----------
    pH : float or np.ndarray
        pH values.
    Vmax_base : float
        Base Vmax value (peak height).
    pH_opt : float
        Optimal pH where the peak occurs.
    sigma : float
        Standard deviation controlling the peak width.

    Returns
    -------
    float or np.ndarray
        Adjusted Vmax values.
    """
    return Vmax_base * np.exp(-((pH - pH_opt) ** 2) / (2 * sigma**2))


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


# Create sliders for parameters with value labels enabled
def create_slider(min_val, max_val, step, value, description):
    """
    Create a slider widget.

    Parameters
    ----------
    min_val : float
        Minimum value of the slider.
    max_val : float
        Maximum value of the slider.
    step : float
        Step size for the slider.
    value : float
        Initial value of the slider.
    description : str
        Description label for the slider.

    Returns
    -------
    ipywidgets.FloatSlider
        Configured slider widget.
    """
    return FloatSlider(
        min=min_val,
        max=max_val,
        step=step,
        value=value,
        description=description,
        style={"description_width": "initial"},
        layout=Layout(width="250px"),
    )


# Create Enzyme Widgets
def create_enzyme_widgets(num_enzymes, config=None):
    """
    Create and display dynamic widgets for a given number of enzymes with configurable parameters.

    Parameters
    ----------
    num_enzymes : int
        Number of enzymes to create widgets for.
    config : dict, optional
        Dictionary with configuration for each enzyme's sliders. If None, defaults are used.
        Example:
        {
            "Enzyme_1": {
                "Vmax": {"min": 0.5, "max": 2.0, "step": 0.1, "value": 1.0},
                "Km": {"min": 0.1, "max": 1.0, "step": 0.1, "value": 0.5},
                "pH_opt": {"min": 4.0, "max": 8.0, "step": 0.1, "value": 6.0},
                "sigma": {"min": 0.1, "max": 2.0, "step": 0.1, "value": 0.5},
            },
            ...
        }

    Returns
    -------
    ipywidgets.VBox
        A VBox containing all the enzyme widgets organized in HBoxes.
    """
    widgets_dict = {}
    for i in range(num_enzymes):
        enzyme_key = f"Enzyme_{i+1}"
        enzyme_config = config.get(enzyme_key, {}) if config else {}

        # Default slider parameters
        widgets_dict[enzyme_key] = {
            "Vmax": create_slider(
                enzyme_config.get("Vmax", {}).get("min", 0.5),
                enzyme_config.get("Vmax", {}).get("max", 2.0),
                enzyme_config.get("Vmax", {}).get("step", 0.1),
                enzyme_config.get("Vmax", {}).get("value", 1.0),
                f"Vmax{i+1}",
            ),
            "Km": create_slider(
                enzyme_config.get("Km", {}).get("min", 0.1),
                enzyme_config.get("Km", {}).get("max", 1.0),
                enzyme_config.get("Km", {}).get("step", 0.1),
                enzyme_config.get("Km", {}).get("value", 0.5),
                f"Km{i+1}",
            ),
            "pH_opt": create_slider(
                enzyme_config.get("pH_opt", {}).get("min", 4.0),
                enzyme_config.get("pH_opt", {}).get("max", 8.0),
                enzyme_config.get("pH_opt", {}).get("step", 0.1),
                enzyme_config.get("pH_opt", {}).get("value", 6.0),
                f"pH_opt{i+1}",
            ),
            "sigma": create_slider(
                enzyme_config.get("sigma", {}).get("min", 0.1),
                enzyme_config.get("sigma", {}).get("max", 2.0),
                enzyme_config.get("sigma", {}).get("step", 0.1),
                enzyme_config.get("sigma", {}).get("value", 0.5),
                f"Sigma{i+1}",
            ),
        }

    # Create layout for widgets
    sliders_layout = VBox(
        [
            HBox(
                [
                    VBox(list(widgets_dict[f"Enzyme_{i+1}"].values()))
                    for i in range(num_enzymes)
                ]
            )
        ]
    )

    return sliders_layout, widgets_dict


def enzyme_truth(pH, enzyme_params):
    """
    Compute total and individual reaction rates for an arbitrary number of enzymes.

    Parameters
    ----------
    pH : np.ndarray
        pH values.
    enzyme_params : list of dict
        List of dictionaries, each containing the parameters for one enzyme:
        [{'Vmax': float, 'Km': float, 'pH_opt': float, 'sigma': float}, ...]

    Returns
    -------
    np.ndarray
        Total reaction rate and individual rates for each enzyme.
    """
    S = 1.0  # Fixed substrate concentration

    # Compute rates dynamically for all enzymes
    rates = [
        compute_rate(
            S,
            gaussian_vmax(pH, params["Vmax"], params["pH_opt"], params["sigma"]),
            params["Km"],
        )
        for params in enzyme_params
    ]
    total_rate = np.sum(rates, axis=0)
    return total_rate, rates


def plot_enzyme_truth(pH_range, enzyme_params):
    """
    Plot enzyme truth for a given pH_range with the selected parameters.

    Parameters
    ----------
    pH_range : np.ndarray
        Range of pH values.
    enzyme_params : list of dict
        Parameters for each enzyme.
    """
    # Compute the total rate and individual enzyme rates
    total_rate, individual_rates = enzyme_truth(pH_range, enzyme_params)

    # Plot the total reaction rate
    plt.figure(figsize=(10, 6))
    plt.plot(
        pH_range, total_rate, label="Total Reaction Rate", color="black", linewidth=2
    )

    # Plot individual enzyme rates
    for i, rates in enumerate(individual_rates, start=1):
        plt.plot(pH_range, rates, label=f"Enzyme {i} Rate", linestyle="--")

    # Formatting
    plt.xlabel("pH")
    plt.ylabel("Reaction Rate")
    plt.grid()
    plt.legend()
    plt.show()


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
    seed: int = None,
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
    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0, scale=std_dev)

    return noise


def symmetric_noise(
    x: np.ndarray,
    sigma_0: float = 0.03,
    sigma_1: float = 0.1,
    max_noise: float = 0.4,
    seed: int = None,
) -> np.ndarray:
    """
    Generate symmetric noise that increases towards the edges of the parameter space
    and is minimal in the center.

    Parameters
    ----------
    x : np.ndarray
        Input values.
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
    np.ndarray
        Symmetric noise values for each input value.
    """
    # Normalize x to [0, 1]
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))

    def pol_variance(x_norm: np.ndarray) -> np.ndarray:
        return sigma_0 + sigma_1 * np.abs(2 * x_norm - 1)

    # Compute standard deviation
    std_dev = np.clip(pol_variance(x_norm), 0, max_noise)
    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0, scale=std_dev)

    return noise


def generate_noisy_observations(
    x: np.ndarray,
    truth_fn: Callable[
        [np.ndarray, List[Dict[str, float]]], Tuple[np.ndarray, List[np.ndarray]]
    ],
    noise_fn: Callable[[np.ndarray, float, float, float, int], np.ndarray],
    truth_params: List[Dict[str, float]],
    sigma_0: Optional[float] = None,
    sigma_1: Optional[float] = None,
    max_noise: Optional[float] = None,
    noise_params: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate noisy observations from a ground truth function and noise function,
    replacing negative values with samples from a half-normal distribution.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    truth_fn : Callable
        Ground truth function to compute true values. Accepts `x` and `truth_params`.
    noise_fn : Callable
        Function to generate heteroskedastic noise.
    truth_params : List[Dict[str, float]]
        Parameters for the ground truth function.
    sigma_0 : float, optional
        Base noise level.
    sigma_1 : float, optional
        Coefficient for the quadratic term in the noise function.
    max_noise : float, optional
        Maximum allowable noise level.
    noise_params : Dict[str, float], optional
        Dictionary of noise parameters containing `sigma_0`, `sigma_1`, and `max_noise`.
        Overrides individual parameters if provided.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy observations with non-negative values.
    """
    # Consolidate noise parameters
    sigma_0 = noise_params.get("sigma_0", sigma_0) if noise_params else sigma_0
    sigma_1 = noise_params.get("sigma_1", sigma_1) if noise_params else sigma_1
    max_noise = noise_params.get("max_noise", max_noise) if noise_params else max_noise

    # Ensure all noise parameters are provided
    missing_params = [
        param
        for param, value in {
            "sigma_0": sigma_0,
            "sigma_1": sigma_1,
            "max_noise": max_noise,
        }.items()
        if value is None
    ]
    if missing_params:
        raise ValueError(f"Missing noise parameters: {', '.join(missing_params)}")

    # Compute ground truth
    y_true, _ = truth_fn(x, truth_params)

    # Compute heteroskedastic noise
    noise = noise_fn(x, sigma_0, sigma_1, max_noise, seed)
    y_noisy = y_true + noise

    # Replace negative values with half-normal samples
    negative_indices = y_noisy < 0
    if np.any(negative_indices):
        rng = np.random.default_rng(seed)  # Modern RNG
        replacement_values = rng.normal(
            loc=0, scale=sigma_0, size=np.sum(negative_indices)
        )
        y_noisy[negative_indices] = np.abs(replacement_values)

    return y_noisy


def extract_widget_values(widget_dict):
    """
    Recursively extract current values from a dictionary of widgets.

    Parameters
    ----------
    widget_dict : dict
        A dictionary (potentially nested) of widgets.

    Returns
    -------
    dict
        Dictionary of current widget values with the same structure as the input.
    """
    values = {}
    for key, widget in widget_dict.items():
        if isinstance(widget, dict):
            # Recursively handle nested dictionaries
            values[key] = extract_widget_values(widget)
        elif hasattr(widget, "value"):
            # Extract the value if the widget has a 'value' attribute
            values[key] = widget.value
        else:
            # Handle cases where the widget has no 'value' (fallback to None)
            values[key] = None
    return values


def plot_symmetric_noise(
    x_range: np.ndarray,
    sigma_0: float = 0.01,
    sigma_1: float = 0.6,
    max_noise: float = 0.4,
    seed: int = None,
):
    """
    Plot the symmetric noise over a given range.

    Parameters
    ----------
    x_range : np.ndarray
        Input range of x values.
    sigma_0 : float
        Minimum noise level at the center.
    sigma_1 : float
        Coefficient controlling the increase towards the edges.
    max_noise : float
        Maximum allowable noise.
    """
    noise = symmetric_noise(x_range, sigma_0, sigma_1, max_noise, seed)
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, noise, label="Symmetric Noise", color="orange")
    plt.xlabel("X")
    plt.ylabel("Noise")
    plt.title("Symmetric Noise Distribution")
    plt.grid()
    plt.legend()
    plt.show()


def generate_sobol_points(
    num_points: int, range_min: float, range_max: float, seed: int = None
) -> np.ndarray:
    """
    Generate Sobol sequence points within a specified range.

    Parameters
    ----------
    num_points : int
        Number of points to generate.
    range_min : float
        Minimum value of the range.
    range_max : float
        Maximum value of the range.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sobol points scaled to the specified range.
    """
    range_min = np.atleast_1d(range_min)
    range_max = np.atleast_1d(range_max)

    dimensions = len(range_min)

    rng = np.random.RandomState(seed)
    sobol_gen = Sobol(d=dimensions, scramble=True, seed=rng)

    # bitwise operation to check if num_points is a power of 2
    if (num_points & (num_points - 1)) == 0:
        points = sobol_gen.random_base2(m=int(np.log2(num_points)))
    else:
        points = sobol_gen.random(n=num_points)

    return range_min + points.flatten() * (range_max - range_min)
