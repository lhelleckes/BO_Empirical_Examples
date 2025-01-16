import typing

import ipywidgets
import matplotlib.pyplot
import numpy
import torch

from bo import fit_gp_model, generate_noisy_observations
from kinetics import enzyme_truth, symmetric_noise


class Colors:
    light_red = numpy.array((223, 83, 62)) / 255
    light_blue = numpy.array((69, 145, 247)) / 255
    dark_red = numpy.array((122, 25, 24)) / 255
    dark_blue = numpy.array((0, 0, 255)) / 255
    alt_blue = numpy.array((59, 117, 175)) / 255


def plot_symmetric_noise(
    x_range: numpy.ndarray,
    bounds: typing.Tuple[float, float],
    sigma_0: float = 0.01,
    sigma_1: float = 0.6,
    max_noise: float = 0.4,
    seed: int = None,
):
    """
    Plot the symmetric noise over a given range.

    Parameters
    ----------
    bounds : Tuple[float, float]
        Lower and upper bounds of the parameter space.
    sigma_0 : float
        Minimum noise level at the center.
    sigma_1 : float
        Coefficient controlling the increase towards the edges.
    max_noise : float
        Maximum allowable noise.
    """
    noise = symmetric_noise(x_range, bounds, sigma_0, sigma_1, max_noise, seed)
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.plot(x_range, noise, label="Symmetric Noise", color="orange")
    matplotlib.pyplot.xlabel("X")
    matplotlib.pyplot.ylabel("Noise")
    matplotlib.pyplot.title("Symmetric Noise Distribution")
    matplotlib.pyplot.grid()
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


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

    def get_value(widget):
        if isinstance(widget, dict):
            # Recursively process nested dictionaries
            return extract_widget_values(widget)
        return getattr(widget, "value", None)  # Get 'value' or None if not available

    # Use dictionary comprehension for concise mapping
    return {key: get_value(widget) for key, widget in widget_dict.items()}


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
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.plot(
        pH_range, total_rate, label="Total Reaction Rate", color="black", linewidth=2
    )

    # Plot individual enzyme rates
    for i, rates in enumerate(individual_rates, start=1):
        matplotlib.pyplot.plot(pH_range, rates, label=f"Enzyme {i} Rate", linestyle="--")

    # Formatting
    matplotlib.pyplot.xlabel("pH")
    matplotlib.pyplot.ylabel("Reaction Rate [U mL$^{-1}$]")
    matplotlib.pyplot.grid()
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


def plot_noisy_samples(sigma_0, sigma_1, max_noise, bounds, enzyme_params, seed=None):
    x = numpy.linspace(bounds[0], bounds[1], 500)

    y_noisy = generate_noisy_observations(
        x=x,
        bounds=bounds,
        truth_fn=enzyme_truth,
        noise_fn=symmetric_noise,
        truth_params=enzyme_params,
        noise_params={
            "sigma_0": sigma_0,
            "sigma_1": sigma_1,
            "max_noise": max_noise,
        },
        seed=seed,
    )

    y_true, _ = enzyme_truth(x, enzyme_params)  # Unpack total rate and individual rates

    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.plot(x, y_true, label="ground truth", color=Colors.light_blue, linewidth=2)
    matplotlib.pyplot.scatter(x, y_noisy, color=Colors.dark_blue, label="noisy observations")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


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


# Create Enzyme Widgets
def create_enzyme_widgets(config):
    """
    Create and display dynamic widgets for a given number of enzymes with configurable parameters.

    Parameters
    ----------
    config : dict
        Dictionary with configuration for each enzyme's sliders.
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

    for enzyme, params in config.items():
        sliders = {
            param: create_slider(**params[param], description=f"{param} ({enzyme})")
            for param in params
        }
        widgets_dict[enzyme] = sliders

    # Organize sliders into a layout
    sliders_layout = ipywidgets.VBox(
        [
            ipywidgets.HBox([slider for slider in sliders.values()])
            for sliders in widgets_dict.values()
        ]
    )

    # Create layout for widgets
    sliders_layout = ipywidgets.VBox(
        [
            ipywidgets.HBox(
                [
                    ipywidgets.VBox(list(widgets_dict[f"Enzyme_{i+1}"].values()))
                    for i in range(len(config))
                ]
            )
        ]
    )

    return sliders_layout, widgets_dict


# Create sliders for parameters with value labels enabled
def create_slider(min, max, step, value, description):
    """
    Create a slider widget.

    Parameters
    ----------
    min : float
        Minimum value of the slider.
    max : float
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
    return ipywidgets.FloatSlider(
        min=min,
        max=max,
        step=step,
        value=value,
        description=description,
        style={"description_width": "initial"},
        layout=ipywidgets.Layout(width="250px"),
    )


def plot_gp_fit(
    gp_model,
    train_x,
    train_y,
    bounds,
    ground_truth_fn=None,
    ground_truth_params=None,
    highlight_points=None,
    proposed_experiment=None,
    ax=None,
    title="GP Fit",
    ylabel="Output",
    xlabel="Input",
    show_legend=True,
):
    """
    Plot the GP fit, confidence intervals, and training data.

    Parameters
    ----------
    gp_model : botorch.models.SingleTaskGP
        The fitted GP model to visualize.
    train_x : torch.Tensor
        Training input points.
    train_y : torch.Tensor
        Training output values.
    bounds : tuple
        Bounds for the x-axis (min, max).
    ground_truth_fn : callable, optional
        Ground truth function for comparison.
    ground_truth_params : dict, optional
        Parameters for the ground truth function.
    highlight_points : list of tuple, optional
        Points to highlight as [(x, y), ...].
    proposed_experiment : float, optional
        Proposed experiment point to highlight.
    ax : matplotlib.axes.Axes, optional
        External axes to plot on. Creates a new figure if None.
    title : str, optional
        Title of the plot.
    ylabel : str, optional
        Y-axis label.
    xlabel : str, optional
        X-axis label.
    """
    if ax is None:
        fig, ax = matplotlib.pyplot.subplots(figsize=(10, 6))

    x_test = torch.linspace(bounds[0], bounds[1], 800).unsqueeze(-1)

    gp_model = fit_gp_model(train_x, train_y, bounds)

    gp_model.eval()
    with torch.no_grad():
        posterior = gp_model.posterior(x_test)
        mean = posterior.mean.numpy()
        lower, upper = posterior.mvn.confidence_region()

    ax.fill_between(
        x_test.squeeze(-1).numpy(),
        lower,
        upper,
        color=Colors.light_red,
        label="Confidence Interval",
    )

    if ground_truth_fn and ground_truth_params:
        x_ground_truth = numpy.linspace(bounds[0], bounds[1], 500)
        y_ground_truth, _ = ground_truth_fn(x_ground_truth, ground_truth_params)
        ax.plot(
            x_ground_truth,
            y_ground_truth,
            label="Ground Truth",
            color=Colors.light_blue,
            linewidth=2,
        )
    ax.scatter(
        train_x.numpy(),
        train_y.numpy(),
        color=Colors.dark_blue,
        label="Training Data",
        s=80,
        zorder=10,
    )
    # we might want to highlight the lastest observation
    if highlight_points:
        for x, y in highlight_points:
            ax.scatter(
                [x],
                [y],
                color="orange",
                s=100,
                label="Latest Observation",
                zorder=20,
            )

    # Highlight proposed experiment
    if proposed_experiment is not None:
        proposed_experiment = proposed_experiment.numpy()

        for exp in proposed_experiment:
            ax.axvline(
                x=exp,
                color="red",
                linestyle="--",
                linewidth=2,
            )

    ax.plot(
        x_test.numpy(),
        mean,
        label="GP Mean",
        color=Colors.dark_red,
        linewidth=2,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_legend:
        ax.legend()

    if ax is None:
        matplotlib.pyplot.show()


def plot_acquisition_function(
    acquisition_fn,
    candidates,
    bounds,
    ax,
    title=None,
    ylabel="Acquisition Value",
    xlabel="pH [-]",
    show_legend=True,
):
    """
    Plot the acquisition function.

    Parameters
    ----------
    acquisition_fn : Callable
        The acquisition function to evaluate.
    candidates : torch.Tensor
        Proposed candidates.
    bounds : tuple
        Bounds of the input space.
    ax : matplotlib.axes.Axes
        The axis to plot on.
    title : str, optional
        Title for the plot.
    ylabel : str, optional
        Label for the y-axis.
    xlabel : str, optional
        Label for the x-axis.
    show_legend : bool, optional
        Whether to display the legend.
    """
    x_test = torch.linspace(bounds[0], bounds[1], 500).unsqueeze(-1)
    with torch.no_grad():
        acquisition_values = acquisition_fn(x_test.unsqueeze(-2)).detach().numpy().squeeze()
    ax.plot(
        x_test.squeeze(-1).numpy(),
        acquisition_values,
        color=Colors.alt_blue,
        label="Acquisition Value" if show_legend else None,
    )

    candidates = candidates.numpy()

    for candidate in candidates:
        ax.axvline(
            x=candidate,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Proposed Experiment" if show_legend else None,
        )
    if title:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.set_xlabel(xlabel)
    if show_legend:
        ax.legend()


def plot_combined_gp_and_acquisition_from_results(
    results, bounds, truth_fn=None, truth_params=None
):
    """
    Plot GP models and acquisition functions for multiple rounds of BO using results dictionary.

    Parameters
    ----------
    results : dict
        Dictionary containing:
        - gp_models: List of GP models from each round.
        - acquisition_fns: List of acquisition functions from each round.
        - train_x_per_round: List of training inputs for each round.
        - train_y_per_round: List of training outputs for each round.
        - candidates_per_round: List of proposed candidates for each round.
    bounds : tuple
        Bounds of the input domain (min, max).
    truth_fn : callable, optional
        Ground truth function for comparison (if available).
    truth_params : dict, optional
        Parameters for the ground truth function (if any).
    """
    gp_models = results["gp_models"]
    acquisition_fns = results["acquisition_fns"]
    train_x_per_round = results["train_x_per_round"]
    train_y_per_round = results["train_y_per_round"]
    candidates_per_round = results["candidates_per_round"]

    num_rounds = len(gp_models)
    fig, axes = matplotlib.pyplot.subplots(
        num_rounds,
        2,
        figsize=(12, num_rounds * 2.5),
        gridspec_kw={"wspace": 0.3, "hspace": 0.6},
    )

    # Ensure axes is a 2D array even if num_rounds == 1
    if num_rounds == 1:
        axes = [axes]

    for round_idx in range(num_rounds):
        gp_ax, acq_ax = axes[round_idx]

        # Highlight the latest observation for the current round
        highlight_points = None
        if round_idx > 0:
            latest_candidates = train_x_per_round[round_idx][-len(candidates_per_round[round_idx]) :]
            latest_observation = train_y_per_round[round_idx][
                -len(candidates_per_round[round_idx]) :
            ]

            highlight_points = [
                (x.item(), y.item()) for x, y in zip(latest_candidates, latest_observation)
            ]

        # GP plot
        plot_gp_fit(
            gp_model=gp_models[round_idx],
            train_x=train_x_per_round[round_idx],
            train_y=train_y_per_round[round_idx],
            bounds=bounds,
            ground_truth_fn=truth_fn,
            ground_truth_params=truth_params,
            highlight_points=highlight_points,
            proposed_experiment=candidates_per_round[round_idx],
            ax=gp_ax,
            title=f"Round {round_idx + 1} - GP Model",
            ylabel="Reaction Rate",
            xlabel="pH [-]",  # X-axis label for every plot
            show_legend=False,
        )

        # Acquisition function plot
        plot_acquisition_function(
            acquisition_fn=acquisition_fns[round_idx],
            candidates=candidates_per_round[round_idx],
            bounds=bounds,
            ax=acq_ax,
            title="Acquisition Function",
            ylabel="Acquisition Value",
            xlabel="pH [-]",  # X-axis label for every plot
            show_legend=False,
        )

    matplotlib.pyplot.show()


def plot_selected_rounds(results, bounds, selected_rounds, truth_fn=None, truth_params=None):
    gp_models = results["gp_models"]
    acquisition_fns = results["acquisition_fns"]
    train_x_per_round = results["train_x_per_round"]
    train_y_per_round = results["train_y_per_round"]
    candidates_per_round = results["candidates_per_round"]

    num_selected = len(selected_rounds)
    fig, axes = matplotlib.pyplot.subplots(
        num_selected * 2, 1, figsize=(8, num_selected * 7.5), gridspec_kw={"hspace": 0.3}
    )
    axes = axes if num_selected > 1 else [axes]  # Handle single-row case

    for idx, round_idx in enumerate(selected_rounds):
        gp_ax = axes[idx * 2]
        acq_ax = axes[idx * 2 + 1]

        highlight_points = None
        if round_idx > 0:
            latest_x = train_x_per_round[round_idx][-1].item()
            latest_y = train_y_per_round[round_idx][-1].item()
            highlight_points = [(latest_x, latest_y)]

        # GP plot
        plot_gp_fit(
            gp_model=gp_models[round_idx],
            train_x=train_x_per_round[round_idx],
            train_y=train_y_per_round[round_idx],
            bounds=bounds,
            ground_truth_fn=truth_fn,
            ground_truth_params=truth_params,
            highlight_points=highlight_points,
            proposed_experiment=candidates_per_round[round_idx],
            ax=gp_ax,
            title=f"Round {round_idx + 1} - GP Model",
            ylabel="Reaction Rate",
            xlabel=None,
            show_legend=(idx == 1),  # Show legend only once
        )

        # Acquisition function plot
        plot_acquisition_function(
            acquisition_fn=acquisition_fns[round_idx],
            candidates=candidates_per_round[round_idx],
            bounds=bounds,
            ax=acq_ax,
            show_legend=(idx == 0),  # Show legend only once
        )

    matplotlib.pyplot.savefig("simple_example.png")
    matplotlib.pyplot.show()
