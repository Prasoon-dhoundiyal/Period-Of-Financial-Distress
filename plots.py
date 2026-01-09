import numpy as np
import matplotlib.pyplot as plt


def plot_single_run_prices(
    p_unconstrained,
    p_constrained,
    frac_constrained,
    constraint_start,
    crash_time,
    replacement_time,
    F,
    title=None,
    ax=None
):
    """
    Plot single-run price dynamics with constraint, crash,
    and replacement markers.
    """
    pass


def plot_wealth_distributions(
    W_hist,
    limit_wealth,
    time_points,
    ax=None
):
    """
    Plot wealth distribution histograms at selected time points.
    """
    pass


def plot_monte_carlo_paths(
    results,
    F=10.0,
    alpha_paths=0.08,
    ax=None
):
    """
    Plot Monte Carlo ensemble of valid realizations.
    Uses stored outputs only (no reruns).
    """
    pass


def plot_beta_comparison_constrained(
    price_paths_by_beta,
    F=10.0,
    title=None,
    ax=None
):
    """
    Plot constrained price paths for different beta values,
    using precomputed outputs.

    Parameters
    ----------
    price_paths_by_beta : dict
        {beta_value: price_path_array}
    F : float
        Fundamental price
    """
    pass
