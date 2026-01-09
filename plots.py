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

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(11, 6))
    else:
        ax1 = ax

    # --- Price paths ---
    ax1.plot(
        p_unconstrained,
        color="#4B5563",
        lw=1.1,
        label="Price (unconstrained)"
    )

    ax1.plot(
        p_constrained,
        color="#111827",
        lw=1.8,
        label="Price (constrained)"
    )

    # --- Event markers ---
    ax1.axvline(
        constraint_start,
        ls="--",
        lw=0.8,
        alpha=0.5,
        label=f"Constraint start (t={constraint_start})"
    )

    ax1.axvline(
        crash_time,
        ls="-.",
        lw=0.9,
        alpha=0.6,
        label=f"Crash – worst fall (t={crash_time})"
    )

    ax1.axvline(
        replacement_time,
        ls=":",
        lw=1.0,
        alpha=0.6,
        label=f"Replacement (t={replacement_time})"
    )

    # --- Fundamental ---
    ax1.axhline(
        0.6 * F,
        ls=":",
        lw=0.8,
        alpha=0.5,
        label="0.6 × Fundamental"
    )

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")

    # --- Fraction constrained (secondary axis) ---
    ax2 = ax1.twinx()
    ax2.plot(
        frac_constrained,
        color="#2563EB",
        lw=1.2,
        alpha=0.75,
        label="Fraction constrained"
    )
    ax2.set_ylabel("Fraction constrained")

    # --- Legend ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        frameon=False
    )

    if title is not None:
        ax1.set_title(title)

    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_wealth_distributions(
    W_hist,
    limit_wealth,
    time_points,
    ax=None
):
    """
    Plot wealth distribution histograms at selected time points.

    Parameters
    ----------
    W_hist : array (T x N)
        Wealth history from the model
    limit_wealth : float
        Liquidity constraint threshold (theta * W0)
    time_points : list
        Time indices at which to plot wealth distributions
    ax : matplotlib axis, optional
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for t in time_points:
        if t < W_hist.shape[0]:
            ax.hist(
                W_hist[t, :],
                bins=30,
                alpha=0.5,
                density=True,
                label=f"t = {t}"
            )
        else:
            print(f"Warning: t={t} exceeds simulation horizon")

    ax.axvline(
        limit_wealth,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Constraint (θW₀ = {limit_wealth:.0f})"
    )

    ax.set_title("Wealth Distribution Dynamics")
    ax.set_xlabel("Wealth")
    ax.set_ylabel("Density")
    ax.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()


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
    title : str, optional
    ax : matplotlib axis, optional
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    colors = [
        "#111827",  # near-black
        "#1D4ED8",  # blue
        "#7C2D12",  # brown-red
        "#065F46",  # green
        "#6D28D9"   # purple
    ]

    for i, beta in enumerate(sorted(price_paths_by_beta.keys())):
        ax.plot(
            price_paths_by_beta[beta],
            color=colors[i % len(colors)],
            lw=0.9,
            label=f"β = {beta}"
        )

    ax.axhline(
        F,
        color="#9CA3AF",
        linestyle=":",
        linewidth=1.0,
        alpha=0.6,
        label="Fundamental"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")

    if title is not None:
        ax.set_title(title)

    ax.legend(frameon=False)

    if ax is None:
        plt.tight_layout()
        plt.show()


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
