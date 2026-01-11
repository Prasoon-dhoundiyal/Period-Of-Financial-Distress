import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Plot styling
# ============================================================

COLOR_UNCONSTRAINED = "#6B7280"   # mid gray
COLOR_CONSTRAINED   = "#111827"   # near black

COLOR_CONSTRAINT = "#374151"      # dark gray
COLOR_CRASH      = "#7F1D1D"      # dark red
COLOR_REPLACEMENT= "#065F46"      # dark green

COLOR_FUNDAMENTAL = "#4B5563"     # dark neutral


def plot_single_run_prices(
    p_unconstrained,
    p_constrained,
    frac_constrained,
    fraction_replaced,
    constraint_start,
    crash_time,
    replacement_time,
    F,
    title=None
):
    """
    Plot single-run price dynamics with constraint, crash,
    and replacement markers.

    Top panel:
        - Prices
        - Fraction constrained (secondary axis)

    Bottom panel:
        - Fraction replaced
    """

    fig, (ax1, ax3) = plt.subplots(
        2, 1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # ====================================================
    # TOP PANEL — Prices + Fraction Constrained
    # ====================================================

    # --- Price paths ---
    ax1.plot(
        p_constrained,
        color="#111827",
        lw=1.8,
        label="Price (constrained)"
    )

    ax1.plot(
        p_unconstrained,
        color="#6B7280",
        lw=1.4,
        linestyle="--",
        label="Price (unconstrained)"
    )

    # --- Event markers ---
    ax1.axvline(
        constraint_start,
        color=COLOR_CONSTRAINT,
        linestyle="--",
        linewidth=1.4,
        alpha=0.95,
        label=f"Constraint start (t={constraint_start})"
    )

    ax1.axvline(
        crash_time,
        color=COLOR_CRASH,
        linestyle="-.",
        linewidth=1.6,
        alpha=0.95,
        label=f"Crash (t={crash_time})"
    )

    ax1.axvline(
        replacement_time,
        color=COLOR_REPLACEMENT,
        linestyle=":",
        linewidth=1.6,
        alpha=0.95,
        label=f"Replacement start (t={replacement_time})"
    )

    # --- Fundamental reference ---
    ax1.axhline(
        0.6 * F,
        color=COLOR_FUNDAMENTAL,
        linestyle=":",
        linewidth=1.2,
        alpha=0.9,
        label="0.6 × Fundamental"
    )

    ax1.set_ylabel("Price")

    # --- Fraction constrained (secondary axis) ---
    ax2 = ax1.twinx()
    ax2.plot(
        frac_constrained,
        color="#2563EB",
        lw=1.2,
        alpha=0.8,
        label="Fraction constrained"
    )
    ax2.set_ylabel("Fraction constrained")

    # --- Combined legend (top panel) ---
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

    # ====================================================
    # BOTTOM PANEL — Fraction Replaced
    # ====================================================

    ax3.plot(
        fraction_replaced,
        color=COLOR_REPLACEMENT,
        lw=1.4,
        linestyle="-",
        label="Fraction replaced"
    )

    ax3.axvline(
        replacement_time,
        color=COLOR_REPLACEMENT,
        linestyle=":",
        linewidth=1.4,
        alpha=0.9
    )

    ax3.set_ylabel("Fraction replaced")
    ax3.set_xlabel("Time")
    ax3.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_logdiff_4_comparison(
    p_unconstrained_log,
    p_constrained_log,
    constraint_start,
    crash_time,
    replacement_time,
    title=None,
    ax=None
):
    """
    Plot 4-period log price differences for constrained vs unconstrained series.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))

    T = len(p_constrained_log)

    g_uncon = np.full(T, np.nan)
    g_con = np.full(T, np.nan)

    for t in range(4, T):
        g_uncon[t] = p_unconstrained_log[t] - p_unconstrained_log[t - 4]
        g_con[t] = p_constrained_log[t] - p_constrained_log[t - 4]

    # --- Plot ---
    ax.plot(
        g_uncon,
        color="#065F46",
        lw=1.0,
        alpha=0.8,
        label="Unconstrained (Δ₄ log price)"
    )

    ax.plot(
        g_con,
        color="#111827",
        lw=0.8,
        alpha=1.0,
        label="Constrained (Δ₄ log price)"
    )

    # --- Event markers ---
    ax.axvline(
        constraint_start,
        color=COLOR_CONSTRAINT,
        linestyle="--",
        linewidth=1.4,
        alpha=0.95,
        label=f"Constraint start (t={constraint_start})"
    )

    ax.axvline(
        crash_time,
        color=COLOR_CRASH,
        linestyle="-.",
        linewidth=1.6,
        alpha=0.95,
        label=f"Crash (t={crash_time})"
    )

    ax.axvline(
        replacement_time,
        color=COLOR_REPLACEMENT,
        linestyle=":",
        linewidth=1.6,
        alpha=0.95,
        label=f"Replacement (t={replacement_time})"
    )

    # --- Zero return reference ---
    ax.axhline(
        0.0,
        color=COLOR_FUNDAMENTAL,
        linewidth=1.2,
        alpha=0.9
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("4-period log difference")

    if title is not None:
        ax.set_title(title)

    ax.legend(frameon=False)

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

    Parameters
    ----------
    results : dict
        Output from monte_carlo_gallegati (valid runs only)
    F : float
        Fundamental price
    alpha_paths : float
        Transparency for individual paths
    ax : matplotlib axis, optional
    """

    if len(results) == 0:
        raise ValueError("No valid Monte Carlo runs to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # --- Collect price paths and event times ---
    price_paths = []
    constraint_times = []
    crash_times = []
    replacement_times = []

    for seed, entry in results.items():
        out = entry["output"]

        price_paths.append(np.exp(out["p_log"]))
        constraint_times.append(entry["constraint_start"])
        crash_times.append(entry["crash_time"])
        replacement_times.append(entry["replacement_time"])

    price_paths = np.array(price_paths)

    # --- Averages ---
    mean_price = price_paths.mean(axis=0)
    avg_constraint = int(np.mean(constraint_times))
    avg_crash = int(np.mean(crash_times))
    avg_replacement = int(np.mean(replacement_times))

    # --- Plot all paths ---
    for path in price_paths:
        ax.plot(
            path,
            color="#6B7280",
            alpha=alpha_paths,
            linewidth=0.8
        )

    # --- Mean path ---
    ax.plot(
        mean_price,
        color="#111827",
        linewidth=2.6,
        label="Mean price (valid runs)"
    )

    # --- Average event timings across valid runs ---
    ax.axvline(
        avg_constraint,
        color=COLOR_CONSTRAINT,
        linestyle="--",
        linewidth=1.6,
        alpha=0.9,
        label=f"Avg constraint start (t={avg_constraint})"
    )

    ax.axvline(
        avg_crash,
        color=COLOR_CRASH,
        linestyle="-.",
        linewidth=1.8,
        alpha=0.9,
        label=f"Avg crash (t={avg_crash})"
    )

    ax.axvline(
        avg_replacement,
        color=COLOR_REPLACEMENT,
        linestyle=":",
        linewidth=1.8,
        alpha=0.9,
        label=f"Avg replacement (t={avg_replacement})"
    )

    # --- Fundamental price reference ---
    ax.axhline(
        F,
        color=COLOR_FUNDAMENTAL,
        linestyle=":",
        linewidth=1.4,
        alpha=0.85,
        label="Fundamental"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title(
        "Monte Carlo Valid Realizations\n"
        f"(N = {len(results)} valid runs)"
    )

    ax.legend(frameon=False, loc="upper right")

    if ax is None:
        plt.tight_layout()
        plt.show()


