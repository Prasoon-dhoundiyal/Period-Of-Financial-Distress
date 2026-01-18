import numpy as np
import matplotlib.pyplot as plt
from events import detect_events   # REQUIRED, already in your project

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
    Single-run plot with TWO IDENTICAL PANELS.
    Top panel overlays fraction constrained.
    Bottom panel overlays fraction replaced.
    All prices, vertical lines, and fundamentals appear in BOTH panels.
    """

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(11, 9),
        sharex=True
    )

    def draw_panel(ax, fraction, fraction_label, fraction_color):

        ax.plot(p_constrained, color=COLOR_CONSTRAINED, lw=1.8,
                label="Price (constrained)")
        ax.plot(p_unconstrained, color=COLOR_UNCONSTRAINED, lw=1.4,
                linestyle="--", label="Price (unconstrained)")

        ax.axvline(constraint_start, color=COLOR_CONSTRAINT,
                   linestyle="--", linewidth=1.4,
                   label=f"Constraint start (t={constraint_start})")

        ax.axvline(crash_time, color=COLOR_CRASH,
                   linestyle="-.", linewidth=1.6,
                   label=f"Crash (t={crash_time})")

        ax.axvline(replacement_time, color=COLOR_REPLACEMENT,
                   linestyle=":", linewidth=1.6,
                   label=f"Replacement start (t={replacement_time})")

        ax.axhline(0.6 * F, color=COLOR_FUNDAMENTAL,
                   linestyle=":", linewidth=1.2,
                   label="0.6 × Fundamental")

        ax.set_ylabel("Price")

        ax_f = ax.twinx()
        ax_f.plot(fraction, color=fraction_color, lw=1.3,
                  alpha=0.85, label=fraction_label)
        ax_f.set_ylabel(fraction_label)

        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax_f.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2,
                  loc="upper right", frameon=False)

    draw_panel(ax_top, frac_constrained,
               "Fraction constrained", "#2563EB")

    if title is not None:
        ax_top.set_title(title)

    draw_panel(ax_bot, fraction_replaced,
               "Fraction replaced", COLOR_REPLACEMENT)

    ax_bot.set_xlabel("Time")

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

    ax.plot(g_uncon, color=COLOR_REPLACEMENT, lw=1.0,
            alpha=0.8, label="Unconstrained (Δ₄ log price)")
    ax.plot(g_con, color=COLOR_CONSTRAINED, lw=0.8,
            alpha=1.0, label="Constrained (Δ₄ log price)")

    ax.axvline(constraint_start, color=COLOR_CONSTRAINT,
               linestyle="--", linewidth=1.4)
    ax.axvline(crash_time, color=COLOR_CRASH,
               linestyle="-.", linewidth=1.6)
    ax.axvline(replacement_time, color=COLOR_REPLACEMENT,
               linestyle=":", linewidth=1.6)

    ax.axhline(0.0, color=COLOR_FUNDAMENTAL,
               linewidth=1.2, alpha=0.9)

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
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for t in time_points:
        if t < W_hist.shape[0]:
            ax.hist(W_hist[t, :], bins=30,
                    alpha=0.5, density=True,
                    label=f"t = {t}")

    ax.axvline(limit_wealth, color="red",
               linestyle="--", linewidth=1.2,
               label=f"Constraint (θW₀ = {limit_wealth:.0f})")

    ax.set_title("Wealth Distribution Dynamics")
    ax.set_xlabel("Wealth")
    ax.set_ylabel("Density")
    ax.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()


# ============================================================
# UPDATED FUNCTION (ONLY CHANGE)
# ============================================================

def plot_beta_comparison_constrained(
    price_paths_by_beta,
    p_log,
    p_log_expect,
    frac_constrained,
    F=10.0,
    title=None
):
    """
    SINGLE-RUN beta comparison with event detection.

    Top panel:
        Price vs expected price (4-period log difference)
        with detected events

    Bottom panel:
        Constrained price paths for different beta values
        (unchanged)
    """

    # --- Event detection (existing logic) ---
    constraint_start, crash_time, _, replacement_time = detect_events(
        p_log,
        p_log_expect,
        frac_constrained,
        F=F
    )

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True
    )

    # --- Top panel: Δ4 log(price vs expectation) ---
    T = len(p_log)
    g_price = np.full(T, np.nan)
    g_expect = np.full(T, np.nan)

    for t in range(4, T):
        g_price[t] = p_log[t] - p_log[t - 4]
        g_expect[t] = p_log_expect[t] - p_log_expect[t - 4]

    ax_top.plot(g_price, color=COLOR_CONSTRAINED,
                lw=1.4, label="Price (Δ₄ log)")
    ax_top.plot(g_expect, color="#1D4ED8",
                lw=1.4, alpha=0.85,
                label="Expected price (Δ₄ log)")

    ax_top.axvline(constraint_start, color=COLOR_CONSTRAINT,
                   linestyle="--", linewidth=1.4)
    ax_top.axvline(crash_time, color=COLOR_CRASH,
                   linestyle="-.", linewidth=1.6)
    ax_top.axvline(replacement_time, color=COLOR_REPLACEMENT,
                   linestyle=":", linewidth=1.6)

    ax_top.axhline(0.0, color=COLOR_FUNDAMENTAL,
                   linewidth=1.2, alpha=0.9)

    ax_top.set_ylabel("4-period log difference")
    ax_top.set_title("Price vs expected price (single run)")
    ax_top.legend(frameon=False)

    # --- Bottom panel: beta comparison (UNCHANGED) ---
    colors = [
        COLOR_CONSTRAINED,
        "#1D4ED8",
        "#7C2D12",
        "#065F46",
        "#6D28D9"
    ]

    for i, beta in enumerate(sorted(price_paths_by_beta.keys())):
        ax_bot.plot(price_paths_by_beta[beta],
                    color=colors[i % len(colors)],
                    lw=0.9, label=f"β = {beta}")

    ax_bot.axhline(F, color=COLOR_FUNDAMENTAL,
                   linestyle=":", linewidth=1.0,
                   alpha=0.6, label="Fundamental")

    ax_bot.set_xlabel("Time")
    ax_bot.set_ylabel("Price")

    if title is not None:
        ax_bot.set_title(title)

    ax_bot.legend(frameon=False)

    plt.tight_layout()
    plt.show()


def plot_monte_carlo_paths(
    results,
    F=10.0,
    alpha_paths=0.08,
    ax=None
):
    if len(results) == 0:
        raise ValueError("No valid Monte Carlo runs to plot.")

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_ax = True

    price_paths = []
    constraint_times = []
    crash_times = []
    replacement_times = []

    for seed, entry in results.items():
        out = entry["output"]
        price_paths.append(np.exp(out["p_log"]))
        constraint_times.append(entry["constraint_start"])
        crash_times.append(entry["crash_time"])
        replacement_times.append(entry["replacement_start"])

    price_paths = np.asarray(price_paths)

    mean_price = price_paths.mean(axis=0)
    avg_constraint = int(np.mean(constraint_times))
    avg_crash = int(np.mean(crash_times))
    avg_replacement = int(np.mean(replacement_times))

    for path in price_paths:
        ax.plot(path, color=COLOR_UNCONSTRAINED,
                alpha=alpha_paths, linewidth=0.8)

    ax.plot(mean_price, color=COLOR_CONSTRAINED,
            linewidth=2.6, label="Mean price (valid runs)")

    ax.axvline(avg_constraint, color=COLOR_CONSTRAINT,
               linestyle="--", linewidth=1.6)
    ax.axvline(avg_crash, color=COLOR_CRASH,
               linestyle="-.", linewidth=1.8)
    ax.axvline(avg_replacement, color=COLOR_REPLACEMENT,
               linestyle=":", linewidth=1.8)

    ax.axhline(F, color=COLOR_FUNDAMENTAL,
               linestyle=":", linewidth=1.4,
               alpha=0.85, label="Fundamental")

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title(
        "Monte Carlo Valid Realizations\n"
        f"(N = {len(results)} valid runs)"
    )

    ax.legend(frameon=False, loc="upper right")

    if created_ax:
        plt.tight_layout()
        plt.show()
