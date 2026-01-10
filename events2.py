"""
Event detection for Model 2.
Detects:
- constraint_start
- replacement_time (expectations < 60% F, only after constraint start)
- crash_time (first new worst 4-step negative outlier, benchmarked on all past values,
              but occurring only within the window)
"""

import numpy as np


def detect_events_model2(p_log, p_expect, frac_constrained, F=10.0):
    T = len(p_log)

    # ----------------------------
    # 4-step growth rate
    # ----------------------------
    g = np.full(T, np.nan)
    for t in range(4, T):
        g[t] = p_log[t] - p_log[t - 4]

    # ----------------------------
    # Constraint start
    # ----------------------------
    idx = np.where(frac_constrained > 0)[0]
    constraint_start = idx[0] if len(idx) > 0 else None

    # ----------------------------
    # Replacement time
    # (expectations < 60% F, only AFTER constraint start)
    # ----------------------------
    replacement_time = None
    if constraint_start is not None:
        threshold = np.log(0.6 * F)
        for t in range(constraint_start + 1, T):
            if p_expect[t] < threshold:
                replacement_time = t
                break

    # ----------------------------
    # Crash time
    # First time a 4-step return becomes
    # the worst observed so far across ALL history,
    # but only if it occurs inside the window
    # [constraint_start, replacement_time)
    # ----------------------------
    crash_time = None
    if constraint_start is not None:
        end = replacement_time if replacement_time is not None else T

        # worst value over all past history BEFORE the window
        past_worst = np.nanmin(g[:constraint_start + 1])

        for t in range(constraint_start + 1, end):
            if np.isnan(g[t]):
                continue
            if g[t] < past_worst:
                crash_time = t
                break

    return constraint_start, crash_time, replacement_time
