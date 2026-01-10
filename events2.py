"""
Event detection for Model 2.
"""

import numpy as np


def detect_events_model2(p_log, p_expect, frac_constrained, F=10.0):
    T = len(p_log)

    # ----------------------------
    # Growth rate (same as Model 1)
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
    # Replacement time (Model 2)
    # Only AFTER constraint start
    # ----------------------------
    replacement_time = None
    if constraint_start is not None:
        threshold = np.log(0.6 * F)
        for t in range(constraint_start, T):
            if p_expect[t] < threshold:
                replacement_time = t
                break

    # ----------------------------
    # Crash time
    # Most negative outlier BETWEEN
    # constraint_start and replacement_time
    # ----------------------------
    crash_time = None
    if constraint_start is not None:
        end = replacement_time if replacement_time is not None else T
        for t in range(constraint_start, end):
            valid = g[:t + 1][~np.isnan(g[:t + 1])]
            worst_idx = np.where(~np.isnan(g[:t + 1]))[0][np.argmin(valid)]
            if worst_idx >= constraint_start and worst_idx < end:
                crash_time = worst_idx
                break

    return constraint_start, crash_time, replacement_time
