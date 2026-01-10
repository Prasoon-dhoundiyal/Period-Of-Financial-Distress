"""
Event detection for Model 2.

Detects:
- constraint_start
- replacement_time (first time constraints switch off)
- crash_time (first new worst 4-step negative outlier,
               benchmarked on all past values,
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
    # (actual regime switch: constraints off)
    # ----------------------------
    replacement_time = None
    if constraint_start is not None:
        for t in range(constraint_start + 1, T):
            if frac_constrained[t] == 0.0:
                replacement_time = t
                break

    # ----------------------------
    # Crash time
    # First new worst 4-step outlier,
    # benchmarked on ALL past values,
    # but only within the window
    # ----------------------------
    crash_time = None
    if constraint_start is not None:
        end = replacement_time if replacement_time is not None else T

        past_worst = np.nanmin(g[:constraint_start + 1])

        for t in range(constraint_start + 1, end):
            if np.isnan(g[t]):
                continue
            if g[t] < past_worst:
                crash_time = t
                break

    return constraint_start, crash_time, replacement_time
