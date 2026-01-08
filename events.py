import numpy as np

"""
Crash, constraint, and replacement detection logic.
"""

# ================= Event Detection =================
def detect_events(p_log, p_expect, frac_constrained, F=10.0):
    T = len(p_log)

    g = np.full(T, np.nan)
    for t in range(4, T):
        g[t] = p_log[t] - p_log[t-4]

    constraint_start = np.where(frac_constrained > 0)[0][0]

    # Worst fall so far (post-constraint)
    crash_time = None
    for t in range(constraint_start, T):
        valid = g[:t+1][~np.isnan(g[:t+1])]
        worst_idx = np.where(~np.isnan(g[:t+1]))[0][np.argmin(valid)]
        if worst_idx >= constraint_start:
            crash_time = worst_idx
            break

    # EXPECTATION-BASED fundamental rule (Original paper based)
    fundamental_log = np.log(0.6 * F)
    fundamental_cross = next(
        (t for t in range(constraint_start, T) if p_expect[t] < fundamental_log),
        None
    )

    replacement_time = max(crash_time, fundamental_cross) + 1
    return constraint_start, crash_time, fundamental_cross, replacement_time
