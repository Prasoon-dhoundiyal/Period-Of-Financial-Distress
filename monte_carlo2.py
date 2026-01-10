"""
Monte Carlo simulation driver for Model 2.
Stores valid runs only.
"""

from tqdm import tqdm
import numpy as np
from model2 import GallegatiModel2 as GallegatiModel
from events2 import detect_events_model2 as detect_events


def monte_carlo_gallegati(
    n_runs,
    model_params,
    F=10.0,
    min_gap=10
):
    """
    Monte Carlo driver.
    Stores ONLY valid runs.

    Valid run criteria (as defined by you):
    1. At least one agent becomes constrained
    2. A crash is detected
    3. crash_time - constraint_start >= min_gap
    """

    results = {}

    max_gap = -np.inf
    max_gap_seed = None

    for seed in tqdm(range(n_runs), desc="Monte Carlo runs"):

        model = GallegatiModel(seed=seed, **model_params)
        out = model.run(
            return_vars=[
                "p_log",
                "p_expect",
                "frac_constrained"
            ]
        )

        frac = out["frac_constrained"]

        # Skip runs with no constraints at all
        if np.all(frac == 0):
            continue

        # Model 2 event detection
        constraint_start, crash_time, replacement_time = detect_events(
            out["p_log"],
            out["p_expect"],
            frac,
            F=F
        )

        # Crash is required for validity
        if constraint_start is None or crash_time is None:
            continue

        steps_constraint_to_crash = crash_time - constraint_start

        # VALIDITY CONDITION
        if steps_constraint_to_crash < min_gap:
            continue

        # Replacement is optional in Model 2
        if replacement_time is not None:
            steps_crash_to_replacement = replacement_time - crash_time
            steps_constraint_to_replacement = replacement_time - constraint_start
        else:
            steps_crash_to_replacement = None
            steps_constraint_to_replacement = None

        if steps_constraint_to_crash > max_gap:
            max_gap = steps_constraint_to_crash
            max_gap_seed = seed

        # STORE ONLY VALID RUN
        results[seed] = {
            "constraint_start": constraint_start,
            "crash_time": crash_time,
            "replacement_time": replacement_time,

            "steps_constraint_to_crash": steps_constraint_to_crash,
            "steps_crash_to_replacement": steps_crash_to_replacement,
            "steps_constraint_to_replacement": steps_constraint_to_replacement,

            "output": out
        }

    summary = {
        "n_runs": n_runs,
        "n_valid": len(results),
        "max_steps_constraint_to_crash": max_gap,
        "max_gap_seed": max_gap_seed
    }

    return results, summary
