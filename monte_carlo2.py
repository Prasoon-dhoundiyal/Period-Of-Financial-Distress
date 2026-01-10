"""
Monte Carlo simulation driver.
Stores valid runs only.
"""

from tqdm import tqdm
import numpy as np
from model2 import GallegatiModel
from events import detect_events

def monte_carlo_gallegati(
    n_runs,
    model_params,
    F=10.0,
    min_gap=10
):
    """
    Monte Carlo driver.
    Stores ONLY valid runs.
    """

    results = {}

    max_gap = -np.inf
    max_gap_seed = None

    for seed in tqdm(range(n_runs), desc="Monte Carlo runs"):

        model = GallegatiModel(seed=seed, **model_params)
        out = model.run(return_vars=[
                                      "p_log",
                                      "p_expect",
                                      "frac_constrained"
                                ])

        frac = out["frac_constrained"]

        # Skip runs with no constraints at all
        if np.all(frac == 0):
            continue

        constraint_start, crash_time, fundamental_cross, replacement_time = \
            detect_events(
                out["p_log"],
                out["p_expect"],
                frac,
                F=F
            )

        steps_constraint_to_crash = crash_time - constraint_start

        # VALIDITY CONDITION
        if steps_constraint_to_crash < min_gap:
            continue

        steps_crash_to_replacement = replacement_time - crash_time
        steps_constraint_to_replacement = replacement_time - constraint_start

        if steps_constraint_to_crash > max_gap:
            max_gap = steps_constraint_to_crash
            max_gap_seed = seed

        # STORE ONLY VALID RUN
        results[seed] = {
            "constraint_start": constraint_start,
            "crash_time": crash_time,
            "fundamental_cross": fundamental_cross,
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
