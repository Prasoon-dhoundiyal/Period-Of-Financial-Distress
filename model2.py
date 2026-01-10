"""
Gallegati et al. (2011) Financial Distress Model
Variant with endogenous replacement and valid-run checks.
"""

import numpy as np


# ================= Model =================

class GallegatiModel2:
    def __init__(
        self,
        N=100,
        T=550,
        c=0.8,
        beta=0.1,
        J=0.5,
        k=0.4,
        rho=0.7,
        theta=0.7,
        sigma1=2e-6,
        sigma2=4e-2,
        y=1 / 36000,
        W0=1000,
        seed=6,
        constraints_active=True,
        replacement_time=None,   # <-- ignored, kept for compatibility
        fundamental=10.0,
    ):

        self.N = N
        self.T = T
        self.c = c
        self.beta = beta
        self.J = J
        self.k = k
        self.rho = rho
        self.theta = theta
        self.limit_wealth = theta * W0
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.y = y
        self.W0 = W0
        self.seed = seed
        self.constraints_active = constraints_active
        self.fundamental = fundamental

    # ======================================================
    # Core simulation
    # ======================================================
    def run(self, return_vars=None):

        np.random.seed(self.seed)

        # --------------------------------------------------
        # Shocks and random draws
        # --------------------------------------------------
        z1 = np.random.normal(0, 1, self.T)
        z2 = np.random.normal(0, 1, self.T)
        rand_choices = np.random.random((self.T, self.N))

        # --------------------------------------------------
        # State variables
        # --------------------------------------------------
        W = np.full(self.N, self.W0, dtype=float)
        w = np.where(np.random.random(self.N) > 0.5, 1, -1)

        # --------------------------------------------------
        # Histories (market-level)
        # --------------------------------------------------
        p_hist = np.zeros(self.T)
        p_expect_hist = np.zeros(self.T)
        prob_buy_hist = np.zeros(self.T)

        # --------------------------------------------------
        # Histories (agent-level)
        # --------------------------------------------------
        W_history = np.zeros((self.T, self.N))
        profit_history = np.zeros((self.T, self.N))
        w_history = np.zeros((self.T, self.N))
        constraint_history = np.zeros((self.T, self.N))

        # --------------------------------------------------
        # Aggregates
        # --------------------------------------------------
        mean_wealth = np.zeros(self.T)
        mean_profit = np.zeros(self.T)
        mean_position = np.zeros(self.T)
        frac_constrained = np.zeros(self.T)

        # --------------------------------------------------
        # Initial conditions
        # --------------------------------------------------
        p_prev = np.log(self.fundamental)
        p_expect = np.log(self.fundamental * 1.01)
        w_agg_prev = np.mean(w)
        constrained_active = self.constraints_active

        # --------------------------------------------------
        # Tracking variables (NEW)
        # --------------------------------------------------
        fundamental_log = np.log(0.6 * self.fundamental)
        constraint_start = None
        replacement_time = None
        replacement_done = False

        # ==================================================
        # Time loop
        # ==================================================
        for t in range(self.T):

            # ---- decision rule ----
            signal = (p_expect - p_prev) + self.J * w_agg_prev
            prob_buy = 1.0 / (
                1.0 + np.exp(np.clip(-2 * self.beta * signal, -100, 100))
            )
            prob_buy_hist[t] = prob_buy

            w_new = np.where(rand_choices[t] < prob_buy, 1, -1)

            if constrained_active:
                w_new[W <= self.limit_wealth] = -1

            # ---- price dynamics ----
            w_agg = np.mean(w_new)
            p_next = p_prev + self.k * w_agg + self.sigma1 * z1[t]

            # ---- profits / wealth ----
            profits = (
                w_new * (np.exp(p_next) - np.exp(p_prev) + self.y)
                - self.c
            )
            W += profits
            W = np.maximum(W, 0.0)

            # ---- bookkeeping ----
            W_history[t] = W
            profit_history[t] = profits
            w_history[t] = w_new
            constraint_history[t] = (W <= self.limit_wealth).astype(int)

            mean_wealth[t] = W.mean()
            mean_profit[t] = profits.mean()
            mean_position[t] = w_new.mean()
            frac_constrained[t] = (
                constraint_history[t].mean()
                if constrained_active
                else 0.0
            )

            # ---- constraint start tracking (NEW) ----
            if constraint_start is None and np.any(W <= self.limit_wealth):
                constraint_start = t

            # ---- expectations ----
            p_expect = (
                p_expect
                - self.rho * (p_expect - p_next)
                + self.sigma2 * z2[t]
            )

            # ---- endogenous replacement (NEW) ----
            if (
                not replacement_done
                and p_expect < fundamental_log
            ):
                replacement_time = t
                W[:] = self.W0
                w_new[:] = np.where(np.random.random(self.N) > 0.5, 1, -1)
                constrained_active = False
                replacement_done = True

            # ---- advance ----
            p_hist[t] = p_next
            p_expect_hist[t] = p_expect
            p_prev = p_next
            w_agg_prev = w_agg

        # ==================================================
        # Outlier computation (UNCHANGED logic)
        # ==================================================
        g = np.full(self.T, np.nan)
        for t in range(4, self.T):
            g[t] = p_hist[t] - p_hist[t - 4]

        t_star = np.nanargmin(g)

        # ==================================================
        # Valid-run logic (NEW, exact rules)
        # ==================================================
        valid_run = True

        if constraint_start is None:
            valid_run = False

        elif p_expect_hist[constraint_start] < fundamental_log:
            valid_run = False

        elif replacement_time is None:
            valid_run = False

        elif not (constraint_start <= t_star < replacement_time):
            valid_run = False

        elif (t_star - constraint_start) < 10:
            valid_run = False

        # ==================================================
        # Full output dictionary
        # ==================================================
        full_output = {
            "p_log": p_hist,
            "p_expect": p_expect_hist,
            "W": W_history,
            "profits": profit_history,
            "positions": w_history,
            "constraints": constraint_history,
            "mean_wealth": mean_wealth,
            "mean_profit": mean_profit,
            "mean_position": mean_position,
            "frac_constrained": frac_constrained,
            "prob_buy": prob_buy_hist,
            "price_shock": z1,
            "exp_shock": z2,
            "constraint_start": constraint_start,
            "replacement_time": replacement_time,
            "t_star": t_star,
            "valid_run": valid_run,
        }

        # ==================================================
        # Selective return (same pattern as model.py)
        # ==================================================
        if return_vars is None:
            return {
                "p_log": p_hist,
                "p_expect": p_expect_hist,
                "frac_constrained": frac_constrained,
                "valid_run": valid_run,
            }

        missing = set(return_vars) - full_output.keys()
        if missing:
            raise KeyError(
                f"Requested variables not available: {sorted(missing)}"
            )

        return {k: full_output[k] for k in return_vars}
