"""
Gallegati et al. (2011) Financial Distress Model
Variant with ENDOGENOUS, AGENT-LEVEL replacement.
All other logic identical to model.py.
"""

import numpy as np


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
        replacement_time=None,   # accepted for backward compatibility, ignored
        constraints_active=True,
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

        # ----------------------------
        # Shocks and random draws
        # ----------------------------
        z1 = np.random.normal(0, 1, self.T)
        z2 = np.random.normal(0, 1, self.T)
        rand_choices = np.random.random((self.T, self.N))

        # ----------------------------
        # State variables
        # ----------------------------
        W = np.full(self.N, self.W0, dtype=float)
        w = np.where(np.random.random(self.N) > 0.5, 1, -1)

        # ----------------------------
        # Market histories
        # ----------------------------
        p_hist = np.zeros(self.T)
        p_expect_hist = np.zeros(self.T)
        prob_buy_hist = np.zeros(self.T)

        # ----------------------------
        # Agent histories
        # ----------------------------
        W_history = np.zeros((self.T, self.N))
        profit_history = np.zeros((self.T, self.N))
        w_history = np.zeros((self.T, self.N))
        constraint_history = np.zeros((self.T, self.N))

        # ----------------------------
        # Aggregates
        # ----------------------------
        mean_wealth = np.zeros(self.T)
        mean_profit = np.zeros(self.T)
        mean_position = np.zeros(self.T)
        frac_constrained = np.zeros(self.T)
        fraction_replaced = np.zeros(self.T)

        # ----------------------------
        # Replacement state (agent-level)
        # ----------------------------
        replaced = np.zeros(self.N, dtype=bool)

        # ----------------------------
        # Initial conditions
        # ----------------------------
        p_prev = np.log(self.fundamental)
        p_expect = np.log(self.fundamental * 1.01)
        w_agg_prev = np.mean(w)

        fundamental_log = np.log(0.6 * self.fundamental)

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

            # ---- apply constraints (ONLY if active) ----
            constrained_idx = (
                self.constraints_active
                & (W <= self.limit_wealth)
                & (~replaced)
            )
            w_new[constrained_idx] = -1

            # ---- price dynamics ----
            w_agg = np.mean(w_new)
            p_next = p_prev + self.k * w_agg + self.sigma1 * z1[t]

            # ---- profits and wealth ----
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

            constrained_idx = (
                self.constraints_active
                & (W <= self.limit_wealth)
                & (~replaced)
            )
            constraint_history[t] = constrained_idx.astype(int)
            frac_constrained[t] = constraint_history[t].mean()

            mean_wealth[t] = W.mean()
            mean_profit[t] = profits.mean()
            mean_position[t] = w_new.mean()

            # ---- expectations ----
            p_expect = (
                p_expect
                - self.rho * (p_expect - p_next)
                + self.sigma2 * z2[t]
            )

            # ---- AGENT-LEVEL replacement ----
            replace_idx = (
                self.constraints_active
                & (p_expect < fundamental_log)
                & constrained_idx
            )

            if np.any(replace_idx):
                W[replace_idx] = self.W0
                replaced[replace_idx] = True
                fraction_replaced[t] = replace_idx.mean()

            # ---- advance ----
            p_hist[t] = p_next
            p_expect_hist[t] = p_expect
            p_prev = p_next
            w_agg_prev = w_agg

        # ==================================================
        # Output
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
            "fraction_replaced": fraction_replaced,
            "prob_buy": prob_buy_hist,
            "price_shock": z1,
            "exp_shock": z2,
        }

        if return_vars is None:
            return {
                "p_log": p_hist,
                "p_expect": p_expect_hist,
                "frac_constrained": frac_constrained,
                "fraction_replaced": fraction_replaced,
            }

        missing = set(return_vars) - full_output.keys()
        if missing:
            raise KeyError(
                f"Requested variables not available: {sorted(missing)}"
            )

        return {k: full_output[k] for k in return_vars}
