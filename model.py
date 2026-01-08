import numpy as np

"""
Gallegati et al. (2011) Financial Distress Model
Fully instrumented agent-based implementation.
"""

# ================= Model =================

class GallegatiModel:
    def __init__(self, N=100, T=550,
                 c=0.8, beta=0.1, J=0.5,
                 k=0.4, rho=0.7, theta=0.7,
                 sigma1=2e-6, sigma2=4e-2,
                 y=1/36000, W0=1000,
                 seed=6,
                 replacement_time=None,
                 constraints_active=True):

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
        self.replacement_time = replacement_time
        self.constraints_active = constraints_active

    def run(self):
        np.random.seed(self.seed)

        # shocks
        z1 = np.random.normal(0, 1, self.T)   # price shocks
        z2 = np.random.normal(0, 1, self.T)   # expectation shocks
        rand_choices = np.random.random((self.T, self.N))

        # state variables
        W = np.full(self.N, self.W0, dtype=float)
        w = np.where(np.random.random(self.N) > 0.5, 1, -1)

        # histories (market)
        p_hist = np.zeros(self.T)
        p_expect_hist = np.zeros(self.T)
        prob_buy_hist = np.zeros(self.T)

        # histories (agents)
        W_history = np.zeros((self.T, self.N))
        profit_history = np.zeros((self.T, self.N))
        w_history = np.zeros((self.T, self.N))
        constraint_history = np.zeros((self.T, self.N))

        # aggregates
        mean_wealth = np.zeros(self.T)
        mean_profit = np.zeros(self.T)
        mean_position = np.zeros(self.T)
        frac_constrained = np.zeros(self.T)

        # initial conditions
        p_prev = np.log(10.0)
        p_expect = np.log(10.1)
        w_agg_prev = np.mean(w)
        constrained_active = self.constraints_active

        for t in range(self.T):

            # ---- decision rule ----
            signal = (p_expect - p_prev) + self.J * w_agg_prev
            prob_buy = 1.0 / (1.0 + np.exp(np.clip(-2 * self.beta * signal, -100, 100)))
            prob_buy_hist[t] = prob_buy

            w_new = np.where(rand_choices[t] < prob_buy, 1, -1)

            if constrained_active:
                w_new[W <= self.limit_wealth] = -1

            # ---- price dynamics ----
            w_agg = np.mean(w_new)
            p_next = p_prev + self.k * w_agg + self.sigma1 * z1[t]

            # ---- profits / utility ----
            profits = w_new * (np.exp(p_next) - np.exp(p_prev) + self.y) - self.c
            W += profits
            W = np.maximum(W, 0.0)

            # ---- bookkeeping ----
            W_history[t] = W
            profit_history[t] = profits
            w_history[t] = w_new
            constraint_history[t] = (W <= self.limit_wealth).astype(int)

            mean_wealth[t] = W.mean()
            mean_profit[t] = profits.mean()
            mean_utility[t] = profits.mean()
            mean_position[t] = w_new.mean()
            frac_constrained[t] = constraint_history[t].mean() if constrained_active else 0.0

            # ---- replacement ----
            if self.replacement_time is not None and t == self.replacement_time:
                W[:] = self.W0
                w_new[:] = np.where(np.random.random(self.N) > 0.5, 1, -1)
                constrained_active = False

            # ---- expectations ----
            p_expect = p_expect - self.rho * (p_expect - p_next) + self.sigma2 * z2[t]

            # ---- advance ----
            p_hist[t] = p_next
            p_expect_hist[t] = p_expect
            p_prev = p_next
            w_agg_prev = w_agg

        # ---- return EVERYTHING needed ----
        return {
            # prices & expectations
            "p_log": p_hist,
            "p_expect": p_expect_hist,

            # agent-level
            "W": W_history,
            # "profits": profit_history,
            # "positions": w_history,
            # "constraints": constraint_history,

            # aggregates
            # "mean_wealth": mean_wealth,
            # "mean_profit": mean_profit,
            # "mean_position": mean_position,
            "frac_constrained": frac_constrained,

            # diagnostics
            # "prob_buy": prob_buy_hist,
            # "price_shock": z1,
            # "exp_shock": z2
        }

