---

# Gallegati Financial Distress Model

This repository contains a fully instrumented agent-based implementation of the **Gallegati et al. (2011)** financial distress model.
The implementation is designed to replicate the emergence of **endogenous periods of financial distress and crashes**, while preserving causal integrity, reproducibility, and compatibility with Monte Carlo analysis.

---

## Overview

The model simulates a financial market populated by heterogeneous agents who trade a single risky asset. Prices, expectations, and wealth evolve endogenously through the interaction of:

* stochastic price shocks,
* adaptive expectations,
* imitation (herding),
* transaction costs,
* and endogenous liquidity constraints.

A key feature of this implementation is the **strict separation between market dynamics and crash identification**, which mirrors the methodological structure of the original paper and standard practice in empirical crisis analysis.

---

## Features

* Agent-based trading with imitation and adaptive expectations
* Endogenous liquidity (wealth-based) constraints
* Crash identification via worst historical price decline (outlier-based)
* Expectation-based fundamental replacement rule
* Monte Carlo filtering of slow-burn crises
* Full agent-level and aggregate diagnostics

---

## Repository Structure

* `model.py` – Core agent-based market model
* `events.py` – Crash and replacement detection logic
* `monte_carlo.py` – Monte Carlo simulation engine
* `plots.py` – Publication-quality plotting functions

---

## Model Architecture

### Two-Stage Design

The model is intentionally structured as a **two-stage process**:

1. **Forward Simulation (Market Dynamics)**
2. **Ex-Post Event Identification (Crash and Replacement Detection)**

This separation is a deliberate methodological choice and should not be collapsed into a single on-the-run mechanism.

---

## Stage 1: Forward Simulation (Market Dynamics)

The core simulation (`GallegatiModel.run`) evolves the market forward in discrete time without foresight.

At each time step:

* Agents choose buy/sell positions using a Brock–Durlauf logit decision rule.
* Decisions depend on:

  * expected price changes,
  * past aggregate behavior (imitation),
  * and idiosyncratic randomness.
* Prices update through excess demand.
* Wealth evolves through realized trading profits net of transaction costs.
* Liquidity constraints bind endogenously once agent wealth falls below a fixed fraction of initial wealth.
* Expectations adjust adaptively toward realized prices with stochastic shocks.

**Importantly:**

* The model itself has no concept of a “crash.”
* No thresholds, regime switches, or replacement logic are applied during the run.
* Agents and prices use only contemporaneous and past information.

This ensures the simulation is **causally clean** and free of look-ahead bias.

---

## Stage 2: Event Identification (Ex-Post Analysis)

Once the full price and expectation trajectories are realized, a separate event-detection step identifies key market events from the perspective of the analyst.

### Identified Events

1. **Constraint Start**
   The first time at which at least one agent becomes liquidity constrained.

2. **Crash Time**
   Defined as the **most negative price return** occurring *after constraints have begun*.
   This is a global property of the realized price path and is not observable in real time.

3. **Fundamental Breach**
   The first time at which **expected prices** fall below
   [
   0.6 \times \text{fundamental value}
   ]
   consistent with the identification strategy in Gallegati et al. (2011).

4. **Replacement Time**
   Defined as:
   [
   \text{replacement_time} = \max(\text{crash_time}, \text{fundamental_cross}) + 1
   ]

This logic is implemented in `events.py` and applied **only after the simulation has completed**.

---

## Why Crash Identification Is Ex-Post

Although the model generates crashes endogenously, **the crash itself is not a stopping time**.

Even during a period of financial distress:

* Constraints bind gradually across agents.
* Selling pressure is heterogeneous.
* Expectations may temporarily stabilize or rebound.
* Aggregate demand can partially recover before deteriorating further.

As a result, the worst negative return may occur **after apparent stabilization**.
No rule based solely on current or recent information can guarantee that a given drop is the worst one that will occur.

Declaring crashes “during the run” would introduce implicit foresight and systematically bias crash timing earlier.
For this reason, crashes are identified **ex post**, from the analyst’s perspective, using the full realized path.

This approach is standard in:

* financial crisis dating,
* drawdown analysis,
* and empirical identification of extreme market events.

---

## Replacement Methodology

Replacement is treated as an **institutional response**, not an endogenous market mechanism.

Key principles:

* Replacement is **never triggered during the forward simulation**.
* It is applied only after crash identification is complete.
* Replacement does not influence the formation of the crash itself.

### Implementation

After detecting the replacement time:

* The model is re-run with identical parameters, shocks, and initial conditions.
* At the replacement time:

  * agent wealth is reset,
  * positions are re-initialized,
  * liquidity constraints are lifted.

A separate **unconstrained counterfactual run** is also performed for comparison.

All runs differ **only** in the institutional rule applied; the stochastic realization and market dynamics remain identical.

---

## Why the Model Is Run More Than Once

Running the model multiple times is **intentional and unavoidable**:

* Event identification requires observing the full trajectory.
* Replacement is defined relative to detected crash timing.
* Counterfactual comparisons require separate runs.

This design avoids:

* look-ahead bias,
* endogenous regime switching with foresight,
* contamination of market dynamics by analyst-level constructs.

The separation between **simulation** and **interpretation** preserves reproducibility and theoretical consistency.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
