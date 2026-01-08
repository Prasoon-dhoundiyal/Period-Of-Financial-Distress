# Gallegati Financial Distress Model

This repository contains a fully instrumented agent-based implementation of the
Gallegati et al. (2011) financial distress model, including liquidity constraints,
crash detection, and Monte Carlo analysis.

## Features
- Agent-based trading with imitation and adaptive expectations
- Endogenous liquidity constraints
- Crash identification via worst historical price decline using grubbs one sided outlier test.
- Expectation-based fundamental replacement rule
- Monte Carlo filtering of slow-burn crises
- Full agent-level and aggregate diagnostics

## Repository Structure
- `model.py` – Core agent-based model
- `events.py` – Crash and replacement detection logic
- `monte_carlo.py` – Monte Carlo simulation engine
- `plots.py` – Publication-quality plotting functions

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
