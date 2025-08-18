import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.optimizer import minimize_cvar
from src.simulate import simulate_portfolio_losses, simulate_t_dist_losses
from src.risk_metrics import compute_var, compute_cvar



def realized_horizon_loss(returns_block, weights):
    # sum daily log returns over the horizon for each asset; portfolio return = dot; loss = -return
    asset_logret = returns_block.sum(axis=0).values
    return float(-(np.dot(asset_logret, weights)))


def rolling_optimize_and_dashboard(returns_df, baseline_weights, alpha=0.95, window=252, horizon=1, N=50_000,
    dist="normal",     # "normal" | "t-dist"
    df=5, save_dir="reports", seed=123):
    """
    Rolling CVaR-minimizing optimization with risk evaluation vs a baseline.

    returns_df: DataFrame of DAILY log returns (index: dates; columns: tickers)
    baseline_weights: np.array aligned to returns_df.columns
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    dates = []
    w_rolling = []
    var_fore = []
    cvar_fore = []
    rlz_losses_rolling = []
    rlz_losses_baseline = []
    turnover = []

    # for cumulative performance, we build a log-returns path
    roll_log_returns = []
    base_log_returns = []

    cols = list(returns_df.columns)
    baseline_weights = np.asarray(baseline_weights, dtype=float)
    baseline_weights = baseline_weights / baseline_weights.sum()

    # simple baseline weight path. keep baseline constant
    prev_w = None

    if len(returns_df) < window + horizon:
        raise ValueError("Not enough data: increase history or reduce window/horizon")
    
    for t in range(len(window), len(returns_df)-horizon+1):
        