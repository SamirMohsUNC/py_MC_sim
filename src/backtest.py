import numpy as np
import pandas as pd
from src.simulate import simulate_portfolio_losses, simulate_t_dist_losses
from src.risk_metrics import compute_var, compute_cvar


def backtest_var_cvar(returns_df, weights, alpha=0.95, window=252)
