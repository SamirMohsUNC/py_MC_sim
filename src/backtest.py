import numpy as np
import pandas as pd
from src.simulate import simulate_portfolio_losses, simulate_t_dist_losses
from src.risk_metrics import compute_var, compute_cvar
from math import log
from scipy.stats import chisquare


def backtest_var_cvar(returns_df, weights, alpha=0.95, window=252, horizon=1, N=100_100, dist='normal', df=5, seed=123):
    """
    Rolling Monte Carlo backtest for VaR and Expected Shortfall(ES)
    
    Fits mu and sigma on a rolling window, simulates horizon day loss distribution, records VaR_t and ES_t,
    then compares to realized horizon loss. Returns a dict of series and summary stats (hit rate, Kupiec, Christoffersen)
    """

    def kupiec_pof(exceedances, alpha):

        x = int(exceedances.sum())
        n = int(exceedances.size)

        if n == 0:
            return np.nan, np.nan
        
        p_hat = x / N
        p_0 = 1 - alpha
        # safe logs for edge cases
        eps = 1e-12
        log_1 = (n-x) * log(max(1-p_hat, eps)) + x * log(max(p_hat, eps))
        log_0 = (n-x) * log(max(1-p_0, eps)) + x * log(max(p_0, eps))
        log_r = -2 * (log_0-log_1)
        return float(log_r), float(1 - chisquare.cdf(log_r, 1))
    

    def christoffersen_ind(exceedances):
        # Independence of exceptions via 2x2 Markov chain test
        if exceedances.size <= 1:
            return np.nan, np.nan
        
        x_t = exceedances[1:]
        x_tm = exceedances[:-1]
        n00 = int(((x_tm == 0) & (x_t == 0)).sum())
        n01 = int(((x_tm == 0) & (x_t == 1)).sum())
        n10 = int(((x_tm == 1) & (x_t == 0)).sum())
        n11 = int(((x_tm == 1) & (x_t == 1)).sum())

        n0, n1 = n00 + n01, n10 + n11
        if n0 == 0 or n1 == 0:
            return np.nan, np.nan
        p01 = n01 / n0
        p11 = n11 / n1
        p = (n01 + n11) / (n0 + n1)
        eps = 1e-12
        def ll(p01, p11):
            return (n00 * np.log(max(1-p01, eps)),
                    n01 * np.log(max(p01, eps)),
                    n10 * np.log(max(1-p11, eps)),
                    n11 * np.log(max(p11, eps)))
        
        log_r = -2 * (ll(p, p) - ll(p01, p11))
        return float(log_r), float(1 - chisquare.cdf(log_r, 1))
    
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, float)
    returns_df = returns_df.copy()
    returns_df = returns_df.dropna().astype(float)

    dates = []
    var_series = []
    es_series = []
    realized = []
    
    
