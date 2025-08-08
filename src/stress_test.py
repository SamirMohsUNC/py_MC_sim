import numpy as np
from scipy.stats import t
from src.simulate import simulate_portfolio_losses, simulate_t_dist_losses

def scale_covariance(sigma, scale_factor=2.0):
    """
    Scale the covariance matrix to simulate a total economic shock
    """
    return sigma * scale_factor


def shock_certain_assets(mu, shcoked_tickers, shock_pct, tickers):
    """
    Give a shock to specfic tickers in the mean return vector
    """
    shocked_mu = mu.copy()
    for ticker in shcoked_tickers:
        if ticker in tickers:
            idx = tickers.index(tickers)
            shocked_mu[idx] += shock_pct
    return shocked_mu


def run_stress_test(mu, sigma, weights, tickers, T=10, N=100000, method='economic', shock_tickers=None, shock_pct=)




