# src/stress_test.py
import numpy as np
from src.simulate import simulate_portfolio_losses, simulate_t_dist_losses

def scale_covariance(sigma, scale_factor=2.0):
    """
    Scale the covariance matrix to simulate a total economic shock.
    """
    return sigma * scale_factor

def shock_certain_assets(mu, shocked_tickers, shock_pct, tickers):
    """
    Apply a return shock to specific tickers in the mean return vector.

    Args:
        mu (ndarray): Original mean returns, shape (d,)
        shocked_tickers (list[str]): Tickers to shock
        shock_pct (float): Decimal shock amount (e.g., -0.10 = -10%)
        tickers (list[str]): Full list of tickers (for index lookup)

    Returns:
        ndarray: Adjusted mean returns with shocks applied
    """
    shocked_mu = mu.copy()
    for ticker in shocked_tickers:
        if ticker in tickers:
            idx = tickers.index(ticker)      
            shocked_mu[idx] += shock_pct
    return shocked_mu

def run_stress_test(
    mu, sigma, weights, tickers,
    T=10, N=100000,
    method="economic",                 
    shock_tickers=None, shock_pct=-0.10,
    scale_factor=2.0,
    dist="normal", df=5
):
    """
    Run a modular stress test.

    - method = "economic": scale entire covariance via `scale_factor`
    - method = "stock": shock select tickers' mean returns by `shock_pct`

    dist: "normal" or "t-dist"
    """
    if method == "economic":
        sigma_stressed = scale_covariance(sigma, scale_factor)
        if dist == "normal":
            return simulate_portfolio_losses(mu, sigma_stressed, weights, T, N)
        elif dist == "t-dist":
            return simulate_t_dist_losses(mu, sigma_stressed, weights, T, N, df)
        else:
            raise ValueError("dist must be 'normal' or 't-dist'.")

    elif method == "stock":
        if not shock_tickers:
            raise ValueError("shock_tickers must be provided for 'stock' method.")
        shocked_mu = shock_certain_assets(mu, shock_tickers, shock_pct, tickers)
        if dist == "normal":
            return simulate_portfolio_losses(shocked_mu, sigma, weights, T, N)
        elif dist == "t-dist":
            return simulate_t_dist_losses(shocked_mu, sigma, weights, T, N, df)
        else:
            raise ValueError("dist must be 'normal' or 't-dist'.")

    else:
        raise ValueError("Unsupported stress test method. Use 'economic' or 'stock'.")
