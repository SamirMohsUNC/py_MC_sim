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


def run_stress_test(mu, sigma, weights, tickers, T=10, N=100000, method="economic", shock_tickers=None, 
                    shock_pct=-0.10, scale_factor=2.0, dist="normal", df=5):
    """
    Run a modular stress test
    
    Economic scales the entire covariance matrix

    Stock will shock a specific ticker
    """
    if method == "economic":
        sigma_stressed = scale_covariance(sigma, scale_factor)
    if dist == "normal":
        return simulate_portfolio_losses(mu, sigma_stressed, weights, T, N)
    elif dist == "t-dist":
        return simulate_t_dist_losses(mu, sigma_stressed, weights, T, N, df)

    elif method == "stock":
        if shock_tickers is None:
            raise ValueError("shock_tickers must be provided for stock method.")
        shocked_mu = shock_certain_assets(mu, shock_tickers, shock_pct, tickers)
        if dist == "normal":
            return simulate_portfolio_losses(shocked_mu, sigma, weights, T, N)
        elif dist == "t-dist":
            return simulate_t_dist_losses(shocked_mu, sigma, weights, T, N, df)

    else:
        raise ValueError("Unsupported stress test method. Use 'economic' or 'stock'.")