import yfinance as yf
import pandas as pd
import numpy as np


def simulate_portfolio_losses(mu, sigma, weights, T=10, N=10000, seed=42):
    """
    Simulate the losses on the portfolio over a T day period using Monte Carlo

    Inputs:
    mu(array): Expected daily return vector
    sigma(array): Covariance matrix
    weights(array): Portfolio weights
    T(int): Time period in days, such as 10 day VaR
    N(int): Number of Monte Carlo simulations

    Returns:
    losses(array): Simulated losses
    """

    np.random.seed(seed)
    mu_scaled = T * mu
    sigma_scaled = T * sigma

    # Simulate N paths of our T day returns
    simulated_returns = np.random.multivariate_normal(mu_scaled, sigma_scaled, size=N)

    # Portfolio returns is dot product with weights
    portfolio_returns = np.dot(simulated_returns, weights)

    # Portfolio loss is negative returns
    losses = -portfolio_returns

    return losses
