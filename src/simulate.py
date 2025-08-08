import numpy as np

def simulate_portfolio_losses(mu, sigma, weights, T=10, N=10000, seed=42):
    """
    Simulates portfolio losses over a T-day horizon using Monte Carlo.

    Inputs:
        mu (array): Expected daily return vector (shape: [d])
        sigma (array): Covariance matrix (shape: [d, d])
        weights (array): Portfolio weights (shape: [d])
        T (int): Time horizon in days (e.g., 10-day VaR)
        N (int): Number of Monte Carlo simulations
        seed (int): Random seed for reproducibility

    Returns:
        losses (array): Simulated losses (shape: [N])
    """
    np.random.seed(seed)
    mu_scaled = T * mu
    sigma_scaled = T * sigma

    # Simulate N paths of T-day returns
    simulated_returns = np.random.multivariate_normal(mu_scaled, sigma_scaled, size=N)
    
    # Portfolio return = dot product with weights
    portfolio_returns = np.dot(simulated_returns, weights)
    
    # Portfolio loss = negative return
    losses = -portfolio_returns

    return losses


def simulate_t_dist_losses(mu, sigma, weights, T=10, N=10000, df=5, seed=42):
    """
    Monte Carlo portfolio losses using a multivariate Student-t model
    (fat tails via normal and chi-square mixture)
    """
    np.random.seed(seed)
    d = len(mu)

    mu_scaled = mu * T
    sigma_scaled = sigma * T

    Z = np.random.multivariate_normal(np.zeros(d), sigma_scaled, size=N)

    chi = np.random.chisquare(df, size=N) / df

    t_samples = mu_scaled + Z / np.sqrt(chi)[:, None]

    portfolio_returns = np.dot(t_samples, weights)

    return -portfolio_returns

