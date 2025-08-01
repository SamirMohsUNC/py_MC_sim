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


