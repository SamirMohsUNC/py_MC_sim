import numpy as np
from scipy.stats import t

def scale_covariance(sigma, scale_factor=2.0):
    """
    Stress test by scaling the covariance matrix.

    Inputs:
        sigma (ndarray): Original covariance matrix (d x d)
        scale_factor (float): Multiplier for volatility (e.g., 2.0 = double volatility)

    Returns:
        ndarray: Scaled covariance matrix
    """

    return sigma * scale_factor


def simulate_t_dist_losses(mu, sigma, weights, T=10, N=10000, df=5, seed=42):
    """
    Simulate portfolio losses under a multivariate t-distribution.

    Inputs:
        mu (ndarray): Mean return vector (d,)
        sigma (ndarray): Covariance matrix (d, d)
        weights (ndarray): Portfolio weights (d,)
        T (int): Time horizon in days
        N (int): Number of simulations
        df (int): Degrees of freedom (lower = fatter tails)
        seed (int): Random seed

    Returns:
        ndarray: Simulated losses (N,)
    """

    np.random.seed(seed)
    d = len(mu)
    
    # Scale parameters
    mu_scaled = T * mu
    sigma_scaled = T * sigma

    # Sample from multivariate t-distribution:
    # Step 1: Standard normal
    Z = np.random.multivariate_normal(np.zeros(d), sigma_scaled, size=N)
    
    # Step 2: Chi-squared samples
    chi_samples = np.random.chisquare(df, size=N) / df

    # Step 3: Create t-distributed samples
    t_samples = mu_scaled + Z / np.sqrt(chi_samples)[:, None]

    # Portfolio returns and losses
    portfolio_returns = np.dot(t_samples, weights)
    losses = -portfolio_returns

    return losses
