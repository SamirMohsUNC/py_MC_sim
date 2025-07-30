import numpy as np
from scipy.stats import t

def scale_covariance(sigma, scale_factor=2.0):
    """
    Stress test by scaling the covariance matrix

    Inputs: 
    sigma(array): Original covariance matrix
    scale_factor(array): Multiplier for volatility. For example, 2.0 is double volatility

    Returns:
    array: Scaled covariance matrix
    """

    return sigma * scale_factor

def simulate_t_dist_losses(mu, sigma, weights, T=10, N=10000, df=5, seed=42):
    """
    Simulate portfolio losses under a multivariate t-distribution

    Inputs:
    mu(array): Expected daily return vector
    sigma(array): Covariance matrix
    weights(array): Portfolio weights
    T(int): Time period in days, such as 10 day VaR
    N(int): Number of Monte Carlo simulations
    df(int): Degrees of freedom (lower = fatter tails)
    seed(int): Random seed

    Returns:
    array: Simulated losses
    """
    
    np.random.seed(seed)
    d = len(mu)

    # Scale paramters
    mu_scaled = T * mu
    sigma_scaled = T * sigma

    # Sample from multivariate t-dist:
    # Step 1 with standard normal
    Z = np.random.multivariate_normal(np.zeros(d), sigma_scaled, size=N)

    # Step 2 with chi-squared samples
    chi_samples = np.random.chisquare(df, size=N) / df

    # Step 3 creating t-dist samples
    t_samples = mu_scaled + Z / np.sqrt(chi_samples)[:, None]

    # Portfolio returns and losses
    portfolio_returns = np.dot(t_samples, weights)
    losses = -portfolio_returns

    return losses


