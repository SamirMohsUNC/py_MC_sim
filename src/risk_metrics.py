import numpy as np

def compute_var(losses, alpha=0.95):
    """
    Compute the value at risk (VaR) at a given confidence level

    Inputs:
    losses(array): Simulated portfolio losses
    alpha(float): Confidence level

    Returns:
    float: VaR threshold. Positive number implies loss
    """

    return np.percentile(losses, 100 * alpha)

def compute_cvar(losses, alpha=0.95):
    """
    Compute the conditional value at risk (CVaR) - expected loss in worst (1-alpha)% cases
    
    Inputs:
    losses(array): Simulated portfolio losses
    alpha(float): Confidence level

    Returns:
    float: CVaR value
    """
    
    var = compute_var(losses, alpha)
    tail_losses = losses[losses >= var]
    
    return np.mean(tail_losses)

