import numpy as np

def compute_var(losses, alpha=0.95):
    """
    Compute Value-at-Risk (VaR) at a given confidence level.

    Inputs:
        losses (ndarray): Simulated portfolio losses (shape: [N])
        alpha (float): Confidence level (e.g., 0.95 for 95%)

    Returns:
        float: VaR threshold (positive number = loss)
    """

    return np.percentile(losses, 100 * alpha)


def compute_cvar(losses, alpha=0.95):
    """
    Compute Conditional Value-at-Risk (CVaR) — expected loss in worst (1 - alpha)% cases.

    Inputs:
        losses (ndarray): Simulated portfolio losses
        alpha (float): Confidence level

    Returns:
        float: CVaR value
    """

    var = compute_var(losses, alpha)
    tail_losses = losses[losses >= var]
    
    return tail_losses.mean()
