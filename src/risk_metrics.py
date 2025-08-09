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


def bootstrap_confidence_interval(losses, alpha=0.95, n_boot=1000, ci_level=0.95, seed=42):
    """
    Bootstraps VaR and CVaR confidence intervals
    """
    np.random.seed(seed)
    boot_var = []
    boot_cvar = []

    N = len(losses)
    for _ in range(n_boot):
        sample = np.random.choice(losses, size=N, replace=True)
        v = compute_var(sample, alpha)
        c = compute_cvar(sample, alpha)
        boot_var.append(v)
        boot_cvar.append(c)

        lower_idx = int(((1 - ci_level) / 2) * n_boot)
        upper_idx = int((1 - (1 - ci_level) / 2) * n_boot)

        var_ci = (np.sort(boot_var)[lower_idx], np.sort(boot_var)[upper_idx])
        cvar_ci = (np.sort(boot_cvar)[lower_idx], np.sort(boot_cvar)[upper_idx])
        

        return {'var_mean':np.mean(boot_var), 'var_ci': var_ci, 'cvar_mean': np.mean(boot_cvar), 'cvar_ci': cvar_ci}
