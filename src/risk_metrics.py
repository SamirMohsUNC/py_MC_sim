import numpy as np

def compute_var(losses, alpha=0.95):
    """
    Compute Value-at-Risk (VaR) at any given confidence level.
    """

    return np.percentile(losses, 100 * alpha)


def compute_cvar(losses, alpha=0.95):
    """
    Compute Conditional Value-at-Risk (CVaR) — expected loss in worst (1 - alpha)% cases.
    """

    var = compute_var(losses, alpha)
    tail_losses = losses[losses >= var]

    return tail_losses.mean()


def bootstrap_confidence_intervals(losses, alpha=0.95, n_boot=1000, ci_level=0.95, seed=42):
    """
    Bootstrap CIs for VaR and CVaR from simulated losses.
    Returns dict with var_mean, var_ci, cvar_mean, cvar_ci.
    """
    losses = np.asarray(losses)
    if losses.size == 0:
        raise ValueError("`losses` is empty.")
    if np.isnan(losses).any():
        raise ValueError("`losses` contains NaNs.")

    rng = np.random.default_rng(seed)
    N = losses.size

    boot_var = np.empty(n_boot, dtype=float)
    boot_cvar = np.empty(n_boot, dtype=float)

    def _var(x):  # VaR at level alpha
        return np.percentile(x, 100 * alpha)

    def _cvar(x):  # CVaR at level alpha
        v = _var(x)
        tail = x[x >= v]
        return tail.mean() if tail.size > 0 else v  # fallback if tail empty

    # bootstrap resamples
    for b in range(n_boot):
        sample = rng.choice(losses, size=N, replace=True)
        boot_var[b] = _var(sample)
        boot_cvar[b] = _cvar(sample)

    # confidence interval via quantiles
    lo_q = (1 - ci_level) / 2.0
    hi_q = 1.0 - lo_q

    var_ci = tuple(np.quantile(boot_var, [lo_q, hi_q]))
    cvar_ci = tuple(np.quantile(boot_cvar, [lo_q, hi_q]))

    return {
        "var_mean": float(boot_var.mean()),
        "var_ci": var_ci,
        "cvar_mean": float(boot_cvar.mean()),
        "cvar_ci": cvar_ci,
    }