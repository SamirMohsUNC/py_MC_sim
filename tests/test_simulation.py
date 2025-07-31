import sys
from pathlib import Path
import numpy as np

# Dynamically add the root directory (i.e., PY_MC_SIM) to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.simulate import simulate_portfolio_losses
from src.risk_metrics import compute_var, compute_cvar


def test_simulate_portfolio_losses_shape_and_range():
    mu = np.array([0.001, 0.002])
    sigma = np.array([[0.0001, 0.00005], [0.00005, 0.0002]])
    weights = np.array([0.5, 0.5])

    losses = simulate_portfolio_losses(mu, sigma, weights, T=10, N=5000, seed=42)

    assert isinstance(losses, np.ndarray)
    assert losses.shape == (5000,)
    assert np.all(np.isfinite(losses))
    assert np.all(losses >= -1.0)  # Cannot lose more than 100%


def test_compute_var_and_cvar_known_input():
    losses = np.array([1, 2, 3, 4, 5])
    var_80 = compute_var(losses, alpha=0.8)
    cvar_80 = compute_cvar(losses, alpha=0.8)

    assert var_80 == 4.0
    assert np.isclose(cvar_80, 4.5)


def test_cvar_not_less_than_var():
    rng = np.random.default_rng(123)
    losses = rng.normal(loc=0.01, scale=0.03, size=10000)

    var_95 = compute_var(losses, alpha=0.95)
    cvar_95 = compute_cvar(losses, alpha=0.95)

    assert cvar_95 >= var_95