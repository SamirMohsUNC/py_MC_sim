import sys
from pathlib import Path
import numpy as np

# Dynamically add the root directory (i.e., PY_MC_SIM) to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.simulate import simulate_portfolio_losses
from src.stress_test import scale_covariance, simulate_t_dist_losses
from src.risk_metrics import compute_var


def test_scale_covariance_doubles_matrix():
    sigma = np.array([[0.01, 0.002], [0.002, 0.015]])
    scaled = scale_covariance(sigma, scale_factor=2.0)

    assert np.allclose(scaled, sigma * 2)


def test_t_distribution_fatter_tails_than_normal():
    mu = np.array([0.001, 0.001])
    sigma = np.array([[0.0001, 0.0], [0.0, 0.0001]])
    weights = np.array([0.5, 0.5])

    losses_norm = simulate_portfolio_losses(mu, sigma, weights, T=10, N=100000, seed=1)
    losses_t = simulate_t_dist_losses(mu, sigma, weights, T=10, N=100000, df=4, seed=1)

    var_norm = compute_var(losses_norm, alpha=0.99)
    var_t = compute_var(losses_t, alpha=0.99)

    assert var_t > var_norm, "t-distribution should have fatter tails"


def test_extreme_t_tail_risk():
    mu = np.array([0.001])
    sigma = np.array([[0.0002]])
    weights = np.array([1.0])

    losses = simulate_t_dist_losses(mu, sigma, weights, T=10, N=100000, df=2, seed=7)

    assert np.percentile(losses, 99.9) > 0.5  # extreme tail values should exist
    