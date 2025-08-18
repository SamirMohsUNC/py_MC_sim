import numpy as np
from scipy.optimize import minimize
from src.simulate import simulate_portfolio_losses
from src.risk_metrics import compute_cvar

def minimize_cvar(mu, sigma, T, N=100_000):
    d = len(mu)

    def objective(w):
        losses = simulate_portfolio_losses(mu, sigma, w, T=T, N=N)
        return compute_cvar(losses, 0.95)
    

    constraints = [{'type': 'eq', 'fun':lambda w: np.sum(w) - 1}]

    # Set dynamic constraints based on portfolio size
    if d <= 5:
        max_weight = 0.20
    elif d == 6:
        max_weight = 0.19
    elif d == 7:
        max_weight = 0.18
    elif d == 8:
        max_weight = 0.17
    elif d == 9:
        max_weight = 0.16
    elif d == 10:
        max_weight = 0.15
    elif d == 11: 
        max_weight = 0.14
    elif d == 12:
        max_weight = 0.13
    elif d == 13:
        max_weight = 0.12
    elif d == 14:
        max_weight = 0.11
    elif d >= 15:
        max_weight = 0.10


    bounds = [(0.0, max_weight)] * d
    initial_guess = [1/d] * d

    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        raise RuntimeError("Optimization failed:", result.message)
    
    return result.x
