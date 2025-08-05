import numpy as np
from scipy.optimize import minimize
from src.simulate  import simulate_portfolio_losses
from src.risk_metrics import compute_cvar

def minimize_cvar(mu, sigma, T, N=100000):
    d = len(mu)

    def objective(w):
        losses = simulate_portfolio_losses(mu, sigma, w, T=T, N=N)
        return compute_cvar(losses, 0.95)
    
    
    constraints = [{'type': 'eq', 'fun':lambda w: np.sum(w) - 1}]

    # Set dynamic constraints based on portfolio size
    if d<= 5:
        max_weight = 0.50
    elif d==6:
        max_weight = 0.45
    elif d==7:
        max_weight = 0.40
    elif d==8:
        max_weight = 0.35
    elif d==9:
        max_weight = 0.30
    elif d>=10:
        max_weight = 0.25


    bounds = [(0.0, max_weight)] * d
    initial_guess = [1/d] * d

    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        raise RuntimeError("Optimization failed:", result.message)
    
    return result.x
