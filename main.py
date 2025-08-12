import numpy as np
import pandas as pd 
import yfinance as yf
from datetime import datetime, timedelta

from src.simulate import simulate_portfolio_losses
from src.risk_metrics import compute_var, compute_cvar, bootstrap_confidence_intervals
from src.optimizer import minimize_cvar
from src.stress_test import run_stress_test


### Create helper questions for user to answer
def get_user_input():
    tickers = input("Enter stock tickers separated by commas (e.g. AAPL,MSFT,GOOGL): ").strip().upper().split(',')
    tickers = [t.strip() for t in tickers]

    print("\nNow enter how much you're investing in each (same order):")
    investments = list(map(float, input("Investment ammounts (comma separated): ").strip().split(',')))

    assert len(tickers) == len(investments), "Number of tickers must match numberof investments."

    print("\nChoose historical data window:")
    print("1 - Last 6 months\n2 - Last 1 year\n3 - Last 2 years")
    window_choice = input("Enter 1, 2, or 3: ").strip()
    window_days = {'1': 182, '2': 365, '3': 730}[window_choice]

    print("\nChoose investment horizon for risk calculation")
    print("1 - 1 day\n2 - 2 days\n3 - 4 days\n4 - 7 days\n5 - 10 days")
    horizon_map = {'1':1, '2': 2, '3': 4, '4': 7, '5': 10}
    horizon_days = horizon_map[input("Enter 1-5: ").strip()]

    return tickers, investments, window_days, horizon_days


def fetch_data(tickers, window_days):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=window_days)
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data.dropna(inplace=True)
    return data


def prepare_inputs(data, investments):
    log_returns = np.log(data / data.shift(1)).dropna()
    mu = log_returns.mean().values
    sigma = log_returns.cov().values
    weights = np.array(investments) / np.sum(investments)
    return mu, sigma, weights


def display_portfolio(tickers, weights, total_value, title="Portfolio Allocation"):
    print(f"\n{title}:")
    for t, w in zip(tickers, weights):
        print(f"{t}: {w:.2%} -> ${w * total_value:.2f}")

 
# Interactive stress test 
def run_stress_prompt(mu, sigma, weights, tickers, horizon, total_value, label="Current"):
    do_stress = input(f"\nWould you like to run a stress test on the {label} portfolio? (y or n): ").strip().lower()
    if do_stress != 'y':
        return
    
    method = input("Stress method ('economic' to scale covariance, 'stock' to shock specific tickers): ").strip().lower()
    dist = input("Stress distribution ('normal' or 't-dist'): ").strip().lower()
    if dist not in ('normal', 't-dist'):
        print("Invalid distribution. Skipping test stress.")
        return
    
    df = 5
    if dist == 't-dist':
        df_in = input("Degrees of freedom for t-dist (press Enter or Return for 5): ").strip()
        if df_in:
            try:
                df = max(2, int(df_in))
            except ValueError:
                print("Invalid df. Using 5.")

    if method == 'economic':
        try:
            scale_factor = float(input("Enter volatility scale factor (e.g. 2.0 doubles volatility): ").strip())
        except ValueError:
            print("Invalid scale factor. Skipping stress test.")
            return
        
        stressed_losses = run_stress_test(mu, sigma, weights, tickers, T=horizon, N=100_000, method='economic', scale_factor=scale_factor, dist=dist, df=df)

    elif method == 'stock':
        tickers_input = input(f"Tickers to shock (subset of {tickers}, comma separated): ").strip().upper()
        shock_tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
        unknown = [t for t in shock_tickers if t not in tickers]
        if unknown:
            print(f"Unknown ticker(s) for shock: {unknown}. Skipping stress test.")
            return
        
        try:
            shock_pct = float(input("Enter shock to those tickers as decimal (e.g. -0.08 for -8%): ").strip())
        except ValueError:
            print("Invalid shock size. Skipping stress test.")
            return
        
        stressed_losses = run_stress_test(mu, sigma, weights, tickers, T=horizon, N=100_000, method='stock', shock_tickers=shock_tickers, shock_pct=shock_pct, dist=dist, df=df)

    else:
        print("Invalid method. Skipping stress test.")
        return
    
    stressed_var = compute_var(stressed_losses, 0.95)
    stressed_cvar = compute_cvar(stressed_losses, 0.95)
    stressed_ci_results = bootstrap_confidence_intervals(stressed_losses, alpha=0.95, n_boot=2000)

    print(f"VaR 95% = {stressed_var:.4%} USD {stressed_var * total_value:.2f}")
    print(f"   95% CI for VaR: {stressed_ci_results['var_ci'][0]:.4%} to {stressed_ci_results['var_ci'][1]:.4%}")
    print(f"CVaR 95% = {stressed_cvar:.4%} USD {stressed_cvar * total_value:.2f}")
    print(f"   95% CI for CVaR: {stressed_ci_results['cvar_ci'][0]:.4%} to {stressed_ci_results['cvar_ci'][1]:.4%}")



### Now to run user logic in main
def main():
    print("Interactive Monte Carlo Portfolio Risk Simulation With Optimization")

    tickers, investments, window_days, horizon = get_user_input()
    print(f"\nFetching data for {tickers}...")
    data = fetch_data(tickers, window_days)

    mu, sigma, weights = prepare_inputs(data, investments)
    total_value = np.sum(investments)

    print(f"\nRunning Monte Carlo simulation for a {horizon}-day investment horizon...")
    losses = simulate_portfolio_losses(mu, sigma, weights, T=horizon, N=100_000)

    var_95 = compute_var(losses, 0.95)
    cvar_95 = compute_cvar(losses, 0.95)
    reg_ci_results = bootstrap_confidence_intervals(losses, alpha=0.95, n_boot=2000)

    display_portfolio(tickers, weights, total_value, "Initial Portfolio Allocation)")

    print(f"VaR 95% = {var_95:.4%} USD {var_95 * total_value:.2f}")
    print(f"   95% CI for VaR: {reg_ci_results['var_ci'][0]:.4%} to {reg_ci_results['var_ci'][1]:.4%}")
    print(f"CVaR 95% = {cvar_95:.4%} USD {cvar_95 * total_value:.2f}")
    print(f"   95% CI for CVaR: {reg_ci_results['cvar_ci'][0]:.4%} to {reg_ci_results['cvar_ci'][1]:.4%}")


#Optional portfolio optimization
    opt_weights = None
    choice = input("\nWould you like to optimize your portfolio to minimize CVaR? (y or n): ").strip().lower()
    if choice == 'y':
        print("\nOptimizing portfolio...")
        opt_weights = minimize_cvar(mu, sigma, T=horizon)

        opt_losses = simulate_portfolio_losses(mu, sigma, opt_weights, T=horizon, N=100_000)
        opt_var = compute_var(opt_losses, 0.95)
        opt_cvar = compute_cvar(opt_losses, 0.95)
        opt_ci_results = bootstrap_confidence_intervals(opt_losses, alpha=0.95, n_boot=2000)

        display_portfolio(tickers, opt_weights, total_value, "Optimized Portfolio Allocation")
        print(f"VaR 95% = {opt_var:.4%} USD {opt_var * total_value:.2f}")
        print(f"   95% CI for VaR: {opt_ci_results['var_ci'][0]:.4%} to {opt_ci_results['var_ci'][1]:.4%}")
        print(f"CVaR 95% = {opt_cvar:.4%} USD {opt_cvar * total_value:.2f}")
        print(f"   95% CI for CVaR: {opt_ci_results['cvar_ci'][0]:.4%} to {opt_ci_results['cvar_ci'][1]:.4%}")
    else:
        print("\nNo optimization performed.")

    # Now to run the stressor after optimization
    run_stress_prompt(mu, sigma, weights, tickers, horizon, total_value, label='Initial')

    if opt_weights is not None:
        run_stress_prompt(mu, sigma, weights, tickers, horizon, total_value, label='Optimized')
    


### Call all functions to iteract with user
if __name__ == "__main__":
    main()

