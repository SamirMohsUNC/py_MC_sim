import numpy as np
import pandas as pd 
import yfinance as yf
from datetime import datetime, timedelta

from src.simulate import simulate_portfolio_losses
from src.risk_metrics import compute_var, compute_cvar
from src.optimizer import minimize_cvar


# Create questions for user to answer
def get_user_input():
    tickers = input("Enter stock tickers separated by commas (e.g. AAPL,MSFT,GOOGL): ").strip().upper().split(',')
    tickers = [t.strip() for t in tickers]

    print("\nNow enter how much you're investing in each (same order):")
    investments = list(map(float, input("Investment ammounts (comma separated): ").strip().split(',')))

    assert len(tickers) == len(investments), "Number of tickers must match numberof investments."

    print("\nChoose historical data window:")
    print("1 - Last 6 months\n2 - Last 1 year\n3 Last 2 years")
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


# Now to run user logic
def main():
    print("Interactive Monte Carlo Portfolio Risk Simulation With Optimization")

    tickers, investments, window_days, horizon = get_user_input()
    print(f"\nFetching data for {tickers}...")
    data = fetch_data(tickers, window_days)

    mu, sigma, weights = prepare_inputs(data, investments)
    total_value = np.sum(investments)

    print(f"\nRunning Monte Carlo simulation for a {horizon}-day investment horizon...")
    losses = simulate_portfolio_losses(mu, sigma, weights, T=horizon, N=100000)

    var_95 = compute_var(losses, 0.95)
    cvar_95 = compute_cvar(losses, 0.95)

    display_portfolio(tickers, weights, total_value, "Initial Portfolio Allocation)")

    print(f"\n Portfolio VaR 95% = {var_95:.4%} ({var_95 * total_value:.2f} USD)")
    print(f"\n Portfolio CVaR 95% = {cvar_95:.4%} ({cvar_95 * total_value:.2f} USD)")

    choice = input("\nWould you like to optimize your portfolio to minimize CVaR? (y/n): ").strip().lower()
    if choice == 'y':
        print("\nOptimizing portfolio...")
        opt_weights = minimize_cvar(mu, sigma, T=horizon)

        opt_losses = simulate_portfolio_losses(mu, sigma, opt_weights, T=horizon, N=100000)
        opt_var = compute_var(opt_losses, 0.95)
        opt_cvar = compute_cvar(opt_losses, 0.95)

        display_portfolio(tickers, opt_weights, total_value, "Optimized Portfolio Allocation")
        print(f"\nOptimized VaR 95% ={opt_var:.4%} ({opt_var * total_value:.2f} USD)")
        print(f"\nOptimized CVaR 95% ={opt_cvar:.4%} ({opt_cvar * total_value:.2f} USD)")
    else:
        print("\nNo optimization performed.")


if __name__ == "__main__":
    main()