import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from src.simulate import simulate_portfolio_losses
from src.risk_metrics import compute_var, compute_cvar, bootstrap_confidence_intervals
from src.optimizer import optimize_cvar, minimize_cvar_from_returns  
from src.stress_test import run_stress_test
from src.backtest import backtest_var_cvar
from src.rolling import rolling_optimize_and_dashboard


# -----------------------------
# Helpers
# -----------------------------
def get_user_input():
    tickers = input("Enter stock tickers separated by commas (e.g. AAPL,MSFT,GOOGL): ").strip().upper().split(',')
    tickers = [t.strip() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError("No tickers provided.")

    print("\nNow enter how much you're investing in each (same order):")
    investments = list(map(float, input("Investment ammounts (comma separated): ").strip().split(',')))
    if len(tickers) != len(investments):
        raise ValueError("Number of tickers must match number of investments.")

    print("\nChoose historical data window:")
    print("1 - Last 3 years")
    print("2 - Last 5 year")
    print("3 - Last 7 years")
    window_choice = input("Enter 1, 2, or 3: ").strip()
    window_days_map = {'1': 782, '2': 1304, '3': 1825}
    window_days = window_days_map.get(window_choice)
    if window_days is None:
        raise ValueError("Invalid window selection.")

    print("\nChoose investment horizon for risk calculation")
    print("1 - 1 day")
    print("2 - 2 days")
    print("3 - 3 days")
    print("4 - 6 days")
    print("5 - 10 days")
    horizon_map = {'1': 1, '2': 2, '3': 3, '4': 6, '5': 10}
    horizon_days = horizon_map.get(input("Enter 1-5: ").strip())
    if horizon_days is None:
        raise ValueError("Invalid horizon selection.")

    return tickers, investments, window_days, horizon_days


def fetch_data(tickers, window_days):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=window_days)
    print(f"\nFetching data for {tickers}...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        px = data['Close']
    else:
        px = data
    # keep only rows with all tickers to avoid NaNs in covariance
    px = px.dropna(how='any')
    data = px
    if data.shape[0] < 60:
        raise ValueError("Not enough historical price data after cleaning.")
    return data


def portfolio_weights_from_amounts(investments):
    w = np.array(investments, dtype=float)
    w = w / w.sum()
    return w


def display_portfolio(tickers, weights, total_value, title="Portfolio Allocation"):
    print(f"\n{title}:")
    for t, w in zip(tickers, weights):
        print(f"{t}: {w:.2%} -> ${w * total_value:.2f}")


# -----------------------------
# Main flow
# -----------------------------
def main():
    print(" Monte Carlo Portfolio Risk Simulation")
    tickers, investments, window_days, horizon = get_user_input()

    data = fetch_data(tickers, window_days)
    returns_df = np.log(data / data.shift(1)).dropna()

    weights = portfolio_weights_from_amounts(investments)
    total_value = float(sum(investments))

    # Choose how to estimate μ/Σ for baseline reporting (sample or EWMA short-window) 
    print("\nHow should we estimate μ/Σ for baseline risk?")
    print("1 - Sample (equal-weighted over full lookback)")
    print("2 - EWMA (exponentially weighted, short recent window)")
    baseline_stats_choice = input("Enter 1 or 2: ").strip()
    if baseline_stats_choice == '2':
        # Short, reactive baseline
        try:
            base_wdays = int(input("Baseline EWMA window (10–45 trading days): ").strip())
        except Exception:
            base_wdays = 17
        base_wdays = max(10, min(45, base_wdays))
        try:
            base_lam = float(input("Baseline EWMA decay λ (0.825–0.985 typical; Enter for 0.94): ").strip() or "0.94")
        except Exception:
            base_lam = 0.94
        base_lam = max(0.80, min(0.995, base_lam))

        recent = returns_df.iloc[-base_wdays:]
        # compute EWMA μ/Σ inline (same as optimizer’s helper)
        R = recent.values
        Tn = R.shape[0]
        w = (1.0 - base_lam) * base_lam ** np.arange(Tn - 1, -1, -1)
        w = w / w.sum()
        mu = (R * w[:, None]).sum(axis=0)
        X = R - mu
        sigma = np.dot((X.T * w), X)
        # light diagonal shrinkage
        gamma = 0.10
        sigma = (1 - gamma) * sigma + gamma * np.diag(np.diag(sigma))
    else:
        # Full-sample stats
        mu = returns_df.mean().values
        sigma = returns_df.cov().values

    # Baseline simulation (with static or dynamic)
    print(f"\nRunning Monte Carlo simulation for a {horizon}-day investment horizon...")
    losses = simulate_portfolio_losses(mu, sigma, weights, T=horizon, N=100_000)
    var_95 = compute_var(losses, 0.95)
    cvar_95 = compute_cvar(losses, 0.95)
    ci_results = bootstrap_confidence_intervals(losses, alpha=0.95, n_boot=2000)

    display_portfolio(tickers, weights, total_value, "Initial Portfolio Allocation)")
    print(f"VaR 95% = {var_95:.4%} USD {var_95 * total_value:.2f}")
    print(f"   95% CI for VaR: {ci_results['var_ci'][0]:.4%} to {ci_results['var_ci'][1]:.4%}")
    print(f"CVaR 95% = {cvar_95:.4%} USD {cvar_95 * total_value:.2f}")
    print(f"   95% CI for CVaR: {ci_results['cvar_ci'][0]:.4%} to {ci_results['cvar_ci'][1]:.4%}")

    # Optional backtest (initial portfolio)
    do_bt_i = input("\nRun a rolling VaR/CVaR backtest on the lookback window for your initial portfolio? (y or n): ").strip().lower()
    if do_bt_i == 'y':
        window_bt_i = min(252, max(60, len(returns_df) // 2))
        bt_i = backtest_var_cvar(
            returns_df, weights,
            alpha=0.95, window=window_bt_i, horizon=2,
            N=50_000, dist='t-dist', df=5
        )
        print(f"\nBacktest (alpha={bt_i['alpha']}, horizon={bt_i['horizon']}d, window={bt_i['window']}):")
        print(f" - VaR hit rate: {bt_i['hit_rate']:.3f} (expected {1-bt_i['alpha']:.3f})")
        print(f" - Kupiec  POF: LR={bt_i['kupiec_LR']:.3f}, p={bt_i['kupiec_p']:.3f}")
        print(f" - Christoffersen IND: LR={bt_i['christ_LR']:.3f}, p={bt_i['christ_p']:.3f}")
        if bt_i['avg_realized_tail'] == bt_i['avg_realized_tail']:
            print(f" - Tail realized vs CVaR: {bt_i['avg_realized_tail']:.4%} vs {bt_i['avg_forecast_cvar']:.4%} (gap {bt_i['es_gap']:.4%})")
    else:
        print("\nNo backtest desired")

    # Optional portfolio optimization (static vs dynamic) 
    opt_weights = None
    choice = input("\nWould you like to optimize your portfolio to minimize CVaR? (y or n): ").strip().lower()
    if choice == 'y':
        print("\nChoose optimization mode:")
        print("1 - Static (use current μ/Σ you computed above)")
        print("2 - Dynamic (EWMA μ/Σ from a short recent window, 10–45 trading days)")
        mode_choice = input("Enter 1 or 2: ").strip()

        if mode_choice == '1':
            # Static: uses the same mu, sigma already computed in this run
            print("\nOptimizing portfolio (Static μ/Σ)...")
            opt_weights = optimize_cvar(
                mode="static",
                T=horizon,
                N=8_000,       # lighter inside optimizer; simulate with 100k for reporting
                alpha=0.95,
                mu=mu,
                sigma=sigma
            )
            mu_used, sigma_used = mu, sigma

        else:
            # Dynamic: short-window EWMA stats for reactive trading
            try:
                window_days = int(input("Dynamic window (10–45 trading days): ").strip())
            except Exception:
                window_days = 17
            window_days = max(10, min(45, window_days))

            try:
                lam = float(input("EWMA decay λ (0.825–0.985 typical; Enter for 0.94): ").strip() or "0.94")
            except Exception:
                lam = 0.94
            lam = max(0.80, min(0.995, lam))

            print(f"\nOptimizing portfolio (Dynamic EWMA, window={window_days}d, λ={lam:.3f})...")
            opt_weights = optimize_cvar(
                mode="dynamic",
                T=horizon,
                N=8_000,       # faster objective for frequent rebalances
                alpha=0.95,
                returns_df=returns_df,
                window_days=window_days,
                lam=lam,
                method="ewma"
            )
            # For a fair evaluation of the optimized weights, use the same short-window EWMA μ/Σ for reporting:
            _, mu_used, sigma_used = minimize_cvar_from_returns(
                returns_df=returns_df, T=horizon, N=8_000, alpha=0.95,
                method="ewma", window_days=window_days, lam=lam
            )

        # Optimize portfolio for reporting metrics in simulation
        opt_losses = simulate_portfolio_losses(mu_used, sigma_used, opt_weights, T=horizon, N=100_000)
        opt_var = compute_var(opt_losses, 0.95)
        opt_cvar = compute_cvar(opt_losses, 0.95)
        opt_ci_results = bootstrap_confidence_intervals(opt_losses, alpha=0.95, n_boot=2000)

        display_portfolio(tickers, opt_weights, total_value, "Optimized Portfolio Allocation:")
        print(f"VaR 95% = {opt_var:.4%} USD {opt_var * total_value:.2f}")
        print(f"   95% CI for VaR: {opt_ci_results['var_ci'][0]:.4%} to {opt_ci_results['var_ci'][1]:.4%}")
        print(f"CVaR 95% = {opt_cvar:.4%} USD {opt_cvar * total_value:.2f}")
        print(f"   95% CI for CVaR: {opt_ci_results['cvar_ci'][0]:.4%} to {opt_ci_results['cvar_ci'][1]:.4%}")
    else:
        print("\nNo optimization performed.")

    # Optional backtest (after optimization) 
    if opt_weights is not None:
        do_bt_o = input("\nRun a rolling VaR/CVaR backtest on the lookback window for your optimized portfolio? (y or n): ").strip().lower()
        if do_bt_o == 'y':
            window_bt_o = min(252, max(60, len(returns_df) // 2))
            bt_o = backtest_var_cvar(
                returns_df, opt_weights,
                alpha=0.95, window=window_bt_o, horizon=2, N=50_000, dist='t-dist', df=5
            )
            print(f"\nBacktest (alpha={bt_o['alpha']}, horizon={bt_o['horizon']}d, window={bt_o['window']}):")
            print(f" - VaR hit rate: {bt_o['hit_rate']:.3f} (expected {1-bt_o['alpha']:.3f})")
            print(f" - Kupiec  POF: LR={bt_o['kupiec_LR']:.3f}, p={bt_o['kupiec_p']:.3f}")
            print(f" - Christoffersen IND: LR={bt_o['christ_LR']:.3f}, p={bt_o['christ_p']:.3f}")
            if bt_o['avg_realized_tail'] == bt_o['avg_realized_tail']:
                print(f" - Tail realized vs CVaR: {bt_o['avg_realized_tail']:.4%} vs {bt_o['avg_forecast_cvar']:.4%} (gap {bt_o['es_gap']:.4%})")
        else:
            print("\nNo backtest desired")

    # Optional stress testing (after optimization) 
    def stress_prompt(muX, sigmaX, weightsX, label):
        ans = input(f"\nWould you like to run a stress test on the {label} portfolio? (y or n): ").strip().lower()
        if ans != 'y':
            return
        method = input("Stress method ('economic' to scale covariance, 'stock' to shock specific tickers): ").strip().lower()
        dist = input("Stress distribution ('normal' or 't-dist'): ").strip().lower()
        df = 5
        if dist == 't-dist':
            try:
                df = max(2, int(input("Degrees of freedom for t-dist (Enter for 5): ").strip() or "5"))
            except Exception:
                df = 5

        if method == 'economic':
            try:
                scale_factor = float(input("Enter volatility scale factor (e.g., 2.0 doubles volatility): ").strip())
            except ValueError:
                print("Invalid scale factor. Skipping stress test.")
                return
            stressed_losses = run_stress_test(muX, sigmaX, weightsX, tickers, T=horizon, N=100_000,
                                              method="economic", scale_factor=scale_factor, dist=dist, df=df)
        elif method == 'stock':
            tickers_input = input(f"Tickers to shock (subset of {tickers}, comma-separated): ").strip().upper()
            shock_tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
            unknown = [t for t in shock_tickers if t not in tickers]
            if unknown:
                print(f"Unknown ticker(s) for shock: {unknown}. Skipping stress test.")
                return
            try:
                shock_pct = float(input("Enter shock to those tickers as decimal (e.g., -0.08 for -8%): ").strip())
            except ValueError:
                print("Invalid shock size. Skipping stress test.")
                return
            stressed_losses = run_stress_test(muX, sigmaX, weightsX, tickers, T=horizon, N=100_000,
                                              method="stock", shock_tickers=shock_tickers, shock_pct=shock_pct,
                                              dist=dist, df=df)
        else:
            print("Invalid method. Skipping stress test.")
            return

        s_var = compute_var(stressed_losses, 0.95)
        s_cvar = compute_cvar(stressed_losses, 0.95)
        print(f"\n⚠️ {label} portfolio — Stressed VaR 95% = {s_var:.4%} ({s_var * total_value:.2f} USD)")
        print(f"⚠️ {label} portfolio — Stressed CVaR 95% = {s_cvar:.4%} ({s_cvar * total_value:.2f} USD)")

    # stress initial
    stress_prompt(mu, sigma, weights, "Initial")
    # stress optimized (if exists)
    if opt_weights is not None:
        # Use the same μ/Σ employed for optimized reporting
        stress_prompt(mu_used, sigma_used, opt_weights, "Optimized")

    # --- Optional rolling dashboard ---
    do_roll = input("\nRun rolling adaptive optimization & risk dashboard? (y or n): ").strip().lower()
    if do_roll == 'y':
        # Rolling module compares a rolling strategy vs a fixed baseline
        baseline_for_roll = opt_weights if opt_weights is not None else weights
        window_bt = min(252, max(60, len(returns_df) // 2))
        summary, dashboard_path = rolling_optimize_and_dashboard(
            returns_df=returns_df,
            baseline_weights=baseline_for_roll,
            alpha=0.95,
            window=window_bt,
            horizon=1,
            N=20_000,         # lighter per-step sims inside function will handle speed
            dist="normal",
            df=5,
            save_dir="reports",
            seed=123,
        )
        print("\nRolling optimization summary:")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f" - {k}: {v:.6f}")
            else:
                print(f" - {k}: {v}")
        print(f"\nDashboard saved to: {dashboard_path}")


if __name__ == "__main__":
    main()
