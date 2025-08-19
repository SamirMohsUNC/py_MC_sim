import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.optimizer import minimize_cvar
from src.simulate import simulate_portfolio_losses, simulate_t_dist_losses
from src.risk_metrics import compute_var, compute_cvar



def realized_horizon_loss(returns_block, weights):
    # sum daily log returns over the horizon for each asset; portfolio return = dot; loss = -return
    asset_logret = returns_block.sum(axis=0).values
    return float(-(np.dot(asset_logret, weights)))


def rolling_optimize_and_dashboard(returns_df, baseline_weights, alpha=0.95, window=252, horizon=1, N=50_000,
    dist="normal",     # "normal" | "t-dist"
    df=5, save_dir="reports", seed=123):
    """
    Rolling CVaR-minimizing optimization with risk evaluation vs a baseline.

    returns_df: DataFrame of DAILY log returns (index: dates; columns: tickers)
    baseline_weights: np.array aligned to returns_df.columns
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    dates = []
    w_rolling = []
    var_fore = []
    cvar_fore = []
    rlz_losses_rolling = []
    rlz_losses_baseline = []
    turnover = []

    # for cumulative performance, we build a log-returns path
    roll_log_returns = []
    base_log_returns = []

    cols = list(returns_df.columns)
    baseline_weights = np.asarray(baseline_weights, dtype=float)
    baseline_weights = baseline_weights / baseline_weights.sum()

    # simple baseline weight path. keep baseline constant
    prev_w = None

    if len(returns_df) < window + horizon:
        raise ValueError("Not enough data: increase history or reduce window/horizon")
    
    for t in range(window, len(returns_df)-horizon+1):
        train = returns_df.iloc[t - window: t]
        future = returns_df.iloc[t: t + horizon]
        
        mu = train.mean().values
        sigma = train.cov().values

        #optimize weights for this step
        w_opt = minimize_cvar(mu, sigma, T=horizon, N=N)

        #forecast VaR/CVaR for optimized weights
        local_seed = int(rng.integers(0, 1_000_000_000))
        if dist == 'normal': 
            losses = simulate_portfolio_losses(mu, sigma, w_opt, T=horizon, N=N, seed=local_seed)
        else:
            losses = simulate_portfolio_losses(mu, sigma, w_opt, T=horizon, N=N, df=df, seed=local_seed)

        v = compute_var(losses, alpha)
        es = compute_cvar(losses, alpha)

        #realized losses over nect horizon
        rlz_roll = realized_horizon_loss(future, w_opt)
        rlz_base = realized_horizon_loss(train, baseline_weights)

        #store
        dates.append(returns_df.index[t + horizon - 1])
        w_rolling.append(w_opt)
        var_fore.append(v)
        cvar_fore.append(es)
        rlz_losses_rolling.append(rlz_roll)
        rlz_losses_baseline.append(rlz_base)

        #realized log returns
        roll_log_returns.append(-rlz_roll)
        base_log_returns.append(-rlz_base)

        #turnover from previous step
        if prev_w is None:
            turnover.append(0.0)
        else:
            turnover.append(float(np.abs(w_opt - prev_w).sum()))
        prev_w = w_opt


    #arrays/series
    dates = pd.to_datetime(pd.Index(dates))
    var_fore = np.array(var_fore)
    cvar_fore = np.array(cvar_fore)
    rlz_roll = np.array(rlz_losses_rolling)
    rlz_base = np.array(rlz_losses_baseline)
    exceed_roll = (rlz_roll >= var_fore).astype(int)

    #cumulative performance (log space)
    cum_roll = np.exp(np.cumsum(np.array(roll_log_returns)))
    cum_base = np.exp(np.cumsum(np.array(base_log_returns)))

    #summary stats
    hit_rate = exceed_roll.mean()
    avg_turnover = float(np.mean(turnover))
    avg_var = float(np.mean(var_fore))
    avg_cvar = float(np.mean(cvar_fore))
    total_outperf = float(cum_roll[-1] / cum_base[-1] - 1.0)

    summary = {
        "alpha": alpha,
        "window": window,
        "horizon": horizon,
        "N": N,
        "dist": dist,
        "df": df,
        "hit rate": float(hit_rate),
        "avg_turnover_L1": avg_turnover,
        "avg_VaR": avg_var,
        "avg_CVaR": avg_cvar,
        "rolling_final_growth": float(cum_roll[-1]),
        "baseline_final_growth": float(cum_base[-1]),
        "rolling_vs_baseline_outperformance": total_outperf
        }
    
    
    #dashboard creation
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_dir, f"rolling_dashboard_{ts}.png")

    fig, axes = plt.subplots(3, 1, figsize = (10, 12))

    # 1) performance
    axes[0].plot(dates, cum_roll, label = "Rolling-Optimized")
    axes[0].plot(dates, cum_base, label = "Baseline (Static)")
    axes[0].set_title("Cumulative Growth (log-return compounding)")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Growth of 1$")
    axes[0].legend()

    # 2) VaR forecast vs realized loss (rolling)
    axes[1].plot(dates, var_fore, label = "Forecast VaR (alpha)")
    axes[1].plot(dates, rlz_roll, label = "Realized Loss")
    axes[1].set_title(f"Roling Forecast VaR (a={alpha}) vs. Realized Loss")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    # 3) Turnover (L1 distance step to step)
    axes[2].plot(dates, turnover, label = "Turnover (L1)")
    axes[2].set_title("Portfolio Turnover")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Sum")
    axes[2].legend()

    # Formatting
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return summary, out_path

