import numpy as np
import pandas as pd
from src.simulate import simulate_portfolio_losses, simulate_t_dist_losses
from src.risk_metrics import compute_var, compute_cvar
from scipy.stats import chi2


def backtest_var_cvar(
    returns_df,
    weights,
    alpha=0.95,
    window=252,
    horizon=1,
    N=50_000,
    dist="t-dist",   # "normal" or "t-dist"
    df=5,
    seed=123,
):
    """
    Rolling Monte Carlo VaR/CVaR backtest:
      - Fit μ, Σ on rolling 'window' days
      - Simulate horizon-day losses
      - Record VaR_t, CVaR_t
      - Compare to realized horizon loss
    Returns dict with series and Kupiec/Christoffersen diagnostics.
    """

    def kupiec_pof(exceed, alpha):
        # Unconditional coverage test: exceptions ~ Bernoulli(1-alpha)
        x = int(exceed.sum())
        n = int(exceed.size)
        if n == 0:
            return np.nan, np.nan
        pi_hat = x / n
        pi0 = 1 - alpha
        eps = 1e-12
        # Log-likelihoods
        ll1 = (n - x) * np.log(max(1 - pi_hat, eps)) + x * np.log(max(pi_hat, eps))
        ll0 = (n - x) * np.log(max(1 - pi0,  eps)) + x * np.log(max(pi0,  eps))
        LR  = -2.0 * (ll0 - ll1)
        return float(LR), float(chi2.sf(LR, df=1))

    def christoffersen_ind(exceed):
        # Independence test via 2x2 transition matrix
        if exceed.size <= 1:
            return np.nan, np.nan
        x_t  = exceed[1:]
        x_tm = exceed[:-1]
        n00 = int(((x_tm == 0) & (x_t == 0)).sum())
        n01 = int(((x_tm == 0) & (x_t == 1)).sum())
        n10 = int(((x_tm == 1) & (x_t == 0)).sum())
        n11 = int(((x_tm == 1) & (x_t == 1)).sum())
        n0, n1 = n00 + n01, n10 + n11
        if n0 == 0 or n1 == 0:
            return np.nan, np.nan

        p01 = n01 / n0
        p11 = n11 / n1
        p   = (n01 + n11) / (n0 + n1)
        eps = 1e-12

        def ll(p01v, p11v):
            # Return a scalar 
            return (
                n00 * np.log(max(1 - p01v, eps)) +
                n01 * np.log(max(p01v,      eps)) +
                n10 * np.log(max(1 - p11v, eps)) +
                n11 * np.log(max(p11v,      eps))
            )

        LR  = -2.0 * (ll(p, p) - ll(p01, p11))
        return float(LR), float(chi2.sf(LR, df=1))

    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, float)
    returns_df = returns_df.dropna().astype(float)

    dates, var_list, cvar_list, realized_list = [], [], [], []

    # guard for short samples
    if len(returns_df) < window + horizon:
        raise ValueError("Not enough data for the chosen window and horizon.")

    for t in range(window, len(returns_df) - horizon + 1):
        train  = returns_df.iloc[t - window: t]
        future = returns_df.iloc[t: t + horizon]

        mu    = train.mean().values
        sigma = train.cov().values

        local_seed = int(rng.integers(0, 1_000_000_000))
        if dist == "normal":
            losses = simulate_portfolio_losses(mu, sigma, weights, T=horizon, N=N, seed=local_seed)
        else:
            losses = simulate_t_dist_losses(mu, sigma, weights, T=horizon, N=N, df=df, seed=local_seed)

        v  = compute_var(losses, alpha)
        es = compute_cvar(losses, alpha)

        # realized horizon loss: sum log-returns over horizon, weight, negate
        asset_logret = future.sum(axis=0).values
        rlz = float(-(asset_logret @ weights))

        var_list.append(v)
        cvar_list.append(es)
        realized_list.append(rlz)
        dates.append(returns_df.index[t + horizon - 1])

    var_arr   = np.array(var_list)
    cvar_arr  = np.array(cvar_list)
    realized  = np.array(realized_list)
    exceed    = (realized >= var_arr).astype(int)

    hit_rate = float(exceed.mean()) if exceed.size else np.nan
    kup_LR, kup_p = kupiec_pof(exceed, alpha)
    chr_LR, chr_p = christoffersen_ind(exceed)

    tail = realized >= var_arr
    avg_real_tail = float(realized[tail].mean()) if tail.any() else np.nan
    avg_es_tail   = float(cvar_arr[tail].mean()) if tail.any() else np.nan
    es_gap = float(avg_real_tail - avg_es_tail) if tail.any() else np.nan

    return {
        "dates": np.array(dates),
        "realized_losses": realized,
        "var": var_arr,
        "cvar": cvar_arr,
        "exceedances": exceed,
        "hit_rate": hit_rate,
        "kupiec_LR": kup_LR, "kupiec_p": kup_p,
        "christ_LR": chr_LR, "christ_p": chr_p,
        "avg_realized_tail": avg_real_tail,
        "avg_forecast_cvar": avg_es_tail,
        "es_gap": es_gap,
        "alpha": alpha, "window": window, "horizon": horizon,
        "N": N, "dist": dist, "df": df,
        "tickers": list(returns_df.columns),
    }

