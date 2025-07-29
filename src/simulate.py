import yfinance as yf
import pandas as pd
import numpy as np

# Choose my tickers
tickers = ['AMZN', 'SBUX', 'NKE', 'AAPL', 'TSLA',
            'GOOGL', 'META', 'WMT', 'ADDYY']

# Gather data
data = yf.download(tickers, start = '2023-01-01', end = '2025-06-01')['Close']

# Find log returns
log_returns = np.log(data / data.shift(1)).dropna()

# Expected returns and covariance matrix
mu = log_returns.mean().values
sigma = log_returns.cov().values

# Convert to csv for later use
log_returns.to_csv('data/historical_returns.csv')
np.save('data/mu.npy', mu)
np.save('data/sigma.npy', sigma)