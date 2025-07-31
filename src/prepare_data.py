import yfinance as yf
import pandas as pd
import numpy as np
import os

# Choose tickers
tickers = ['AMZN', 'SBUX', 'NKE', 'AAPL', 'TSLA', 'GOOGL', 'META', 'WMT', 'ADDYY']

# Download price data
data = yf.download(tickers, start='2023-01-01', end='2025-06-01')['Close']
data.dropna(inplace=True)

# Compute log returns
log_returns = np.log(data / data.shift(1)).dropna()

# Compute expected return vector and covariance matrix
mu = log_returns.mean().values
sigma = log_returns.cov().values

# Save for simulation use
log_returns.to_csv('data/historical_returns.csv')
np.save('data/mu.npy', mu)
np.save('data/sigma.npy', sigma)

print("Data prepared and saved to /data")
