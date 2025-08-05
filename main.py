import numpy as np
import pandas as pd 
import yfinance as yf
from datetime import datetime, timedelta
from src.simulate import simulate_portfolio_losses
from src.risk_metrics import compute_var, compute_cvar

def get_user_input():
    tickers = input("Enter stock tickers separated by commas (e.g. AAPL,MSFT,GOOGL): ").strip().upper().split(',')
    tickers = [t.strip() for t in tickers]

    print("\nNow enter how much you're investing in each (same order):")
    investments = list(map(float, input("Investment ammounts")))