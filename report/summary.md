# Project Summary — Monte Carlo Portfolio Risk Simulation

This project is a complete **portfolio risk management toolkit** implemented in Python.  

It integrates:
- **Risk measurement**: VaR, CVaR, bootstrap confidence intervals.  
- **Optimization**: minimize CVaR with dynamic weight constraints.  
- **Stress testing**: economic (covariance scaling) and asset-specific shocks.  
- **Backtesting**: Kupiec and Christoffersen tests to validate model accuracy.  
- **Rolling strategies**: adaptive optimization with dashboard output.  

### Key Innovations
- **Dynamic optimization using EWMA (10–45 day windows)** to capture recent market behavior.  
- **Interactive CLI** in `main.py` for user-driven scenarios.  
- **Robust error handling** for ticker alignment, missing data, and covariance conditioning.  

### Technical Stack
- Python (numpy, pandas, scipy, scikit-learn, matplotlib, seaborn)  
- yfinance for market data  
- Optimization with SLSQP (scipy.optimize)  

### Use Case
This tool demonstrates how professional risk teams and hedge funds might:
- Measure tail risk exposure.  
- Rebalance portfolios to reduce CVaR.  
- Validate models through statistical tests.  
- Stress portfolios against shocks.  

### Resume Value
This project highlights:
- **Quant finance concepts** (VaR, CVaR, EWMA, backtesting).  
- **Numerical optimization** and constraints handling.  
- **Statistical modeling and testing**.  
- **Software engineering** best practices (modular design, testing, documentation).  

It is both a **learning project** and a **practical risk simulation engine** that can be extended with machine learning, factor models, or alternative risk measures.
