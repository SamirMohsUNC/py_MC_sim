# Monte Carlo Portfolio Risk Simulation

A Python-based framework for **portfolio risk analysis and optimization**.  
This project integrates **Monte Carlo simulation**, **Value-at-Risk (VaR)**, **Conditional Value-at-Risk (CVaR)**,  
**portfolio optimization**, **stress testing**, and **backtesting** into a modular, user-friendly tool.

---

## 🚀 Features
- **Monte Carlo Simulation**  
  - Simulates portfolio losses across different horizons (1, 2, 4, 7, 10 days).  
  - Supports both **Normal** and **t-distribution** sampling.

- **Risk Metrics**
  - Value-at-Risk (VaR)  
  - Conditional Value-at-Risk (CVaR)  
  - Bootstrap confidence intervals for VaR & CVaR  

- **Portfolio Optimization**
  - Minimize CVaR under dynamic weight constraints.  
  - Choose between:
    - **Static optimization** (sample mean/covariance)  
    - **Dynamic optimization** using **EWMA (Exponentially Weighted Moving Average)**  
      with short rolling windows (10–45 trading days).  
  - Enforces realistic single-asset caps based on portfolio size.  

- **Stress Testing**
  - Apply **economic shocks** by scaling volatility.  
  - Apply **stock-specific shocks** (e.g., -10% on chosen tickers).  

- **Backtesting**
  - Kupiec POF test for hit rate accuracy.  
  - Christoffersen test for independence of exceptions.  
  - Rolling VaR & CVaR validation.  

- **Rolling Adaptive Optimization**
  - Run rolling re-optimization with short horizons.  
  - Compare performance against a static baseline.  
  - Generates a dashboard visualization of risk & return.

---

## 📂 Project Structure
# Monte Carlo Portfolio Risk Simulation

A Python-based framework for **portfolio risk analysis and optimization**.  
This project integrates **Monte Carlo simulation**, **Value-at-Risk (VaR)**, **Conditional Value-at-Risk (CVaR)**,  
**portfolio optimization**, **stress testing**, and **backtesting** into a modular, user-friendly tool.

---

## 🚀 Features
- **Monte Carlo Simulation**  
  - Simulates portfolio losses across different horizons (1, 2, 4, 7, 10 days).  
  - Supports both **Normal** and **t-distribution** sampling.

- **Risk Metrics**
  - Value-at-Risk (VaR)  
  - Conditional Value-at-Risk (CVaR)  
  - Bootstrap confidence intervals for VaR & CVaR  

- **Portfolio Optimization**
  - Minimize CVaR under dynamic weight constraints.  
  - Choose between:
    - **Static optimization** (sample mean/covariance)  
    - **Dynamic optimization** using **EWMA (Exponentially Weighted Moving Average)**  
      with short rolling windows (10–45 trading days).  
  - Enforces realistic single-asset caps based on portfolio size.  

- **Stress Testing**
  - Apply **economic shocks** by scaling volatility.  
  - Apply **stock-specific shocks** (e.g., -10% on chosen tickers).  

- **Backtesting**
  - Kupiec POF test for hit rate accuracy.  
  - Christoffersen test for independence of exceptions.  
  - Rolling VaR & CVaR validation.  

- **Rolling Adaptive Optimization**
  - Run rolling re-optimization with short horizons.  
  - Compare performance against a static baseline.  
  - Generates a dashboard visualization of risk & return.

---

## 📂 Project Structure
py_MC_sim/
│
├── main.py # Interactive entry point
├── src/
│ ├── simulate.py # Monte Carlo simulation functions
│ ├── risk_metrics.py # VaR, CVaR, confidence intervals
│ ├── optimizer.py # CVaR optimization (static & dynamic EWMA)
│ ├── stress_test.py # Stress testing methods
│ ├── backtest.py # Backtesting module
│ ├── rolling.py # Rolling optimization + dashboard
│
├── tests/
│ ├── test_simulation.py
│ ├── test_stress.py
│
├── data/ # Saved historical returns, mu, sigma (if needed)
├── reports/ # Rolling dashboard plots
└── README.md

---

## 🛠️ Installation
```bash
git clone https://github.com/your-username/py_MC_sim.git
cd py_MC_sim
pip install -r requirements.txt


---

## 🛠️ Installation
```bash
git clone https://github.com/your-username/py_MC_sim.git
cd py_MC_sim
pip install -r requirements.txt

Requirements
Python 3.9+
numpy, pandas, yfinance
matplotlib, seaborn
scikit-learn
scipy

Usage

Run the interactive program:
python main.py

You’ll be prompted to:
Enter tickers and investment amounts.
Choose historical data length (3y, 5y, 7y).
Select horizon (1–10 days).
Compute baseline VaR & CVaR.
Optionally: run backtests, optimize portfolio, apply stress tests, or generate a rolling dashboard.


Example Output

Initial Portfolio Allocation:
AAPL: 15.00% -> $150.00
MSFT: 10.00% -> $100.00
...

VaR 95% = 2.45% (USD 245.00)
CVaR 95% = 3.20% (USD 320.00)
95% CI for VaR: 2.40% to 2.50%
95% CI for CVaR: 3.15% to 3.25%


Why This Project Matters
Shows skills in quantitative finance, risk management, and numerical optimization.
Bridges theory (VaR, CVaR, EWMA, statistical backtesting) with practice (real financial data, realistic caps, stress tests).
Built to resemble a mini risk engine like those used in hedge funds or risk departments.


References
J.P. Morgan (1996). RiskMetrics Technical Document.
Christoffersen, P. (1998). Evaluating interval forecasts.
Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering.