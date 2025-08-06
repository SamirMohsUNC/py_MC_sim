Creating a Monte Carlo risk simulation with the ultimate goal of reducing portfolio risk.

Details of a portfolio can be input, along with answers to general simulation questions.

The program will then optimize the given portfolio to minimize Value at Risk and Conditional Value at Risk.

All methodology is clearly explained in src and test files, with optional stress testing available. 

Since these simulations are minimizing VaR and CVaR, this is meant for short term trading rather than holding stocks long term. Trading is also meant to be performed with stocks, rather than ETFs or mutual funds. Since stocks should only make up 5-15% of an entire portfolio, this does not put additional risk on the optimization constraints on quantity of a stock.

Of course, the market has inherent risks and this should not be used for serious investing. 