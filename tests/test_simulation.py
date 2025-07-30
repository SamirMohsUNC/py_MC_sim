import sys
from pathlib import Path

# Add correct path to sys.path — the full 'src' folder
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from simulate import simulate_portfolio_losses
from risk_metrics import compute_var, compute_cvar
