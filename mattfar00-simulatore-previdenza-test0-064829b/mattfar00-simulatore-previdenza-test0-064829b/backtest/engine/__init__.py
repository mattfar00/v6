"""backtest.engine — esecuzione deterministica del backtest storico."""
from .engine import (
    backtest_lump_sum,
    backtest_con_contributi,
    equity_to_returns,
)

__all__ = [
    "backtest_lump_sum", "backtest_con_contributi", "equity_to_returns",
]
