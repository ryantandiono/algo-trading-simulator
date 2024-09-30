import backtrader as bt
from typing import List


class PortfolioValueAnalyzer(bt.Analyzer):
    """
    Custom Analyzer to track portfolio value over time.
    This analyzer records the portfolio's value at each time step during the backtest.
    It can be extended to include additional performance metrics as needed.
    """

    def __init__(self):
        """
        Initializes the PortfolioValueAnalyzer.

        Sets up a list to store portfolio values.
        """
        super().__init__()
        self.portfolio_values: List[float] = []

    def next(self):
        """
        Called at each new bar.
        Records the current portfolio value.
        """
        current_value = self.strategy.broker.getvalue()
        self.portfolio_values.append(current_value)
        # Debugging statement (optional)
        # print(f"Portfolio Value at {self.datas[0].datetime.date(0)}: {current_value:.2f}")

    def get_analysis(self) -> List[float]:
        """
        Retrieves the recorded portfolio values.
        """
        return self.portfolio_values
