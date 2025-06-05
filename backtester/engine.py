import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, df, prediction_col='prediction', return_col='Close', initial_cash=10000):
        self.df = df.copy()
        self.prediction_col = prediction_col
        self.return_col = return_col
        self.initial_cash = initial_cash
        self.results = None

    def prepare_data(self):
        # Calculate daily returns
        self.df['return'] = self.df[self.return_col].pct_change().fillna(0)

        # Shift prediction so we act on the signal at the open of next bar
        self.df['signal'] = self.df[self.prediction_col].shift(1).fillna(0)

        # Simulate strategy returns
        self.df['strategy_return'] = self.df['signal'] * self.df['return']

        # Cumulative returns
        self.df['cumulative_market'] = (1 + self.df['return']).cumprod()
        self.df['cumulative_strategy'] = (1 + self.df['strategy_return']).cumprod()

        self.results = self.df

    def get_results(self):
        if self.results is None:
            self.prepare_data()
        return self.results

    def summary_stats(self):
        if self.results is None:
            self.prepare_data()

        total_return = self.results['cumulative_strategy'].iloc[-1] - 1
        sharpe = self.results['strategy_return'].mean() / self.results['strategy_return'].std()
        max_dd = self._max_drawdown(self.results['cumulative_strategy'])

        return {
            'Total Return (%)': round(total_return * 100, 2),
            'Sharpe Ratio': round(sharpe, 3),
            'Max Drawdown (%)': round(max_dd * 100, 2)
        }

    def _max_drawdown(self, equity_curve):
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()
