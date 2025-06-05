import pandas as pd
from backtester.engine import Backtester
import matplotlib.pyplot as plt

# Load your processed data with model predictions
df = pd.read_csv("data/processed/BTC-USD_processed.csv", index_col=0, parse_dates=True)
df['prediction'] = pd.read_csv("models/xgb_predictions.csv")['prediction']  # OR from model output

bt = Backtester(df, prediction_col='prediction', return_col='Close')
bt.prepare_data()

results = bt.get_results()
stats = bt.summary_stats()

print("Strategy Performance:")
for k, v in stats.items():
    print(f"{k}: {v}")

# Plot
results[['cumulative_market', 'cumulative_strategy']].plot(figsize=(10, 5), title="Equity Curve")
plt.grid()
plt.show()
