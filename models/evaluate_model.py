import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def evaluate_strategy(df, y_pred, y_test):
    df = df.iloc[-len(y_test):].copy()
    df['prediction'] = y_pred
    df['return'] = df['Close'].pct_change().fillna(0)

    df['strategy_return'] = df['return'] * df['prediction'].shift(1).fillna(0)
    df['cumulative_market'] = (1 + df['return']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod()

    print(classification_report(y_test, y_pred))

    df[['cumulative_market', 'cumulative_strategy']].plot(figsize=(10, 5))
    plt.title("Strategy vs. Market Returns")
    plt.grid()
    plt.show()

    return df

if __name__ == "__main__":
    from train_model import load_data, train_xgboost_model
    df = load_data("data/processed/BTC-USD_processed.csv")
    model, X_test, y_test, y_pred = train_xgboost_model(df)
    evaluate_strategy(df, y_pred, y_test)
