import yfinance as yf
import pandas as pd
import os

def download_asset(asset: str, start: str, end: str, save_path: str):
    df = yf.download(asset, start=start, end=end)
    df.to_csv(os.path.join(save_path, f"{asset}.csv"))
    return df
