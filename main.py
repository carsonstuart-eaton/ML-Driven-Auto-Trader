import yaml
from pipeline.downloader import download_asset
from pipeline.cleaner import clean_data
from pipeline.feature_engineer import add_technical_indicators
import os

with open("pipeline/config.yaml", "r") as f:
    config = yaml.safe_load(f)

for asset in config["assets"]:
    raw_path = os.path.join(config["data_dir"], "raw")
    processed_path = os.path.join(config["data_dir"], "processed")

    df = download_asset(asset, config["start_date"], config["end_date"], raw_path)
    df = clean_data(df)
    # df = add_technical_indicators(df)

    df.to_csv(os.path.join(processed_path, f"{asset}_processed.csv"))
