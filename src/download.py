import yaml
import os
import requests
from pathlib import Path

data_list = ["listings.csv.gz", "calendar.csv.gz", "reviews.csv.gz"]

def download_raw(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_root = config.get("data_root")
    snapshots = config.get("snapshots")

    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    for snapshot in snapshots:
        snapshot_folder = raw_dir / str(snapshot)
        snapshot_folder.mkdir(parents=True, exist_ok=True)
        for file_name in data_list:
            URL = f"https://data.insideairbnb.com/{data_root}/{snapshot}/data/{file_name}"
            response = requests.get(URL, timeout=20)
            response.raise_for_status()

            save_file = snapshot_folder / file_name
            save_file.write_bytes(response.content)


if __name__ == "__main__":  
    config_path = Path(__file__).resolve().parent.parent / "configs" / "porto.yaml"
    download_raw(config_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    snapshots = config.get("snapshots")
    for snapshot in snapshots:
        snapshot_folder = raw_dir / str(snapshot)
        for file_name in data_list:
            save_file = snapshot_folder / file_name
            assert save_file.exists(), f"Thieu file {snapshot}/{file_name}"
    print("Downloaded raw successfully")