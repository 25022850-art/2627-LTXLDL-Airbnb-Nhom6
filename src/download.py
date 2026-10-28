import yaml
import os
import requests
import pandas as pd
from pathlib import Path

data_list = ["listings.csv.gz", "calendar.csv.gz", "reviews.csv.gz"]

def download_raw(config_path, listing_ids_path, review_ids_path):
    df_listing = pd.read_csv(listing_ids_path, dtype={"listing_id": str}, encoding="utf-8")
    df_review = pd.read_csv(review_ids_path, dtype={"review_id": str}, encoding="utf-8")

    df_listing_target = df_listing[
            (df_listing["Lớp"].astype(str).str.contains("3")) & 
            (df_listing["Nhóm"].astype(str) == "6")
        ].copy()
    df_review_target = df_review[
            (df_review["Lớp"].astype(str).str.contains("3")) &
            (df_review["Nhóm"].astype(str) == "6")
        ].copy()
    print(df_review_target["review_id"].isna().sum())
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

            df = pd.read_csv(save_file, compression='gzip', low_memory=False)
            
            if file_name == "listings.csv.gz":
                df = df[df['id'].isin(df_listing_target['listing_id'])]
                # TODO: Thêm logic merge để gắn cột 'analysis_area' theo yêu cầu
                
            elif file_name == "calendar.csv.gz":
                df = df[df['listing_id'].isin(df_listing_target['listing_id'])]
                
            elif file_name == "reviews.csv.gz":
                df = df[df['id'].isin(df_review_target['review_id'])]

            df.to_csv(save_file, index=False, compression='gzip')
            print(f"Đã tải và lọc xong {file_name} mốc {snapshot}.")

if __name__ == "__main__":  
    config_path = Path(__file__).resolve().parent.parent / "configs" / "porto.yaml"

    sheet_id = "1xaMiF78Ta90IWm2oHESKD35IHCGEcOMbGiYfUbmiwxA"
    gid_review = "2096394307"
    gid_listing = "454441734"
    url_listing = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid_listing}"
    url_review = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid_review}"

    download_raw(config_path, url_listing, url_review)

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