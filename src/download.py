import yaml
import requests
import pandas as pd
from pathlib import Path

data_list = ["listings.csv.gz", "calendar.csv.gz", "reviews.csv.gz"]

def check_reviews(df, snapshot):
    assert len(df) == 5000, f"Số dòng file listings mốc {snapshot} khác 5000; số dòng thực tế: {len(df)}"
    assert (df["Mẫu LLM"].str.strip().str.lower()=='x').sum() == 1000, \
                f"Số mẫu LLM khác 1000 (mốc: {snapshot}); số lượng thực tế: {(df['Mẫu LLM'].str.strip().str.lower()=='x').sum()}"
    assert (df["Nhãn tay"].str.strip().str.lower()=='x').sum() == 100,\
                f"Số 'Nhãn tay' khác 100 (mốc: {snapshot}), số lượng thực tế: {(df['Nhãn tay'].str.strip().str.lower()=='x').sum()}"

def check_listings(df, snapshot):
    assert len(df) == 1000, f"Số dòng file listings mốc {snapshot} khác 1000; số dòng thực tế: {len(df)}"

    room_type_count = df["room_type"].value_counts()
    assert room_type_count["Entire home/apt"] == 875, \
                f"Số loại 'Entire home/apt' ở mốc {snapshot} khác 875, số lượng thực tế: {room_type_count['Entire home/apt']}"
    assert room_type_count["Private room"] == 125, \
                f"Số loại 'Private room' ở mốc {snapshot} khác 125, số lượng thực tế: {room_type_count['Private room']}"

    if str(snapshot) == "2025-09-21":
        assert df["number_of_reviews"].eq(0).sum() == 50, \
                f"Số 'number_of_reviews' = 0 ở mốc {snapshot} khác 50, số lượng thực tế: {df['number_of_reviews'].eq(0).sum()}"
    elif str(snapshot) == "2025-12-25":
        assert df["number_of_reviews"].eq(0).sum() == 30, \
                f"Số 'number_of_reviews' = 0 ở mốc {snapshot} khác 30, số lượng thực tế: {df['number_of_reviews'].eq(0).sum()}"
    elif str(snapshot) == "2026-03-26":
        assert df["number_of_reviews"].eq(0).sum() == 25, \
                f"Số 'number_of_reviews' = 0 ở mốc {snapshot} khác 25, số lượng thực tế: {df['number_of_reviews'].eq(0).sum()}"
    elif str(snapshot) == "2026-06-23":
        assert df["number_of_reviews"].eq(0).sum() == 25, \
                f"Số 'number_of_reviews' = 0 ở mốc {snapshot} khác 25, số lượng thực tế: {df['number_of_reviews'].eq(0).sum()}"
        assert df["price"].isna().sum() == 20, \
                f"Số giá thiếu ở mốc cuối khác 20, số lượng thực tế: {df['price'].isna().sum()}"

    check_vung = df.groupby("analysis_area")["id"].size().between(120, 130).all()
    assert check_vung, f"Điều kiện số lượng listing của mỗi vùng nằm trong 120-130 chưa thỏa mãn"

def download_raw(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_root = config.get("data_root")
    snapshots = config.get("snapshots")
    listing_ids_path = config.get("listings_id_sheet")
    review_ids_path = config.get("reviews_id_sheet")

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
            
            if file_name == "listings.csv.gz":
                df = pd.read_csv(save_file, compression='gzip', low_memory=False,
                                    dtype={"id": str, "host_id": str}, encoding='utf-8')
                df = df[df['id'].isin(df_listing_target['listing_id'])]

                merged = df.merge(df_listing_target[["listing_id", "analysis_area"]],
                                    left_on="id", right_on="listing_id", how="left")
                check_listings(merged, snapshot)
    
            elif file_name == "calendar.csv.gz":
                df = pd.read_csv(save_file, compression='gzip', low_memory=False, dtype={"listing_id": str}, encoding='utf-8')
                df = df[df['listing_id'].isin(df_listing_target['listing_id'])]
                merged = df
                assert len(merged) == 365000, f"Số dòng file listings mốc {snapshot} khác 365000; số dòng thực tế: {len(merged)}"
                
            elif file_name == "reviews.csv.gz":
                df = pd.read_csv(save_file, compression='gzip', low_memory=False, 
                                    dtype={"id": str, "listing_id": str, "reviewer_id": str}, encoding='utf-8')
                df = df[df['id'].isin(df_review_target['review_id'])]

                merged = df.merge(df_review_target[["review_id", "Mẫu LLM", "Nhãn tay"]], 
                                    left_on="id", right_on="review_id", how="left")
                check_reviews(merged, snapshot)

            merged.to_csv(save_file, index=False, compression='gzip', encoding='utf-8')
            print(f"Đã tải và lọc xong {file_name} mốc {snapshot}.")

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