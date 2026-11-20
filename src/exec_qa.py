from pathlib import Path

import pandas as pd
from src.qa import (
    kiem_tra_cau_truc_all,
    clean_listings,
    clean_reviews,
    clean_calendar,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"

def read_listings(path):
    return pd.read_csv(path, encoding="utf-8", dtype={"id": str, "host_id": str, "listing_id": str})

def read_reviews(path):
   return pd.read_csv(path, encoding="utf-8", dtype={"id": str, "listing_id": str, "comments": str}) 

def read_calendar(path):
    return pd.read_csv(path, encoding="utf-8", dtype={"listing_id": str}) 


def exec_qa(config):
    snapshots = config.get("snapshots")
    for snapshot in snapshots:
        snapshot_folder_raw = RAW_DIR / str(snapshot)
        snapshot_folder_raw.mkdir(parents=True, exist_ok=True)
        
        snapshot_folder_processed = PROCESSED_DIR / str(snapshot)
        snapshot_folder_processed.mkdir(parents=True, exist_ok=True)

        snapshot_folder_reports = REPORT_DIR / str(snapshot)
        snapshot_folder_reports.mkdir(parents=True, exist_ok=True)

        listings = read_listings(snapshot_folder_raw / "listings.csv.gz")
        reviews = read_reviews(snapshot_folder_raw / "reviews.csv.gz")
        calendar = read_calendar(snapshot_folder_raw / "calendar.csv.gz")

        qa_report_listings_path = snapshot_folder_reports / "qa_report_listings.csv"
        qa_report_reviews_path = snapshot_folder_reports / "qa_report_reviews.csv"
        qa_report_calendar_path = snapshot_folder_reports / "qa_report_calendar.csv"

        qa_processed_listings = snapshot_folder_processed / "processed_listings.csv"
        qa_processed_reviews = snapshot_folder_processed / "processed_reviews.csv"
        qa_processed_calendar = snapshot_folder_processed / "processed_calendar.csv"

        kiem_tra_cau_truc_all(listings, calendar, reviews)
        clean_listings(listings, qa_report_listings_path, qa_processed_listings)
        clean_reviews(reviews, listings, qa_report_reviews_path, qa_processed_reviews)
        clean_calendar(calendar, listings, qa_report_calendar_path, qa_processed_calendar)