from pathlib import Path
import pandas as pd
import yaml
#from src import qa

try:
    from src.kpi import (
        calculate_review_concentration,
        calculate_dormant_listing_rate_over_time,
        calculate_overpriced_rate,
        calculate_revpar_single_snapshot,
        calculate_professional_host_share_over_time,
        calculate_top_n_host_share_over_time,
    )
except ImportError:
    from kpi import (
        calculate_review_concentration,
        calculate_dormant_listing_rate_over_time,
        calculate_overpriced_rate,
        calculate_revpar_single_snapshot,
        calculate_professional_host_share_over_time,
        calculate_top_n_host_share_over_time,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "configs" / "porto.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
RAW_PATH = PROJECT_ROOT / "data" / "raw"

snapshot_folder = RAW_PATH / "2026-06-23"

calendar_path = snapshot_folder / "calendar.csv.gz"
listings_path = snapshot_folder / "listings.csv.gz"
reviews_path = snapshot_folder / "reviews.csv.gz"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def exec_kpi():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    #calendar = pd.read_csv(calendar_path, encoding='utf-8', dtype={"listing_id": str})
    listings = pd.read_csv(listings_path, encoding='utf-8', dtype={"id": str, "host_id": str, "listing_id": str})
    reviews = pd.read_csv(reviews_path, encoding='utf-8', dtype={"id": str, "listing_id": str, "reviewer_id": str, "review_id": str})

    # 1. KPI Review Concentration
    review_stats, review_concentration_pivot = calculate_review_concentration(
        listings_df=listings,
        reviews_df=reviews,
        output_path=OUTPUT_DIR / "kpi_review_concentration.csv",
    )

    # 2. KPI Dormant Listing Rate (chuỗi thời gian qua các snapshot)
    dormant_rate_pivot = calculate_dormant_listing_rate_over_time(
        config_path=CONFIG_PATH,
        output_path=OUTPUT_DIR / "kpi_dormant_listing_rate.csv",
    )

    # 3. KPI Overpriced Rate (single snapshot)
    overpriced_df = calculate_overpriced_rate(
        listings_df=listings,
        **config["overpriced_rate"],
        output_path=OUTPUT_DIR / "kpi_overpriced_rate.csv",
    )

    # 4. KPI RevPAR (single snapshot)
    revpar_df = calculate_revpar_single_snapshot(
        listings_df=listings,
        output_path=OUTPUT_DIR / "kpi_revpar.csv",
    )

    # 5. KPI Professional Host Share (chuỗi thời gian qua các snapshot)
    pro_host_share_df = calculate_professional_host_share_over_time(
        config_path=CONFIG_PATH,
        **config["professional_host"],
        latest_only=False,
        output_path=OUTPUT_DIR / "kpi_professional_host_share.csv",
    )

    # 6. KPI Top N Host Share (toàn thành phố theo các mốc snapshot)
    top_n_host_share_df = calculate_top_n_host_share_over_time(
        config_path=CONFIG_PATH,
        **config["top_n_host"],
        latest_only=False,
        output_path=OUTPUT_DIR / "kpi_top_n_host_share.csv",
    )
    print(f"Hoàn tất tính toán toàn bộ KPI. File CSV đã lưu tại: {OUTPUT_DIR}")
    return {
        "review_concentration": (review_stats, review_concentration_pivot),
        "dormant_listing_rate": dormant_rate_pivot,
        "overpriced_rate": overpriced_df,
        "revpar": revpar_df,
        "professional_host_share": pro_host_share_df,
        "top_n_host_share": top_n_host_share_df,
    }


if __name__ == "__main__":
    exec_kpi()














