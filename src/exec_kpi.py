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
        calculate_unavailability_rate,
        label_distribution_area,
        aspect_sentiment_correlation,
    )
    from src.qa import (
        concat_calendar
    )
except ImportError:
    from kpi import (
        calculate_review_concentration,
        calculate_dormant_listing_rate_over_time,
        calculate_overpriced_rate,
        calculate_revpar_single_snapshot,
        calculate_professional_host_share_over_time,
        calculate_top_n_host_share_over_time,
        calculate_unavailability_rate,
        label_distribution_area,
        aspect_sentiment_correlation,
    )
    from qa import (
        concat_calendar
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "configs" / "porto.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
kpi_cfg = config["kpi"]

SNAPSHOT = kpi_cfg["snapshot"]
snapshot_folder = PROCESSED_DIR / str(SNAPSHOT)

calendar_path = snapshot_folder / "processed_calendar.csv"
listings_path = snapshot_folder / "processed_listings.csv"
reviews_path = snapshot_folder / "processed_reviews.csv"

llm_output_path = PROCESSED_DIR / "llm_output.csv"
llm_cache_path = PROCESSED_DIR / "llm_cache.jsonl"

llm_path = llm_output_path if llm_output_path.exists() else llm_cache_path

def exec_kpi():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calendar = pd.read_csv(
        calendar_path, encoding='utf-8',
        dtype={"listing_id": str},
        parse_dates=["clean_date"],
    )
    listings = pd.read_csv(
        listings_path, encoding='utf-8',
        dtype={"id": str, "host_id": str},
        parse_dates=["last_review_processed", "last_scraped_processed"],
    )
    reviews = pd.read_csv(
        reviews_path, encoding='utf-8',
        dtype={"id": str, "listing_id": str},
        usecols=["id", "listing_id", "date"],
        parse_dates=["date"],
    )
    if llm_path.suffix == ".jsonl":
        llm_df = pd.read_json(llm_path, lines=True)
    else:
        llm_df = pd.read_csv(llm_path)

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
        **kpi_cfg["overpriced_rate"],
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
        **kpi_cfg["professional_host"],
        latest_only=False,
        output_path=OUTPUT_DIR / "kpi_professional_host_share.csv",
    )

    # 6. KPI Top N Host Share (toàn thành phố theo các mốc snapshot)
    top_n_host_share_df = calculate_top_n_host_share_over_time(
        config_path=CONFIG_PATH,
        **kpi_cfg["top_n_host"],
        latest_only=False,
        output_path=OUTPUT_DIR / "kpi_top_n_host_share.csv",
    )

    full_calendar = concat_calendar()
    #7. KPI tỉ lệ phòng không khả dụng 
    unavailability_rate_df = calculate_unavailability_rate(
    full_calendar,
    listings,
    output_path=OUTPUT_DIR / "kpi_unavailability_rate.csv",
    )

    #8. KPI tỉ lệ các nhãn llm 
    label_distribution_df = label_distribution_area(
        review_df= reviews,
        listings_df=listings,
        llm_output=llm_df,
        output_path= OUTPUT_DIR / "kpi_label_distribution_area.csv"
    )

    #9 KPI tỉ lệ các nhãn trong aspects là review non-positive
    aspect_sentiment_correlation_df = aspect_sentiment_correlation(
        llm_output = llm_df,
        output_path= OUTPUT_DIR / "kpi_aspect_sentiment_correlation.csv"
    )

    print(f"Hoàn tất tính toán toàn bộ KPI. File CSV đã lưu tại: {OUTPUT_DIR}")
    return {
        "review_concentration": (review_stats, review_concentration_pivot),
        "dormant_listing_rate": dormant_rate_pivot,
        "overpriced_rate": overpriced_df,
        "revpar": revpar_df,
        "professional_host_share": pro_host_share_df,
        "top_n_host_share": top_n_host_share_df,
        "unavailability_rate" : unavailability_rate_df,
        "label_distribution" : label_distribution_df,
        "aspect_sentiment_correlation" : aspect_sentiment_correlation_df,
    }


if __name__ == "__main__":
    exec_kpi()














