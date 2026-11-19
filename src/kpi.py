
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import json


def calculate_review_concentration(listings_df, reviews_df, output_path = "data/processed/kpi_review_concentration.csv"):
    reviews_df["date"] = pd.to_datetime(reviews_df["date"], format="%Y-%m-%d", errors="coerce")
    reviews_df["year"] = reviews_df["date"].dt.to_period('Y')
    df = reviews_df.merge(listings_df[["id", "analysis_area"]], left_on = "listing_id", right_on = "id", how = "inner")

    area_yearly_reviews = df.groupby(["year", "analysis_area"]).size().reset_index(name="area_reviews")
    city_yearly_reviews = df.groupby("year").size().reset_index(name="city_total_reviews")

    stats = pd.merge(area_yearly_reviews, city_yearly_reviews, on="year")
    stats["concentration_pct"] = (stats["area_reviews"] / stats["city_total_reviews"]) * 100

    concentration_pivot = stats.pivot(
        index="year", 
        columns="analysis_area", 
        values="concentration_pct"
    ).fillna(0)

    concentration_pivot = concentration_pivot.round(2)
    concentration_pivot.to_csv(output_path, index=True)

    return stats, concentration_pivot

def _dormant_rate_by_group(df, group_col):
    kpi_df = df.groupby(group_col, observed=False).agg(
        total_listings=("id", "count"),          
        dormant_listings=("is_dormant", "sum")   
    ).reset_index()

    kpi_df["dormant_rate_%"] = ((kpi_df["dormant_listings"] / kpi_df["total_listings"]) * 100).round(2)
    return kpi_df


def calculate_dormant_listing_rate_single_snapshot(listings_df, snapshot=None):
    df = listings_df.copy()

    df["last_review"] = pd.to_datetime(df["last_review"])
    df["last_scraped"] = pd.to_datetime(df["last_scraped"])
    days_since_review = (df["last_scraped"] - df["last_review"]).dt.days

    dormant_condition = (
        (df["number_of_reviews_ltm"] == 0)
        & (df["last_review"].isna() | (days_since_review > 540))
    )

    df["is_dormant"] = np.where(dormant_condition, 1, 0)

    # Lát cắt 1: Theo analysis_area
    area_kpi = _dormant_rate_by_group(df, "analysis_area")
    area_kpi.insert(1, "room_type", "All")

    # Lát cắt 2: Theo room_type trên toàn thành phố (2 dòng mỗi snapshot)
    room_kpi = _dormant_rate_by_group(df, "room_type")
    room_kpi.insert(0, "analysis_area", "All")

    kpi_df = pd.concat([area_kpi, room_kpi], ignore_index=True)
    kpi_df.insert(0, "snapshot", str(snapshot))
    return kpi_df


def calculate_dormant_listing_rate_over_time(
    config_path,
    output_path="data/processed/kpi_dormant_listing_rate.csv",
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    snapshots = config.get("snapshots")

    all_kpi_results = []
    for date in snapshots:
        base_dir = Path(__file__).resolve().parent.parent
        folder_path = base_dir / "data" / "raw" / str(date)

        listings_df = pd.read_csv(folder_path / "listings.csv.gz")
        kpi_1_thang = calculate_dormant_listing_rate_single_snapshot(listings_df, snapshot=date)
        all_kpi_results.append(kpi_1_thang)

    final_time_series_kpi = pd.concat(all_kpi_results, ignore_index=True)
    pivot_kpi = (
        final_time_series_kpi
        .pivot(index=["analysis_area", "room_type"], columns="snapshot", values="dormant_rate_%")
        .round(2)
    )
    final_time_series_kpi.to_csv(output_path, index=False)
    return pivot_kpi


def _overpriced_rate_by_group(df, group_col, premium_ratio=1.25, low_value_threshold=4.6):
    sub_df = df.copy()
    sub_df["area_median_price"] = sub_df.groupby(group_col, observed=False)["price"].transform("median")
    sub_df["price_index"] = sub_df["price"] / sub_df["area_median_price"]

    sub_df["is_premium"] = sub_df["price_index"] >= premium_ratio
    sub_df["is_overpriced"] = sub_df["is_premium"] & (sub_df["review_scores_value"] < low_value_threshold)

    kpi_df = sub_df.groupby(group_col, observed=False).agg(
        total_listings=("id", "count"),
        area_median_price=("area_median_price", "first"),
        premium_listings=("is_premium", "sum"),          
        overpriced_listings=("is_overpriced", "sum"),   
    ).reset_index()

    kpi_df["overpriced_rate_%"] = (
        kpi_df["overpriced_listings"] / kpi_df["premium_listings"] * 100
    ).round(2)
    return kpi_df


def calculate_overpriced_rate(
    listings_df,
    premium_ratio=1.25,
    low_value_threshold=4.6,
    output_path="data/processed/kpi_overpriced_rate.csv",
):
    df = listings_df.copy()

    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )

    df = df[df["price"].notna() & df["review_scores_value"].notna()].copy()

    # Lát cắt 1: Theo analysis_area
    area_kpi = _overpriced_rate_by_group(
        df, "analysis_area", premium_ratio=premium_ratio, low_value_threshold=low_value_threshold
    )
    area_kpi.insert(1, "room_type", "All")

    # Lát cắt 2: Theo room_type trên toàn thành phố
    room_kpi = _overpriced_rate_by_group(
        df, "room_type", premium_ratio=premium_ratio, low_value_threshold=low_value_threshold
    )
    room_kpi.insert(0, "analysis_area", "All")

    kpi_df = pd.concat([area_kpi, room_kpi], ignore_index=True)

    kpi_df.to_csv(output_path, index=False)
    return kpi_df

def _revpar_by_group(df, group_col):
    kpi_df = df.groupby(group_col, observed=False).agg(
        total_listings=("id", "count"),                                   # N_{k,t}
        total_revenue=("estimated_revenue_l365d", "sum"),                 # Σ doanh thu 365 ngày
        occupied_nights=("estimated_occupancy_l365d", "sum"),             # Σ đêm có khách (ước tính)
    ).reset_index()

    # RevPAR = Σ doanh thu / (N × 365)
    kpi_df["revpar"] = (kpi_df["total_revenue"] / (kpi_df["total_listings"] * 365)).round(2)

    # Phân rã: RevPAR = ADR × tỷ lệ lấp đầy
    kpi_df["adr"] = (kpi_df["total_revenue"] / kpi_df["occupied_nights"].replace(0, np.nan)).round(2)
    kpi_df["occupancy_rate_%"] = (
        kpi_df["occupied_nights"] / (kpi_df["total_listings"] * 365) * 100
    ).round(2)
    return kpi_df


def calculate_revpar_single_snapshot(
    listings_df,
    output_path="data/processed/kpi_revpar.csv",
):
    df = listings_df.copy()
    df = df[df["estimated_revenue_l365d"].notna()].copy()

    # Lát cắt 1: Theo analysis_area
    area_kpi = _revpar_by_group(df, "analysis_area")
    area_kpi.insert(1, "room_type", "All")

    # Lát cắt 2: Theo room_type trên toàn thành phố
    room_kpi = _revpar_by_group(df, "room_type")
    room_kpi.insert(0, "analysis_area", "All")

    kpi_df = pd.concat([area_kpi, room_kpi], ignore_index=True)

    kpi_df.to_csv(output_path, index=False)
    return kpi_df


def calculate_professional_host_share_single_snapshot(listings_df, snapshot=None, pro_threshold=10):

    df = listings_df[listings_df["calculated_host_listings_count"].notna()].copy()

    df["host_tier"] = pd.cut(
        df["calculated_host_listings_count"],
        bins=[0, 1, pro_threshold - 1, np.inf],
        labels=["individual", "small_multi", "professional"],
    )

    kpi_df = (
        df.groupby(["analysis_area", "host_tier"], observed=False)
        .size()
        .unstack("host_tier")
    )
    kpi_df.columns = [f"{tier}_listings" for tier in kpi_df.columns]
    kpi_df["total_listings"] = kpi_df.sum(axis=1)

    for tier in ["individual", "small_multi", "professional"]:
        kpi_df[f"{tier}_share_%"] = (
            kpi_df[f"{tier}_listings"] / kpi_df["total_listings"] * 100
        ).round(2)

    kpi_df = kpi_df.reset_index()
    kpi_df.insert(0, "snapshot", str(snapshot))
    return kpi_df


def calculate_professional_host_share_over_time(
    config_path,
    pro_threshold=10,
    latest_only=False,
    output_path="data/processed/kpi_professional_host_share.csv",
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    snapshots = sorted(config.get("snapshots"), key=str)
    if latest_only:
        snapshots = snapshots[-1:]

    cols = ["id", "analysis_area", "calculated_host_listings_count"]
    all_results = []
    for date in snapshots:
        folder_path = Path(__file__).resolve().parent.parent / "data" / "raw" / str(date)
        listings_df = pd.read_csv(folder_path / "listings.csv.gz", usecols=cols)
        all_results.append(calculate_professional_host_share_single_snapshot(
            listings_df, snapshot=date, pro_threshold=pro_threshold))

    final_kpi = pd.concat(all_results, ignore_index=True)
    final_kpi.to_csv(output_path, index=False)
    return final_kpi

def calculate_top_n_host_share_single_snapshot(
    listings_df,
    snapshot=None,
    top_n=10,
    output_path=None,
):
    df = listings_df[listings_df["host_id"].notna()].copy()

    host_counts = df.groupby("host_id")["id"].count()
    total_hosts = int(host_counts.shape[0])
    total_listings = int(host_counts.sum())
    top_n_listings = int(host_counts.nlargest(top_n).sum())
    top_n_share = round((top_n_listings / total_listings * 100), 2) if total_listings > 0 else 0.0

    kpi_df = pd.DataFrame([{
        "snapshot": str(snapshot),
        "top_n": top_n,
        "total_hosts": total_hosts,
        "total_listings": total_listings,
        "top_n_listings": top_n_listings,
        "top_n_share_%": top_n_share,
    }])

    if output_path is not None:
        kpi_df.to_csv(output_path, index=False)
    return kpi_df


def calculate_top_n_host_share_over_time(
    config_path,
    top_n=10,
    latest_only=False,
    output_path="data/processed/kpi_top_n_host_share.csv",
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    snapshots = sorted(config.get("snapshots"), key=str)
    if latest_only:
        snapshots = snapshots[-1:]

    cols = ["id", "host_id"]
    all_results = []
    for date in snapshots:
        folder_path = Path(__file__).resolve().parent.parent / "data" / "raw" / str(date)
        listings_df = pd.read_csv(folder_path / "listings.csv.gz", usecols=cols)
        all_results.append(
            calculate_top_n_host_share_single_snapshot(
                listings_df, snapshot=date, top_n=top_n
            )
        )

    final_kpi = pd.concat(all_results, ignore_index=True)
    final_kpi.to_csv(output_path, index=False)
    return final_kpi

