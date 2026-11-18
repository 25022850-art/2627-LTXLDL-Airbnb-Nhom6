
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

    print(f"Đã lưu KPI Tỉ lệ tập trung đánh giá tại: {output_path}")
    return stats, concentration_pivot

def calculate_dormant_listing_rate_single_snapshot(listings_df, snapshot = None):
    df = listings_df.copy()

    df["last_review"] = pd.to_datetime(df["last_review"])
    df["last_scraped"] = pd.to_datetime(df["last_scraped"])
    days_since_review = (df["last_scraped"] - df["last_review"]).dt.days
    
    dormant_condition = (
    (df["number_of_reviews_ltm"] == 0)
    & (df["last_review"].isna() | (days_since_review > 540))
    )
    
    df['is_dormant'] = np.where(dormant_condition, 1, 0)
    
    kpi_df = df.groupby('analysis_area').agg(
        total_listings=('id', 'count'),          # Mẫu số N_{k,t}: Tổng số listing tại vùng k
        dormant_listings=('is_dormant', 'sum')   # Tử số: Số listing thỏa mãn điều kiện ngủ đông
    ).reset_index()

    kpi_df['dormant_rate_%'] = (kpi_df['dormant_listings'] / kpi_df['total_listings']) * 100
    kpi_df['dormant_rate_%'] = kpi_df['dormant_rate_%'].round(2)

    kpi_df.insert(0, "snapshot", str(snapshot))                     
    
    return kpi_df
def calculate_dormant_listing_rate_over_time(config_path, output_path = "data/processed/kpi_dormant_listing_rate.csv"):
    with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    snapshots = config.get("snapshots")

    all_kpi_results = []
    for date in snapshots:
        base_dir = Path.cwd().parent 
        folder_path = base_dir / "data" / "raw" / str(date) 
   
        listings_df = pd.read_csv(folder_path / "listings.csv.gz")
        kpi_1_thang = calculate_dormant_listing_rate_single_snapshot(listings_df, snapshot=date)
        all_kpi_results.append(kpi_1_thang)

    final_time_series_kpi = pd.concat(all_kpi_results, ignore_index=True)
    pivot_kpi = (
        final_time_series_kpi
        .pivot(index="analysis_area", columns="snapshot", values="dormant_rate_%")
        .round(2)
    )
    final_time_series_kpi.to_csv(output_path, index=False)
    return pivot_kpi


# def calculate_value_score_by_price_tier_single_snapshot(
#     listings_df,
#     output_path="data/processed/kpi_value_score_price_tier.csv",
# ):
#     df = listings_df.copy()
#     df["price"] = pd.to_numeric(
#     df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
#     errors="coerce",
# )

#     df = df[df["price"].notna() & df["review_scores_value"].notna()].copy()

#     tier_labels = ["Q1_Binh_dan", "Q2_Trung_cap", "Q3_Cao_cap", "Q4_Hang_sang"]
#     df["price_tier"] = pd.qcut(df["price"], q=4, labels=tier_labels)
#     df["weighted_score"] = df["review_scores_value"] * df["number_of_reviews"]
#     df["is_low_value"] = (df["review_scores_value"] < 4.5).astype(int)


#     kpi_df = df.groupby("price_tier", observed=True).agg(
#         total_listings=("id", "count"),
#         price_min=("price", "min"),
#         price_max=("price", "max"),
#         total_reviews=("number_of_reviews", "sum"),
#         sum_weighted_score=("weighted_score", "sum"),
#         low_value_share_pct=("is_low_value", lambda s: s.mean() * 100),
#     ).reset_index()

#     kpi_df["weighted_value_score"] = (
#         kpi_df["sum_weighted_score"] / kpi_df["total_reviews"]
#     ).round(3)
#     kpi_df = kpi_df.drop(columns="sum_weighted_score")

#     kpi_df.to_csv(output_path, index=False)
#     print(f"Đã lưu KPI Value-for-Money Score tại: {output_path}")
#     return kpi_df


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

    df["area_median_price"] = df.groupby("analysis_area")["price"].transform("median")
    df["price_index"] = df["price"] / df["area_median_price"]

    df["is_premium"] = df["price_index"] >= premium_ratio                      # giá cao
    df["is_overpriced"] = df["is_premium"] & (df["review_scores_value"] < low_value_threshold)  # giá cao + điểm value thấp

    kpi_df = df.groupby("analysis_area").agg(
        total_listings=("id", "count"),
        area_median_price=("area_median_price", "first"),
        premium_listings=("is_premium", "sum"),          # Mẫu số
        overpriced_listings=("is_overpriced", "sum"),    # Tử số
    ).reset_index()

    kpi_df["overpriced_rate_%"] = (
        kpi_df["overpriced_listings"] / kpi_df["premium_listings"] * 100
    ).round(2)

    kpi_df.to_csv(output_path, index=False)
    print(f"Đã lưu KPI Overpriced Rate tại: {output_path}")
    return kpi_df

def calculate_revpar_single_snapshot(
    listings_df,
    output_path="data/processed/kpi_revpar.csv"
):
    df = listings_df.copy()

    df = df[df["estimated_revenue_l365d"].notna()].copy()

    kpi_df = df.groupby("analysis_area").agg(
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
        folder_path = Path.cwd().parent / "data" / "raw" / str(date)
        listings_df = pd.read_csv(folder_path / "listings.csv.gz", usecols=cols)
        all_results.append(calculate_professional_host_share_single_snapshot(
            listings_df, snapshot=date, pro_threshold=pro_threshold))

    final_kpi = pd.concat(all_results, ignore_index=True)
    final_kpi.to_csv(output_path, index=False)
    return final_kpi