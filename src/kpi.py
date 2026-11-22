from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import json


try:
    from src.qa import exclude_nan_price, exclude_missing_review_scores, exclude_missing_days_in_month, exclude_nan_estimated_rev
except ImportError:
    from qa import exclude_nan_price, exclude_missing_review_scores, exclude_missing_days_in_month, exclude_nan_estimated_rev


def _read_processed_listings(snapshot, usecols=None):
    path = (
        Path(__file__).resolve().parent.parent
        / "data" / "processed" / str(snapshot) / "processed_listings.csv"
    )
    date_cols = ["last_review_processed", "last_scraped_processed"]
    if usecols is not None:
        date_cols = [c for c in date_cols if c in usecols]
    return pd.read_csv(
        path,
        dtype={"id": str, "host_id": str},
        usecols=usecols,
        parse_dates=date_cols,
    )


def calculate_review_concentration(listings_df, reviews_df, output_path = "data/processed/kpi_review_concentration.csv"):
    rv = reviews_df.copy()

    rv["date"] = pd.to_datetime(rv["date"], errors="coerce")
    rv["year"] = rv["date"].dt.to_period('Y')
    df = rv.merge(listings_df[["id", "analysis_area_processed"]], left_on = "listing_id", right_on = "id", how = "inner")

    area_yearly_reviews = df.groupby(["year", "analysis_area_processed"]).size().reset_index(name="area_reviews")
    city_yearly_reviews = df.groupby("year").size().reset_index(name="city_total_reviews")

    stats = pd.merge(area_yearly_reviews, city_yearly_reviews, on="year")
    stats["concentration_pct"] = (stats["area_reviews"] / stats["city_total_reviews"]) * 100

    concentration_pivot = stats.pivot(
        index="year", 
        columns="analysis_area_processed", 
        values="concentration_pct"
    ).fillna(0)

    concentration_pivot = concentration_pivot.round(2)

    stats_out = stats.rename(columns={"analysis_area_processed": "analysis_area"}).copy()
    stats_out["year"] = stats_out["year"].dt.year        # Period -> số nguyên
    stats_out["concentration_pct"] = stats_out["concentration_pct"].round(2)
    stats_out = stats_out.sort_values(["year", "analysis_area"]).reset_index(drop=True)
    stats_out.to_csv(output_path, index=False)

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

    df["last_review_processed"] = pd.to_datetime(df["last_review_processed"], errors="coerce")
    df["last_scraped_processed"] = pd.to_datetime(df["last_scraped_processed"], errors="coerce")
    days_since_review = (df["last_scraped_processed"] - df["last_review_processed"]).dt.days

    dormant_condition = (
        (df["number_of_reviews_ltm_processed"] == 0)
        & (df["last_review_processed"].isna() | (days_since_review > 540))
    )

    df["is_dormant"] = dormant_condition

    # Lát cắt 1: Theo analysis_area
    area_kpi = _dormant_rate_by_group(df, "analysis_area_processed")
    area_kpi = area_kpi.rename(columns={"analysis_area_processed": "analysis_area"})
    area_kpi.insert(1, "room_type", "All")  

    # Lát cắt 2: Theo room_type trên toàn thành phố (2 dòng mỗi snapshot)
    room_kpi = _dormant_rate_by_group(df, "room_type_processed")
    room_kpi = room_kpi.rename(columns={"room_type_processed": "room_type"})
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
        listings_df = _read_processed_listings(date)
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
    sub_df["group_median_price"] = sub_df.groupby(group_col, observed=False)["price_processed"].transform("median")
    sub_df["price_index"] = sub_df["price_processed"] / sub_df["group_median_price"]

    sub_df["is_premium"] = sub_df["price_index"] >= premium_ratio
    sub_df["is_overpriced"] = sub_df["is_premium"] & (sub_df["review_scores_value_processed"] < low_value_threshold)

    kpi_df = sub_df.groupby(group_col, observed=False).agg(
        total_listings=("id", "count"),
        group_median_price=("group_median_price", "first"),
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

    df = exclude_nan_price(df)
    df = exclude_missing_review_scores(df)

    # Lát cắt 1: Theo analysis_area
    area_kpi = _overpriced_rate_by_group(
        df, "analysis_area_processed", premium_ratio=premium_ratio, low_value_threshold=low_value_threshold
    )
    area_kpi = area_kpi.rename(columns={"analysis_area_processed": "analysis_area"})
    area_kpi.insert(1, "room_type", "All")

    # Lát cắt 2: Theo room_type trên toàn thành phố
    room_kpi = _overpriced_rate_by_group(
        df, "room_type_processed", premium_ratio=premium_ratio, low_value_threshold=low_value_threshold
    )
    room_kpi = room_kpi.rename(columns={"room_type_processed": "room_type"})
    room_kpi.insert(0, "analysis_area", "All")

    kpi_df = pd.concat([area_kpi, room_kpi], ignore_index=True)

    kpi_df.to_csv(output_path, index=False)
    return kpi_df

def _revpar_by_group(df, group_col):
    kpi_df = df.groupby(group_col, observed=False).agg(
        total_listings=("id", "count"),                                   # N_{k,t}
        total_revenue=("estimated_revenue_l365d_processed", "sum"),                 # Σ doanh thu 365 ngày
        occupied_nights=("estimated_occupancy_l365d_processed", "sum"),             # Σ đêm có khách (ước tính)
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
    df = exclude_nan_estimated_rev(df)
    
    # Lát cắt 1: Theo analysis_area
    area_kpi = _revpar_by_group(df, "analysis_area_processed")
    area_kpi = area_kpi.rename(columns={"analysis_area_processed": "analysis_area"})
    area_kpi.insert(1, "room_type", "All")

    # Lát cắt 2: Theo room_type trên toàn thành phố
    room_kpi = _revpar_by_group(df, "room_type_processed")
    room_kpi = room_kpi.rename(columns={"room_type_processed": "room_type"})
    room_kpi.insert(0, "analysis_area", "All")

    kpi_df = pd.concat([area_kpi, room_kpi], ignore_index=True)

    kpi_df.to_csv(output_path, index=False)
    return kpi_df


def calculate_host_share_single_snapshot(listings_df, snapshot=None, pro_threshold=10):

    df = listings_df.copy()

    df["host_tier"] = pd.cut(
        df["calculated_host_listings_count_processed"],
        bins=[0, 1, pro_threshold - 1, np.inf],
        labels=["individual", "small_multi", "professional"],
    )

    kpi_df = (
        df.groupby(["analysis_area_processed", "host_tier"], observed=False)
        .size()
        .reset_index(name="listings")
    )
    kpi_df["total_listings"] = (
        kpi_df.groupby("analysis_area_processed", observed=False)["listings"]
        .transform("sum")
    )

    kpi_df["share_%"] = (
        kpi_df["listings"] / kpi_df["total_listings"].replace(0, np.nan) * 100
    ).round(2)

    kpi_df = kpi_df.rename(columns={"analysis_area_processed": "analysis_area"})
    kpi_df.insert(0, "snapshot", str(snapshot))
    return kpi_df


def calculate_host_share_over_time(
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

    cols = ["id", "analysis_area_processed", "calculated_host_listings_count_processed"]
    all_results = []
    for date in snapshots:
        listings_df = _read_processed_listings(date, usecols=cols)
        all_results.append(calculate_host_share_single_snapshot(
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
    df = listings_df.copy()

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
        listings_df = _read_processed_listings(date, usecols=cols)
        all_results.append(
            calculate_top_n_host_share_single_snapshot(
                listings_df, snapshot=date, top_n=top_n
            )
        )

    final_kpi = pd.concat(all_results, ignore_index=True)
    final_kpi.to_csv(output_path, index=False)
    return final_kpi

def calculate_unavailability_rate(
    calendar_df,
    listings_df,
    output_path="data/processed/kpi_unavailability_rate.csv",
):
    df = calendar_df.merge(
        listings_df[["id", "analysis_area_processed"]],
        left_on="listing_id", right_on="id", how="inner",
    )
    
    df["clean_date"] = pd.to_datetime(df["clean_date"], errors="coerce")

    df["year_month"] = df["clean_date"].dt.to_period("M")
    df["is_unavailable"] = ~df["clean_available"]

    # giữ các tháng đủ ngày
    df = exclude_missing_days_in_month(df)

    kpi_df = df.groupby(["year_month", "analysis_area_processed"]).agg(
        n_listings=("listing_id", "nunique"),
        total_days=("listing_id", "count"),
        unavailable_days=("is_unavailable", "sum"),
    ).reset_index()

    kpi_df = kpi_df.rename(columns={"analysis_area_processed": "analysis_area"})

    kpi_df["unavailability_rate_%"] = (
        kpi_df["unavailable_days"] / kpi_df["total_days"] * 100
    ).round(2)

    kpi_df = kpi_df.sort_values(["year_month", "analysis_area"]).reset_index(drop=True)
    kpi_df.to_csv(output_path, index=False)
    return kpi_df


def label_distribution_area(review_df, listings_df, llm_output, output_path="data/processed/kpi_label_distribution_area.csv"):
    review_df = review_df.copy()
    listings_df = listings_df.copy()
    llm_output = llm_output.copy()

    review_df = review_df.rename(columns={"id": "review_id"})

    review_df["review_id"] = review_df["review_id"].astype(str)
    review_df["listing_id"] = review_df["listing_id"].astype(str)
    listings_df["id"] = listings_df["id"].astype(str)
    llm_output["id"] = llm_output["id"].astype(str)

    # Merge 1: Reviews + Listings
    review_area_df = review_df[
        ["review_id", "listing_id"]
    ].merge(
        listings_df[["id", "analysis_area_processed"]],
        left_on="listing_id",
        right_on="id",
        how="inner",
        validate="many_to_one"
    )

    # Merge 2: LLM Output + Review Areas
    df = llm_output.merge(
        review_area_df[
            ["review_id", "analysis_area_processed"]
        ],
        left_on="id",
        right_on="review_id",
        how="inner",
        validate="one_to_one"
    )
    aspects = ['general_sentiment', 'accessibility', 'noise']

    results = []

    for area, area_df in df.groupby('analysis_area_processed'):
        area_total = len(area_df)

        for aspect in aspects:
            counts = area_df[aspect].value_counts()

            for label, count in counts.items():
                if label == 'Not mentioned':
                    continue

                results.append({
                    'analysis_area_processed': area,
                    'aspect': aspect,
                    'label': label,
                    'count': count,
                    'percentage': round(count / area_total * 100, 2)
                })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    return result_df

def aspect_sentiment_correlation(llm_output, output_path = "data/processed/kpi_aspect_sentiment_correlation.csv" ):
    df = llm_output.copy()
    aspects = ['accessibility', 'noise']
    sentiment_col = 'general_sentiment'

    non_positive_labels = ['Mixed', 'Negative']

    # Baseline: non-positive rate across all reviews
    baseline = df[sentiment_col].isin(non_positive_labels).mean()

    results = []

    for aspect in aspects:
        for label, group in df.groupby(aspect):
            if label == 'Not mentioned':
                continue

            total = len(group)

            positive = (group[sentiment_col] == 'Positive').sum()
            mixed = (group[sentiment_col] == 'Mixed').sum()
            negative = (group[sentiment_col] == 'Negative').sum()

            non_positive_rate = (mixed + negative) / total
            risk_lift = (
                non_positive_rate / baseline
                if baseline > 0 else float('nan')
            )

            results.append({
                'aspect': aspect,
                'label': label,
                'total_reviews': total,
                'positive_rate': round(positive / total * 100, 2),
                'mixed_rate': round(mixed / total * 100, 2),
                'negative_rate': round(negative / total * 100, 2),
                'non_positive_rate': round(non_positive_rate * 100, 2),
                'risk_lift': round(risk_lift, 2)
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    return result_df