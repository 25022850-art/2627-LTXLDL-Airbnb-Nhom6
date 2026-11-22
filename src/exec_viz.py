from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from src.viz import (
    plot_overpriced_analysis,
    plot_unavailability_rate,
    plot_sentiment_correlation,
    plot_dormant_listing_rate,
    plot_label_distribution,
    plot_professional_host_share,
    plot_review_concentration,
    plot_revpar,
    plot_top_n_host_share
)
from src.viz_geojson import (
    get_latest_geojson_path,
    load_analysis_area_gdf,
    plot_revpar_geojson,
    plot_label_distribution_area_geojson,
)

config_path = Path(__file__).parent.parent / "configs" / "porto.yaml"
raw_folder = Path(__file__).parent.parent / "data" / "raw"
csv_folder = Path(__file__).parent.parent / "data" / "processed"
output_folder = Path(__file__).parent.parent / "figures"

def csv_to_df(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def exec_viz():
    # Load config
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Load KPI data
    #kpi_csv_path = csv_folder / "kpi_results.csv"
    #kpi_df = csv_to_df(kpi_csv_path)

    # Load Overpriced Analysis data
    # print("Tạo biểu đồ cột cho overprice")
    overpriced_csv_path = csv_folder / "kpi_overpriced_rate.csv"
    overpriced_df = csv_to_df(overpriced_csv_path)
    overpriced_output_path = output_folder / "overpriced_analysis.png"
    # Generate visualizations
    plot_overpriced_analysis(overpriced_df, overpriced_output_path)

    unavailable_csv = csv_folder / "kpi_unavailability_rate.csv"
    unavailable_df = csv_to_df(unavailable_csv)
    unavailable_output_path = output_folder / "unavailability_rate.png"

    plot_unavailability_rate(unavailable_df,unavailable_output_path)

    # data/processed/kpi_aspect_sentiment_correlation.csv
    sentiment_csv = csv_folder / "kpi_aspect_sentiment_correlation.csv"
    sentiment_df = csv_to_df(sentiment_csv)
    sentiment_output_path = output_folder / "aspect_sentiment_correlation.png"

    plot_sentiment_correlation(sentiment_df,sentiment_output_path)

    dormant_csv = csv_folder / "kpi_dormant_listing_rate.csv"
    dormant_df = csv_to_df(dormant_csv)
    dormant_output_path = output_folder / "dormant_listing_rate.png"
    plot_dormant_listing_rate(dormant_df, dormant_output_path)

    label_csv = csv_folder / "kpi_label_distribution_area.csv"
    label_df = csv_to_df(label_csv)
    label_output_path = output_folder / "label_distribution_area.png"
    plot_label_distribution(label_df, label_output_path)

    prof_host_csv = csv_folder / "kpi_host_share.csv"
    prof_host_df = csv_to_df(prof_host_csv)
    prof_host_output_path = output_folder / "host_share.png"
    plot_professional_host_share(prof_host_df, prof_host_output_path)

    review_conc_csv = csv_folder / "kpi_review_concentration.csv"
    review_conc_df = csv_to_df(review_conc_csv)
    review_conc_output_path = output_folder / "review_concentration.png"
    plot_review_concentration(review_conc_df, review_conc_output_path)

    revpar_csv = csv_folder / "kpi_revpar.csv"
    revpar_df = csv_to_df(revpar_csv)
    revpar_output_path = output_folder / "revpar.png"
    plot_revpar(revpar_df, revpar_output_path)

    top_n_csv = csv_folder / "kpi_top_n_host_share.csv"
    top_n_df = csv_to_df(top_n_csv)
    top_n_output_path = output_folder / "top_n_host_share.png"
    plot_top_n_host_share(top_n_df, top_n_output_path)

    # --- Vẽ bản đồ GeoJSON (mốc mới nhất) cho RevPAR và Phân bố nhãn theo Area ---
    latest_geojson_path = get_latest_geojson_path(raw_folder)
    snapshot_label = latest_geojson_path.parent.name
    print(f"Đọc GeoJSON mốc mới nhất: {latest_geojson_path}")
    gdf_areas, gdf_raw = load_analysis_area_gdf(latest_geojson_path)

    revpar_geojson_output_path = output_folder / "revpar_geojson.png"
    plot_revpar_geojson(
        revpar_df,
        gdf_areas,
        revpar_geojson_output_path,
        gdf_raw=gdf_raw,
        snapshot_label=snapshot_label,
    )

    label_geojson_output_path = output_folder / "label_distribution_area_geojson.png"
    plot_label_distribution_area_geojson(
        label_df,
        gdf_areas,
        label_geojson_output_path,
        gdf_raw=gdf_raw,
        snapshot_label=snapshot_label,
    )

