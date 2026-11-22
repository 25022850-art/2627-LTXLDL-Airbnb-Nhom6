from __future__ import annotations

from pathlib import Path
from typing import Mapping

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from shapely.geometry import box

try:
    import contextily as ctx
except ImportError:
    ctx = None

try:
    from shapely import make_valid as _make_valid
except ImportError:  # Shapely < 2.0 fallback
    def _make_valid(geom):
        return geom.buffer(0)


PROJECT_ROOT = Path(__file__).parent.parent
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
CSV_FOLDER = PROJECT_ROOT / "data" / "processed"
OUTPUT_FOLDER = PROJECT_ROOT / "figures"

# Bounding boxes (lon_min, lat_min, lon_max, lat_max) ánh xạ từ bản đồ thật Porto sang analysis_area
PORTO_AREA_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "A01": (-8.79409, 40.78018104, -8.61030595, 41.143183),
    "A02": (-8.79409, 41.143183, -8.61030595, 41.14906785),
    "A03": (-8.61030595, 40.78018104, -8.60340415, 41.14906785),
    "A04": (-8.60340415, 40.78018104, -8.097183878, 41.14906785),
    "A05": (-8.79409, 41.14906785, -8.61262, 41.16046615),
    "A06": (-8.79409, 41.16046615, -8.61262, 41.44549),
    "A07": (-8.61262, 41.14906785, -8.097183878, 41.15492477),
    "A08": (-8.61262, 41.15492477, -8.097183878, 41.44549),
}

# Các vùng ngoại vi rộng (gắn nhãn trên bản đồ toàn cảnh) và các vùng trung tâm hẹp (gắn nhãn trong khung phóng to)
OUTER_AREAS = {"A01", "A04", "A06", "A08"}
CENTRAL_AREAS = {"A02", "A03", "A05", "A07"}
CENTRAL_ZOOM_BOUNDS = (-8.665, 41.128, -8.565, 41.172)  # (lon_min, lat_min, lon_max, lat_max)

_CONTEXTILY_WARNED = False


def setup_map_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11.2,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def csv_to_df(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def get_latest_geojson_path(raw_folder: str | Path = RAW_FOLDER) -> Path:
    """Tìm file neighbourhoods.geojson ở mốc thời gian (snapshot) mới nhất trong data/raw/."""
    raw_dir = Path(raw_folder)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục raw data: {raw_dir}")

    dated_candidates = sorted(
        [
            p / "neighbourhoods.geojson"
            for p in raw_dir.iterdir()
            if p.is_dir() and (p / "neighbourhoods.geojson").exists()
        ],
        key=lambda p: p.parent.name,
        reverse=True,
    )
    if dated_candidates:
        return dated_candidates[0]

    geojson_dir = raw_dir / "geojson"
    if geojson_dir.exists():
        fallback = sorted(geojson_dir.glob("*.geojson"), reverse=True)
        if fallback:
            return fallback[0]

    raise FileNotFoundError(
        f"Không tìm thấy file neighbourhoods.geojson trong các snapshot của {raw_dir}"
    )


def _union_geometries(gdf: gpd.GeoDataFrame):
    cleaned = gdf.geometry.apply(lambda g: _make_valid(g) if g is not None and not g.is_valid else g)
    cleaned_gdf = gpd.GeoDataFrame(geometry=cleaned, crs=gdf.crs)
    if hasattr(cleaned_gdf, "union_all"):
        return cleaned_gdf.union_all()
    return cleaned_gdf.unary_union


def load_analysis_area_gdf(
    geojson_path: str | Path,
    area_bounds: Mapping[str, tuple[float, float, float, float]] = PORTO_AREA_BOUNDS,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Đọc file neighbourhoods.geojson và ánh xạ (clip/intersect) hình học bản đồ thật
    thành 8 vùng analysis_area (A01..A08) theo bảng tọa độ bounding box.

    Returns:
        (gdf_areas, gdf_raw): GeoDataFrame đã ánh xạ theo analysis_area và GeoDataFrame gốc.
    """
    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file GeoJSON: {path}")

    gdf_raw = gpd.read_file(path)
    if gdf_raw.crs is None:
        gdf_raw = gdf_raw.set_crs("EPSG:4326")
    elif gdf_raw.crs.to_epsg() != 4326:
        gdf_raw = gdf_raw.to_crs("EPSG:4326")

    city_union = _union_geometries(gdf_raw)

    records = []
    for area_id, (lon_min, lat_min, lon_max, lat_max) in area_bounds.items():
        area_box = box(lon_min, lat_min, lon_max, lat_max)
        clipped = city_union.intersection(area_box)
        if clipped.is_empty:
            clipped = area_box
        records.append({"analysis_area": area_id, "geometry": clipped})

    gdf_areas = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf_areas, gdf_raw


def _prepare_plot_crs(
    merged: gpd.GeoDataFrame,
    gdf_raw: gpd.GeoDataFrame | None,
    use_basemap: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None, tuple[float, float, float, float], bool]:
    """Chuyển sang EPSG:3857 nếu bật basemap contextily; fallback mượt về EPSG:4326 nếu không có mạng/thư viện."""
    global _CONTEXTILY_WARNED
    zoom_box_4326 = gpd.GeoDataFrame(geometry=[box(*CENTRAL_ZOOM_BOUNDS)], crs="EPSG:4326")
    if use_basemap and ctx is not None:
        merged_proj = merged.to_crs(epsg=3857)
        raw_proj = gdf_raw.to_crs(epsg=3857) if gdf_raw is not None else None
        zb = zoom_box_4326.to_crs(epsg=3857).total_bounds
        return merged_proj, raw_proj, (float(zb[0]), float(zb[1]), float(zb[2]), float(zb[3])), True

    if use_basemap and ctx is None and not _CONTEXTILY_WARNED:
        print("Lưu ý: Chưa cài 'contextily' (pip install contextily), sẽ vẽ bản đồ không có nền basemap.")
        _CONTEXTILY_WARNED = True
    return merged, gdf_raw, CENTRAL_ZOOM_BOUNDS, False


def _get_default_basemap_sources() -> list:
    """Trả về danh sách nguồn nền bản đồ miễn phí, KHÔNG cần API key (ưu tiên Esri WorldTopoMap)."""
    if ctx is None:
        return []
    sources = []
    for group_name, style_name in [
        ("Esri", "WorldTopoMap"),
        ("Esri", "WorldGrayCanvas"),
        ("OpenStreetMap", "Mapnik"),
    ]:
        group = getattr(ctx.providers, group_name, None)
        if group is not None and style_name in group:
            sources.append(group[style_name])
    return sources


def _add_basemap_safe(
    ax: plt.Axes,
    enabled: bool,
    source=None,
    show_attribution: bool = False,
) -> None:
    if not enabled or ctx is None:
        return

    candidates = [source] if source is not None else _get_default_basemap_sources()
    last_exc = None
    for provider in candidates:
        try:
            kwargs = dict(
                source=provider,
                zoom="auto",
                alpha=0.85,
                zorder=1,
            )
            if show_attribution:
                kwargs["attribution_size"] = 5.5
            else:
                kwargs["attribution"] = ""
            ctx.add_basemap(ax, **kwargs)
            return
        except Exception as exc:
            last_exc = exc

    if last_exc is not None:
        print(f"Không thể tải nền bản đồ contextily (kiểm tra kết nối mạng): {last_exc}")


def _annotate_area(
    ax: plt.Axes,
    geom,
    text: str,
    fontsize: float = 8.2,
    text_color: str = "#1A1A1A",
    xytext: tuple[float, float] | None = None,
) -> None:
    if geom is None or geom.is_empty:
        return
    pt = geom.representative_point()
    bbox_kw = dict(
        boxstyle="round,pad=0.22",
        facecolor="white",
        edgecolor="#BBBBBB",
        linewidth=0.6,
        alpha=0.90,
    )
    if xytext is not None:
        ax.annotate(
            text,
            xy=(pt.x, pt.y),
            xytext=xytext,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=text_color,
            bbox=bbox_kw,
            arrowprops=dict(
                arrowstyle="->",
                color="#222222",
                linewidth=0.9,
                shrinkA=2,
                shrinkB=2,
            ),
            zorder=7,
        )
    else:
        ax.text(
            pt.x,
            pt.y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=text_color,
            bbox=bbox_kw,
            zorder=6,
        )


def _draw_choropleth_column(
    ax_main: plt.Axes,
    ax_zoom: plt.Axes,
    merged: gpd.GeoDataFrame,
    gdf_raw: gpd.GeoDataFrame | None,
    value_col: str,
    title: str,
    cbar_label: str,
    cmap: str,
    fmt_func,
    small_sample_areas: set[str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    use_basemap: bool = True,
    basemap_source=None,
) -> None:
    small_sample_areas = small_sample_areas or set()

    merged_plot, raw_plot, zoom_bounds, basemap_active = _prepare_plot_crs(
        merged, gdf_raw, use_basemap=use_basemap
    )
    zx0, zy0, zx1, zy1 = zoom_bounds
    zoom_box = box(zx0, zy0, zx1, zy1)

    valid_vals = merged_plot[value_col].dropna()
    v_min = float(vmin) if vmin is not None else (float(valid_vals.min()) if len(valid_vals) else 0.0)
    v_max = float(vmax) if vmax is not None else (float(valid_vals.max()) if len(valid_vals) else 1.0)
    if v_min == v_max:
        v_max = v_min + 1.0

    fill_alpha = 0.72 if basemap_active else 0.96
    small_mask = merged_plot["analysis_area"].astype(str).isin(small_sample_areas)
    small_gdf = merged_plot[small_mask]

    # --- 1. Bản đồ toàn cảnh Porto (trên) ---
    merged_plot.plot(
        column=value_col,
        ax=ax_main,
        cmap=cmap,
        vmin=v_min,
        vmax=v_max,
        alpha=fill_alpha,
        edgecolor="#222222",
        linewidth=0.9,
        missing_kwds={"color": "#EEEEEE", "edgecolor": "#999999", "hatch": "///"},
        zorder=2,
    )
    if not small_gdf.empty:
        small_gdf.plot(
            ax=ax_main,
            facecolor="none",
            edgecolor="#7A1C1C",
            hatch="///",
            linewidth=0.7,
            alpha=0.65,
            zorder=3,
        )
    if raw_plot is not None and not raw_plot.empty:
        raw_plot.boundary.plot(ax=ax_main, color="#FFFFFF", linewidth=0.35, alpha=0.45, zorder=3)

    _add_basemap_safe(ax_main, enabled=basemap_active, source=basemap_source, show_attribution=False)

    for _, row in merged_plot.iterrows():
        area = str(row["analysis_area"])
        if area in OUTER_AREAS and pd.notna(row[value_col]):
            prefix = "⚠ " if area in small_sample_areas else ""
            label_text = f"{prefix}{area}\n{fmt_func(row[value_col])}"
            color = "#B00020" if area in small_sample_areas else "#1A1A1A"
            _annotate_area(ax_main, row.geometry, label_text, fontsize=8.3, text_color=color)

    ax_main.add_patch(
        Rectangle(
            (zx0, zy0),
            zx1 - zx0,
            zy1 - zy0,
            linewidth=1.1,
            edgecolor="#111111",
            facecolor="none",
            linestyle="--",
            zorder=5,
        )
    )
    ax_main.set_title(title)
    ax_main.set_xlabel("")
    ax_main.set_ylabel("")
    ax_main.set_axis_off()

    # --- 2. Khung phóng to trung tâm Porto (dưới) hiển thị rõ A02, A03, A05, A07 ---
    merged_plot.plot(
        column=value_col,
        ax=ax_zoom,
        cmap=cmap,
        vmin=v_min,
        vmax=v_max,
        alpha=fill_alpha,
        edgecolor="#222222",
        linewidth=0.9,
        missing_kwds={"color": "#EEEEEE", "edgecolor": "#999999"},
        zorder=2,
    )
    if not small_gdf.empty:
        small_gdf.plot(
            ax=ax_zoom,
            facecolor="none",
            edgecolor="#7A1C1C",
            hatch="///",
            linewidth=0.7,
            alpha=0.65,
            zorder=3,
        )
    if raw_plot is not None and not raw_plot.empty:
        raw_plot.boundary.plot(ax=ax_zoom, color="#FFFFFF", linewidth=0.35, alpha=0.45, zorder=3)

    ax_zoom.set_xlim(zx0, zx1)
    ax_zoom.set_ylim(zy0, zy1)
    _add_basemap_safe(ax_zoom, enabled=basemap_active, source=basemap_source, show_attribution=False)

    zoom_w = zx1 - zx0
    zoom_h = zy1 - zy0
    for _, row in merged_plot.iterrows():
        area = str(row["analysis_area"])
        if area in CENTRAL_AREAS and pd.notna(row[value_col]):
            visible_part = row.geometry.intersection(zoom_box)
            if not visible_part.is_empty:
                prefix = "⚠ " if area in small_sample_areas else ""
                label_text = f"{prefix}{area}: {fmt_func(row[value_col])}"
                color = "#B00020" if area in small_sample_areas else "#1A1A1A"

                # Dịch nhẹ nhãn A03 sang phải kèm mũi tên chỉ vào dải hẹp A03 để tránh đè biên A04
                if area == "A03":
                    pt = visible_part.representative_point()
                    xytext = (pt.x + zoom_w * 0.21, pt.y - zoom_h * 0.05)
                    _annotate_area(
                        ax_zoom,
                        visible_part,
                        label_text,
                        fontsize=7.6,
                        text_color=color,
                        xytext=xytext,
                    )
                else:
                    _annotate_area(ax_zoom, visible_part, label_text, fontsize=7.6, text_color=color)

    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    ax_zoom.set_xlabel("")
    ax_zoom.set_ylabel("")
    ax_zoom.set_title("Central Porto Zoom (A02, A03, A05, A07)", fontsize=9.0, fontweight="bold", pad=4)
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor("#444444")
        spine.set_linewidth(0.9)
        spine.set_linestyle("--")

    # Colorbar nằm ngang bên dưới khung phóng to
    sm = ScalarMappable(norm=Normalize(vmin=v_min, vmax=v_max), cmap=cmap)
    sm.set_array([])
    cbar = ax_zoom.figure.colorbar(
        sm,
        ax=ax_zoom,
        orientation="horizontal",
        fraction=0.09,
        pad=0.07,
        aspect=26,
    )
    cbar.set_label(cbar_label, fontsize=9, fontweight="bold")
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=8.5)


def plot_revpar_geojson(
    df: pd.DataFrame,
    gdf_areas: gpd.GeoDataFrame,
    output_path: str | Path,
    gdf_raw: gpd.GeoDataFrame | None = None,
    snapshot_label: str | None = None,
    currency_symbol: str = "€",
    use_basemap: bool = True,
    basemap_source=None,
) -> None:
    """
    Vẽ 3 bản đồ đặt cạnh nhau (1 hàng × 3 cột, kèm khung phóng to trung tâm dưới mỗi bản đồ)
    cho KPI RevPAR theo analysis_area:
    - Bản đồ 1: ADR (Blues)
    - Bản đồ 2: Occupancy Rate % (Greens)
    - Bản đồ 3: RevPAR (OrRd)
    """
    setup_map_style()
    df = df.copy()
    df.columns = df.columns.str.strip()

    df_area = df[(df["room_type"] == "All") & (df["analysis_area"] != "All")].copy()
    merged = gdf_areas.merge(df_area, on="analysis_area", how="left")

    fig = plt.figure(figsize=(18.0, 8.8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2.35, 1.15], hspace=0.14, wspace=0.08)

    snap_text = f" — Snapshot {snapshot_label}" if snapshot_label else ""
    fig.suptitle(
        f"Porto Revenue Performance by Analysis Area (ADR, Occupancy & RevPAR){snap_text}",
        y=0.97,
    )

    panels = [
        (
            "adr",
            f"1. ADR (Average Daily Rate, {currency_symbol})",
            f"ADR ({currency_symbol})",
            "Blues",
            lambda v: f"{currency_symbol}{v:.0f}",
        ),
        (
            "occupancy_rate_%",
            "2. Occupancy Rate (%)",
            "Occupancy Rate (%)",
            "Greens",
            lambda v: f"{v:.1f}%",
        ),
        (
            "revpar",
            f"3. RevPAR (Revenue / Available Night, {currency_symbol})",
            f"RevPAR ({currency_symbol})",
            "OrRd",
            lambda v: f"{currency_symbol}{v:.1f}",
        ),
    ]

    for col_idx, (col, title, cbar_label, cmap, fmt_func) in enumerate(panels):
        ax_main = fig.add_subplot(gs[0, col_idx])
        ax_zoom = fig.add_subplot(gs[1, col_idx])
        _draw_choropleth_column(
            ax_main=ax_main,
            ax_zoom=ax_zoom,
            merged=merged,
            gdf_raw=gdf_raw,
            value_col=col,
            title=title,
            cbar_label=cbar_label,
            cmap=cmap,
            fmt_func=fmt_func,
            use_basemap=use_basemap,
            basemap_source=basemap_source,
        )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.06)
    print(f"Đang lưu bản đồ GeoJSON RevPAR vào: {output_path}")
    save_figure(fig, output_path)


def plot_label_distribution_area_geojson(
    df: pd.DataFrame,
    gdf_areas: gpd.GeoDataFrame,
    output_path: str | Path,
    gdf_raw: gpd.GeoDataFrame | None = None,
    snapshot_label: str | None = None,
    use_basemap: bool = True,
    basemap_source=None,
) -> None:
    """
    Vẽ 3 bản đồ đặt cạnh nhau (1 hàng × 3 cột, kèm khung phóng to trung tâm dưới mỗi bản đồ)
    cho KPI Phân bố nhãn theo Area:
    - Bản đồ 1: General Sentiment — tỷ lệ nhãn Positive (%) (Greens, thang 85%–100% để tránh tương phản giả)
    - Bản đồ 2: Accessibility — tỷ lệ nhãn Highly accessible (%) (Blues, thang 85%–100%)
    - Bản đồ 3: Noise — tỷ lệ nhãn Quiet (%) (Purples)
    Các vùng có mẫu nhỏ (n < 70) được phủ gạch chéo (///) kèm cảnh báo ⚠.
    """
    setup_map_style()
    df = df.copy()
    df.columns = df.columns.str.strip()

    df = df[df["label"].astype(str).str.strip().str.lower() != "not mentioned"]
    df = df[df["analysis_area_processed"].astype(str) != "All"]

    SMALL_N = 70
    aspect_specs = [
        (
            "general_sentiment",
            "positive",
            "1. General Sentiment — Positive % (Scale: 85%–100%)",
            "Positive Share (%)",
            "Greens",
            85.0,
            100.0,
        ),
        (
            "accessibility",
            "highly accessible",
            "2. Accessibility — Highly Accessible % (Scale: 85%–100%)",
            "Highly Accessible Share (%)",
            "Blues",
            85.0,
            100.0,
        ),
        (
            "noise",
            "quiet",
            "3. Noise — Quiet %",
            "Quiet Share (%)",
            "Purples",
            None,
            None,
        ),
    ]

    fig = plt.figure(figsize=(18.0, 9.3))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2.35, 1.15], hspace=0.14, wspace=0.08)

    snap_text = f" — Snapshot {snapshot_label}" if snapshot_label else ""
    fig.suptitle(
        f"Review Sentiment & Experience Aspect Distribution by Analysis Area{snap_text}",
        y=0.97,
    )

    for col_idx, (aspect, target_label, title, cbar_label, cmap, vmin, vmax) in enumerate(aspect_specs):
        sub = df[df["aspect"] == aspect].copy()
        sub["label_norm"] = sub["label"].astype(str).str.strip().str.lower()

        totals = sub.groupby("analysis_area_processed")["count"].sum()
        pos_counts = (
            sub[sub["label_norm"] == target_label]
            .groupby("analysis_area_processed")["count"]
            .sum()
        )

        summary = pd.DataFrame({"total_mentions": totals}).join(
            pos_counts.rename("target_count"), how="left"
        ).fillna(0)
        summary["share_%"] = np.where(
            summary["total_mentions"] > 0,
            summary["target_count"] / summary["total_mentions"] * 100.0,
            np.nan,
        )
        summary = summary.reset_index().rename(columns={"analysis_area_processed": "analysis_area"})

        small_areas = set(
            summary.loc[summary["total_mentions"] < SMALL_N, "analysis_area"].astype(str)
        )
        merged = gdf_areas.merge(summary, on="analysis_area", how="left")

        ax_main = fig.add_subplot(gs[0, col_idx])
        ax_zoom = fig.add_subplot(gs[1, col_idx])
        _draw_choropleth_column(
            ax_main=ax_main,
            ax_zoom=ax_zoom,
            merged=merged,
            gdf_raw=gdf_raw,
            value_col="share_%",
            title=title,
            cbar_label=cbar_label,
            cmap=cmap,
            fmt_func=lambda v: f"{v:.1f}%",
            small_sample_areas=small_areas,
            vmin=vmin,
            vmax=vmax,
            use_basemap=use_basemap,
            basemap_source=basemap_source,
        )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.11)
    fig.text(
        0.5,
        0.015,
        "Shares are computed within each analysis area over reviews mentioning the aspect ('Not mentioned' excluded). "
        f"⚠ Hatched areas (///) indicate small sample (n < {SMALL_N}) — interpret with caution.",
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
        color="#555555",
    )
    print(f"Đang lưu bản đồ GeoJSON Phân bố nhãn theo Area vào: {output_path}")
    save_figure(fig, output_path)


def main() -> None:
    geojson_path = get_latest_geojson_path(RAW_FOLDER)
    snapshot_label = geojson_path.parent.name
    print(f"Sử dụng GeoJSON mốc mới nhất: {geojson_path}")

    gdf_areas, gdf_raw = load_analysis_area_gdf(geojson_path)

    revpar_df = csv_to_df(CSV_FOLDER / "kpi_revpar.csv")
    plot_revpar_geojson(
        revpar_df,
        gdf_areas,
        OUTPUT_FOLDER / "revpar_geojson.png",
        gdf_raw=gdf_raw,
        snapshot_label=snapshot_label,
    )

    label_df = csv_to_df(CSV_FOLDER / "kpi_label_distribution_area.csv")
    plot_label_distribution_area_geojson(
        label_df,
        gdf_areas,
        OUTPUT_FOLDER / "label_distribution_area_geojson.png",
        gdf_raw=gdf_raw,
        snapshot_label=snapshot_label,
    )


if __name__ == "__main__":
    main()