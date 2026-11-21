import argparse
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

output_folder = Path(__file__).parent.parent / "figures"
csv_folder = Path(__file__).parent.parent / "data" / "processed"


def setup_plot_style():
    plt.rcParams.update({
        # 1. Font chữ & Kích thước
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9.5,

        # 2. Khung viền (Spines)
        "axes.spines.top": False,
        "axes.spines.right": False,

        # 3. Lưới (Grid)
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,

        # 4. Đường vẽ (Line plots)
        "lines.linewidth": 2.2,

        # 5. Xuất file ảnh
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    
setup_plot_style()

def save_figure(fig: plt.Figure, output_path: str | Path):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def csv_to_df(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def plot_overpriced_analysis(df: pd.DataFrame, output_path: Path):
    df_area = df[(df["room_type"] == "All") & (df["analysis_area"] != "All")].copy()
    df_room = df[(df["analysis_area"] == "All") & (df["room_type"] != "All")].copy()

    if "overpriced_rate_%" not in df.columns:
        raise KeyError(f"Không thấy cột 'overpriced_rate_%'. Các cột hiện có: {df.columns.tolist()}")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.4, 1]}
    )
    fig.suptitle("Overpriced Listings Rate, snapshot 23/06/2026", y=0.99)

    # --- BIỂU ĐỒ 1 (TRÁI): Combo Bar (Overpriced %) + Line (Median Price) theo Khu vực ---
    x = np.arange(len(df_area))
    bars = ax1.bar(
        x, df_area["overpriced_rate_%"],
        width=0.55, color="#E07A5F", alpha=0.88, label="Overpriced Rate (%)"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_area["analysis_area"])
    ax1.set_ylabel("Overpriced Rate (%)", color="#E07A5F")
    ax1.set_ylim(0, df_area["overpriced_rate_%"].max() * 1.2)
    ax1.set_title("Overpriced Rate and Median Price by Area")
    ax1.grid(axis="y")

    for bar, (_, row) in zip(bars, df_area.iterrows()):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, h + 0.6,
            f"{h:.1f}%\n(Count: {int(row['overpriced_listings'])})",
            ha="center", fontsize=8.5
        )

    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        x, df_area["group_median_price"],
        color="#3D405B", marker="o", label="Median Price ($)"
    )
    ax1_twin.set_ylabel("Area Median Price ($)", color="#3D405B")
    ax1_twin.set_ylim(0, df_area["group_median_price"].max() * 1.2)
    ax1_twin.spines["right"].set_visible(True)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True)

    # --- BIỂU ĐỒ 2 (PHẢI): So sánh theo Loại phòng (Room Type) ---
    x_room = np.arange(len(df_room))
    bars_room = ax2.bar(
        x_room, df_room["overpriced_rate_%"],
        width=0.45, color=["#517ABD", "#DD8452"]
    )
    ax2.set_xticks(x_room)
    ax2.set_xticklabels(df_room["room_type"])
    ax2.set_ylim(0, df_room["overpriced_rate_%"].max() * 1.25)
    ax2.set_title("Overpriced Rate by Room Type (Citywide)")
    ax2.grid(axis="y")

    for bar, (_, row) in zip(bars_room, df_room.iterrows()):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2, h + 0.6,
            f"{h:.1f}%\n(Med: ${row['group_median_price']:.0f})\n(Listing Count: {int(row['overpriced_listings'])})",
            ha="center", va="bottom", fontsize=8.5
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.text(
        0.5, 0.012,
        "Overpriced: Price ≥ 1,25 × Median Price by area and review_scores_value < 4,6. ",
        ha="center", va="bottom", fontsize=9, style="italic", color="#555555"
    )
    print(f"Đang lưu biểu đồ phân tích Overpriced vào: {output_path}")
    save_figure(fig, output_path)


def plot_unavailability_rate(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["year_month"] = df["year_month"].astype(str)

    pivot_df = df.pivot(
        index="analysis_area",
        columns="year_month",
        values="unavailability_rate_%"
    ).sort_index()

    areas = pivot_df.index.tolist()
    months = pivot_df.columns.tolist()
    matrix = pivot_df.to_numpy(dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    # --- BIỂU ĐỒ 1 (TRÁI): Line Chart xu hướng theo thời gian ---
    x = np.arange(len(months))
    cmap_lines = plt.get_cmap("tab10")

    for i, area in enumerate(areas):
        ax1.plot(
            x,
            pivot_df.loc[area].to_numpy(),
            marker="o",
            label=area,
            color=cmap_lines(i % 10)
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(months, rotation=30, ha="right")
    ax1.set_ylabel("Unavailability Rate (%)")
    ax1.set_title("Xu hướng Unavailability Rate (%) theo Tháng")
    ax1.grid(True)
    ax1.legend(title="Area", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    # --- BIỂU ĐỒ 2 (PHẢI): Heatmap ma trận Khu vực x Tháng ---
    im = ax2.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax2.set_xticks(np.arange(len(months)))
    ax2.set_xticklabels(months, rotation=30, ha="right")
    ax2.set_yticks(np.arange(len(areas)))
    ax2.set_yticklabels(areas)
    ax2.set_title("Heatmap Unavailability Rate (%)")

    threshold = np.nanmean(matrix)
    for r in range(len(areas)):
        for c in range(len(months)):
            val = matrix[r, c]
            if not np.isnan(val):
                text_color = "white" if val > threshold * 1.15 else "#222222"
                ax2.text(
                    c, r, f"{val:.1f}%",
                    ha="center", va="center", color=text_color, fontsize=8.5
                )

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Rate (%)", fontsize=9)

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_sentiment_correlation(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.sort_values(["aspect", "risk_lift"], ascending=[True, True]).reset_index(drop=True)

    SMALL_N = 70
    small_flags = (df["total_reviews"].astype(int) < SMALL_N).to_numpy()

    y_labels = [
        f"{'⚠ ' if small else ''}[{row['aspect']}] {row['label']} (n={int(row['total_reviews'])})"
        for small, (_, row) in zip(small_flags, df.iterrows())
    ]
    y = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, max(5, len(df) * 0.65)),
        sharey=True,
        gridspec_kw={"width_ratios": [1.5, 1]}
    )
    fig.suptitle("Correlation Between Experience Aspects and Review Sentiment", y=0.99)

    # --- BIỂU ĐỒ 1 (TRÁI): Horizontal Stacked Bar phân bổ tỷ lệ Sentiment ---
    pos = df["positive_rate"].to_numpy()
    mix = df["mixed_rate"].to_numpy()
    neg = df["negative_rate"].to_numpy()

    ax1.barh(y, pos, color="#2A9D8F", height=0.62, label="Positive (%)")
    ax1.barh(y, mix, left=pos, color="#E9C46A", height=0.62, label="Mixed (%)")
    ax1.barh(y, neg, left=pos + mix, color="#E76F51", height=0.62, label="Negative (%)")

    aspects = df["aspect"].tolist()
    for i in range(1, len(aspects)):
        if aspects[i] != aspects[i - 1]:
            ax1.axhline(i - 0.5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.6)
            ax2.axhline(i - 0.5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.6)

    for i in range(len(df)):
        if pos[i] > 8:
            ax1.text(pos[i] / 2, y[i], f"{pos[i]:.1f}%", va="center", ha="center", color="white", fontsize=8.5, fontweight="bold")
        if mix[i] > 8:
            ax1.text(pos[i] + mix[i] / 2, y[i], f"{mix[i]:.1f}%", va="center", ha="center", color="#222222", fontsize=8.5, fontweight="bold")
        if neg[i] > 8:
            ax1.text(pos[i] + mix[i] + neg[i] / 2, y[i], f"{neg[i]:.1f}%", va="center", ha="center", color="white", fontsize=8.5, fontweight="bold")

    ax1.set_yticks(y)
    ax1.set_yticklabels(y_labels)
    for tick, small in zip(ax1.get_yticklabels(), small_flags):
        if small:
            tick.set_color("#C1121F")

    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Review Percentage (%)")
    ax1.set_title("Sentiment Distribution by Experience Aspect")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    # --- BIỂU ĐỒ 2 (PHẢI): Risk Lift (Hệ số rủi ro Non-positive) ---
    lifts = df["risk_lift"].to_numpy()
    bar_colors = ["#D62828" if val > 1.0 else "#6C757D" for val in lifts]

    nonpos = mix + neg
    valid = lifts > 0
    baseline_nonpos = float(np.median(nonpos[valid] / lifts[valid])) if valid.any() else float("nan")

    bars2 = ax2.barh(y, lifts, color=bar_colors, height=0.62)
    ax2.axvline(
        1.0, color="#222222", linestyle=":", linewidth=1.2,
        label=f"Baseline (Lift = 1.0 | Non-positive = {baseline_nonpos:.1f}%)"
    )

    max_lift = lifts.max() if len(lifts) > 0 and lifts.max() > 0 else 1.0
    ax2.set_xlim(0, max_lift * 1.25)

    for bar, val in zip(bars2, lifts):
        ax2.text(
            bar.get_width() + max_lift * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}x",
            va="center", ha="left", fontsize=9, fontweight="bold",
            color="#D62828" if val > 1.0 else "#333333"
        )

    ax2.set_xlabel("Risk Lift (x)")
    ax2.set_title("Non-Positive Review Risk Lift")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax2.grid(axis="x")

    plt.tight_layout()
    fig.text(
        0.01, 0.01, f"⚠ Small sample (n < {SMALL_N}): Result needs to be elaborated carefully",
        ha="left", va="bottom", fontsize=9, style="italic", color="#C1121F"
    )
    save_figure(fig, output_path)


def plot_dormant_listing_rate(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["snapshot"] = df["snapshot"].astype(str)
    snapshots = sorted(df["snapshot"].unique())

    df_area = df[(df["room_type"] == "All") & (df["analysis_area"] != "All")]
    df_room = df[(df["analysis_area"] == "All") & (df["room_type"] != "All")]

    colors = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(snapshots)))
    Y_MAX, Y_STEP = 24, 4
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [2.4, 1]})
    fig.suptitle("Dormant Listing Rate Across Four Snapshots", y=0.99)

    for ax, sub_df, group_col, title, xlabel, ylabel in [
        (axes[0], df_area, "analysis_area", "Dormant Rate (%) by Area", "Analysis Area", "Dormant Rate (%)"),
        (axes[1], df_room, "room_type", "Dormant Rate (%) by Room Type", "Room Type", ""),
    ]:
        categories = sorted(sub_df[group_col].unique())
        x = np.arange(len(categories))
        bar_width = 0.78 / max(len(snapshots), 1)

        for i, snap in enumerate(snapshots):
            vals = (
                sub_df[sub_df["snapshot"] == snap]
                .set_index(group_col)
                .reindex(categories)["dormant_rate_%"]
                .fillna(0)
                .to_numpy()
            )
            offset = (i - (len(snapshots) - 1) / 2) * bar_width
            bars = ax.bar(x + offset, vals, width=bar_width * 0.92, label=snap, color=colors[i % len(colors)])
            if i == len(snapshots) - 1:
                for bar, val in zip(bars, vals):
                    if val > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}",
                            ha="center", va="bottom", fontsize=8, fontweight="bold"
                        )

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        ax.set_ylim(0, Y_MAX)
        ax.set_yticks(np.arange(0, Y_MAX + 1, Y_STEP))
        ax.grid(axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Snapshot", loc="lower center",
        bbox_to_anchor=(0.5, 0.0), ncol=len(snapshots), frameon=True
    )
    plt.tight_layout(rect=[0, 0.16, 1, 0.95])
    save_figure(fig, output_path)


def plot_label_distribution(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()

    df = df[df["label"].astype(str).str.strip().str.lower() != "not mentioned"]
    df = df[df["analysis_area_processed"].astype(str) != "All"]

    aspects = list(df["aspect"].unique())
    areas = sorted(df["analysis_area_processed"].unique())

    TEAL, YELLOW, RED_ORANGE = "#2A9D8F", "#E9C46A", "#E76F51"
    label_style = {
        "positive": (TEAL, 0), "highly accessible": (TEAL, 0), "quiet": (TEAL, 0),
        "mixed": (YELLOW, 1), "neutral": (YELLOW, 1), "acceptable": (YELLOW, 1),
        "negative": (RED_ORANGE, 2), "poor access": (RED_ORANGE, 2), "noisy": (RED_ORANGE, 2),
    }
    SMALL_N = 70

    fig, axes = plt.subplots(
        1, len(aspects), figsize=(6 * len(aspects), 0.6 * len(areas) + 3.6), sharex=True
    )
    if len(aspects) == 1:
        axes = [axes]

    fig.suptitle("Distribution of Sentiment and Experience Aspect Labels by Area", y=0.99)

    y = np.arange(len(areas))
    for ax, aspect in zip(axes, aspects):
        sub = df[df["aspect"] == aspect]
        pivot = sub.pivot_table(
            index="analysis_area_processed", columns="label", values="count", aggfunc="sum"
        ).reindex(areas).fillna(0)
        cols = sorted(pivot.columns, key=lambda l: label_style.get(str(l).strip().lower(), ("#999999", 3))[1])
        pivot = pivot[cols]
        totals = pivot.sum(axis=1)
        share = pivot.div(totals.replace(0, np.nan), axis=0).fillna(0) * 100

        left = np.zeros(len(areas))
        for lab in cols:
            color = label_style.get(str(lab).strip().lower(), ("#999999", 3))[0]
            vals = share[lab].to_numpy()
            ax.barh(y, vals, left=left, color=color, height=0.62, label=lab)
            for i, v in enumerate(vals):
                if v >= 8:
                    ax.text(
                        left[i] + v / 2, y[i], f"{v:.0f}%",
                        ha="center", va="center", fontsize=8.5, fontweight="bold",
                        color="#222222" if color == YELLOW else "white"
                    )
            left += vals

        ylabels = [f"{'⚠ ' if totals[a] < SMALL_N else ''}{a} (n={int(totals[a])})" for a in areas]
        ax.set_yticks(y)
        ax.set_yticklabels(ylabels)
        for tick, a in zip(ax.get_yticklabels(), areas):
            if totals[a] < SMALL_N:
                tick.set_color("#C1121F")

        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xlabel("Share (%)")
        ax.set_title(aspect.replace("_", " ").title())
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(cols), frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.text(
        0.5, 0.012,
        "Shares are computed within each area over reviews that mention the aspect ('Not mentioned' excluded). "
        f"⚠ Small sample (n < {SMALL_N}).",
        ha="center", va="bottom", fontsize=9, style="italic", color="#555555"
    )
    save_figure(fig, output_path)


def plot_professional_host_share(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["snapshot"] = df["snapshot"].astype(str)
    snapshots = sorted(df["snapshot"].unique())
    areas = sorted(df["analysis_area"].unique())
    tiers = ["individual", "small_multi", "professional"]
    tier_colors = {"individual": "#8ECAE6", "small_multi": "#219EBC", "professional": "#023047"}

    ncols = 2
    nrows = int(np.ceil(len(snapshots) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    x = np.arange(len(areas))
    for idx, snap in enumerate(snapshots):
        ax = axes[idx]
        sub = (
            df[df["snapshot"] == snap]
            .pivot(index="analysis_area", columns="host_tier", values="share_%")
            .reindex(areas)
            .fillna(0)
        )
        bottom = np.zeros(len(areas))

        for tier in tiers:
            if tier in sub.columns:
                vals = sub[tier].to_numpy()
                ax.bar(x, vals, bottom=bottom, width=0.6, label=tier, color=tier_colors.get(tier, "#666666"))
                for i, v in enumerate(vals):
                    if v > 10:
                        ax.text(
                            x[i], bottom[i] + v / 2, f"{v:.0f}%",
                            ha="center", va="center", color="white", fontsize=8, fontweight="bold"
                        )
                bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(areas)
        ax.set_ylim(0, 105)
        ax.set_title(f"Snapshot: {snap}")
        if idx % ncols == 0:
            ax.set_ylabel("Share (%)")

    for j in range(len(snapshots), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Host Tier Market Share Distribution by Area Across Four Snapshots", y=1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Host Tier", loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    save_figure(fig, output_path)


def plot_review_concentration(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df[(df["year"] >= 2015) & (df["year"] <= 2025)]

    pivot_df = df.pivot(index="year", columns="analysis_area", values="concentration_pct").fillna(0).sort_index()
    years = pivot_df.index.to_numpy()
    areas = sorted(pivot_df.columns.tolist())
    cmap = plt.get_cmap("tab10")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle("Review Volume and Distribution Trends by Area (2015–2025)", y=0.99)

    stacks = [pivot_df[a].to_numpy() for a in areas]
    ax1.stackplot(years, stacks, labels=areas, colors=[cmap(i % 10) for i in range(len(areas))], alpha=0.88)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(years.min(), years.max())
    ax1.set_ylabel("Review Concentration (%)")
    ax1.set_xlabel("Year")
    ax1.set_title("Review Share Distribution by Area Over Time")

    h_leg, l_leg = ax1.get_legend_handles_labels()
    ax1.legend(h_leg[::-1], l_leg[::-1], title="Area", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8.5)

    yearly_total = df[["year", "city_total_reviews"]].drop_duplicates().sort_values("year")
    ax2.bar(yearly_total["year"].astype(str), yearly_total["city_total_reviews"], color="#3D405B", width=0.65)
    ax2.set_ylabel("City Total Reviews")
    ax2.set_xlabel("Year")
    ax2.set_title("City Total Reviews")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y")

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_revpar(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df_area = df[(df["room_type"] == "All") & (df["analysis_area"] != "All")].sort_values("analysis_area")
    df_room = df[(df["analysis_area"] == "All") & (df["room_type"] != "All")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [2.4, 1]})
    fig.suptitle("Revenue Performance (RevPAR) and Occupancy Rate, Snapshot 23/6/2026", y=0.99)

    Y_STEP = 10
    y_top = float(np.ceil(max(df_area["revpar"].max(), df_room["revpar"].max()) * 1.1 / Y_STEP) * Y_STEP)

    x = np.arange(len(df_area))
    bars = ax1.bar(x, df_area["revpar"], width=0.65, color="#2A9D8F", label="RevPAR ($)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_area["analysis_area"])
    ax1.set_ylabel("RevPAR ($)", color="#2A9D8F")
    ax1.set_ylim(0, y_top)
    ax1.set_title("RevPAR ($) & Occupancy Rate (%) by Area")
    ax1.grid(axis="y")

    for bar, (_, row) in zip(bars, df_area.iterrows()):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, y_top * 0.015,
            f"${h:.1f}\n(ADR ${row['adr']:.0f})",
            ha="center", va="bottom", fontsize=8, color="white", fontweight="bold"
        )

    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, df_area["occupancy_rate_%"], color="#E76F51", marker="o", label="Occupancy (%)")
    ax1_twin.set_ylabel("Occupancy Rate (%)", color="#E76F51")
    ax1_twin.set_ylim(0, df_area["occupancy_rate_%"].max() * 1.25)
    ax1_twin.spines["right"].set_visible(True)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    x_r = np.arange(len(df_room))
    bars_r = ax2.bar(x_r, df_room["revpar"], width=0.45, color=["#4C72B0", "#DD8452"])
    ax2.set_xticks(x_r)
    ax2.set_xticklabels(df_room["room_type"])
    ax2.set_ylim(0, y_top)
    ax2.set_yticks(np.arange(0, y_top + 1, Y_STEP))
    ax2.set_title("RevPAR by Room Type")
    ax2.grid(axis="y")

    for bar, (_, row) in zip(bars_r, df_room.iterrows()):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2, h + 0.8,
            f"${h:.2f}\n(ADR ${row['adr']:.0f} | Occ {row['occupancy_rate_%']:.1f}%)",
            ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_top_n_host_share(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["snapshot"] = df["snapshot"].astype(str)
    df = df.sort_values("snapshot")

    top_n = int(df["top_n"].iloc[0]) if "top_n" in df.columns else 10
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    bars = ax1.bar(x, df["top_n_share_%"], width=0.48, color="#4C72B0", label=f"Top {top_n} Share (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["snapshot"])
    ax1.set_ylabel(f"Top {top_n} Host Market Share (%)", color="#4C72B0")
    ax1.set_ylim(0, max(df["top_n_share_%"].max() * 1.3, 15))
    ax1.set_title(f"Top {top_n} Hosts Market Share Across Four Snapshots")
    ax1.grid(axis="y")

    for bar, (_, row) in zip(bars, df.iterrows()):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, h - 0.3,
            f"{h:.2f}%\n({int(row['top_n_listings'])}/{int(row['total_listings'])})",
            ha="center", va="top", fontsize=9, color="white", fontweight="bold"
        )

    ax2 = ax1.twinx()
    ax2.plot(x, df["total_hosts"], color="#E76F51", marker="s", label="Total Hosts")
    ax2.set_ylabel("Total Unique Hosts", color="#E76F51")
    ax2.set_ylim(0, df["total_hosts"].max() * 1.25)
    ax2.spines["right"].set_visible(True)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

    plt.tight_layout()
    save_figure(fig, output_path)


def main():
    setup_plot_style()

    print("Tạo biểu đồ cột cho overprice")
    plot_overpriced_analysis(
        csv_to_df(csv_folder / "kpi_overpriced_rate.csv"),
        output_folder / "overpriced_analysis.png",
    )
    plot_unavailability_rate(
        csv_to_df(csv_folder / "kpi_unavailability_rate.csv"),
        output_folder / "unavailability_rate.png",
    )
    plot_sentiment_correlation(
        csv_to_df(csv_folder / "kpi_aspect_sentiment_correlation.csv"),
        output_folder / "aspect_sentiment_correlation.png",
    )
    plot_dormant_listing_rate(
        csv_to_df(csv_folder / "kpi_dormant_listing_rate.csv"),
        output_folder / "dormant_listing_rate.png",
    )
    plot_label_distribution(
        csv_to_df(csv_folder / "kpi_label_distribution.csv"),
        output_folder / "label_distribution.png",
    )
    plot_professional_host_share(
        csv_to_df(csv_folder / "kpi_professional_host_share.csv"),
        output_folder / "professional_host_share.png",
    )
    plot_review_concentration(
        csv_to_df(csv_folder / "kpi_review_concentration.csv"),
        output_folder / "review_concentration.png",
    )
    plot_revpar(
        csv_to_df(csv_folder / "kpi_revpar.csv"),
        output_folder / "revpar.png",
    )
    plot_top_n_host_share(
        csv_to_df(csv_folder / "kpi_top_n_host_share.csv"),
        output_folder / "top_n_host_share.png",
    )


if __name__ == "__main__":
    main()