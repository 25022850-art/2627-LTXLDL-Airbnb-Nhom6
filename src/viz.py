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

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def csv_to_df(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def plot_overpriced_analysis(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "overpriced_rate_%" not in df.columns:
        raise KeyError(f"Không thấy cột 'overpriced_rate_%'. Các cột hiện có: {df.columns.tolist()}")

    # 1. Tách 2 lát cắt dữ liệu
    df_area = df[(df["room_type"] == "All") & (df["analysis_area"] != "All")].copy()
    df_room = df[(df["analysis_area"] == "All") & (df["room_type"] != "All")].copy()

    # Ánh xạ tên loại phòng sang tiếng Việt
    room_type_map = {
        "Entire home/apt": "Cả căn nhà",
        "Private room": "Phòng riêng tư",
    }
    df_room["room_type_vi"] = df_room["room_type"].replace(room_type_map)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.5, 5.5), gridspec_kw={"width_ratios": [2.2, 1.15]}
    )

    # --- BIỂU ĐỒ 1 (TRÁI): Biểu đồ cột nhóm (Tỷ lệ đắt đỏ % & Giá trung vị $) theo Khu vực ---
    x = np.arange(len(df_area))
    width = 0.38

    bars_rate = ax1.bar(
        x - width / 2, df_area["overpriced_rate_%"],
        width=width, color="#E07A5F", alpha=0.9, label="Tỷ lệ đắt đỏ (%)"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_area["analysis_area"])
    ax1.set_xlabel("Khu vực phân tích", fontweight="bold", labelpad=8)
    ax1.set_ylabel("Tỷ lệ đắt đỏ (%)", color="#E07A5F", fontweight="bold")
    ax1.set_ylim(0, df_area["overpriced_rate_%"].max() * 1.25)
    ax1.set_title("Tỷ lệ đắt đỏ (%) và Giá trung vị ($) theo Khu vực", fontweight="bold", pad=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)

    for bar, (_, row) in zip(bars_rate, df_area.iterrows()):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, h + 0.6,
            f"{h:.1f}%\n({int(row['overpriced_listings'])})",
            ha="center", va="bottom", fontsize=7.5, color="#B84A2E"
        )

    ax1_twin = ax1.twinx()
    bars_price = ax1_twin.bar(
        x + width / 2, df_area["group_median_price"],
        width=width, color="#3D405B", alpha=0.88, label="Giá trung vị ($)"
    )
    ax1_twin.set_ylabel("Giá trung vị ($)", color="#3D405B", fontweight="bold")
    ax1_twin.set_ylim(0, df_area["group_median_price"].max() * 1.25)
    ax1_twin.spines["top"].set_visible(False)

    for bar in bars_price:
        h = bar.get_height()
        ax1_twin.text(
            bar.get_x() + bar.get_width() / 2, h + 2,
            f"${h:.0f}",
            ha="center", va="bottom", fontsize=7.5, color="#3D405B", fontweight="bold"
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True)

    # --- BIỂU ĐỒ 2 (PHẢI): Biểu đồ cột nhóm (Tỷ lệ đắt đỏ % & Giá trung vị $) theo Loại phòng ---
    x_room = np.arange(len(df_room))
    width_room = 0.34

    bars_room_rate = ax2.bar(
        x_room - width_room / 2, df_room["overpriced_rate_%"],
        width=width_room, color="#E07A5F", alpha=0.9, label="Tỷ lệ đắt đỏ (%)"
    )
    ax2.set_xticks(x_room)
    ax2.set_xticklabels(df_room["room_type_vi"])
    ax2.set_xlabel("Loại phòng", fontweight="bold", labelpad=8)
    ax2.set_ylabel("Tỷ lệ đắt đỏ (%)", color="#E07A5F", fontweight="bold")
    ax2.set_ylim(0, df_room["overpriced_rate_%"].max() * 1.28)
    ax2.set_title("Tỷ lệ đắt đỏ (%) và Giá trung vị ($) theo Loại phòng", fontweight="bold", pad=12)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False)

    for bar, (_, row) in zip(bars_room_rate, df_room.iterrows()):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2, h + 0.6,
            f"{h:.1f}%\n({int(row['overpriced_listings'])})",
            ha="center", va="bottom", fontsize=8, color="#B84A2E"
        )

    ax2_twin = ax2.twinx()
    bars_room_price = ax2_twin.bar(
        x_room + width_room / 2, df_room["group_median_price"],
        width=width_room, color="#3D405B", alpha=0.88, label="Giá trung vị ($)"
    )
    ax2_twin.set_ylabel("Giá trung vị ($)", color="#3D405B", fontweight="bold")
    ax2_twin.set_ylim(0, df_room["group_median_price"].max() * 1.28)
    ax2_twin.spines["top"].set_visible(False)

    for bar in bars_room_price:
        h = bar.get_height()
        ax2_twin.text(
            bar.get_x() + bar.get_width() / 2, h + 2,
            f"${h:.0f}",
            ha="center", va="bottom", fontsize=8, color="#3D405B", fontweight="bold"
        )

    h_r1, l_r1 = ax2.get_legend_handles_labels()
    h_r2, l_r2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(h_r1 + h_r2, l_r1 + l_r2, loc="upper right", frameon=True)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Đang lưu biểu đồ phân tích đắt đỏ vào: {out}")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_unavailability_rate(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["year_month"] = df["year_month"].astype(str)

    # Tạo bảng pivot: hàng = analysis_area, cột = year_month, giá trị = unavailability_rate_%
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
            linewidth=2,
            label=area,
            color=cmap_lines(i % 10)
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(months, rotation=30, ha="right")
    ax1.set_ylabel("Tỷ lệ phòng không có sẵn (%)", fontweight="bold")
    ax1.set_title("Xu hướng phòng không có sẵn (%) theo Tháng", fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(title="Area", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    # --- BIỂU ĐỒ 2 (PHẢI): Heatmap ma trận Khu vực x Tháng ---
    im = ax2.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax2.set_xticks(np.arange(len(months)))
    ax2.set_xticklabels(months, rotation=30, ha="right")
    ax2.set_yticks(np.arange(len(areas)))
    ax2.set_yticklabels(areas)
    ax2.set_title("Tỷ lệ không có sẵn trên biểu đồ nhiệt (%)", fontweight="bold", pad=12)

    # Ghi trực tiếp giá trị % vào từng ô của Heatmap
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
    cbar.set_label("Tỷ lệ phòng không có sẵn (%)", fontsize=9)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_sentiment_correlation(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Ánh xạ tiếng Việt cho khía cạnh và nhãn đánh giá
    aspect_map = {
        "accessibility": "Khả năng tiếp cận",
        "noise": "Độ ồn",
    }
    label_map = {
        "Highly accessible": "Rất dễ tiếp cận",
        "Neutral": "Trung tính",
        "Poor access": "Khó tiếp cận",
        "Acceptable": "Chấp nhận được",
        "Noisy": "Ồn ào",
        "Quiet": "Yên tĩnh",
    }

    df["aspect_vi"] = df["aspect"].replace(aspect_map)
    df["label_vi"] = df["label"].replace(label_map)
    df = df.sort_values(["aspect", "risk_lift"], ascending=[True, True]).reset_index(drop=True)

    n_rows = len(df)
    ncols = 3
    nrows_pie = int(np.ceil(n_rows / ncols))

    # Bố cục: Trái là lưới biểu đồ tròn (Pie charts), Phải là biểu đồ cột ngang Risk Lift
    fig = plt.figure(figsize=(16.5, max(6.5, nrows_pie * 3.4)))
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.28)
    left_gs = outer_gs[0].subgridspec(nrows_pie, ncols, hspace=0.45, wspace=0.25)

    pie_colors = ["#2A9D8F", "#E9C46A", "#E76F51"]
    pie_labels = ["Tích cực (%)", "Trung lập / Hỗn hợp (%)", "Tiêu cực (%)"]
    legend_handles = None

    # --- PHẦN 1 (TRÁI): Lưới biểu đồ tròn phân bổ cảm xúc cho từng nhóm ---
    for idx, row in df.iterrows():
        r, c = divmod(idx, ncols)
        ax_pie = fig.add_subplot(left_gs[r, c])

        sizes = [row["positive_rate"], row["mixed_rate"], row["negative_rate"]]
        wedges, texts, autotexts = ax_pie.pie(
            sizes,
            colors=pie_colors,
            startangle=90,
            counterclock=False,
            autopct=lambda p: f"{p:.1f}%" if p >= 5 else "",
            pctdistance=0.68,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
            textprops={"fontsize": 8, "fontweight": "bold", "color": "#111111"}
        )
        if legend_handles is None:
            legend_handles = wedges

        ax_pie.set_title(
            f"[{row['aspect_vi']}]\n{row['label_vi']} (n={int(row['total_reviews'])})",
            fontsize=9.5,
            fontweight="bold",
            pad=6
        )

    # Tắt các ô trống nếu số lượng nhóm không chia hết cho 3
    for idx in range(n_rows, nrows_pie * ncols):
        r, c = divmod(idx, ncols)
        fig.add_subplot(left_gs[r, c]).axis("off")

    fig.legend(
        legend_handles,
        pie_labels,
        title="Phân loại cảm xúc",
        loc="lower left",
        bbox_to_anchor=(0.06, -0.02),
        ncol=3,
        frameon=True,
        fontsize=9
    )

    # --- PHẦN 2 (PHẢI): Biểu đồ cột ngang Hệ số rủi ro (Risk Lift) ---
    ax_bar = fig.add_subplot(outer_gs[1])
    y = np.arange(n_rows)
    # Chỉ hiển thị tên nhãn trên trục Y (bỏ tiền tố [Khía cạnh])
    y_labels = df["label_vi"].tolist()
    lifts = df["risk_lift"].to_numpy()
    bar_colors = ["#D62828" if val > 1.0 else "#6C757D" for val in lifts]

    bars = ax_bar.barh(y, lifts, color=bar_colors, height=0.58)
    ax_bar.axvline(1.0, color="#222222", linestyle=":", linewidth=1.3, label="Mức cơ sở (Hệ số = 1.0)")

    max_lift = lifts.max() if len(lifts) > 0 and lifts.max() > 0 else 1.0
    ax_bar.set_xlim(0, max_lift * 1.32)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(y_labels, fontsize=9.5)

    # Đường phân cách và tiêu đề nhóm khía cạnh gộp trực tiếp trên biểu đồ
    for aspect_vi, group in df.groupby("aspect_vi", sort=False):
        y_min = group.index.min()
        y_max = group.index.max()
        y_center = (y_min + y_max) / 2

        if y_min > 0:
            ax_bar.axhline(y_min - 0.5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.6)

        # Vẽ khung bao nhẹ cho từng nhóm và đặt tên khía cạnh ở bên phải biểu đồ
        ax_bar.axhspan(y_min - 0.48, y_max + 0.48, color="#F4F6F9", alpha=0.5, zorder=0)
        ax_bar.text(
            max_lift * 1.29,
            y_center,
            aspect_vi,
            va="center",
            ha="right",
            fontsize=9.5,
            fontweight="bold",
            color="#2B2D42",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#E9ECEF", edgecolor="#ADB5BD", linewidth=0.8)
        )

    ax_bar.set_xlabel("Hệ số rủi ro không tích cực (x lần)", fontweight="bold")
    ax_bar.set_title("Hệ số rủi ro đánh giá không tích cực", fontweight="bold", pad=12)
    ax_bar.legend(loc="lower right", frameon=True, fontsize=9)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    for bar, val in zip(bars, lifts):
        ax_bar.text(
            bar.get_width() + max_lift * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}x",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#D62828" if val > 1.0 else "#333333"
        )

    fig.suptitle("Tương quan Khía cạnh Đánh giá và Cảm xúc Khách hàng", fontsize=13, fontweight="bold", y=1.01)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_dormant_listing_rate(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["snapshot"] = df["snapshot"].astype(str)
    snapshots = sorted(df["snapshot"].unique())

    df_area = df[(df["room_type"] == "All") & (df["analysis_area"] != "All")].copy()
    df_room = df[(df["analysis_area"] == "All") & (df["room_type"] != "All")].copy()

    # Ánh xạ loại phòng sang tiếng Việt
    room_type_map = {
        "Entire home/apt": "Cả căn nhà",
        "Private room": "Phòng riêng tư",
    }
    df_room["room_type"] = df_room["room_type"].replace(room_type_map)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    # Tăng chiều rộng figure một chút để các cột ở biểu đồ 1 trông dày và thoáng hơn
    fig, axes = plt.subplots(1, 2, figsize=(20, 5.4), gridspec_kw={"width_ratios": [2.6, 1]})

    configs = [
        (axes[0], df_area, "analysis_area", "Tỷ lệ phòng không hoạt động (%) theo Khu vực", "Khu vực phân tích", "Tỷ lệ phòng không hoạt động (%)", 0.86),
        (axes[1], df_room, "room_type", "Tỷ lệ phòng không hoạt động (%) theo Loại phòng", "Loại phòng", "", 0.65),
    ]

    for ax, sub_df, group_col, title, xlabel, ylabel, group_span in configs:
        categories = sorted(sub_df[group_col].unique())
        x = np.arange(len(categories))
        bar_width = group_span / max(len(snapshots), 1)

        for i, snap in enumerate(snapshots):
            vals = (
                sub_df[sub_df["snapshot"] == snap]
                .set_index(group_col)
                .reindex(categories)["dormant_rate_%"]
                .fillna(0)
                .to_numpy()
            )
            offset = (i - (len(snapshots) - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset,
                vals,
                width=bar_width * 0.96,  # Tăng tỷ lệ lấp đầy giữa các cột trong nhóm
                label=snap,
                color=colors[i % len(colors)]
            )
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.25,
                        f"{val:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7.2
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, fontweight="bold", labelpad=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_ylim(0, sub_df["dormant_rate_%"].max() * 1.2)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Thời điểm (Snapshot)", loc="lower right", bbox_to_anchor=(0.99, 0.15))
    plt.tight_layout(rect=[0, 0, 0.90, 1])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_label_distribution(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Ánh xạ tên khía cạnh và nhãn đánh giá sang tiếng Việt
    aspect_map = {
        "general_sentiment": "Cảm xúc chung",
        "accessibility": "Khả năng tiếp cận",
        "noise": "Độ ồn",
    }
    label_map = {
        "Positive": "Tích cực",
        "Mixed": "Trung lập / Hỗn hợp",
        "Negative": "Tiêu cực",
        "Highly accessible": "Rất dễ tiếp cận",
        "Poor access": "Khó tiếp cận",
        "Neutral": "Trung tính",
        "Quiet": "Yên tĩnh",
        "Noisy": "Ồn ào",
        "Acceptable": "Chấp nhận được",
    }

    df["aspect_vi"] = df["aspect"].replace(aspect_map)
    df["label_vi"] = df["label"].replace(label_map)
    aspects = df["aspect"].unique()

    fig, axes = plt.subplots(1, len(aspects), figsize=(5.4 * len(aspects), 4.6))
    if len(aspects) == 1:
        axes = [axes]

    palette = ["#2A9D8F", "#E9C46A", "#E76F51", "#4C72B0"]
    for ax, aspect in zip(axes, aspects):
        sub = df[df["aspect"] == aspect].sort_values("percentage", ascending=True)
        y = np.arange(len(sub))
        bars = ax.barh(y, sub["percentage"], color=palette[: len(sub)], height=0.55)

        ax.set_yticks(y)
        ax.set_yticklabels(sub["label_vi"], fontsize=9.5)
        ax.set_xlim(0, max(sub["percentage"].max() * 1.32, 15))
        ax.set_xlabel("Tỷ lệ (%)", fontweight="bold")
        ax.set_title(sub["aspect_vi"].iloc[0], fontweight="bold", pad=10)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, (_, row) in zip(bars, sub.iterrows()):
            w = bar.get_width()
            ax.text(
                w + 1.2,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}% ({int(row['count'])})",
                va="center",
                fontsize=8.5
            )

    fig.suptitle("Phân bổ Nhãn Đánh giá theo Từng Khía cạnh", fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_professional_host_share(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["snapshot"] = df["snapshot"].astype(str)
    snapshots = sorted(df["snapshot"].unique())
    areas = sorted(df["analysis_area"].unique())

    tiers = ["individual", "small_multi", "professional"]
    tier_labels = {
        "individual": "Cá nhân",
        "small_multi": "Quy mô nhỏ",
        "professional": "Chuyên nghiệp",
    }
    tier_colors = {
        "individual": "#8ECAE6",
        "small_multi": "#219EBC",
        "professional": "#023047",
    }

    ncols = 2
    nrows = int(np.ceil(len(snapshots) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 4.6 * nrows), sharey=True)
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
                ax.bar(
                    x,
                    vals,
                    bottom=bottom,
                    width=0.6,
                    label=tier_labels.get(tier, tier),
                    color=tier_colors.get(tier, "#666666")
                )
                for i, v in enumerate(vals):
                    if v > 10:
                        ax.text(
                            x[i],
                            bottom[i] + v / 2,
                            f"{v:.0f}%",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=8,
                            fontweight="bold"
                        )
                bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(areas)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Khu vực phân tích", fontweight="bold", labelpad=6)
        ax.set_title(f"Thời điểm: {snap}", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx % ncols == 0:
            ax.set_ylabel("Tỷ trọng (%)", fontweight="bold")

    for j in range(len(snapshots), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Nhóm quy mô chủ nhà",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=True
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_review_concentration(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    pivot_df = df.pivot(index="year", columns="analysis_area", values="concentration_pct").fillna(0).sort_index()
    years = pivot_df.index.to_numpy()
    areas = sorted(pivot_df.columns.tolist())
    cmap = plt.get_cmap("tab10")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.4, 1]})

    # Biểu đồ trái: Stacked Area thể hiện tỷ trọng tập trung theo năm
    stacks = [pivot_df[a].to_numpy() for a in areas]
    ax1.stackplot(years, stacks, labels=areas, colors=[cmap(i % 10) for i in range(len(areas))], alpha=0.88)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(years.min(), years.max())
    ax1.set_ylabel("Tỷ trọng đánh giá (%)", fontweight="bold")
    ax1.set_xlabel("Năm", fontweight="bold")
    ax1.set_title("Cơ cấu thị phần Đánh giá theo Khu vực qua các năm", fontweight="bold", pad=12)
    ax1.legend(title="Khu vực", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Biểu đồ phải: Biểu đồ đường thể hiện tổng số đánh giá toàn thành phố qua các năm
    yearly_total = df[["year", "city_total_reviews"]].drop_duplicates().sort_values("year")
    x_years = yearly_total["year"].to_numpy()
    y_totals = yearly_total["city_total_reviews"].to_numpy()

    ax2.plot(x_years, y_totals, color="#3D405B", marker="o", linewidth=2.2, markersize=6)
    ax2.set_xticks(x_years)
    ax2.set_xticklabels(x_years.astype(str), rotation=45, ha="right")
    ax2.set_ylim(0, y_totals.max() * 1.18 if len(y_totals) > 0 else 10)
    ax2.set_ylabel("Tổng số đánh giá toàn thành phố", fontweight="bold")
    ax2.set_xlabel("Năm", fontweight="bold")
    ax2.set_title("Tổng lượng Đánh giá toàn thành phố qua các năm", fontweight="bold", pad=12)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Gắn nhãn giá trị lên từng điểm trên biểu đồ đường
    for x_val, y_val in zip(x_years, y_totals):
        ax2.text(
            x_val, y_val + (y_totals.max() * 0.025),
            f"{int(y_val):,}",
            ha="center", va="bottom", fontsize=8, color="#000000", fontweight="bold"
        )

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_revpar(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df_area = df[(df["room_type"] == "All") & (df["analysis_area"] != "All")].sort_values("analysis_area").copy()
    df_room = df[(df["analysis_area"] == "All") & (df["room_type"] != "All")].copy()

    # Ánh xạ tên loại phòng sang tiếng Việt
    room_type_map = {
        "Entire home/apt": "Cả căn nhà",
        "Private room": "Phòng riêng tư",
    }
    df_room["room_type_vi"] = df_room["room_type"].replace(room_type_map)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16.5, 5.8), gridspec_kw={"width_ratios": [2.4, 1.15]}
    )

    # --- BIỂU ĐỒ 1 (TRÁI): Cột nhóm 3 chỉ số (RevPAR, ADR, Tỷ lệ lấp đầy) theo Khu vực ---
    x = np.arange(len(df_area))
    width = 0.26

    # Trục Y trái dùng chung thang đo tiền tệ ($) cho cả RevPAR và ADR
    max_dollar_1 = max(df_area["revpar"].max(), df_area["adr"].max())
    bars1_revpar = ax1.bar(x - width, df_area["revpar"], width=width, color="#2A9D8F", label="RevPAR ($)")
    bars1_adr = ax1.bar(x, df_area["adr"], width=width, color="#3D405B", label="ADR ($)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(df_area["analysis_area"])
    ax1.set_xlabel("Khu vực phân tích", fontweight="bold", labelpad=8)
    ax1.set_ylabel("Giá trị ($)", color="#3D405B", fontweight="bold")
    ax1.set_ylim(0, max_dollar_1 * 1.25)
    ax1.set_title(f"RevPAR (\\$), ADR (\\$) và Tỷ lệ lấp đầy (%) theo Khu vực", fontweight="bold", pad=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)

    for bar in bars1_revpar:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + max_dollar_1 * 0.015, f"${h:.1f}", ha="center", va="bottom", fontsize=7, color="#2A9D8F", fontweight="bold")

    for bar in bars1_adr:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + max_dollar_1 * 0.015, f"${h:.0f}", ha="center", va="bottom", fontsize=7, color="#3D405B", fontweight="bold")

    # Trục Y phải vẽ cột Tỷ lệ lấp đầy (%)
    ax1_twin = ax1.twinx()
    max_occ_1 = df_area["occupancy_rate_%"].max()
    bars1_occ = ax1_twin.bar(x + width, df_area["occupancy_rate_%"], width=width, color="#E76F51", label="Tỷ lệ lấp đầy (%)")
    ax1_twin.set_ylabel("Tỷ lệ lấp đầy (%)", color="#E76F51", fontweight="bold")
    ax1_twin.set_ylim(0, max_occ_1 * 1.25)
    ax1_twin.spines["top"].set_visible(False)

    for bar in bars1_occ:
        h = bar.get_height()
        ax1_twin.text(bar.get_x() + bar.get_width() / 2, h + max_occ_1 * 0.015, f"{h:.1f}%", ha="center", va="bottom", fontsize=7, color="#E76F51", fontweight="bold")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    # --- BIỂU ĐỒ 2 (PHẢI): Cột nhóm 3 chỉ số (RevPAR, ADR, Tỷ lệ lấp đầy) theo Loại phòng ---
    x_r = np.arange(len(df_room))
    width_r = 0.24

    max_dollar_2 = max(df_room["revpar"].max(), df_room["adr"].max())
    bars2_revpar = ax2.bar(x_r - width_r, df_room["revpar"], width=width_r, color="#2A9D8F", label="RevPAR ($)")
    bars2_adr = ax2.bar(x_r, df_room["adr"], width=width_r, color="#3D405B", label="ADR ($)")

    ax2.set_xticks(x_r)
    ax2.set_xticklabels(df_room["room_type_vi"])
    ax2.set_xlabel("Loại phòng", fontweight="bold", labelpad=8)
    ax2.set_ylabel("Giá trị ($)", color="#3D405B", fontweight="bold")
    ax2.set_ylim(0, max_dollar_2 * 1.25)
    ax2.set_title(f"RevPAR (\\$), ADR (\\$) và Tỷ lệ lấp đầy (%) theo Loại phòng", fontweight="bold", pad=12)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False)

    for bar in bars2_revpar:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + max_dollar_2 * 0.015, f"${h:.1f}", ha="center", va="bottom", fontsize=7.5, color="#2A9D8F", fontweight="bold")

    for bar in bars2_adr:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + max_dollar_2 * 0.015, f"${h:.0f}", ha="center", va="bottom", fontsize=7.5, color="#3D405B", fontweight="bold")

    # Trục Y phải vẽ cột Tỷ lệ lấp đầy (%) cho Loại phòng
    ax2_twin = ax2.twinx()
    max_occ_2 = df_room["occupancy_rate_%"].max()
    bars2_occ = ax2_twin.bar(x_r + width_r, df_room["occupancy_rate_%"], width=width_r, color="#E76F51", label="Tỷ lệ lấp đầy (%)")
    ax2_twin.set_ylabel("Tỷ lệ lấp đầy (%)", color="#E76F51", fontweight="bold")
    ax2_twin.set_ylim(0, max_occ_2 * 1.25)
    ax2_twin.spines["top"].set_visible(False)

    for bar in bars2_occ:
        h = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width() / 2, h + max_occ_2 * 0.015, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, color="#E76F51", fontweight="bold")

    h_r1, l_r1 = ax2.get_legend_handles_labels()
    h_r2, l_r2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(h_r1 + h_r2, l_r1 + l_r2, loc="upper right", frameon=True)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_top_n_host_share(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["snapshot"] = df["snapshot"].astype(str)
    df = df.sort_values("snapshot")

    top_n = int(df["top_n"].iloc[0]) if "top_n" in df.columns else 10
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    bars = ax1.bar(x, df["top_n_share_%"], width=0.48, color="#4C72B0", label=f"Thị phần Top {top_n} (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["snapshot"])
    ax1.set_xlabel("Thời điểm thu thập (Snapshot)", fontweight="bold", labelpad=8)
    ax1.set_ylabel(f"Thị phần của Top {top_n} chủ nhà (%)", color="#4C72B0", fontweight="bold")
    ax1.set_ylim(0, max(df["top_n_share_%"].max() * 1.3, 15))
    ax1.set_title(f"Thị phần của Top {top_n} chủ nhà qua các thời điểm", fontweight="bold", pad=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)

    for bar, (_, row) in zip(bars, df.iterrows()):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, h + 0.25,
            f"{h:.2f}%\n({int(row['top_n_listings'])}/{int(row['total_listings'])} phòng)",
            ha="center", va="bottom", fontsize=8.5
        )

    ax2 = ax1.twinx()
    ax2.plot(x, df["total_hosts"], color="#E76F51", marker="s", linewidth=2.2, label="Tổng số chủ nhà")
    ax2.set_ylabel("Tổng số chủ nhà duy nhất", color="#E76F51", fontweight="bold")
    ax2.set_ylim(0, df["total_hosts"].max() * 1.25)
    ax2.spines["top"].set_visible(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    # Đọc tham số --config từ dòng lệnh bash
    """
    parser = argparse.ArgumentParser(description="Visualize space, time and LLM from YAML config")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Đường dẫn tới file cấu hình YAML"
    )
    """
    #args = parser.parse_args()

    # Load Overpriced Analysis data
    #print("Tạo biểu đồ cột cho overprice")
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

    label_csv = csv_folder / "kpi_label_distribution.csv"
    label_df = csv_to_df(label_csv)
    label_output_path = output_folder / "label_distribution.png"
    plot_label_distribution(label_df, label_output_path)

    prof_host_csv = csv_folder / "kpi_professional_host_share.csv"
    prof_host_df = csv_to_df(prof_host_csv)
    prof_host_output_path = output_folder / "professional_host_share.png"
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

# Điểm bắt đầu khi chạy lệnh `python -m src.viz`
if __name__ == "__main__":
    main()