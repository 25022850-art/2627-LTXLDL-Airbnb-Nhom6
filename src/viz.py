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

        # 2. Khung viền (Spines) & Ticks: bỏ trên/phải, làm dịu trái/dưới
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#9A9A9A",
        "xtick.color": "#9A9A9A",         # màu vạch tick
        "ytick.color": "#9A9A9A",
        "xtick.labelcolor": "#222222",    # chữ tick giữ đậm để dễ đọc
        "ytick.labelcolor": "#222222",
        "xtick.major.size": 3,
        "ytick.major.size": 3,

        # 3. Lưới (Grid): mảnh, nhạt, nằm dưới dữ liệu
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": "#E4E4E4",
        "grid.linestyle": "-",
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,

        # 3b. Legend: nền trắng mờ, bỏ viền đen/xám
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "none",
        "legend.framealpha": 0.9,

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
def add_main_title(fig, title, y=0.985):
    fig.text(
        0.5, y,
        title,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#111111"
    )

def plot_overpriced_analysis(df: pd.DataFrame, output_path: Path):
    df_area = df[(df["room_type"] == "All") & (df["analysis_area"] != "All")].copy()
    df_room = df[(df["analysis_area"] == "All") & (df["room_type"] != "All")].copy()

    if "overpriced_rate_%" not in df.columns:
        raise KeyError(f"Không thấy cột 'overpriced_rate_%'. Các cột hiện có: {df.columns.tolist()}")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.4, 1]}
    )

    add_main_title(
        fig,
        "A08 records the highest overpriced rate (37.1%), while private rooms are more frequently overpriced than entire homes."
    )
    # --- BIỂU ĐỒ 1 (TRÁI): Combo Bar (Overpriced %) + Line (Median Price) theo Khu vực ---
    x = np.arange(len(df_area))
    bars = ax1.bar(
        x, df_area["overpriced_rate_%"],
        width=0.55, color="#4C72B0", alpha=0.92, label="Overpriced Rate (%)"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_area["analysis_area"])
    ax1.set_ylabel("Overpriced Rate (%)", color="#4C72B0")
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
        color="#E76F51", marker="o", label="Median Price (€)"
    )
    ax1_twin.set_ylabel("Area Median Price (€)", color="#E76F51")
    ax1_twin.set_ylim(0, df_area["group_median_price"].max() * 1.2)
    ax1_twin.spines["right"].set_visible(True)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True)

    # --- BIỂU ĐỒ 2 (PHẢI): So sánh theo Loại phòng (Room Type) ---
    x_room = np.arange(len(df_room))
    bars_room = ax2.bar(
        x_room, df_room["overpriced_rate_%"],
        width=0.45, color=["#4C72B0", "#8FB3E0"]
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
            f"{h:.1f}%\n(Med: €{row['group_median_price']:.0f})\n(Listing Count: {int(row['overpriced_listings'])})",
            ha="center", va="bottom", fontsize=8.5
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.87])
    fig.text(
        0.5, 0.012,
        "Note: Overpriced Rate = Overpriced / Premium listings (Price ≥ 1.25 × group median). "
        "Overpriced: Premium listings with review_scores_value < 4.6.",
        ha="center", va="bottom", fontsize=9, style="italic", color="#555555"
    )
    # print(f"Đang lưu biểu đồ phân tích Overpriced vào: {output_path}")
    save_figure(fig, output_path)



def plot_booking_curve(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    months = sorted(df["target_month"].astype(str).unique())
    areas = sorted(a for a in df["analysis_area"].unique() if a != "All")
    ncols = len(months)
    month_label = {m: pd.Period(m, freq="M").strftime("%b %Y") for m in months}

    fig, axes = plt.subplots(
        2, ncols, figsize=(7.4 * ncols, 9.6), squeeze=False,
        gridspec_kw={"height_ratios": [1.25, 1]},
    )
    add_main_title(
    fig,
    "Unavailability rises sharply near stay dates, reaching 57.0% in July and 46.1% in August.",
    y=0.985
    )

    # Trục y dùng chung cho hàng trên
    y_top = float(np.ceil(df["unavailability_rate_%"].max() * 1.1 / 10) * 10)
    cmap_lines = plt.get_cmap("tab10")

    # ---------- Hàng trên: đường cong theo lead time ----------
    for c, m in enumerate(months):
        ax = axes[0, c]
        sub = df[df["target_month"].astype(str) == m]
        leads = sorted(sub["lead_days"].unique(), reverse=True)
        snap_of_lead = sub.drop_duplicates("lead_days").set_index("lead_days")["snapshot"].astype(str)

        for i, a in enumerate(areas):
            s = sub[sub["analysis_area"] == a].sort_values("lead_days", ascending=False)
            ax.plot(s["lead_days"], s["unavailability_rate_%"], marker="o", markersize=4.5,
                    linewidth=1.6, color=cmap_lines(i % 10), label=a, alpha=0.9)

        # Toàn thành phố của CHÍNH tháng này 
        s_all = sub[sub["analysis_area"] == "All"].sort_values("lead_days", ascending=False)
        ax.plot(s_all["lead_days"], s_all["unavailability_rate_%"], color="#222222", linestyle="--",
                linewidth=2.2, marker="s", markersize=5.5, label="Citywide", zorder=5)
        for _, r in s_all.iterrows():
            ax.annotate(f"{r['unavailability_rate_%']:.1f}%", (r["lead_days"], r["unavailability_rate_%"]),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5,
                        fontweight="bold", color="#222222",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=6)

        ax.set_xticks(leads)
        ax.set_xticklabels([f"{int(l)}d\n{snap_of_lead[l]}" for l in leads])
        ax.set_xlim(max(leads) + 12, min(leads) - 12)  # đảo trục: thời gian chạy về phía ngày ở
        ax.set_ylim(0, y_top)
        ax.set_yticks(np.arange(0, y_top + 1, 10))
        ax.set_xlabel("Days before the 1st of the month (snapshot date)")
        if c == 0:
            ax.set_ylabel("Unavailability Rate (%)")
        ax.set_title(f"{month_label[m]} stays")
        ax.grid(True, axis="y")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.925), frameon=False)

    # ---------- Hàng dưới: pickup giữa các snapshot (khu vực x khoảng thời gian) ----------
    pk = df.dropna(subset=["pickup_pp"])
    vmax = float(np.nanmax(np.abs(pk["pickup_pp"]))) if len(pk) else 1.0
    has_neg = bool((pk["pickup_pp"] < 0).any())
    if has_neg:
        from matplotlib.colors import TwoSlopeNorm
        cmap_h, norm = plt.get_cmap("RdBu_r"), TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        from matplotlib.colors import Normalize
        cmap_h, norm = plt.get_cmap("YlOrRd"), Normalize(vmin=0, vmax=vmax)

    im = None
    row_order = areas + ["All"]
    for c, m in enumerate(months):
        ax = axes[1, c]
        sub = df[df["target_month"].astype(str) == m].sort_values(["analysis_area", "lead_days"], ascending=[True, False])
        pv = sub.pivot(index="analysis_area", columns="lead_days", values="pickup_pp").reindex(row_order)
        cols = sorted(pv.columns, reverse=True)[1:]  # bỏ snapshot đầu (không có pickup)
        pv = pv[cols]
        all_leads = sorted(sub["lead_days"].unique(), reverse=True)
        col_labels = [f"{int(all_leads[i])}d → {int(all_leads[i + 1])}d" for i in range(len(all_leads) - 1)]

        mat = pv.to_numpy(dtype=float)
        im = ax.imshow(mat, cmap=cmap_h, norm=norm, aspect="auto")
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels([("Citywide" if r == "All" else r) for r in row_order])
        ax.axhline(len(areas) - 0.5, color="white", linewidth=3)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isnan(val):
                    dark = norm(val) > 0.7 or (has_neg and norm(val) < 0.25)
                    ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8.5,
                            color="white" if dark else "#222222", fontweight="bold" if row_order[i] == "All" else "normal")
        ax.set_xlabel("Lead-time interval (days before the 1st of the month)")
        ax.set_title(f"{month_label[m]}: pickup between snapshots (pp)")

    cax = fig.add_axes([0.925, 0.12, 0.012, 0.3])  # thanh màu đặt cố định bên phải hàng dưới
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Change in unavailability (pp)")

    n_txt = ", ".join(
        f"{month_label[m]}: n = {int(df[(df['target_month'].astype(str) == m) & (df['analysis_area'] == 'All')]['n_listings'].max()):,}"
        for m in months
    )
    fig.text(
        0.5, 0.0,
        f"Same listings in every snapshot ({n_txt}). Unavailable = booked or blocked by host. "
        "Lead time is measured to the 1st of the month.",
        ha="center", va="top", fontsize=9, style="italic", color="#555555",
    )
    fig.subplots_adjust(
        top=0.83,
        bottom=0.08,
        left=0.07,
        right=0.9,
        hspace=0.6,
        wspace=0.18
    )  # tránh nhãn trục đè tiêu đề hàng dưới
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
    add_main_title(
    fig,
    "Noise complaints show the strongest association with non-positive reviews (10.07x baseline risk)."
    )
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

    plt.tight_layout(rect=[0, 0, 1, 0.87])
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
    add_main_title(
    fig,
    "Private rooms consistently show higher dormant rates than entire homes, reaching 20.8% in June 2026."
)

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
    plt.tight_layout(rect=[0, 0.16, 1, 0.87])
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

    add_main_title(
    fig,
    "Positive sentiment dominates all areas, while noise-related experiences show the greatest variation."
)
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

    plt.tight_layout(rect=[0, 0.06, 1, 0.87])
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
    tier_colors = {"individual": "#8FB3E0", "small_multi": "#4C72B0", "professional": "#1B3B6F"}

    ncols = 2
    nrows = int(np.ceil(len(snapshots) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), sharey=True)
    add_main_title(
        fig,
        "Host composition remains broadly stable, with small multi-listing hosts holding the largest share in most areas.",
        y=0.985
    )
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

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Host Tier", loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.84])
    save_figure(fig, output_path)


def plot_review_concentration(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df[(df["year"] >= 2015) & (df["year"] <= 2025)]

    pivot_df = df.pivot(index="year", columns="analysis_area", values="concentration_pct").fillna(0).sort_index()
    years = pivot_df.index.to_numpy()
    areas = sorted(pivot_df.columns.tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.4, 1]})
    add_main_title(
    fig,
    "Review volume peaked in 2024 (911 reviews) before declining in 2025, with A02 contributing a substantial share."
    )

    # Biểu đồ trái: Heatmap
    mat = pivot_df[areas].T.to_numpy()
    vmax = float(mat.max()) if mat.size else 1.0
    im = ax1.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)

    ax1.set_xticks(np.arange(len(years)))
    ax1.set_xticklabels([str(int(y)) for y in years])
    ax1.set_yticks(np.arange(len(areas)))
    ax1.set_yticklabels(areas)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Area", fontweight="bold")
    ax1.set_title("Review Share Distribution by Area Over Time", fontweight="bold", pad=12)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax1.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                     color="white" if v > vmax * 0.55 else "#222222")

    cbar = fig.colorbar(im, ax=ax1, pad=0.02, fraction=0.04)
    cbar.set_label("Review Share (%)", fontweight="bold")
    for sp in ax1.spines.values():
        sp.set_visible(False)
    ax1.tick_params(length=0)

    # Biểu đồ phải: Bar chart chuyển sang xanh biển
    yearly_total = df[["year", "city_total_reviews"]].drop_duplicates().sort_values("year")
    bars = ax2.bar(
        yearly_total["year"].astype(str),
        yearly_total["city_total_reviews"],
        color="#4C72B0",
        width=0.65
    )

    max_val = yearly_total["city_total_reviews"].max()
    ax2.set_ylim(0, max_val * 1.12)

    for bar in bars:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + max_val * 0.015,
            f"{int(h):,}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#222222"
        )

    ax2.set_ylabel("City Total Reviews")
    ax2.set_xlabel("Year")
    ax2.set_title("City Total Reviews")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.87])
    save_figure(fig, output_path)



def plot_revpar(df: pd.DataFrame, output_path: str | Path):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df_area = df[(df["room_type"] == "All") & (df["analysis_area"] != "All")].sort_values("analysis_area")
    df_room = df[(df["analysis_area"] == "All") & (df["room_type"] != "All")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [2.4, 1]})
    add_main_title(
    fig,
    "A03 leads in RevPAR ($55.0) and occupancy (35%), while entire homes generate higher revenue than private rooms."
)
    Y_STEP = 10
    y_top = float(np.ceil(max(df_area["revpar"].max(), df_room["revpar"].max()) * 1.1 / Y_STEP) * Y_STEP)

    x = np.arange(len(df_area))
    bars = ax1.bar(x, df_area["revpar"], width=0.65, color="#4C72B0", label="RevPAR (€)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_area["analysis_area"])
    ax1.set_ylabel("RevPAR (€)", color="#4C72B0")
    ax1.set_ylim(0, y_top)
    ax1.set_title("RevPAR (€) & Occupancy Rate (%) by Area")
    ax1.grid(axis="y")

    for bar, (_, row) in zip(bars, df_area.iterrows()):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, y_top * 0.015,
            f"€{h:.1f}\n(ADR €{row['adr']:.0f})",
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
    bars_r = ax2.bar(x_r, df_room["revpar"], width=0.45, color=["#4C72B0", "#8FB3E0"])
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
            f"€{h:.2f}\n(ADR €{row['adr']:.0f} | Occ {row['occupancy_rate_%']:.1f}%)",
            ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout(rect=[0, 0, 1, 0.87])
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
    add_main_title(
    fig,
    "Market concentration remains stable, with the top 10 hosts consistently controlling around 10% of listings."
)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["snapshot"])
    ax1.set_ylabel(f"Top {top_n} Host Market Share (%)", color="#4C72B0")
    ax1.set_ylim(0, max(df["top_n_share_%"].max() * 1.3, 15))
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

    for xi, val in zip(x, df["total_hosts"]):
        ax2.annotate(
            f"{int(val)}",
            (xi, val),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center", va="bottom",
            fontsize=9, color="#E76F51", fontweight="bold"
        )
        
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.87])
    save_figure(fig, output_path)


def main():
    setup_plot_style()

    print("Tạo biểu đồ cột cho overprice")
    plot_overpriced_analysis(
        csv_to_df(csv_folder / "kpi_overpriced_rate.csv"),
        output_folder / "overpriced_analysis.png",
    )
    plot_booking_curve(
        csv_to_df(csv_folder / "kpi_booking_curve.csv"),
        output_folder / "booking_curve.png",
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