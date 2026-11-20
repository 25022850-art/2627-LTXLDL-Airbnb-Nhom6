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

def load_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def plot_grouped_bars(ax, df_sub: pd.DataFrame, group_col: str, snapshots: list, colors: list, title: str, xlabel: str, ylabel: str = ""):
    categories = sorted(df_sub[group_col].unique())
    x = np.arange(len(categories))
    n_snaps = len(snapshots)
    
    # Tính độ rộng mỗi cột dựa trên số lượng mốc snapshot
    total_width = 0.78
    bar_width = total_width / max(n_snaps, 1)

    for i, snap in enumerate(snapshots):
        snap_df = df_sub[df_sub["snapshot"] == snap].set_index(group_col)
        # Reindex để đảm bảo đủ danh mục và giữ đúng thứ tự trục X (điền 0 nếu thiếu)
        values = snap_df.reindex(categories)["dormant_rate_%"].fillna(0).to_numpy()
        
        offset = (i - (n_snaps - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width * 0.92,
            label=str(snap),
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.7
        )

        # Hiển thị giá trị % trên đầu mỗi cột
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333"
                )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)

    # Tăng giới hạn trục Y thêm 15% để chữ trên đỉnh cột không chạm viền trên
    max_val = df_sub["dormant_rate_%"].max()
    ax.set_ylim(0, max_val * 1.18 if pd.notna(max_val) else 25)

    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def visualize(df: pd.DataFrame):
    pass

def plot_overpriced_analysis(df: pd.DataFrame, output_path: Path):
    # 1. Tách 2 lát cắt dữ liệu
    df_area = df[(df["room_type"] == "All") & (df["analysis_area"] != "All")].copy()
    df_room = df[(df["analysis_area"] == "All") & (df["room_type"] != "All")].copy()

    if "overpriced_rate_%" not in df.columns:
        raise KeyError(f"Không thấy cột 'overpriced_rate_%'. Các cột hiện có: {df.columns.tolist()}")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.4, 1]}
    )

    # --- BIỂU ĐỒ 1 (TRÁI): Combo Bar (Overpriced %) + Line (Median Price) theo Khu vực ---
    x = np.arange(len(df_area))
    bars = ax1.bar(
        x, df_area["overpriced_rate_%"],
        width=0.55, color="#E07A5F", alpha=0.88, label="Overpriced Rate (%)"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_area["analysis_area"])
    ax1.set_ylabel("Overpriced Rate (%)", color="#E07A5F", fontweight="bold")
    ax1.set_ylim(0, df_area["overpriced_rate_%"].max() * 1.2)
    ax1.set_title("Tỷ lệ Overpriced (%) vs Giá trung vị theo Khu vực", fontweight="bold", pad=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)

    # Gắn nhãn % lên cột
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.1f}%", ha="center", fontsize=8.5)

    # Trục Y thứ 2 vẽ đường Median Price
    ax1_twin = ax1.twinx()
    line = ax1_twin.plot(
        x, df_area["area_median_price"],
        color="#3D405B", marker="o", linewidth=2.2, label="Median Price ($)"
    )
    ax1_twin.set_ylabel("Area Median Price ($)", color="#3D405B", fontweight="bold")
    ax1_twin.set_ylim(0, df_area["area_median_price"].max() * 1.2)
    ax1_twin.spines["top"].set_visible(False)

    # Gộp legend của 2 trục
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
    ax2.set_title("Overpriced (%) theo Loại phòng", fontweight="bold", pad=12)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Ghi cả % và Median Price lên đầu cột
    for bar, (_, row) in zip(bars_room, df_room.iterrows()):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2, h + 0.6,
            f"{h:.1f}%\n(Med: ${row['area_median_price']:.0f})\n()",
            ha="center", va="bottom", fontsize=8.5
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Đang lưu biểu đồ phân tích Overpriced vào: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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

    # Chạy luồng xử lý
    #cfg = load_config(args.config) 
    df = load_csv(csv_folder / "kpi_overpriced_rate.csv")
    plot_overpriced_analysis(df, output_folder / "overpriced_analysis.png")

# Điểm bắt đầu khi chạy lệnh `python -m src.viz`
if __name__ == "__main__":
    main()