from pathlib import Path
import pandas as pd

try:
    from src.baseline import baseline
except ImportError:
    from baseline import baseline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "configs" / "porto.yaml"
INPUT_PATH = PROJECT_ROOT / "data" / "samples" / "rv_llm_baseline.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "baseline_output.csv"


def exec_baseline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rv_to_label = pd.read_csv(INPUT_PATH, encoding="utf-8")
    #rv_to_label["comments"] = rv_to_label["comments"].fillna("").astype(str)

    result_df = baseline(rv_to_label=rv_to_label, config_path=CONFIG_PATH)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Đã lưu kết quả baseline tại: {OUTPUT_PATH}")
    return result_df


if __name__ == "__main__":
    exec_baseline()