from pathlib import Path
import pandas as pd

from src.baseline import baseline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "baseline_output.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def read_reviews(path):
   return pd.read_csv(path, encoding="utf-8", dtype={"id": str, "listing_id": str, "comments": str}) 

def exec_baseline(config, config_path):
   snapshot = config.get("snapshots")[0]
   file_path = OUTPUT_DIR / str(snapshot) / "processed_reviews.csv"
   reviews = read_reviews(file_path)
   reviews_to_label = reviews.loc[reviews["Mẫu LLM"].str.strip().str.lower()=='x', ["id", "comments"]]
   out = baseline(reviews_to_label, config_path)
   out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")