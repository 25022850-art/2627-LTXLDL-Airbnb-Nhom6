from pathlib import Path

import pandas as pd
import llm as LLM

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def read_reviews(path):
   return pd.read_csv(path, encoding="utf-8", dtype={"id": str, "listing_id": str, "comments": str}) 


def exec_llm(config, config_path):
   snapshot = config.get("snapshots")[0]
   file_path = PROCESSED_DIR / snapshot / "reviews.csv.gz"
   reviews = read_reviews(file_path)
   reviews_to_label = reviews.loc[reviews["Mẫu LLM"].str.strip().str.lower()=='x', ["id", "comments"]]
   out = LLM.llm(reviews_to_label, config_path)
    
    
        

