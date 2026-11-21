from pathlib import Path

import pandas as pd
from src.llm import llm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def read_reviews(path):
   return pd.read_csv(path, encoding="utf-8", dtype={"id": str, "listing_id": str, "comments": str}) 


def exec_llm(config, config_path):
   snapshot = config.get("snapshots")[0]
   rand_seed = config.get("seed")
   sample_size = config.get("llm").get("sample_size")

   file_path = PROCESSED_DIR / str(snapshot) / "processed_reviews.csv"
   reviews = read_reviews(file_path)
   reviews_to_label = reviews.loc[reviews["Mẫu LLM"].str.strip().str.lower()=='x', ["id", "comments"]]

   remain_rv = reviews.loc[~reviews["id"].isin(reviews_to_label["id"]), ["id", "comments"]]
   num_of_extra_rv = min(max(sample_size - len(reviews_to_label), 0), len(remain_rv))
   extra_rv_to_label = remain_rv.sample(n=num_of_extra_rv, random_state=rand_seed)

   out = llm(pd.concat([reviews_to_label, extra_rv_to_label], ignore_index=True), config_path)