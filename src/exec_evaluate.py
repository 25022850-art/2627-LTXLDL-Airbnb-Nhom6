from pathlib import Path

import pandas as pd
from src.evaluate import evaluate

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "samples"

PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

llm_output_path = PROCESSED_DIR / "llm_output.csv"
llm_cache_path = PROCESSED_DIR / "llm_cache.jsonl"
llm_path = llm_output_path if llm_output_path.exists() else llm_cache_path

baseline_output_path = PROCESSED_DIR / "baseline_output.csv"
nhan_tay_path = SAMPLES_DIR / "nhan_tay.csv"

def exec_eval_baseline():
    nhan_tay = pd.read_csv(nhan_tay_path, encoding="utf-8", dtype={"id": str})
    preds = pd.read_csv(baseline_output_path, encoding="utf-8", dtype={"id": str})
    result = evaluate(preds, nhan_tay)
    return pd.DataFrame([result])

def exec_eval_llm():
    nhan_tay = pd.read_csv(nhan_tay_path, encoding="utf-8", dtype={"id": str})
    preds = pd.read_csv(llm_path, encoding="utf-8", dtype={"id": str})
    result = evaluate(preds, nhan_tay)
    return pd.DataFrame([result])

def exec_eval():
    baseline_score = exec_eval_baseline()
    baseline_score["predictor"] = "baseline"
    llm_score = exec_eval_llm()
    llm_score["predictor"] = "llm"
    result = pd.concat([baseline_score, llm_score], ignore_index=True)
    result.to_csv(REPORT_DIR / "f1_score.csv", index=False, encoding="utf-8")
